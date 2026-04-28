#!/usr/bin/env python3
"""
main_seld_only.py — Resumable StandaloneSELD Training Pipeline
==============================================================
Trains ``StandaloneSELDModel`` (dual SED+DOA branches, no GP-AT backbone).

Supported dataset modes (set via ``training.dataset_mode`` in the YAML):
  * STARSS23       — SELDDataModule, trains on STARSS23
  * TAU-NIGENS2021 — SELDDataModule, trains on TAU-NIGENS2021
  * MixedSELD      — FinetuneDataModule, trains on STARSS23 + TAU-NIGENS2021

Test is ALWAYS run on BOTH STARSS23 and TAU-NIGENS2021 via a dedicated
two-loader test DataModule (``StandaloneSELDTestDataModule``), regardless
of the training dataset.

Warm-start from any NAS-SELD checkpoint is supported via ``--pretrained-ckpt``.
The checkpoint is automatically filtered (GP-AT keys dropped, spatial weights
duplicated to both branches) by ``filter_checkpoint``.

Directory layout after a run
-----------------------------
::

    runs/standalone_seld/
    ├── experiment_state.json
    ├── checkpoints/
    │   └── seld_starss23/
    │       ├── 05_0.3412.ckpt
    │       └── last.ckpt
    └── metrics/
        └── seld_starss23.json

Usage
-----
::

    # Debug: 1 epoch, 5 batches
    .venv/bin/python main_seld_only.py \\
        --config-file configs/experiments_standalone_seld/foa/seld_starss23.yaml \\
        --epochs 1 --limit-batches 5

    # Warm-start from best NAS checkpoint
    .venv/bin/python main_seld_only.py \\
        --config-file configs/experiments_standalone_seld/foa/seld_starss23.yaml \\
        --pretrained-ckpt runs/foa/deep_spec/checkpoints/exp_e3_l1_s1/98_0.3570.ckpt

    # MixedSELD with custom data paths
    .venv/bin/python main_seld_only.py \\
        --config-file configs/experiments_standalone_seld/foa/seld_mixed.yaml \\
        --starss-root /data/STARSS23 --tau-root /data/TAU-NIGENS2021

    # Run all configs in a directory
    .venv/bin/python main_seld_only.py \\
        --config-dir configs/experiments_standalone_seld/foa/

    # List status
    .venv/bin/python main_seld_only.py --list

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

import atexit
import json
import signal
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Optional

warnings.filterwarnings('ignore', message='.*LeafSpec.*deprecated.*')
warnings.filterwarnings('ignore', message='.*elements was resized.*')
warnings.filterwarnings('ignore', message='.*No positive samples.*')

import torch
import yaml
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_standalone_seld.train import StandaloneSELDModule  # noqa: E402
from datasets.datamodule import SELDDataModule, FinetuneDataModule    # noqa: E402
from datasets.starss23 import STARSS23Dataset                         # noqa: E402
from datasets.tau_nigens import TAUNIGENSDataset                      # noqa: E402
from datasets.utils import SELDDataset                                # noqa: E402
from dotenv import load_dotenv

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

SEED             = 42
VAL_FRACTION     = 0.2
EFFECTIVE_BATCH  = 32

_DEFAULT_EPOCHS      = 200
_DEFAULT_BATCH_SIZE  = 4
_DEFAULT_LR          = 3e-4
_DEFAULT_PATIENCE    = 30
_DEFAULT_NUM_WORKERS = 4
_DEFAULT_WINDOW_S    = 10.0
_DEFAULT_HOP_S       = 5.0

_DEFAULT_STARSS_ROOT = _ROOT / 'data' / 'STARSS23'
_DEFAULT_TAU_ROOT    = _ROOT / 'data' / 'TAU-NIGENS2021'

_DEFAULT_CONFIGS_DIR = _ROOT / 'configs' / 'experiments_standalone_seld' / 'foa'

CONFIGS_DIR = _DEFAULT_CONFIGS_DIR
CKPT_DIR    = _ROOT / 'runs' / 'standalone_seld' / 'checkpoints'
METRICS_DIR = _ROOT / 'runs' / 'standalone_seld' / 'metrics'
STATE_FILE  = _ROOT / 'runs' / 'standalone_seld' / 'experiment_state.json'


# ══════════════════════════════════════════════════════════════════════════════
# Test DataModule — always STARSS23 + TAU-NIGENS
# ══════════════════════════════════════════════════════════════════════════════

class StandaloneSELDTestDataModule(L.LightningDataModule):
    """
    Provides two test DataLoaders regardless of training dataset:
      idx=0 → STARSS23 test split
      idx=1 → TAU-NIGENS2021 test split
    """

    def __init__(
        self,
        starss_root:  Path,
        tau_root:     Path,
        model_sr:     int,
        hop_length:   int,
        n_tracks:     int,
        window_s:     Optional[float],
        hop_s:        Optional[float],
        batch_size:   int,
        num_workers:  int,
        audio_format: str = 'foa',
    ):
        super().__init__()
        self.starss_root  = starss_root
        self.tau_root     = tau_root
        self.model_sr     = model_sr
        self.hop_length   = hop_length
        self.n_tracks     = n_tracks
        self.window_s     = window_s
        self.hop_s        = hop_s
        self.batch_size   = batch_size
        self.num_workers  = num_workers
        self.audio_format = audio_format

        self._starss_ds: Optional[SELDDataset] = None
        self._tau_ds:    Optional[SELDDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        common = dict(
            split        = 'test',
            model_sr     = self.model_sr,
            hop_length   = self.hop_length,
            n_tracks     = self.n_tracks,
            window_s     = self.window_s,
            hop_s        = self.hop_s,
            audio_format = self.audio_format,
            augment      = None,
        )
        self._starss_ds = STARSS23Dataset(root=self.starss_root, **common)
        self._tau_ds    = TAUNIGENSDataset(root=self.tau_root,   **common)

    def _make_loader(self, ds: SELDDataset) -> DataLoader:
        collate = None if ds.windowed else SELDDataset.collate_fn
        return DataLoader(
            ds,
            batch_size         = self.batch_size,
            shuffle            = False,
            num_workers        = self.num_workers,
            collate_fn         = collate,
            pin_memory         = False,
            persistent_workers = self.num_workers > 0,
        )

    def test_dataloader(self) -> List[DataLoader]:
        return [self._make_loader(self._starss_ds),
                self._make_loader(self._tau_ds)]


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description='Resumable StandaloneSELD training pipeline.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Config / output paths ─────────────────────────────────────────────────
    loc = p.add_mutually_exclusive_group()
    loc.add_argument('--config-dir', type=Path, default=None,
                     metavar='DIR',
                     help='Directory with experiment YAMLs.')
    loc.add_argument('--config-file', type=Path, default=None,
                     metavar='PATH',
                     help='Path to a single experiment YAML.')

    p.add_argument('--run-dir', type=Path, default=None,
                   metavar='DIR',
                   help='Output root. Default: runs/standalone_seld/')

    # ── Data roots ────────────────────────────────────────────────────────────
    p.add_argument('--starss-root', type=Path, default=None,
                   metavar='PATH',
                   help=f'STARSS23 dataset root. Default: {_DEFAULT_STARSS_ROOT}')
    p.add_argument('--tau-root', type=Path, default=None,
                   metavar='PATH',
                   help=f'TAU-NIGENS2021 dataset root. Default: {_DEFAULT_TAU_ROOT}')

    # ── Warm-start ────────────────────────────────────────────────────────────
    p.add_argument('--pretrained-ckpt', type=Path, default=None,
                   metavar='PATH',
                   help='NAS-SELD checkpoint for warm-start (auto-filtered via '
                        'filter_checkpoint; keys not matching StandaloneSELD are dropped).')

    # ── Training hyperparams ──────────────────────────────────────────────────
    p.add_argument('--epochs',            type=int,   default=_DEFAULT_EPOCHS)
    p.add_argument('--batch-size',        type=int,   default=_DEFAULT_BATCH_SIZE)
    p.add_argument('--lr',                type=float, default=_DEFAULT_LR)
    p.add_argument('--lr-warmup-epochs',  type=int,   default=5,
                   help='Linear warmup epochs before cosine annealing kicks in.')
    p.add_argument('--patience',     type=int,   default=_DEFAULT_PATIENCE,
                   help='EarlyStopping patience in epochs (0 = disabled).')
    p.add_argument('--limit-batches', type=int,  default=0,
                   help='Limit train/val/test batches per epoch (0 = disabled).')
    p.add_argument('--window-s',     type=float, default=_DEFAULT_WINDOW_S)
    p.add_argument('--hop-s',        type=float, default=_DEFAULT_HOP_S)
    p.add_argument('--num-workers',  type=int,   default=_DEFAULT_NUM_WORKERS)
    p.add_argument('--accum-steps',  type=int,   default=0,
                   help='Gradient accumulation steps (0 = auto: ceil(16/batch_size)).')
    p.add_argument('--device',       type=str,   default=None,
                   metavar='ACCEL',
                   help='Force accelerator: cuda | mps | cpu. Default: auto-detect.')

    # ── Experiment selection ──────────────────────────────────────────────────
    p.add_argument('--exp',   type=str, nargs='+', metavar='NAME',
                   help='Run only these experiment(s) by name.')
    p.add_argument('--force', action='store_true',
                   help='Re-run experiment(s) even if already completed.')
    p.add_argument('--list',  action='store_true',
                   help='Print experiment status table and exit.')

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# State management
# ══════════════════════════════════════════════════════════════════════════════

def _load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as fh:
            return json.load(fh)
    return {}


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix('.tmp')
    with open(tmp, 'w') as fh:
        json.dump(state, fh, indent=2)
    tmp.replace(STATE_FILE)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec='seconds')


def _print_state_table(exp_names: list, state: dict) -> None:
    col = 36
    print(f'\n{"Experiment":<{col}}  {"Status":<12}  {"Info"}')
    print('─' * 75)
    for name in exp_names:
        s      = state.get(name, {})
        status = s.get('status', 'pending')
        info   = ''
        if status == 'completed':
            info = f"done {s.get('completed_at', '')}"
        elif status == 'running':
            info = f"interrupted {s.get('started_at', '')}"
        print(f'{name:<{col}}  {status:<12}  {info}')
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Config building
# ══════════════════════════════════════════════════════════════════════════════

_TRAINING_DEFAULTS = {
    'lr':               _DEFAULT_LR,
    'beta_sed':         0.6,
    'focal_gamma':      0.0,
    'pos_weight_max':   20.0,
    'angle_thresh':     20.0,
    'lr_warmup_epochs': 5,
    'lr_min':           1e-6,
}


def _build_cfg(exp_yaml_path: Path, args, starss_root: Path, tau_root: Path) -> dict:
    """
    Load experiment YAML and inject runtime training + data sections.

    Steps
    -----
    1. Load YAML.
    2. Inject training defaults (non-destructive: YAML values take precedence
       for all keys except epochs, batch_size, lr which are always CLI-driven).
    3. Inject data section from CLI (paths, windowing, workers).
    4. Set semantic.dataset for SELDDataModule compatibility.
    """
    with open(exp_yaml_path) as fh:
        cfg = yaml.safe_load(fh)

    # 1. Merge training defaults (YAML wins for existing keys)
    train_cfg = cfg.setdefault('training', {})
    for k, v in _TRAINING_DEFAULTS.items():
        train_cfg.setdefault(k, v)

    # 2. CLI always wins for these keys
    train_cfg['epochs']           = args.epochs
    train_cfg['batch_size']       = args.batch_size
    train_cfg['lr']               = args.lr
    train_cfg['lr_warmup_epochs'] = args.lr_warmup_epochs

    dataset_mode = train_cfg.get('dataset_mode', 'STARSS23')

    # 3. Data section
    window_s = args.window_s if args.window_s > 0 else None
    hop_s    = args.hop_s    if args.hop_s    > 0 else None
    data_cfg = cfg.setdefault('data', {})
    data_cfg.setdefault('format',       'foa')
    data_cfg['window_s']    = window_s
    data_cfg['hop_s']       = hop_s
    data_cfg['val_fraction'] = data_cfg.get('val_fraction', VAL_FRACTION)
    data_cfg['val_seed']    = data_cfg.get('val_seed', SEED)
    data_cfg['num_workers'] = args.num_workers

    if dataset_mode == 'STARSS23':
        data_cfg['root_path'] = str(starss_root)
    elif dataset_mode == 'TAU-NIGENS2021':
        data_cfg['root_path'] = str(tau_root)
    elif dataset_mode == 'MixedSELD':
        data_cfg['starss_root'] = str(starss_root)
        data_cfg['tau_root']    = str(tau_root)
    else:
        raise ValueError(f'Unknown dataset_mode: {dataset_mode!r}. '
                         'Expected STARSS23 | TAU-NIGENS2021 | MixedSELD.')

    # 4. semantic.dataset — needed by SELDDataModule
    sem_dataset = 'STARSS23' if dataset_mode in ('STARSS23', 'MixedSELD') else 'TAU-NIGENS2021'
    cfg.setdefault('semantic', {})['dataset'] = sem_dataset

    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint resume helper
# ══════════════════════════════════════════════════════════════════════════════

def _find_resume_ckpt(exp_dir: Path) -> Optional[str]:
    last = exp_dir / 'last.ckpt'
    if last.exists():
        return str(last)
    candidates = sorted(p for p in exp_dir.glob('*.ckpt') if p.name != 'last.ckpt')
    return str(candidates[-1]) if candidates else None


# ══════════════════════════════════════════════════════════════════════════════
# TensorBoard subprocess
# ══════════════════════════════════════════════════════════════════════════════

def _start_tensorboard(logdir: Path, port: int = 6008) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, '-m', 'tensorboard.main',
         '--logdir', str(logdir), '--port', str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atexit.register(proc.terminate)
    _orig_sigint = signal.getsignal(signal.SIGINT)

    def _sigint_handler(sig, frame):
        proc.terminate()
        signal.signal(signal.SIGINT, _orig_sigint)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint_handler)
    print(f'TensorBoard running → http://localhost:{port}  (PID {proc.pid})')
    return proc


# ══════════════════════════════════════════════════════════════════════════════
# Single experiment runner
# ══════════════════════════════════════════════════════════════════════════════

def _run_experiment(
    exp_name:        str,
    cfg:             dict,
    args,
    state:           dict,
    was_interrupted: bool,
    starss_root:     Path,
    tau_root:        Path,
) -> None:
    # 1. Mark running
    state[exp_name] = {'status': 'running', 'started_at': _now_iso()}
    _save_state(state)

    # 2. Directories + resume checkpoint
    exp_dir = CKPT_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    resume_ckpt = _find_resume_ckpt(exp_dir) if was_interrupted else None
    if resume_ckpt:
        print(f'  Resuming from: {resume_ckpt}')

    # 3. Module
    metrics_path = METRICS_DIR / f'{exp_name}.json'
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    module = StandaloneSELDModule(cfg)
    module.set_metrics_path(metrics_path)

    # 3a. Warm-start from NAS-SELD checkpoint (only when not resuming)
    if args.pretrained_ckpt and resume_ckpt is None:
        from experiments_standalone_seld.checkpoint_utils import filter_checkpoint
        print(f'  Warm-start from: {args.pretrained_ckpt}')
        filtered_sd = filter_checkpoint(args.pretrained_ckpt)
        missing, unexpected = module.model.load_state_dict(filtered_sd, strict=False)
        if unexpected:
            print(f'  [WARN] warm-start unexpected keys ({len(unexpected)}): '
                  + ', '.join(unexpected[:5]) + ('...' if len(unexpected) > 5 else ''))
        else:
            print(f'  warm-start: 0 unexpected keys')
        if missing:
            print(f'  warm-start missing keys ({len(missing)}): '
                  + ', '.join(missing[:5]) + ('...' if len(missing) > 5 else ''))

    # 3b. Restore metrics log when resuming an interrupted run
    if was_interrupted and metrics_path.exists():
        try:
            with open(metrics_path) as fh:
                module._epoch_metrics_log = json.load(fh)
            print(f'  Metrics log restored: {len(module._epoch_metrics_log)} entries')
        except Exception as exc:
            print(f'  [WARN] Could not restore metrics log: {exc}')

    # 4. Training DataModule
    dataset_mode = cfg['training']['dataset_mode']
    if dataset_mode in ('STARSS23', 'TAU-NIGENS2021'):
        train_dm = SELDDataModule(cfg)
    elif dataset_mode == 'MixedSELD':
        train_dm = FinetuneDataModule(cfg)
    else:
        raise ValueError(f'Unknown dataset_mode: {dataset_mode!r}')

    # 5. Test DataModule (always STARSS23 + TAU-NIGENS)
    in_cfg  = cfg['input']
    data_cfg = cfg.get('data', {})
    test_dm = StandaloneSELDTestDataModule(
        starss_root  = starss_root,
        tau_root     = tau_root,
        model_sr     = in_cfg['sr'],
        hop_length   = in_cfg.get('hop_length', 320),
        n_tracks     = cfg.get('heads', {}).get('n_tracks', 2),
        window_s     = data_cfg.get('window_s'),
        hop_s        = data_cfg.get('hop_s'),
        batch_size   = cfg['training']['batch_size'],
        num_workers  = data_cfg.get('num_workers', 4),
        audio_format = data_cfg.get('format', 'foa'),
    )

    # 6. Callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath                 = str(exp_dir),
        filename                = '{epoch:02d}_{val_seld_score:.4f}',
        monitor                 = 'val_seld_score',
        mode                    = 'min',
        save_top_k              = 1,
        save_last               = True,
        verbose                 = True,
        auto_insert_metric_name = False,
    )
    callbacks = [ckpt_cb]
    if args.patience > 0:
        callbacks.append(EarlyStopping(
            monitor  = 'val_seld_score',
            mode     = 'min',
            patience = args.patience,
            verbose  = True,
        ))

    # 7. Device
    if args.device is not None:
        accel, devices = args.device.lower(), 1
    elif torch.cuda.is_available():
        accel, devices = 'cuda', 1
    elif torch.backends.mps.is_available():
        accel, devices = 'mps', 1
    else:
        accel, devices = 'cpu', 1

    # 8. Gradient accumulation
    batch_size = cfg['training']['batch_size']
    if args.accum_steps > 0:
        accum = args.accum_steps
    elif batch_size >= EFFECTIVE_BATCH:
        accum = 1
    else:
        accum = EFFECTIVE_BATCH // batch_size

    eff_batch = batch_size * accum
    print(f'  batch_size={batch_size}  accum_steps={accum}  '
          f'effective_batch={eff_batch}'
          + ('  [no accumulation]' if accum == 1 else ''))

    # 9. Logger + Trainer
    tb_logger = TensorBoardLogger(
        save_dir = str(CKPT_DIR),
        name     = exp_name,
        version  = 0,
    )
    
    wb_logger = WandbLogger(
        project  = 'spatial-gpat',
        name     = exp_name,
        config   = cfg
    )

    trainer_kwargs = dict(
        max_epochs              = args.epochs,
        callbacks               = callbacks,
        accelerator             = accel,
        devices                 = devices,
        gradient_clip_val       = 1.0,
        accumulate_grad_batches = accum,
        log_every_n_steps       = 5,
        logger                  = [tb_logger, wb_logger],
        num_sanity_val_steps    = 0,
    )
    if args.limit_batches > 0:
        trainer_kwargs.update(
            limit_train_batches = args.limit_batches,
            limit_val_batches   = args.limit_batches,
            limit_test_batches  = args.limit_batches,
        )

    trainer = L.Trainer(**trainer_kwargs)

    # 10. Fit
    trainer.fit(module, train_dm, ckpt_path=resume_ckpt)

    # 11. Test on both STARSS23 + TAU-NIGENS
    trainer.test(module, test_dm, ckpt_path='best')
    wb_logger.experiment.finish()

    # 12. Remove last.ckpt
    last_path = exp_dir / 'last.ckpt'
    if last_path.exists():
        last_path.unlink()

    # 13. Mark completed
    state[exp_name] = {
        'status':       'completed',
        'best_ckpt':    ckpt_cb.best_model_path,
        'metrics_file': str(metrics_path),
        'completed_at': _now_iso(),
    }
    _save_state(state)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    global CONFIGS_DIR, CKPT_DIR, METRICS_DIR, STATE_FILE

    args = _parse_args()

    starss_root = (args.starss_root or _DEFAULT_STARSS_ROOT).resolve()
    tau_root    = (args.tau_root    or _DEFAULT_TAU_ROOT).resolve()

    # ── Resolve config dir and run dir ────────────────────────────────────────
    if args.config_file:
        exp_yaml   = args.config_file.resolve()
        if not exp_yaml.exists():
            print(f'[ERROR] Config file not found: {exp_yaml}')
            sys.exit(1)
        config_dir = exp_yaml.parent
        exp_names  = [exp_yaml.stem]
        run_dir    = args.run_dir or (_ROOT / 'runs' / 'standalone_seld' / exp_yaml.stem)
        STATE_FILE = run_dir / 'experiment_state.json'
    else:
        config_dir = (args.config_dir or _DEFAULT_CONFIGS_DIR).resolve()
        run_dir    = args.run_dir or (_ROOT / 'runs' / 'standalone_seld')
        STATE_FILE = run_dir / 'experiment_state.json'
        exp_names  = sorted(p.stem for p in config_dir.glob('*.yaml'))

    CONFIGS_DIR = config_dir
    CKPT_DIR    = run_dir / 'checkpoints'
    METRICS_DIR = run_dir / 'metrics'

    # ── Filter by --exp ───────────────────────────────────────────────────────
    if args.exp:
        requested = set(args.exp)
        exp_names = [n for n in exp_names if n in requested]
        unknown   = requested - set(exp_names)
        if unknown:
            print(f'[WARN] Unknown experiment name(s): {", ".join(sorted(unknown))}')

    state = _load_state()

    if args.list:
        _print_state_table(exp_names, state)
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print(f'Config dir  : {config_dir}')
    print(f'Run dir     : {run_dir}')
    print(f'STARSS23    : {starss_root}')
    print(f'TAU-NIGENS  : {tau_root}')
    if args.pretrained_ckpt:
        print(f'Pretrained  : {args.pretrained_ckpt}')

    _start_tensorboard(CKPT_DIR)

    for exp_name in exp_names:
        s      = state.get(exp_name, {})
        status = s.get('status')

        if status == 'completed' and not args.force:
            print(f'[SKIP] {exp_name} — already completed')
            continue

        was_interrupted = (status == 'running')

        if args.config_file:
            cfg = _build_cfg(args.config_file.resolve(), args, starss_root, tau_root)
        else:
            cfg = _build_cfg(config_dir / f'{exp_name}.yaml', args, starss_root, tau_root)

        L.seed_everything(cfg['experiment']['seed'], workers=True)

        print(f'\n{"=" * 60}')
        print(f'[RUN] {exp_name}  (mode={cfg["training"]["dataset_mode"]})')
        print(f'{"=" * 60}')

        _run_experiment(exp_name, cfg, args, state, was_interrupted, starss_root, tau_root)

    print('\nAll experiments finished.')


if __name__ == '__main__':
    load_dotenv()
    main()
