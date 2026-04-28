#!/usr/bin/env python3
"""
main_finetune.py — Phase 4 Fine-Tuning Pipeline
================================================
Runs fine-tuning experiments sequentially on the mixed STARSS23 + TAU-NIGENS2021
dataset using the best NAS architecture (exp_r3_t1_cs01).

Differences from main_nas-seld.py
-------------------------
  * DataModule : FinetuneDataModule (mixed STARSS23 + TAU-NIGENS2021 train;
                 STARSS23-only val/test for cross-phase metric comparability)
  * ``--pretrained-ckpt PATH`` : optional; load model weights only (fresh
                                 optimizer, epoch counter reset to 0).
                                 Default = fresh initialisation (same as main_nas-seld.py).
  * ``data.starss_root`` and ``data.tau_root`` injected separately (no --dataset).
  * Default run dir : runs/foa/finetune/
  * Default config  : configs/experiments_nas-seld/foa/finetune/
  * _DEFAULT_BATCH_SIZE = 1  (full-clip mode; effective batch = 16 via accum)
  * _DEFAULT_LR          = 5e-4
  * _DEFAULT_EPOCHS      = 100
  * LR scheduler: ReduceLROnPlateau (mode='min', monitor='val_seld_score',
                  factor=0.75, patience=20, min_lr=1e-5 — all configurable
                  via ``training:`` in the experiment YAML)

Config YAML notes
-----------------
Training-specific keys read from each experiment YAML::

    training:
      focal_gamma       : 0.0   # 0 = standard BCE; 2.0 = focal
      pos_weight_max    : 20.0  # cap on per-class pos_weight
      lr_reduce_factor  : 0.75  # ReduceLROnPlateau multiplicative factor
      lr_patience       : 20    # epochs without improvement before LR reduction
      lr_min            : 1e-5  # LR floor

The ``augmentation:`` block is read verbatim from the YAML (not merged from a
dataset config file) — enabled by default for fine-tuning.

Directory layout after a run
-----------------------------
::

    runs/foa/finetune/
    ├── experiment_state.json
    ├── checkpoints/
    │   └── exp_ft_mixed_bce/
    │       ├── epoch_12_0.3910.ckpt
    │       └── version_0/
    │           └── events.out.tfevents.*
    └── metrics/
        └── exp_ft_mixed_bce.json

Usage
-----
::

    # Step 1 — mixed + BCE, from scratch
    .venv/bin/python main_finetune.py \\
        --config-file configs/experiments_nas-seld/foa/finetune/exp_ft_mixed_bce.yaml

    # Step 2 — mixed + focal, warm-started from Phase 3 winner
    .venv/bin/python main_finetune.py \\
        --config-file configs/experiments_nas-seld/foa/finetune/exp_ft_mixed_focal.yaml \\
        --pretrained-ckpt runs/foa/deep_reg/checkpoints/exp_r3_t1_cs01/<best>.ckpt

    # Step 3 — full-clip
    .venv/bin/python main_finetune.py \\
        --config-file configs/experiments_nas-seld/foa/finetune/exp_ft_fullclip.yaml \\
        --pretrained-ckpt runs/foa/finetune/checkpoints/exp_ft_mixed_bce/<best>.ckpt

    # Debug: 1 epoch, 5 batches
    .venv/bin/python main_finetune.py \\
        --config-file configs/experiments_nas-seld/foa/finetune/exp_ft_mixed_bce.yaml \\
        --epochs 1 --limit-batches 5

    # List experiment statuses
    .venv/bin/python main_finetune.py --list

Author
------
Stefano Giacomelli, Ph.D. candidate — DISIM, University of L'Aquila
"""

import atexit
import json
import signal
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

warnings.filterwarnings('ignore', message='.*LeafSpec.*deprecated.*')
warnings.filterwarnings('ignore', message='.*elements was resized.*')
warnings.filterwarnings('ignore', message='.*No positive samples.*')

import torch
import yaml
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

# ── Project path setup ────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_nas_seld.train import NASLightningModule       # noqa: E402
from datasets.datamodule import FinetuneDataModule              # noqa: E402

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

SEED             = 42
VAL_FRACTION     = 0.2
EFFECTIVE_BATCH  = 16

_DEFAULT_EPOCHS      = 100
_DEFAULT_PATIENCE    = 30
_DEFAULT_BATCH_SIZE  = 1
_DEFAULT_LR          = 1e-3   # 5e-4 is optimal
_DEFAULT_NUM_WORKERS = 4
_DEFAULT_WINDOW_S    = 10.0   # 0 = full-clip mode
_DEFAULT_HOP_S       =  5.0   # 0 = non-overlapping windows; ignored in full-clip mode

# Dataset root paths (both used simultaneously in FinetuneDataModule)
STARSS23_ROOT = _ROOT / 'data' / 'STARSS23'
TAU_ROOT      = _ROOT / 'data' / 'TAU-NIGENS2021'

# Default config / output directories
_DEFAULT_CONFIGS_DIR = _ROOT / 'configs' / 'experiments_nas-seld' / 'foa' / 'finetune'
_FINETUNE_RUN_DIR    = _ROOT / 'runs' / 'foa' / 'finetune'

# Module-level directory vars (reassigned in main() based on args)
CONFIGS_DIR = _DEFAULT_CONFIGS_DIR
CKPT_DIR    = _FINETUNE_RUN_DIR / 'checkpoints'
METRICS_DIR = _FINETUNE_RUN_DIR / 'metrics'
STATE_FILE  = _FINETUNE_RUN_DIR / 'experiment_state.json'


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description='Phase 4 fine-tuning pipeline (STARSS23 + TAU-NIGENS2021).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Config / output paths ─────────────────────────────────────────────────
    loc = p.add_mutually_exclusive_group()
    loc.add_argument('--config-dir', type=Path, default=None, metavar='DIR',
                     help='Directory with experiment YAMLs.')
    loc.add_argument('--config-file', type=Path, default=None, metavar='PATH',
                     help='Path to a single experiment YAML.')

    p.add_argument('--run-dir', type=Path, default=None, metavar='DIR',
                   help='Output root (checkpoints, metrics, state). '
                        f'Default: {_FINETUNE_RUN_DIR.relative_to(_ROOT)}')

    # ── Pre-trained checkpoint ────────────────────────────────────────────────
    p.add_argument('--pretrained-ckpt', type=Path, default=None, metavar='PATH',
                   help='Optional pretrained checkpoint; model weights only are loaded '
                        '(fresh optimizer, epoch 0). Default: fresh initialisation.')

    # ── Training hyperparams ──────────────────────────────────────────────────
    p.add_argument('--epochs',       type=int,   default=_DEFAULT_EPOCHS)
    p.add_argument('--batch-size',   type=int,   default=_DEFAULT_BATCH_SIZE)
    p.add_argument('--lr',           type=float, default=_DEFAULT_LR)
    p.add_argument('--patience',     type=int,   default=_DEFAULT_PATIENCE,
                   help='EarlyStopping patience in epochs (0 = disabled).')
    p.add_argument('--limit-batches', type=int,  default=0,
                   help='Limit train/val/test batches per epoch (0 = disabled).')
    p.add_argument('--window-s',     type=float, default=_DEFAULT_WINDOW_S,
                   help='Window duration in seconds (0 = full-clip mode).')
    p.add_argument('--hop-s',        type=float, default=_DEFAULT_HOP_S,
                   help='Window hop in seconds.')
    p.add_argument('--num-workers',  type=int,   default=_DEFAULT_NUM_WORKERS)
    p.add_argument('--accum-steps',  type=int,   default=0,
                   help='Gradient accumulation steps (0 = auto: ceil(16/batch_size)).')
    p.add_argument('--device',       type=str,   default=None, metavar='ACCEL',
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

def _build_finetune_cfg(exp_yaml_path: Path, args) -> dict:
    """
    Load experiment YAML and inject runtime parameters.

    Steps
    -----
    1. Load YAML (contains architecture + augmentation + partial training keys).
    2. Build base training block from CLI defaults; override with any keys
       present in the YAML's ``training:`` section (focal_gamma, pos_weight_max
       …).  CLI ``batch_size`` always wins (re-asserted last).
    3. Inject ``data:`` runtime paths (starss_root, tau_root), windowing, split
       params, and num_workers.
    """
    with open(exp_yaml_path) as fh:
        cfg = yaml.safe_load(fh)

    # Base training block (CLI-driven defaults)
    base_training = {
        'lr':               args.lr,
        'epochs':           args.epochs,
        'batch_size':       args.batch_size,
        'lambda_spatial':   0.9,
        'lambda_semantic':  0.1,
        'beta_sed':         0.7,
        'angle_thresh':     20.0,
        'lr_reduce_factor': 0.75,   # ReduceLROnPlateau factor
        'lr_patience':      3,      # ReduceLROnPlateau patience (epochs)
        'lr_min':           1e-5,   # ReduceLROnPlateau LR floor
    }
    # YAML training section overrides defaults (focal_gamma, pos_weight_max, …)
    base_training.update(cfg.get('training', {}))
    # CLI batch_size always takes precedence
    base_training['batch_size'] = args.batch_size
    cfg['training'] = base_training

    # Data runtime params — both dataset roots are required
    window_s = args.window_s if args.window_s > 0 else None
    hop_s    = args.hop_s    if args.hop_s    > 0 else None
    cfg.setdefault('data', {}).update({
        'starss_root':  str(STARSS23_ROOT),
        'tau_root':     str(TAU_ROOT),
        'window_s':     window_s,
        'hop_s':        hop_s,
        'val_fraction': VAL_FRACTION,
        'val_seed':     SEED,
        'num_workers':  args.num_workers,
    })
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

def _start_tensorboard(logdir: Path, port: int = 6007) -> subprocess.Popen:
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
    exp_name: str,
    cfg: dict,
    args,
    state: dict,
    was_interrupted: bool,
) -> None:
    # 1. Mark running ──────────────────────────────────────────────────────────
    state[exp_name] = {'status': 'running', 'started_at': _now_iso()}
    _save_state(state)

    # 2. Directories + resume checkpoint ──────────────────────────────────────
    exp_dir = CKPT_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    resume_ckpt = _find_resume_ckpt(exp_dir) if was_interrupted else None
    if resume_ckpt:
        print(f'  Resuming from: {resume_ckpt}')

    # 3. DataModule + model ───────────────────────────────────────────────────
    metrics_path = METRICS_DIR / f'{exp_name}.json'
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    dm    = FinetuneDataModule(cfg)
    model = NASLightningModule(cfg)
    model.set_metrics_path(metrics_path)

    # Restore metrics log when resuming an interrupted run
    if was_interrupted and metrics_path.exists():
        try:
            with open(metrics_path) as fh:
                model._epoch_metrics_log = json.load(fh)
            print(f'  Metrics log restored: {len(model._epoch_metrics_log)} entries')
        except Exception as exc:
            print(f'  [WARN] Could not restore metrics log: {exc}')

    # Warm-start: load pretrained weights (fresh optimizer, epoch 0).
    # Skipped when resuming an interrupted run — the resume ckpt takes precedence.
    if resume_ckpt is None and args.pretrained_ckpt is not None:
        ckpt_data  = torch.load(args.pretrained_ckpt, map_location='cpu')
        model.load_state_dict(ckpt_data['state_dict'], strict=True)
        print(f'  Loaded pretrained weights from {args.pretrained_ckpt}')

    # 4. Callbacks ─────────────────────────────────────────────────────────────
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

    # 5. Device ────────────────────────────────────────────────────────────────
    if args.device is not None:
        accel, devices = args.device.lower(), 1
    elif torch.cuda.is_available():
        accel, devices = 'cuda', 1
    elif torch.backends.mps.is_available():
        accel, devices = 'mps', 1
    else:
        accel, devices = 'cpu', 1

    # 6. Gradient accumulation ────────────────────────────────────────────────
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

    # 7. Logger + Trainer ─────────────────────────────────────────────────────
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

    # 8. Fit ───────────────────────────────────────────────────────────────────
    trainer.fit(model, dm, ckpt_path=resume_ckpt)

    # 9. Test with best checkpoint ─────────────────────────────────────────────
    trainer.test(model, dm, ckpt_path='best')
    wb_logger.experiment.finish()

    # 10. Remove last.ckpt ────────────────────────────────────────────────────
    last_path = exp_dir / 'last.ckpt'
    if last_path.exists():
        last_path.unlink()

    # 11. Mark completed ───────────────────────────────────────────────────────
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

    # ── Resolve config dir and run dir ────────────────────────────────────────
    if args.config_file:
        exp_yaml   = args.config_file.resolve()
        if not exp_yaml.exists():
            print(f'[ERROR] Config file not found: {exp_yaml}')
            sys.exit(1)
        config_dir = exp_yaml.parent
        exp_names  = [exp_yaml.stem]
    else:
        config_dir = (args.config_dir or _DEFAULT_CONFIGS_DIR).resolve()
        exp_names  = sorted(p.stem for p in config_dir.glob('*.yaml'))

    # All finetune runs share a single run dir (unlike main_nas-seld.py per-config-dir layout)
    run_dir    = args.run_dir or _FINETUNE_RUN_DIR
    STATE_FILE = run_dir / 'experiment_state.json'

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
    print(f'STARSS23    : {STARSS23_ROOT}')
    print(f'TAU-NIGENS  : {TAU_ROOT}')
    if args.pretrained_ckpt:
        print(f'Pretrained  : {args.pretrained_ckpt}')

    _start_tensorboard(CKPT_DIR, port=6007)

    for exp_name in exp_names:
        s      = state.get(exp_name, {})
        status = s.get('status')

        if status == 'completed' and not args.force:
            print(f'[SKIP] {exp_name} — already completed')
            continue

        was_interrupted = (status == 'running')

        if args.config_file:
            cfg = _build_finetune_cfg(args.config_file.resolve(), args)
        else:
            cfg = _build_finetune_cfg(config_dir / f'{exp_name}.yaml', args)

        L.seed_everything(cfg['experiment']['seed'], workers=True)

        print(f'\n{"=" * 60}')
        print(f'[RUN] {exp_name}')
        print(f'{"=" * 60}')

        _run_experiment(exp_name, cfg, args, state, was_interrupted)

    print('\nAll experiments finished.')


if __name__ == '__main__':
    main()
