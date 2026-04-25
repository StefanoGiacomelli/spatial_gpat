#!/usr/bin/env python3
"""
main_at2sed.py — AT2SED Fine-tuning (AudioSet-Strong & STARSS23)
=================================================================
Trains ``AT2SEDLightningModule`` on AudioSet-Strong or STARSS23.
Designed to measure classification quality as a function of input window
duration, in preparation for pseudo-real-time inference.

Features
--------
  * ``--config-dir DIR``  : run all *.yaml configs in DIR
  * ``--config-file PATH``: run a single experiment from an explicit path
  * ``--exp NAME [...]``  : experiment name(s); filter when using --config-dir;
                            stem in default configs dir otherwise
  * ``--list``            : print status table and exit
  * ``--force``           : re-run completed experiments
  * HLP superscript       : ``experiment.hlp_enabled`` overrides ``data.hlp_enabled``
  * Dataset selector      : ``experiment.dataset: starss23`` uses STARSS23SEDDataModule
  * Resumable             : interrupted runs continue from ``last.ckpt``
  * Atomic state JSON and metrics JSON (write tmp → rename)
  * TensorBoard launched automatically; killed on exit
  * ``--limit-batches``   : smoke tests (status stays ``running`` → auto-resume)

Directory layout after a run
-----------------------------
::

    runs/at2sed/{exp_name}/
    ├── state.json           ← pending | running | completed
    ├── metrics.json         ← [{epoch: 0, val/map_macro: ...}, ...]
    ├── checkpoints/
    │   ├── epoch_val.ckpt   ← highest val/map_macro checkpoint
    │   └── last.ckpt        ← deleted on successful completion
    └── tb_logs/             ← TensorBoard event files

Usage
-----
::

    # Smoke test — default experiment
    .venv/bin/python main_at2sed.py --epochs 1 --limit-batches 4

    # Single named experiment from default config dir
    .venv/bin/python main_at2sed.py --exp at2sed_bce_10s --epochs 30

    # Batch: all configs in a directory
    .venv/bin/python main_at2sed.py --config-dir configs/experiments_at2sed

    # Batch with name filter
    .venv/bin/python main_at2sed.py --config-dir configs/experiments_at2sed \
        --exp at2sed_bce_10s_hlp at2sed_bce_10s_nohlp

    # Single explicit config file
    .venv/bin/python main_at2sed.py \
        --config-file configs/experiments_at2sed/at2sed_bce_10s_starss23.yaml

    # Print status table
    .venv/bin/python main_at2sed.py --list

    # Force re-run of completed experiments
    .venv/bin/python main_at2sed.py --config-dir configs/experiments_at2sed --force

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
from typing import List, Optional, Tuple

warnings.filterwarnings('ignore', message='.*No positive samples.*')
warnings.filterwarnings('ignore', message='.*elements was resized.*')

import torch
import yaml
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

# ── Project path setup ────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_at2sed import AT2SEDLightningModule   # noqa: E402
from datasets.audioset_strong import AudioSetStrongDataModule   # noqa: E402
from datasets.starss23_sed_datamodule import STARSS23SEDDataModule   # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

_RESULTS_ROOT    = _ROOT / 'runs' / 'at2sed'
_CONFIGS_DIR     = _ROOT / 'configs' / 'experiments_at2sed'
_DATASET_CFG     = _ROOT / 'configs' / 'datasets' / 'AudioSetStrong.yaml'
_DEFAULT_EXP         = 'at2sed_bce'

_DEFAULT_EPOCHS      = 1000
_DEFAULT_BATCH_SIZE  = 32
_DEFAULT_LR          = 1e-3
_DEFAULT_NUM_WORKERS = 4
_DEFAULT_WINDOW_S    = 0.5   # fallback if neither CLI nor YAML specifies window
_DEFAULT_HOP_S       = None  # fallback: equal to window_s (no overlap)
EFFECTIVE_BATCH      = 32   # target for gradient accumulation
EARLY_STOPPING_PATIENCE = 100  # epochs without improvement before stopping


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description='AT2SED fine-tuning on AudioSet-Strong or STARSS23.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Experiment source (mutually exclusive) ────────────────────────────────
    src = p.add_mutually_exclusive_group()
    src.add_argument('--config-dir', type=Path, default=None, metavar='DIR',
                     help='Run all *.yaml configs in DIR.')
    src.add_argument('--config-file', type=Path, default=None, metavar='PATH',
                     help='Run single experiment from explicit YAML path.')

    # ── Experiment name / filter ──────────────────────────────────────────────
    p.add_argument('--exp', type=str, nargs='*', default=None, metavar='NAME',
                   help='Experiment name(s). Alone: stem in default configs dir. '
                        'With --config-dir: filter subset to run.')

    # ── Training overrides ────────────────────────────────────────────────────
    p.add_argument('--window-s',    type=float, default=None,
                   help='Window duration in seconds (overrides YAML data.window_s).')
    p.add_argument('--hop-s',       type=float, default=None,
                   help='Window hop in seconds (overrides YAML data.hop_s).')
    p.add_argument('--lr',          type=float, default=_DEFAULT_LR,
                   help='Adam learning rate.')
    p.add_argument('--epochs',      type=int,   default=_DEFAULT_EPOCHS,
                   help='Max training epochs.')
    p.add_argument('--batch-size',  type=int,   default=_DEFAULT_BATCH_SIZE,
                   help='Batch size per step.')
    p.add_argument('--num-workers', type=int,   default=_DEFAULT_NUM_WORKERS,
                   help='DataLoader worker processes.')
    p.add_argument('--device',      type=str,   default=None, metavar='ACCEL',
                   help='Force accelerator: cuda | mps | cpu. Default: auto-detect.')
    p.add_argument('--force',       action='store_true',
                   help='Re-run experiments even if already completed.')
    p.add_argument('--limit-batches', type=float, default=0.0,
                   help='Fraction (0,1] or number of batches per epoch/phase '
                        'for smoke tests (0 = disabled). Status stays running → auto-resume.')
    p.add_argument('--list',        action='store_true',
                   help='Print experiment status table and exit.')

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# State management
# ══════════════════════════════════════════════════════════════════════════════

def _state_file(exp_dir: Path) -> Path:
    return exp_dir / 'state.json'


def _load_state(exp_dir: Path) -> dict:
    sf = _state_file(exp_dir)
    if sf.exists():
        with open(sf) as fh:
            return json.load(fh)
    return {}


def _save_state(state: dict, exp_dir: Path) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    sf  = _state_file(exp_dir)
    tmp = sf.with_suffix('.tmp')
    with open(tmp, 'w') as fh:
        json.dump(state, fh, indent=2, default=float)
    tmp.replace(sf)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec='seconds')


def _print_state(exp_name: str, exp_dir: Path) -> None:
    state  = _load_state(exp_dir)
    status = state.get('status', 'pending')
    info   = ''
    if status == 'completed':
        monitor = state.get('checkpoint_monitor', 'val_map_macro')
        score   = state.get('best_val_score', state.get('best_val_map', '?'))
        score_s = f'{score:.4f}' if isinstance(score, float) else str(score)
        info = (
            f"done {state.get('completed_at', '')}  "
            f"{monitor}={score_s}"
        )
    elif status == 'running':
        info = f"interrupted {state.get('started_at', '')}"
    print(f'{exp_name:<40}  {status:<12}  {info}')


def _print_state_table(experiments: List[Tuple[str, Optional[Path]]]) -> None:
    print(f'\n{"Experiment":<40}  {"Status":<12}  Info')
    print('-' * 80)
    for exp_name, _ in experiments:
        _print_state(exp_name, _RESULTS_ROOT / exp_name)
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Experiment resolution
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_experiments(args) -> List[Tuple[str, Path]]:
    """
    Returns list of (exp_name, yaml_path) pairs based on CLI flags.

    Priority:
        --config-dir  → all *.yaml in DIR, optionally filtered by --exp
        --config-file → single explicit YAML
        --exp NAME    → stems looked up in default configs dir
        (none)        → default experiment (_DEFAULT_EXP)
    """
    if args.config_dir is not None:
        config_dir = Path(args.config_dir)
        if not config_dir.is_dir():
            print(f'[ERROR] --config-dir not found: {config_dir}')
            sys.exit(1)
        yamls = sorted(config_dir.glob('*.yaml'))
        if args.exp:
            filter_set = set(args.exp)
            yamls = [y for y in yamls if y.stem in filter_set]
        return [(y.stem, y) for y in yamls]

    if args.config_file is not None:
        cf = Path(args.config_file)
        if not cf.exists():
            print(f'[ERROR] --config-file not found: {cf}')
            sys.exit(1)
        return [(cf.stem, cf)]

    # Default / legacy mode: --exp NAME [NAME ...] or fall back to _DEFAULT_EXP
    exp_names = args.exp if args.exp else [_DEFAULT_EXP]
    experiments: List[Tuple[str, Path]] = []
    for name in exp_names:
        yaml_path = _CONFIGS_DIR / f'{name}.yaml'
        if not yaml_path.exists():
            print(f'[ERROR] Experiment config not found: {yaml_path}')
            sys.exit(1)
        experiments.append((name, yaml_path))
    return experiments


# ══════════════════════════════════════════════════════════════════════════════
# Config building
# ══════════════════════════════════════════════════════════════════════════════

def _build_cfg(exp_yaml: Path, args) -> dict:
    """
    Merge experiment YAML + dataset YAML + CLI overrides into a single cfg dict.

    Priority (highest → lowest):
        CLI args > experiment YAML > AudioSetStrong.yaml defaults

    Post-merge: ``experiment.hlp_enabled`` overrides ``data.hlp_enabled``.
    """
    # 1. Experiment YAML
    with open(exp_yaml) as fh:
        cfg = yaml.safe_load(fh) or {}

    # 2. Dataset YAML — provides data paths, augmentation, training defaults
    if _DATASET_CFG.exists():
        with open(_DATASET_CFG) as fh:
            ds_cfg = yaml.safe_load(fh) or {}
        # Merge each top-level section (experiment YAML keys win on collision)
        # 'semantic' is included so pretrained_class_map / model_class_map
        # from AudioSetStrong.yaml flow through to SplitEPANNs.
        for section in ('data', 'augmentation', 'training', 'semantic'):
            merged = {**ds_cfg.get(section, {}), **cfg.get(section, {})}
            if merged:
                cfg[section] = merged

    # 3. HLP superscript: experiment.hlp_enabled wins over data.hlp_enabled
    exp_cfg = cfg.get('experiment', {})
    if 'hlp_enabled' in exp_cfg:
        cfg.setdefault('data', {})['hlp_enabled'] = bool(exp_cfg['hlp_enabled'])

    # 4. Resolve relative paths in data and semantic sections to absolute
    for section in ('data', 'semantic'):
        for key, val in cfg.get(section, {}).items():
            if isinstance(val, str) and not Path(val).is_absolute():
                resolved = (_ROOT / val).resolve()
                if resolved.exists():
                    cfg[section][key] = str(resolved)

    # 5. CLI overrides (win when explicitly passed; YAML values respected otherwise)
    data = cfg.setdefault('data', {})
    if args.window_s is not None:
        data['window_s'] = args.window_s
    elif 'window_s' not in data:
        data['window_s'] = _DEFAULT_WINDOW_S     # fallback: 0.5 s
    if args.hop_s is not None:
        data['hop_s'] = args.hop_s
    elif 'hop_s' not in data:
        data['hop_s'] = data['window_s']         # fallback: non-overlapping
    data['num_workers'] = args.num_workers
    cfg.setdefault('training', {}).update({
        'lr':         args.lr,
        'epochs':     args.epochs,
        'batch_size': args.batch_size,
    })

    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint resume helper
# ══════════════════════════════════════════════════════════════════════════════

def _find_resume_ckpt(ckpt_dir: Path) -> Optional[str]:
    """
    Return the best checkpoint path for resuming, or None.

    Priority:
        1. ``last.ckpt``  — full optimizer state; can continue to next epoch
        2. Any ``*.ckpt`` sorted alphabetically, take last — best epoch checkpoint
        3. None           — fresh start
    """
    last = ckpt_dir / 'last.ckpt'
    if last.exists():
        return str(last)
    candidates = sorted(p for p in ckpt_dir.glob('*.ckpt') if p.name != 'last.ckpt')
    return str(candidates[-1]) if candidates else None


# ══════════════════════════════════════════════════════════════════════════════
# TensorBoard subprocess
# ══════════════════════════════════════════════════════════════════════════════

def _start_tensorboard(logdir: Path, port: int = 6007) -> subprocess.Popen:
    """Launch TensorBoard and register cleanup handlers."""
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
    was_interrupted: bool,
) -> None:
    """
    Train and test one experiment end-to-end.

    Args:
        exp_name        : experiment name (YAML stem)
        cfg             : merged config dict
        args            : parsed CLI namespace
        was_interrupted : if True, attempt to resume from an existing checkpoint
    """
    exp_dir  = _RESULTS_ROOT / exp_name
    ckpt_dir = exp_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Mark running ─────────────────────────────────────────────────────────
    state = {'status': 'running', 'started_at': _now_iso()}
    _save_state(state, exp_dir)

    # 2. Resume checkpoint ────────────────────────────────────────────────────
    resume_ckpt = _find_resume_ckpt(ckpt_dir) if was_interrupted else None
    if resume_ckpt:
        print(f'  Resuming from: {resume_ckpt}')

    # 3. Seed ─────────────────────────────────────────────────────────────────
    seed = int(cfg.get('experiment', {}).get('seed', 42))
    L.seed_everything(seed, workers=True)

    # 4. DataModule ───────────────────────────────────────────────────────────
    dataset_key = cfg.get('experiment', {}).get('dataset', 'audioset_strong').lower()
    if dataset_key == 'starss23':
        dm = STARSS23SEDDataModule(cfg)
    else:
        dm = AudioSetStrongDataModule(cfg)
    dm.setup('fit')

    # 5. Model ────────────────────────────────────────────────────────────────
    model = AT2SEDLightningModule(cfg)

    # Inject pos_weight from training set statistics (after setup)
    if dm.pos_weights is not None:
        model.set_pos_weight(dm.pos_weights)

    # Inject metrics JSON path
    metrics_path = exp_dir / 'metrics.json'
    model.set_metrics_path(metrics_path)

    # Pre-populate metrics log when resuming an interrupted run
    if was_interrupted and metrics_path.exists():
        try:
            with open(metrics_path) as fh:
                model._epoch_metrics_log = json.load(fh)
            print(
                f'  Metrics log restored: '
                f'{len(model._epoch_metrics_log)} entries'
            )
        except Exception as exc:
            print(f'  [WARN] Could not restore metrics log: {exc}')

    # 6. Callbacks ────────────────────────────────────────────────────────────
    ckpt_monitor = str(
        cfg.get('training', {}).get('checkpoint_monitor', 'val_map_macro')
    )
    # Strip a leading 'val/' prefix if accidentally included in the YAML
    if ckpt_monitor.startswith('val/'):
        ckpt_monitor = 'val_' + ckpt_monitor[4:]

    ckpt_mode = 'min' if any(k in ckpt_monitor for k in ('loss', 'error', '_deg', 'mae')) else 'max'

    best_cb = ModelCheckpoint(
        dirpath                 = str(ckpt_dir),
        filename                = '{epoch:02d}_{' + ckpt_monitor + ':.4f}',
        monitor                 = ckpt_monitor,
        mode                    = ckpt_mode,
        save_top_k              = 1,
        save_last               = True,
        verbose                 = True,
        auto_insert_metric_name = False,
    )

    early_cb = EarlyStopping(
        monitor  = ckpt_monitor,
        mode     = ckpt_mode,
        patience = EARLY_STOPPING_PATIENCE,
        verbose  = True,
    )

    # 7. Device ───────────────────────────────────────────────────────────────
    if args.device is not None:
        accel, devices = args.device.lower(), 1
    elif torch.cuda.is_available():
        accel, devices = 'cuda', 1
    elif torch.backends.mps.is_available():
        accel, devices = 'mps', 1
    else:
        accel, devices = 'cpu', 1

    # 8. Gradient accumulation ────────────────────────────────────────────────
    batch_size = int(cfg.get('training', {}).get('batch_size', _DEFAULT_BATCH_SIZE))
    accum      = max(1, EFFECTIVE_BATCH // batch_size)
    eff_batch  = batch_size * accum
    print(
        f'  batch_size={batch_size}  accum_steps={accum}  '
        f'effective_batch={eff_batch}'
        + ('  [no accumulation]' if accum == 1 else '')
    )

    # 9. Logger + Trainer ─────────────────────────────────────────────────────
    logger = TensorBoardLogger(
        save_dir = str(exp_dir),
        name     = 'tb_logs',
        version  = 0,
    )

    trainer_kwargs = dict(
        max_epochs              = args.epochs,
        callbacks               = [best_cb, early_cb],
        accelerator             = accel,
        devices                 = devices,
        gradient_clip_val       = 1.0,
        accumulate_grad_batches = accum,
        log_every_n_steps       = 10,
        logger                  = logger,
        num_sanity_val_steps    = 0,
    )

    limit_batches = args.limit_batches
    using_limit   = limit_batches > 0.0
    if using_limit:
        lb = int(limit_batches) if limit_batches >= 1.0 else limit_batches
        trainer_kwargs.update(
            limit_train_batches = lb,
            limit_val_batches   = lb,
            limit_test_batches  = lb,
        )

    trainer = L.Trainer(**trainer_kwargs)

    # 10. Fit ─────────────────────────────────────────────────────────────────
    trainer.fit(model, dm, ckpt_path=resume_ckpt)

    # 11. Test with best checkpoint ───────────────────────────────────────────
    dm.setup('test')
    trainer.test(model, dm, ckpt_path='best')

    # 12. Cleanup + finalize ──────────────────────────────────────────────────
    last_path = ckpt_dir / 'last.ckpt'

    if using_limit:
        print(
            '  [limit-batches] Smoke test finished.  '
            'Status remains "running" for auto-resume.'
        )
        return

    if last_path.exists():
        last_path.unlink()

    best_score = float(best_cb.best_model_score or 0.0)
    state = {
        'status':             'completed',
        'best_ckpt':          best_cb.best_model_path,
        'best_val_score':     best_score,
        'checkpoint_monitor': ckpt_monitor,
        'completed_at':       _now_iso(),
    }
    _save_state(state, exp_dir)
    print(f'  Completed.  {ckpt_monitor}={best_score:.4f}')


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = _parse_args()

    # ── --list without any experiment filter → show all known runs ────────────
    if args.list and args.config_dir is None and args.config_file is None and not args.exp:
        if _RESULTS_ROOT.exists():
            dirs = [d for d in sorted(_RESULTS_ROOT.iterdir()) if d.is_dir()]
            if dirs:
                print(f'\n{"Experiment":<40}  {"Status":<12}  Info')
                print('-' * 80)
                for d in dirs:
                    _print_state(d.name, d)
                print()
            else:
                print('No experiments yet.')
        else:
            print('No experiments yet.')
        return

    # ── Resolve which experiments to run / list ───────────────────────────────
    experiments = _resolve_experiments(args)

    if not experiments:
        print('[ERROR] No experiment configs found (check --config-dir or --exp).')
        sys.exit(1)

    # ── --list with a specific source ─────────────────────────────────────────
    if args.list:
        print(f'\n{"Experiment":<40}  {"Status":<12}  Info')
        print('-' * 80)
        for exp_name, _ in experiments:
            _print_state(exp_name, _RESULTS_ROOT / exp_name)
        print()
        return

    # ── Start TensorBoard once for all experiments ────────────────────────────
    _start_tensorboard(_RESULTS_ROOT)

    # ── Iterate over experiments ──────────────────────────────────────────────
    for exp_name, exp_yaml in experiments:
        exp_dir = _RESULTS_ROOT / exp_name
        state  = _load_state(exp_dir)
        status = state.get('status', 'pending')

        if status == 'completed' and not args.force:
            print(f'[SKIP] {exp_name} — already completed.  Use --force to re-run.')
            continue

        was_interrupted = (status == 'running')
        cfg = _build_cfg(exp_yaml, args)

        t_cfg        = cfg.get('training', {})
        d_cfg        = cfg.get('data',     {})
        dataset_name = cfg.get('experiment', {}).get('dataset', 'audioset_strong')

        loss_label = str(t_cfg.get('loss', 'bce')).upper()
        if loss_label == 'OBCE':
            loss_label += f'  beta={t_cfg.get("obce_beta", 1.0)}'

        sampler_label = (
            str(t_cfg.get('balanced_sampler_formula', 'maxpos')).upper()
            if t_cfg.get('weighted_sampler', False)
            else 'off'
        )

        hlp_label  = 'ON'  if d_cfg.get('hlp_enabled',  False) else 'off'
        omap_label = (f'every {t_cfg.get("omap_every_n", 5)} epochs'
                      if t_cfg.get('omap_enabled', False) else 'off')
        aug_label  = 'ON'  if cfg.get('augmentation', {}).get('enabled', False) else 'off'
        monitor    = str(t_cfg.get('checkpoint_monitor', 'val_map_macro'))

        ws = d_cfg.get('window_s')
        hs = d_cfg.get('hop_s')
        ws_s = f'{ws:.1f}s' if ws is not None else '?'
        hs_s = f'{hs:.1f}s' if hs is not None else '?'

        print(f'\n{"=" * 60}')
        print(f'[RUN]  {exp_name}')
        print(f'       dataset={dataset_name}  window={ws_s}  hop={hs_s}')
        print(f'       lr={t_cfg["lr"]}  epochs={t_cfg["epochs"]}')
        print(f'       loss={loss_label}  sampler={sampler_label}  '
              f'HLP={hlp_label}  aug={aug_label}')
        print(f'       oMAP={omap_label}  monitor={monitor}')
        print(f'{"=" * 60}\n')

        _run_experiment(exp_name, cfg, args, was_interrupted)

    print('\nDone.')


if __name__ == '__main__':
    main()
