#!/usr/bin/env python3
"""
main.py — Resumable NAS Training Pipeline
==========================================
Runs NAS experiments sequentially with:

  * ``--dataset NAME``     — choose dataset at runtime (STARSS23 | TAU-NIGENS2021)
  * ``--config-dir DIR``   — custom directory with experiment YAMLs
  * ``--config-file PATH`` — run a single experiment from an arbitrary YAML path
  * ``--run-dir DIR``      — custom output root (default: runs/{config_dir_name}/)
  * JSON state file for resume / skip logic
  * PyTorch Lightning Trainer — best device auto-detected (cuda > mps > cpu)
  * Full-clip mode by default (variable-length clips, padding collate_fn)
  * ModelCheckpoint: best model kept; ``last.ckpt`` deleted on completion
  * EarlyStopping on ``val/SELD_score`` (lower is better)
  * TensorBoard launched as a subprocess at startup; killed on exit / Ctrl+C
  * Per-experiment JSON log with val metrics per epoch + test metrics

Directory layout after a run
-----------------------------
::

    runs/experiments/
    ├── experiment_state.json
    ├── checkpoints/
    │   └── exp_spe_cb_fe_mg/
    │       ├── epoch_08_0.4231.ckpt        ← best (kept)
    │       └── version_0/
    │           └── events.out.tfevents.*   ← TensorBoard logs
    └── metrics/
        └── exp_spe_cb_fe_mg.json           ← epoch + test metrics

Usage
-----
::

    # Debug: 1 epoch, 5 batches
    .venv/bin/python main.py --exp exp_spe_cb_fe_mg --epochs 1 --limit-batches 5

    # Use a specific dataset
    .venv/bin/python main.py --dataset TAU-NIGENS2021 --exp exp_spe_cb_fe_mg

    # Single config file from arbitrary path
    .venv/bin/python main.py --config-file configs/experiments/exp_spe_cb_fe_bg.yaml

    # Custom config directory (e.g. for deep NAS configs)
    .venv/bin/python main.py --config-dir configs/deep_experiments/

    # List all statuses
    .venv/bin/python main.py --list

    # Re-run a completed experiment from scratch
    .venv/bin/python main.py --exp exp_spe_cb_fe_mg --force

    # Full run (all experiments in config-dir)
    .venv/bin/python main.py --epochs 20 --batch-size 16 --patience 5

    # Windowed mode (10 s / 5 s hop)
    .venv/bin/python main.py --window-s 10 --hop-s 5

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
from typing import Optional

# Suppress known benign warnings that pollute training output
warnings.filterwarnings('ignore', message='.*LeafSpec.*deprecated.*')
warnings.filterwarnings('ignore', message='.*elements was resized.*')
warnings.filterwarnings('ignore', message='.*No positive samples.*')

import torch
import yaml
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

# ── Project path setup ────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.train import NASLightningModule  # noqa: E402
from datasets.datamodule import SELDDataModule     # noqa: E402

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

SEED             = 42     # global + val_seed for STARSS23 train/val split
VAL_FRACTION     = 0.2    # STARSS23-only (matches test_datasets.py:145)
EFFECTIVE_BATCH  = 16     # target effective batch size for gradient accumulation

# Default CLI values
_DEFAULT_EPOCHS      = 20
_DEFAULT_BATCH_SIZE  = 4
_DEFAULT_LR          = 5e-4
_DEFAULT_PATIENCE    = 0   # 0 = early stopping disabled; >0 = patience in epochs
_DEFAULT_NUM_WORKERS = 4
# Full-clip mode by default (window_s=0 → None → variable-length clips with
# padding collate_fn).  Use --window-s 10 --hop-s 5 to switch to windowed mode
# if GPU/MPS memory is constrained (~60 s clips can require >8 GB per batch).
_DEFAULT_WINDOW_S    = 10.0    # 0 = full-clip mode (no windowing)
_DEFAULT_HOP_S       =  5.0    # unused in full-clip mode

# Dataset root paths
DATASET_ROOTS = {
    'STARSS23':       _ROOT / 'data' / 'STARSS23',
    'TAU-NIGENS2021': _ROOT / 'data' / 'TAU-NIGENS2021',
}

# Default config / output directories (may be overridden in main())
_DEFAULT_CONFIGS_DIR = _ROOT / 'configs' / 'experiments'
_DATASET_CONFIGS_DIR = _ROOT / 'configs' / 'datasets'

# These are reassigned in main() based on --config-dir / --run-dir
CONFIGS_DIR = _DEFAULT_CONFIGS_DIR
CKPT_DIR    = _ROOT / 'runs' / 'experiments' / 'checkpoints'
METRICS_DIR = _ROOT / 'runs' / 'experiments' / 'metrics'
STATE_FILE  = _ROOT / 'runs' / 'experiments' / 'experiment_state.json'


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description='Resumable NAS training pipeline.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    p.add_argument('--dataset', type=str, default='STARSS23',
                   choices=['STARSS23', 'TAU-NIGENS2021'],
                   help='Dataset for training/validation/test. Injects semantic.dataset '
                        'and loads configs/datasets/{NAME}.yaml if present.')

    # ── Config / output paths ─────────────────────────────────────────────────
    loc = p.add_mutually_exclusive_group()
    loc.add_argument('--config-dir', type=Path, default=None,
                     metavar='DIR',
                     help='Directory with experiment YAMLs. '
                          'Default: configs/experiments/')
    loc.add_argument('--config-file', type=Path, default=None,
                     metavar='PATH',
                     help='Path to a single experiment YAML. '
                          'Runs only that experiment; state file is placed next to it.')

    p.add_argument('--run-dir', type=Path, default=None,
                   metavar='DIR',
                   help='Output root (checkpoints, metrics, state). '
                        'Default: runs/{config_dir_name}/')

    # ── Training hyperparams ──────────────────────────────────────────────────
    p.add_argument('--epochs',       type=int,   default=_DEFAULT_EPOCHS,
                   help='Max training epochs per experiment.')
    p.add_argument('--batch-size',   type=int,   default=_DEFAULT_BATCH_SIZE,
                   help='Batch size (samples). Only CLI override allowed.')
    p.add_argument('--lr',           type=float, default=_DEFAULT_LR,
                   help='AdamW learning rate.')
    p.add_argument('--patience',     type=int,   default=_DEFAULT_PATIENCE,
                   help='EarlyStopping patience in epochs (0 = disabled, run all --epochs).')
    p.add_argument('--limit-batches', type=int,  default=0,
                   help='Limit train/val/test batches per epoch (0 = disabled).')
    p.add_argument('--window-s',     type=float, default=_DEFAULT_WINDOW_S,
                   help='Window duration in seconds (0 = full-clip mode).')
    p.add_argument('--hop-s',        type=float, default=_DEFAULT_HOP_S,
                   help='Window hop in seconds.')
    p.add_argument('--num-workers',  type=int,   default=_DEFAULT_NUM_WORKERS,
                   help='DataLoader worker processes.')
    p.add_argument('--accum-steps',  type=int,   default=0,
                   help='Gradient accumulation steps (0 = auto: ceil(16/batch_size)).')
    p.add_argument('--device',       type=str,   default=None,
                   metavar='ACCEL',
                   help='Force accelerator: cuda | mps | cpu. Default: auto-detect.')

    # ── Experiment selection ──────────────────────────────────────────────────
    p.add_argument('--exp',    type=str, nargs='+', metavar='NAME',
                   help='Run only these experiment(s) by name.')
    p.add_argument('--force',  action='store_true',
                   help='Re-run experiment(s) even if already completed.')
    p.add_argument('--list',   action='store_true',
                   help='Print experiment status table and exit.')

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# State management
# ══════════════════════════════════════════════════════════════════════════════

def _load_state() -> dict:
    """Load experiment state from JSON file; return empty dict if absent."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as fh:
            return json.load(fh)
    return {}


def _save_state(state: dict) -> None:
    """Atomically write state dict to STATE_FILE."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix('.tmp')
    with open(tmp, 'w') as fh:
        json.dump(state, fh, indent=2)
    tmp.replace(STATE_FILE)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec='seconds')


def _print_state_table(exp_names: list, state: dict) -> None:
    """Print a formatted status table for all experiments."""
    col = 36
    print(f'\n{"Experiment":<{col}}  {"Status":<12}  {"Info"}')
    print('─' * 75)
    for name in exp_names:
        s      = state.get(name, {})
        status = s.get('status', 'pending')
        info   = ''
        if status == 'completed':
            info = f"done {s.get('completed_at','')}"
        elif status == 'running':
            info = f"interrupted {s.get('started_at','')}"
        print(f'{name:<{col}}  {status:<12}  {info}')
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Config building
# ══════════════════════════════════════════════════════════════════════════════

def _build_cfg(exp_yaml_path: Path, args) -> dict:
    """
    Load experiment YAML and inject training + data sections from CLI args.

    Steps
    -----
    1. Load the experiment YAML.
    2. Inject ``training:`` block from CLI (batch_size is the only CLI-overridable
       training param; others come from the dataset config YAML).
    3. Load ``configs/datasets/{dataset_name}.yaml`` if it exists and deep-merge
       ``augmentation:`` and ``training:`` sections.  ``batch_size`` always wins
       from CLI (step 4 re-asserts it after the merge).
    4. Inject ``data:`` runtime params (root_path, windowing, workers).
    5. Inject ``semantic.dataset`` from CLI ``--dataset`` argument.
    """
    with open(exp_yaml_path) as fh:
        cfg = yaml.safe_load(fh)

    # 1. Dataset name from CLI — inject into semantic section
    dataset_name = args.dataset
    cfg.setdefault('semantic', {})['dataset'] = dataset_name

    # 2. Base training block from CLI
    cfg['training'] = {
        'lr':              args.lr,
        'epochs':          args.epochs,
        'batch_size':      args.batch_size,
        'lambda_spatial':  0.9,   # weight of tPIT (spatial) in total loss
        'lambda_semantic': 0.1,   # weight of BCE (semantic) in total loss
        'beta_sed':        0.3,   # SED/DOA split inside tPIT (DOA = 1-beta = 0.7)
        'angle_thresh':    20.0,
    }

    # 3. Dataset-specific config (augmentation + training overrides)
    ds_cfg_path = _DATASET_CONFIGS_DIR / f'{dataset_name}.yaml'
    if ds_cfg_path.exists():
        with open(ds_cfg_path) as fh:
            ds_cfg = yaml.safe_load(fh) or {}
        cfg.setdefault('augmentation', {}).update(ds_cfg.get('augmentation', {}))
        cfg['training'].update(ds_cfg.get('training', {}))

    # 4. CLI batch_size always takes precedence (re-assert after dataset merge)
    cfg['training']['batch_size'] = args.batch_size

    # 5. Data runtime params
    if dataset_name not in DATASET_ROOTS:
        raise ValueError(f'Unknown dataset: {dataset_name!r}. '
                         f'Add its root to DATASET_ROOTS in main.py.')
    window_s = args.window_s if args.window_s > 0 else None
    hop_s    = args.hop_s    if args.hop_s    > 0 else None
    cfg.setdefault('data', {}).update({
        'root_path':    str(DATASET_ROOTS[dataset_name]),
        'window_s':     window_s,
        'hop_s':        hop_s,
        'val_fraction': VAL_FRACTION,   # STARSS23 only; ignored by TAU-NIGENS
        'val_seed':     SEED,
        'num_workers':  args.num_workers,
    })
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint resume helper
# ══════════════════════════════════════════════════════════════════════════════

def _find_resume_ckpt(exp_dir: Path) -> Optional[str]:
    """
    Return the best available checkpoint path for resuming a run, or None.

    Priority:
        1. ``last.ckpt`` — most recent training state (epoch N+1 can continue)
        2. Any other ``*.ckpt`` (best checkpoint) — sorted alphabetically, take last
        3. None — start fresh
    """
    last = exp_dir / 'last.ckpt'
    if last.exists():
        return str(last)
    candidates = sorted(p for p in exp_dir.glob('*.ckpt') if p.name != 'last.ckpt')
    return str(candidates[-1]) if candidates else None


# ══════════════════════════════════════════════════════════════════════════════
# TensorBoard subprocess
# ══════════════════════════════════════════════════════════════════════════════

def _start_tensorboard(logdir: Path, port: int = 6006) -> subprocess.Popen:
    """
    Launch TensorBoard as a subprocess pointing at ``logdir``.

    Registers cleanup handlers so that TensorBoard is terminated on:
      * Normal script exit (``atexit``)
      * SIGINT / Ctrl+C (signal handler)

    Returns the ``Popen`` object (not typically needed by callers).
    """
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
    """
    Train and test one experiment end-to-end; update *state* in-place.

    Args:
        exp_name        : experiment YAML stem (e.g. ``'exp_spe_cb_fe_mg'``)
        cfg             : merged config dict (from ``_build_cfg``)
        args            : parsed CLI namespace
        state           : mutable global state dict (saved after each step)
        was_interrupted : if True, attempt to resume from an existing checkpoint
    """
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

    dm    = SELDDataModule(cfg)
    model = NASLightningModule(cfg)
    model.set_metrics_path(metrics_path)

    # Pre-populate metrics log from existing file when resuming an interrupted run
    if was_interrupted and metrics_path.exists():
        try:
            with open(metrics_path) as fh:
                model._epoch_metrics_log = json.load(fh)
            print(f'  Metrics log restored: {len(model._epoch_metrics_log)} entries from {metrics_path.name}')
        except Exception as exc:
            print(f'  [WARN] Could not restore metrics log: {exc}')

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
    logger = TensorBoardLogger(
        save_dir = str(CKPT_DIR),
        name     = exp_name,
        version  = 0,
    )

    trainer_kwargs = dict(
        max_epochs              = args.epochs,
        callbacks               = callbacks,
        accelerator             = accel,
        devices                 = devices,
        gradient_clip_val       = 1.0,
        accumulate_grad_batches = accum,
        log_every_n_steps       = 5,
        logger                  = logger,
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

    # 10. Remove last.ckpt (kept only as mid-run resume target) ───────────────
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
        # Single-file mode: derive config_dir from file's parent directory
        exp_yaml   = args.config_file.resolve()
        if not exp_yaml.exists():
            print(f'[ERROR] Config file not found: {exp_yaml}')
            sys.exit(1)
        config_dir = exp_yaml.parent
        exp_names  = [exp_yaml.stem]
        run_dir    = args.run_dir or (_ROOT / 'runs' / exp_yaml.stem)
        STATE_FILE = run_dir / 'experiment_state.json'
    else:
        config_dir = (args.config_dir or _DEFAULT_CONFIGS_DIR).resolve()
        run_dir    = args.run_dir or (_ROOT / 'runs' / config_dir.name)
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
    print(f'Dataset     : {args.dataset}')

    # Start TensorBoard (visible immediately; updates as each experiment trains)
    _start_tensorboard(CKPT_DIR)

    for exp_name in exp_names:
        s      = state.get(exp_name, {})
        status = s.get('status')

        if status == 'completed' and not args.force:
            print(f'[SKIP] {exp_name} — already completed')
            continue

        was_interrupted = (status == 'running')

        if args.config_file:
            cfg = _build_cfg(args.config_file.resolve(), args)
        else:
            cfg = _build_cfg(config_dir / f'{exp_name}.yaml', args)

        # Per-experiment seed from YAML (distinct, reproducible per config)
        L.seed_everything(cfg['experiment']['seed'], workers=True)

        print(f'\n{"=" * 60}')
        print(f'[RUN] {exp_name}')
        print(f'{"=" * 60}')

        _run_experiment(exp_name, cfg, args, state, was_interrupted)

    print('\nAll experiments finished.')


if __name__ == '__main__':
    main()
