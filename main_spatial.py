#!/usr/bin/env python3
"""
main_spatial.py — Standalone Spatial SELD Training
====================================================
Trains ``SpatialLightningModule`` (NeuralSRPEncoder + DOA/activity/distance
heads) on STARSS23 MIC format with class-agnostic spatial targets.

Features
--------
  * Config-file driven: experiment YAML sets all hyper-parameters
  * Resumable: interrupted runs continue from ``last.ckpt``; best kept
  * Atomic state JSON and metrics JSON (write tmp → rename)
  * TensorBoard launched automatically on port 6008; killed on exit
  * ``--limit-batches`` for smoke tests (status stays ``running`` → auto-resume)

Directory layout after a run
------------------------------
::

    runs/spatial/{exp_name}/
    ├── state.json           ← pending | running | completed
    ├── metrics.json         ← [{epoch: 0, val_lf: ..., val_le_deg: ...}, ...]
    ├── checkpoints/
    │   ├── epoch_val_lf.ckpt   ← highest val_lf checkpoint
    │   └── last.ckpt           ← deleted on successful completion
    └── tb_logs/             ← TensorBoard event files

Usage
-----
::

    # Smoke test (1 batch)
    python main_spatial.py \\
        --config-file configs/experiments_spatial/mic/srp_baseline.yaml \\
        --limit-batches 1

    # Full training run
    python main_spatial.py \\
        --config-file configs/experiments_spatial/mic/srp_baseline.yaml \\
        --epochs 100 --lr 1e-4 --batch-size 4

    # Resume / re-run a completed experiment
    python main_spatial.py \\
        --config-file configs/experiments_spatial/mic/srp_baseline.yaml \\
        --force

    # Print state
    python main_spatial.py \\
        --config-file configs/experiments_spatial/mic/srp_baseline.yaml \\
        --list

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

warnings.filterwarnings('ignore', message='.*No positive samples.*')
warnings.filterwarnings('ignore', message='.*elements was resized.*')

import torch
import yaml
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers   import TensorBoardLogger

# ── Project path setup ────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_spatial           import SpatialLightningModule    # noqa: E402
from datasets.spatial_datamodule   import SpatialDataModule         # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

_RESULTS_ROOT        = _ROOT / 'runs' / 'spatial'
_CONFIGS_DIR         = _ROOT / 'configs' / 'experiments_spatial'

_DEFAULT_EPOCHS      = 100
_DEFAULT_BATCH_SIZE  = 4
_DEFAULT_LR          = 1e-4
_DEFAULT_NUM_WORKERS = 4
EFFECTIVE_BATCH      = 16     # target effective batch size for gradient accumulation
_TB_PORT             = 6008   # TensorBoard port (avoids clash with 6006/6007)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description='Standalone spatial SELD training (NeuralSRPEncoder).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        '--config-file', type=str, required=True,
        help='Path to experiment YAML (absolute or relative to project root).',
    )
    p.add_argument('--lr',          type=float, default=None,
                   help='Override Adam learning rate.')
    p.add_argument('--epochs',      type=int,   default=None,
                   help='Override max training epochs.')
    p.add_argument('--batch-size',  type=int,   default=None,
                   help='Override batch size per step.')
    p.add_argument('--num-workers', type=int,   default=_DEFAULT_NUM_WORKERS,
                   help='DataLoader worker processes.')
    p.add_argument('--device',      type=str,   default=None, metavar='ACCEL',
                   help='Force accelerator: cuda | mps | cpu. Default: auto-detect.')
    p.add_argument('--force',       action='store_true',
                   help='Re-run even if already completed.')
    p.add_argument('--limit-batches', type=float, default=0.0,
                   help='Fraction (0,1] or number of batches per epoch for smoke tests '
                        '(0 = disabled). Status stays running → auto-resume.')
    p.add_argument('--list',        action='store_true',
                   help='Print experiment status and exit.')

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# State management
# ══════════════════════════════════════════════════════════════════════════════

def _state_file(exp_dir: Path) -> Path:
    return exp_dir / 'state.json'


def _load_state(exp_dir: Path) -> dict:
    sf = _state_file(exp_dir)
    return json.loads(sf.read_text()) if sf.exists() else {}


def _save_state(state: dict, exp_dir: Path) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    sf  = _state_file(exp_dir)
    tmp = sf.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2, default=float))
    tmp.replace(sf)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec='seconds')


def _print_state(exp_name: str, exp_dir: Path) -> None:
    state  = _load_state(exp_dir)
    status = state.get('status', 'pending')
    info   = ''
    if status == 'completed':
        monitor = state.get('checkpoint_monitor', 'val_lf')
        score   = state.get('best_val_score', '?')
        score_s = f'{score:.4f}' if isinstance(score, float) else str(score)
        info = (
            f"done {state.get('completed_at', '')}  "
            f"{monitor}={score_s}"
        )
    elif status == 'running':
        info = f"interrupted {state.get('started_at', '')}"
    print(f'{exp_name:<48}  {status:<12}  {info}')


# ══════════════════════════════════════════════════════════════════════════════
# Config loading
# ══════════════════════════════════════════════════════════════════════════════

def _build_cfg(config_file: Path, args) -> dict:
    """
    Load experiment YAML and apply CLI overrides.

    Resolves relative paths in ``data.*`` and ``training.hnet_ckpt`` to
    absolute paths under ``_ROOT`` (same pattern as main_at2sed.py).

    Priority: CLI args > YAML values.
    """
    with open(config_file) as fh:
        cfg = yaml.safe_load(fh) or {}

    # Resolve relative paths in data section
    for key, val in cfg.get('data', {}).items():
        if isinstance(val, str) and not Path(val).is_absolute():
            resolved = (_ROOT / val).resolve()
            if resolved.exists():
                cfg['data'][key] = str(resolved)

    # Resolve hnet_ckpt path
    t_cfg = cfg.get('training', {})
    if 'hnet_ckpt' in t_cfg and isinstance(t_cfg['hnet_ckpt'], str):
        p = Path(t_cfg['hnet_ckpt'])
        if not p.is_absolute():
            resolved = (_ROOT / p).resolve()
            if resolved.exists():
                t_cfg['hnet_ckpt'] = str(resolved)

    # CLI overrides
    t_cfg.setdefault('num_workers', args.num_workers)
    cfg.get('data', {}).setdefault('num_workers', args.num_workers)

    if args.lr          is not None: t_cfg['lr']         = args.lr
    if args.epochs      is not None: t_cfg['epochs']     = args.epochs
    if args.batch_size  is not None: t_cfg['batch_size'] = args.batch_size

    cfg['training'] = t_cfg
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint resume helper
# ══════════════════════════════════════════════════════════════════════════════

def _find_resume_ckpt(ckpt_dir: Path) -> Optional[str]:
    last = ckpt_dir / 'last.ckpt'
    if last.exists():
        return str(last)
    candidates = sorted(p for p in ckpt_dir.glob('*.ckpt') if p.name != 'last.ckpt')
    return str(candidates[-1]) if candidates else None


# ══════════════════════════════════════════════════════════════════════════════
# TensorBoard subprocess
# ══════════════════════════════════════════════════════════════════════════════

def _start_tensorboard(logdir: Path, port: int = _TB_PORT) -> subprocess.Popen:
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
    was_interrupted: bool,
) -> None:
    exp_dir  = _RESULTS_ROOT / exp_name
    ckpt_dir = exp_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Mark running ─────────────────────────────────────────────────────────
    _save_state({'status': 'running', 'started_at': _now_iso()}, exp_dir)

    # 2. Resume checkpoint ────────────────────────────────────────────────────
    resume_ckpt = _find_resume_ckpt(ckpt_dir) if was_interrupted else None
    if resume_ckpt:
        print(f'  Resuming from: {resume_ckpt}')

    # 3. Seed ─────────────────────────────────────────────────────────────────
    seed = int(cfg.get('experiment', {}).get('seed', 42))
    L.seed_everything(seed, workers=True)

    # 4. DataModule ───────────────────────────────────────────────────────────
    dm = SpatialDataModule(cfg)
    dm.setup('fit')

    # 5. Model ────────────────────────────────────────────────────────────────
    model = SpatialLightningModule(cfg)
    model.set_metrics_path(exp_dir / 'metrics.json')

    if was_interrupted and (exp_dir / 'metrics.json').exists():
        try:
            with open(exp_dir / 'metrics.json') as fh:
                model._epoch_metrics_log = json.load(fh)
            print(f'  Metrics log restored: {len(model._epoch_metrics_log)} entries')
        except Exception as exc:
            print(f'  [WARN] Could not restore metrics log: {exc}')

    # 6. Checkpoint callback ──────────────────────────────────────────────────
    t_cfg        = cfg.get('training', {})
    ckpt_monitor = str(t_cfg.get('checkpoint_monitor', 'val_lf'))
    if ckpt_monitor.startswith('val/'):
        ckpt_monitor = 'val_' + ckpt_monitor[4:]

    best_cb = ModelCheckpoint(
        dirpath                 = str(ckpt_dir),
        filename                = '{epoch:02d}_{' + ckpt_monitor + ':.4f}',
        monitor                 = ckpt_monitor,
        mode                    = 'max',
        save_top_k              = 1,
        save_last               = True,
        verbose                 = True,
        auto_insert_metric_name = False,
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
    batch_size = int(t_cfg.get('batch_size', _DEFAULT_BATCH_SIZE))
    accum      = max(1, EFFECTIVE_BATCH // batch_size)
    eff_batch  = batch_size * accum
    print(
        f'  batch_size={batch_size}  accum_steps={accum}  '
        f'effective_batch={eff_batch}'
        + ('  [no accumulation]' if accum == 1 else '')
    )

    # 9. Logger + Trainer ─────────────────────────────────────────────────────
    logger = TensorBoardLogger(save_dir=str(exp_dir), name='tb_logs', version=0)
    epochs = int(t_cfg.get('epochs', _DEFAULT_EPOCHS))

    trainer_kwargs = dict(
        max_epochs              = epochs,
        callbacks               = [best_cb],
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

    # 12. Cleanup + finalise ──────────────────────────────────────────────────
    if using_limit:
        print(
            '  [limit-batches] Smoke test finished.  '
            'Status remains "running" for auto-resume.'
        )
        return

    last_path = ckpt_dir / 'last.ckpt'
    if last_path.exists():
        last_path.unlink()

    best_score = float(best_cb.best_model_score or 0.0)
    _save_state({
        'status':             'completed',
        'best_ckpt':          best_cb.best_model_path,
        'best_val_score':     best_score,
        'checkpoint_monitor': ckpt_monitor,
        'completed_at':       _now_iso(),
    }, exp_dir)
    print(f'  Completed.  {ckpt_monitor}={best_score:.4f}')


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = _parse_args()

    config_file = Path(args.config_file)
    if not config_file.is_absolute():
        config_file = (_ROOT / config_file).resolve()

    if not config_file.exists():
        print(f'[ERROR] Config file not found: {config_file}')
        sys.exit(1)

    cfg      = _build_cfg(config_file, args)
    exp_name = str(cfg.get('experiment', {}).get('name', config_file.stem))
    exp_dir  = _RESULTS_ROOT / exp_name

    if args.list:
        _print_state(exp_name, exp_dir)
        return

    state  = _load_state(exp_dir)
    status = state.get('status', 'pending')

    if status == 'completed' and not args.force:
        print(f'[SKIP] {exp_name} — already completed.  Use --force to re-run.')
        return

    was_interrupted = (status == 'running')

    t_cfg = cfg.get('training', {})
    d_cfg = cfg.get('data',     {})

    loss_label = str(t_cfg.get('loss_type', 'pit')).upper()
    monitor    = str(t_cfg.get('checkpoint_monitor', 'val_lf'))

    print(f'\n{"=" * 60}')
    print(f'[RUN]  {exp_name}')
    print(f'       sr={d_cfg.get("sr")}  hop={d_cfg.get("hop_length")}  '
          f'window={d_cfg.get("window_s")}s  lr={t_cfg.get("lr", _DEFAULT_LR)}  '
          f'epochs={t_cfg.get("epochs", _DEFAULT_EPOCHS)}')
    print(f'       loss={loss_label}  monitor={monitor}')
    print(f'{"=" * 60}\n')

    _start_tensorboard(exp_dir / 'tb_logs')
    _run_experiment(exp_name, cfg, args, was_interrupted)
    print('\nDone.')


if __name__ == '__main__':
    main()
