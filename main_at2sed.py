#!/usr/bin/env python3
"""
main_at2sed.py — SED Fine-tuning on AudioSet-Strong
===========================================================
Trains ``AT2SEDLightningModule`` on AudioSet-Strong balanced+strong clips.
Designed to measure classification quality as a function of input window
duration, in preparation for pseudo-real-time inference.

Features
--------
  * Configurable window duration (``--window-s``) and hop (``--hop-s``)
  * Resumable: interrupted runs continue from ``last.ckpt``; best model kept
  * Atomic state JSON and metrics JSON (write tmp → rename)
  * TensorBoard launched automatically; killed on exit
  * ``--limit-batches`` for smoke tests (status stays ``running`` → auto-resume)

Directory layout after a run
-----------------------------
::

    results/at2sed/{exp_name}/
    ├── state.json           ← pending | running | completed
    ├── metrics.json         ← [{epoch: 0, val/map_macro: ...}, ...]
    ├── checkpoints/
    │   ├── epoch_val.ckpt   ← highest val/map_macro checkpoint
    │   └── last.ckpt        ← deleted on successful completion
    └── tb_logs/             ← TensorBoard event files

Usage
-----
::

    # Smoke test (2% batches for 1 epoch)
    .venv/bin/python main_sed_only.py --epochs 1 --limit-batches 0.02 (or --limit-batches 10 for 10 batches)

    # Default run (window=10s, 30 epochs)
    .venv/bin/python main_sed_only.py --epochs 30

    # Pseudo-RT window sweep
    .venv/bin/python main_sed_only.py --window-s 2 --hop-s 2 --epochs 30

    # Force re-run of a completed experiment
    .venv/bin/python main_sed_only.py --exp at2sed_default --force

    # Print state
    .venv/bin/python main_sed_only.py --list

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
from lightning.pytorch.loggers import TensorBoardLogger

# ── Project path setup ────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_at2sed import AT2SEDLightningModule   # noqa: E402
from datasets.audioset_strong import AudioSetStrongDataModule   # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

_RESULTS_ROOT    = _ROOT / 'runs' / 'at2sed'
_CONFIGS_DIR     = _ROOT / 'configs' / 'experiments_at2sed'
_DATASET_CFG     = _ROOT / 'configs' / 'datasets' / 'AudioSetStrong.yaml'

_DEFAULT_EXP         = 'at2sed_default'
_DEFAULT_EPOCHS      = 30
_DEFAULT_BATCH_SIZE  = 32
_DEFAULT_LR          = 1e-3
_DEFAULT_NUM_WORKERS = 4
_DEFAULT_WINDOW_S    = 0.5
_DEFAULT_HOP_S       = 0.5  # hop between windows (no overlap by default)
EFFECTIVE_BATCH      = 16   # target for gradient accumulation


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description='SED-only fine-tuning on AudioSet-Strong.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument('--exp', type=str, default=_DEFAULT_EXP,
                   help='Experiment YAML stem (from configs/experiments_at2sed/).')
    p.add_argument('--window-s',    type=float, default=_DEFAULT_WINDOW_S,
                   help='Window duration in seconds.')
    p.add_argument('--hop-s',       type=float, default=_DEFAULT_HOP_S,
                   help='Window hop in seconds.')
    p.add_argument('--lr',          type=float, default=_DEFAULT_LR,
                   help='Adam learning rate.')
    p.add_argument('--epochs',      type=int,   default=_DEFAULT_EPOCHS,
                   help='Max training epochs.')
    p.add_argument('--batch-size',  type=int,   default=_DEFAULT_BATCH_SIZE,
                   help='Batch size per step.')
    p.add_argument('--num-workers', type=int,   default=_DEFAULT_NUM_WORKERS,
                   help='DataLoader worker processes.')
    p.add_argument('--device',      type=str,   default=None,
                   metavar='ACCEL',
                   help='Force accelerator: cuda | mps | cpu. Default: auto-detect.')
    p.add_argument('--force',       action='store_true',
                   help='Re-run experiment even if already completed.')
    p.add_argument('--limit-batches', type=float, default=0.0,
                   help='Fraction (0,1] or number of batches per epoch/phase '
                        'for smoke tests (0 = disabled). Status stays running → auto-resume.')
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
        info = (
            f"done {state.get('completed_at', '')}  "
            f"best_val_map={state.get('best_val_map', '?'):.4f}"
        )
    elif status == 'running':
        info = f"interrupted {state.get('started_at', '')}"
    print(f'{exp_name:<36}  {status:<12}  {info}')


# ══════════════════════════════════════════════════════════════════════════════
# Config building
# ══════════════════════════════════════════════════════════════════════════════

def _build_cfg(exp_yaml: Path, args) -> dict:
    """
    Merge experiment YAML + dataset YAML + CLI overrides into a single cfg dict.

    Priority (highest → lowest):
        CLI args > experiment YAML > AudioSetStrong.yaml defaults
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

    # Resolve relative paths in data and semantic sections to absolute
    for section in ('data', 'semantic'):
        for key, val in cfg.get(section, {}).items():
            if isinstance(val, str) and not Path(val).is_absolute():
                resolved = (_ROOT / val).resolve()
                if resolved.exists():
                    cfg[section][key] = str(resolved)

    # 3. CLI overrides (always win)
    cfg.setdefault('data', {}).update({
        'window_s':    args.window_s,
        'hop_s':       args.hop_s,
        'num_workers': args.num_workers,
    })
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
    best_cb = ModelCheckpoint(
        dirpath                 = str(ckpt_dir),
        filename                = '{epoch:02d}_{val_map_macro:.4f}',
        monitor                 = 'val_map_macro',
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
        # Float in (0,1] → fraction; >= 1 → int number of batches
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
        # Smoke-test run: keep last.ckpt so next run resumes; status stays running
        print(
            '  [limit-batches] Smoke test finished.  '
            'Status remains "running" for auto-resume.'
        )
        return

    # Full run completed — clean up last.ckpt and mark completed
    if last_path.exists():
        last_path.unlink()

    best_val_map = float(best_cb.best_model_score or 0.0)
    state = {
        'status':       'completed',
        'best_ckpt':    best_cb.best_model_path,
        'best_val_map': best_val_map,
        'completed_at': _now_iso(),
    }
    _save_state(state, exp_dir)
    print(f'  Completed.  best_val_map={best_val_map:.4f}')


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args     = _parse_args()
    exp_name = args.exp
    exp_dir  = _RESULTS_ROOT / exp_name
    exp_yaml = _CONFIGS_DIR / f'{exp_name}.yaml'

    if not exp_yaml.exists():
        print(f'[ERROR] Experiment config not found: {exp_yaml}')
        sys.exit(1)

    if args.list:
        _print_state(exp_name, exp_dir)
        return

    state  = _load_state(exp_dir)
    status = state.get('status', 'pending')

    if status == 'completed' and not args.force:
        print(f'[SKIP] {exp_name} — already completed.  Use --force to re-run.')
        return

    was_interrupted = (status == 'running')

    cfg = _build_cfg(exp_yaml, args)

    print(f'\n{"=" * 60}')
    print(f'[RUN]  {exp_name}')
    print(f'       window={cfg["data"]["window_s"]:.1f}s  '
          f'hop={cfg["data"]["hop_s"]:.1f}s  '
          f'lr={cfg["training"]["lr"]}  '
          f'epochs={cfg["training"]["epochs"]}')
    print(f'{"=" * 60}\n')

    # Start TensorBoard (port 6007 to avoid clash with main.py on 6006)
    _start_tensorboard(exp_dir / 'tb_logs')

    _run_experiment(exp_name, cfg, args, was_interrupted)

    print('\nDone.')


if __name__ == '__main__':
    main()
