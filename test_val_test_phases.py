#!/usr/bin/env python3
"""
test_val_test_phases.py — Smoke-test for validation and test phases only.

Skips training entirely (limit_train_batches=0, max_epochs=1) and runs:
  - full validation loop  → on_validation_epoch_end → mAP/AUROC via CPU lists
  - full test loop        → on_test_epoch_end        → same + confusion matrix

Use --batches N to limit val/test to N batches (default: full split).

DELETE after confirming the fix works.
"""

import sys
import warnings
from pathlib import Path

import torch
import yaml
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

warnings.filterwarnings('ignore', message='.*No positive samples.*')
warnings.filterwarnings('ignore', message='.*elements was resized.*')

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_at2sed import AT2SEDLightningModule
from datasets.audioset_strong import AudioSetStrongDataModule

_CONFIGS_DIR = _ROOT / 'configs' / 'experiments_at2sed'
_DATASET_CFG = _ROOT / 'configs' / 'datasets' / 'AudioSetStrong.yaml'


def _build_cfg(window_s: float, hop_s: float, batch_size: int,
               num_workers: int) -> dict:
    exp_yaml = _CONFIGS_DIR / 'at2sed_default.yaml'
    with open(exp_yaml) as fh:
        cfg = yaml.safe_load(fh) or {}
    if _DATASET_CFG.exists():
        with open(_DATASET_CFG) as fh:
            ds_cfg = yaml.safe_load(fh) or {}
        for section in ('data', 'augmentation', 'training', 'semantic'):
            merged = {**ds_cfg.get(section, {}), **cfg.get(section, {})}
            if merged:
                cfg[section] = merged
    for section in ('data', 'semantic'):
        for key, val in cfg.get(section, {}).items():
            if isinstance(val, str) and not Path(val).is_absolute():
                resolved = (_ROOT / val).resolve()
                if resolved.exists():
                    cfg[section][key] = str(resolved)
    cfg.setdefault('data', {}).update(
        {'window_s': window_s, 'hop_s': hop_s, 'num_workers': num_workers})
    cfg.setdefault('training', {}).update({'batch_size': batch_size})
    return cfg


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--batches',     type=int,   default=0,
                   help='Limit val+test to N batches each (0 = full split).')
    p.add_argument('--window-s',    type=float, default=0.5)
    p.add_argument('--hop-s',       type=float, default=0.5)
    p.add_argument('--batch-size',  type=int,   default=32)
    p.add_argument('--num-workers', type=int,   default=4)
    args = p.parse_args()

    cfg = _build_cfg(args.window_s, args.hop_s, args.batch_size, args.num_workers)

    print(f'\nwindow={args.window_s}s  hop={args.hop_s}s  '
          f'batch={args.batch_size}  limit={args.batches or "full"}')
    print('Setting up DataModule...')

    L.seed_everything(42, workers=True)
    dm = AudioSetStrongDataModule(cfg)
    dm.setup('fit')

    model = AT2SEDLightningModule(cfg)
    if dm.pos_weights is not None:
        model.set_pos_weight(dm.pos_weights)
    model.set_metrics_path(_ROOT / 'runs' / '_probe' / 'metrics.json')

    # auto-detect device
    if torch.cuda.is_available():
        accel = 'cuda'
    elif torch.backends.mps.is_available():
        accel = 'mps'
    else:
        accel = 'cpu'
    print(f'Accelerator: {accel}\n')

    limit_kw = {}
    if args.batches > 0:
        limit_kw['limit_val_batches']  = args.batches
        limit_kw['limit_test_batches'] = args.batches

    trainer = L.Trainer(
        accelerator          = accel,
        devices              = 1,
        log_every_n_steps    = 10,
        num_sanity_val_steps = 0,
        logger               = TensorBoardLogger(
            save_dir=str(_ROOT / 'runs' / '_probe'), name='tb', version=0),
        enable_checkpointing = False,
        **limit_kw,
    )

    # ── Validation phase ──────────────────────────────────────────────────────
    print('=' * 56)
    print('Running VALIDATION phase...')
    print('=' * 56)
    trainer.validate(model, dm)   # calls validation_step + on_validation_epoch_end

    # ── Test phase ────────────────────────────────────────────────────────────
    print()
    print('=' * 56)
    print('Running TEST phase...')
    print('=' * 56)
    dm.setup('test')
    trainer.test(model, dm)

    print('\nDone — both phases completed without errors.')


if __name__ == '__main__':
    main()
