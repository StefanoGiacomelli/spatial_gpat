"""
DataModules — all PyTorch Lightning DataModules for spatial_gpat
================================================================

Catalog
-------
SELDDataModule           : Single-dataset SELD (STARSS23, TAU-NIGENS2020/2021, TAU2019).
BalancedSELDDataModule   : Balanced 3-dataset training anchored on STARSS23 13-class space.
FoldCyclingDataModule    : TAU2019 with 4-fold epoch cycling.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import lightning as L
from torch.utils.data import DataLoader

from .utils          import SELDAugment, SELDDataset
from .starss23       import STARSS23Dataset
from .tau_nigens     import TAUNIGENSDataset
from .tau_nigens2020 import TAUNIGENS2020Dataset
from .tau2019        import TAU2019Dataset
from .balanced   import BalancedSELDDataModule  # re-export so callers can import from datamodule


# ══════════════════════════════════════════════════════════════════════════════
# SELDDataModule
# ══════════════════════════════════════════════════════════════════════════════

class SELDDataModule(L.LightningDataModule):
    """
    LightningDataModule for a single SELD dataset.

    Supported datasets (``semantic.dataset``):
        'STARSS23'       — 13 classes, 24 kHz, room-stratified split
        'TAU-NIGENS2021' — 12 classes, 24 kHz, fold-based split (subdirs)
        'TAU-NIGENS2020' — 14 classes, 24 kHz, fold-based split (flat dir)
        'TAU2019'        — 11 classes, 48→24 kHz, 4-fold CV

    Args:
        cfg : full NAS configuration dict.  Must contain at minimum
              ``data.root_path``, ``input.sr``, ``input.hop_length``,
              ``heads.n_tracks``, and ``semantic.dataset``.
    """

    def __init__(self, cfg: dict):
        super().__init__()

        data_cfg  = cfg.get('data', {})
        in_cfg    = cfg['input']
        head_cfg  = cfg.get('heads', {})
        sem_cfg   = cfg.get('semantic', {})
        train_cfg = cfg.get('training', {})

        self.dataset_name = sem_cfg.get('dataset', 'STARSS23')
        self.root         = Path(data_cfg['root_path'])
        self.model_sr     = in_cfg['sr']
        self.hop_length   = in_cfg.get('hop_length', 320)
        self.n_tracks     = head_cfg.get('n_tracks', 5)
        self.window_s     = data_cfg.get('window_s', None)
        self.hop_s        = data_cfg.get('hop_s', None)
        self.val_fraction = data_cfg.get('val_fraction', 0.2)
        self.val_seed     = data_cfg.get('val_seed', 42)
        self.fold         = data_cfg.get('fold', 1)           # TAU2019 only
        self.batch_size   = train_cfg.get('batch_size', 16)
        self.num_workers  = data_cfg.get('num_workers', 4)

        aug_cfg = cfg.get('augmentation', {})
        if aug_cfg.get('enabled', False):
            self.augment_train: Optional[SELDAugment] = SELDAugment(**aug_cfg)
        else:
            self.augment_train = None

        self._train_ds: Optional[SELDDataset] = None
        self._val_ds:   Optional[SELDDataset] = None
        self._test_ds:  Optional[SELDDataset] = None

    # ── Dataset factory ──────────────────────────────────────────────────────

    def _make_dataset(self, split: str) -> SELDDataset:
        common = dict(
            split      = split,
            model_sr   = self.model_sr,
            hop_length = self.hop_length,
            n_tracks   = self.n_tracks,
            window_s   = self.window_s,
            hop_s      = self.hop_s,
            augment    = self.augment_train if split == 'train' else None,
        )
        if self.dataset_name == 'STARSS23':
            return STARSS23Dataset(
                root         = self.root,
                val_fraction = self.val_fraction,
                val_seed     = self.val_seed,
                **common,
            )
        elif self.dataset_name == 'TAU-NIGENS2021':
            return TAUNIGENSDataset(root=self.root, **common)
        elif self.dataset_name == 'TAU-NIGENS2020':
            return TAUNIGENS2020Dataset(root=self.root, **common)
        elif self.dataset_name == 'TAU2019':
            return TAU2019Dataset(root=self.root, fold=self.fold, **common)
        else:
            raise ValueError(
                f"[SELDDataModule] Unknown dataset '{self.dataset_name}'. "
                "Expected 'STARSS23', 'TAU-NIGENS2021', 'TAU-NIGENS2020', or 'TAU2019'."
            )

    # ── Lightning hooks ──────────────────────────────────────────────────────

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ('fit', None):
            self._train_ds = self._make_dataset('train')
            self._val_ds   = self._make_dataset('val')
        if stage in ('test', None):
            self._test_ds  = self._make_dataset('test')

    def _make_loader(
        self,
        ds: SELDDataset,
        shuffle: bool,
        drop_last: bool = False,
    ) -> DataLoader:
        collate = None if ds.windowed else SELDDataset.collate_fn
        return DataLoader(
            ds,
            batch_size         = self.batch_size,
            shuffle            = shuffle,
            drop_last          = drop_last,
            num_workers        = self.num_workers,
            collate_fn         = collate,
            pin_memory         = False,
            persistent_workers = self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self._train_ds, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self._val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self._test_ds, shuffle=False)

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def train_dataset(self) -> Optional[SELDDataset]:
        return self._train_ds

    @property
    def val_dataset(self) -> Optional[SELDDataset]:
        return self._val_ds

    @property
    def test_dataset(self) -> Optional[SELDDataset]:
        return self._test_ds



# ══════════════════════════════════════════════════════════════════════════════
# FoldCyclingDataModule
# ══════════════════════════════════════════════════════════════════════════════

class FoldCyclingDataModule(SELDDataModule):
    """
    LightningDataModule for TAU2019 with 4-fold epoch cycling.

    train_dataloader() and val_dataloader() create a fresh TAU2019Dataset for
    fold = (trainer.current_epoch % 4) + 1 at the start of each epoch.
    The test split always uses mixed_mode=True (eval/split0).

    Fold schedule
    -------------
    epoch 0 → fold 1 : train=[split3,4], val=[split2]
    epoch 1 → fold 2 : train=[split4,1], val=[split3]
    epoch 2 → fold 3 : train=[split1,2], val=[split4]
    epoch 3 → fold 4 : train=[split2,3], val=[split1]
    epoch 4 → fold 1 (cycle repeats)

    Requires ``Trainer(reload_dataloaders_every_n_epochs=1)`` in the runner.
    Use a large EarlyStopping patience (≥ 20 epochs) because the validation
    split changes every 4 epochs.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        if self.dataset_name != 'TAU2019':
            raise ValueError(
                f"FoldCyclingDataModule expects dataset='TAU2019', "
                f"got {self.dataset_name!r}."
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _current_fold(self) -> int:
        trainer = getattr(self, 'trainer', None)
        epoch   = int(getattr(trainer, 'current_epoch', 0) or 0)
        return (epoch % 4) + 1

    def _make_tau2019(self, split: str, fold: int) -> TAU2019Dataset:
        return TAU2019Dataset(
            root       = self.root,
            split      = split,
            fold       = fold,
            mixed_mode = (split == 'test'),
            model_sr   = self.model_sr,
            hop_length = self.hop_length,
            n_tracks   = self.n_tracks,
            window_s   = self.window_s,
            hop_s      = self.hop_s,
            augment    = self.augment_train if split == 'train' else None,
        )

    # ── Lightning hooks ───────────────────────────────────────────────────────

    def setup(self, stage: Optional[str] = None) -> None:
        # Train/val datasets are built fresh each epoch in train/val_dataloader.
        # Only the test split needs eager setup (fold-independent).
        if stage in ('test', None):
            self._test_ds = self._make_tau2019('test', fold=1)

    def train_dataloader(self) -> DataLoader:
        fold  = self._current_fold()
        epoch = int(getattr(getattr(self, 'trainer', None), 'current_epoch', 0) or 0)
        ds    = self._make_tau2019('train', fold=fold)
        print(f'[FoldCyclingDataModule] epoch={epoch}  fold={fold}  '
              f'train items={len(ds)}')
        return self._make_loader(ds, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        fold = self._current_fold()
        ds   = self._make_tau2019('val', fold=fold)
        return self._make_loader(ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self._test_ds, shuffle=False)

