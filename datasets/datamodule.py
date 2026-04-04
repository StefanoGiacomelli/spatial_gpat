"""
SELDDataModule
==============
PyTorch Lightning DataModule that wraps STARSS23Dataset or TAUNIGENSDataset
and exposes train/val/test DataLoaders.

Configuration
-------------
The DataModule is driven by the same ``cfg`` dict used by NASModel.  The
relevant sections are::

    data:
      root_path    : path/to/STARSS23         # dataset root directory
      format       : foa                      # 'foa' | 'mic'
      window_s     : 10.0                     # seconds; omit for full-clip mode
      hop_s        : 5.0                      # overlap hop; omit for no overlap
      val_fraction : 0.2                      # STARSS23 only
      val_seed     : 42                       # STARSS23 only
      num_workers  : 4

    input:
      sr           : 32000                    # model sample rate
      hop_length   : 320                      # STFT/filter hop at model_sr

    heads:
      n_tracks     : 5                        # Multi-ACCDOA track count

    semantic:
      dataset      : STARSS23                 # 'STARSS23' | 'TAU-NIGENS2021'

    training:
      batch_size   : 16

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

from pathlib import Path
from typing import Optional

import lightning as L
from torch.utils.data import DataLoader, WeightedRandomSampler

from .augment      import SELDAugment
from .seld_dataset import SELDDataset
from .starss23     import STARSS23Dataset
from .tau_nigens   import TAUNIGENSDataset


class SELDDataModule(L.LightningDataModule):
    """
    LightningDataModule for SELD datasets.

    Args:
        cfg : full NAS configuration dict (as loaded from YAML).
              Must contain at minimum ``data.root_path``, ``input.sr``,
              ``input.hop_length``, ``heads.n_tracks``, and
              ``semantic.dataset``.
    """

    def __init__(self, cfg: dict):
        super().__init__()

        data_cfg = cfg.get('data', {})
        in_cfg   = cfg['input']
        head_cfg = cfg.get('heads', {})
        sem_cfg  = cfg.get('semantic', {})
        train_cfg= cfg.get('training', {})

        self.dataset_name = sem_cfg.get('dataset', 'STARSS23')
        self.root         = Path(data_cfg['root_path'])
        self.audio_format = data_cfg.get('format', 'foa')
        self.model_sr     = in_cfg['sr']
        self.hop_length   = in_cfg.get('hop_length', 320)
        self.n_tracks     = head_cfg.get('n_tracks', 5)
        self.window_s     = data_cfg.get('window_s', None)
        self.hop_s        = data_cfg.get('hop_s', None)
        self.val_fraction = data_cfg.get('val_fraction', 0.2)
        self.val_seed     = data_cfg.get('val_seed', 42)
        self.batch_size   = train_cfg.get('batch_size', 16)
        self.num_workers  = data_cfg.get('num_workers', 4)

        # ── Augmentation (train split only) ──────────────────────────────────
        aug_cfg = cfg.get('augmentation', {})
        if aug_cfg.get('enabled', False):
            self.augment_train: Optional[SELDAugment] = SELDAugment(
                audio_format = self.audio_format,
                **aug_cfg,
            )
        else:
            self.augment_train = None

        # ── WeightedRandomSampler ─────────────────────────────────────────────
        self.use_weighted_sampler: bool = train_cfg.get('weighted_sampler', False)

        self._train_ds: Optional[SELDDataset] = None
        self._val_ds:   Optional[SELDDataset] = None
        self._test_ds:  Optional[SELDDataset] = None

    # ── Dataset factory ──────────────────────────────────────────────────────

    def _make_dataset(self, split: str) -> SELDDataset:
        common = dict(
            root         = self.root,
            split        = split,
            model_sr     = self.model_sr,
            hop_length   = self.hop_length,
            n_tracks     = self.n_tracks,
            window_s     = self.window_s,
            hop_s        = self.hop_s,
            audio_format = self.audio_format,
            augment      = self.augment_train if split == 'train' else None,
        )
        if self.dataset_name == 'STARSS23':
            return STARSS23Dataset(
                **common,
                val_fraction = self.val_fraction,
                val_seed     = self.val_seed,
            )
        elif self.dataset_name == 'TAU-NIGENS2021':
            return TAUNIGENSDataset(**common)
        else:
            raise ValueError(
                f"[SELDDataModule] Unknown dataset '{self.dataset_name}'. "
                "Expected 'STARSS23' or 'TAU-NIGENS2021'."
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
        use_sampler: bool = False,
    ) -> DataLoader:
        # Variable-length full clips need custom padding; fixed windows do not.
        collate = None if ds.windowed else SELDDataset.collate_fn

        sampler = None
        if use_sampler and self.use_weighted_sampler:
            weights = ds.clip_weights()
            sampler = WeightedRandomSampler(
                weights,
                num_samples = len(weights),
                replacement = True,
            )
            shuffle = False   # sampler and shuffle are mutually exclusive

        return DataLoader(
            ds,
            batch_size         = self.batch_size,
            shuffle            = shuffle,
            sampler            = sampler,
            drop_last          = drop_last,
            num_workers        = self.num_workers,
            collate_fn         = collate,
            pin_memory         = False,
            persistent_workers = self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        # drop_last=True: avoids partial batches (B=1) that crash BatchNorm layers
        return self._make_loader(self._train_ds, shuffle=True, drop_last=True, use_sampler=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self._val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self._test_ds, shuffle=False)

    # ── Convenience properties (usable after setup) ───────────────────────────

    @property
    def train_dataset(self) -> Optional[SELDDataset]:
        return self._train_ds

    @property
    def val_dataset(self) -> Optional[SELDDataset]:
        return self._val_ds

    @property
    def test_dataset(self) -> Optional[SELDDataset]:
        return self._test_ds
