"""
FinetuneDataModule
==================
PyTorch Lightning DataModule for Phase 4 fine-tuning.

Train split  : MixedSELDDataset (STARSS23 train + TAU-NIGENS2021 train)
               WeightedRandomSampler active by default for class balancing.
Val / test   : STARSS23 only — same canonical split used in all NAS phases,
               ensuring metrics are directly comparable across phases.

Configuration
-------------
The DataModule is driven by the same ``cfg`` dict structure used by the NAS
training scripts.  Relevant keys::

    data:
      starss_root  : path/to/STARSS23        # STARSS23 dataset root
      tau_root     : path/to/TAU-NIGENS2021  # TAU-NIGENS2021 dataset root
      format       : foa                     # 'foa' | 'mic'
      window_s     : 10.0                    # seconds; null → full-clip mode
      hop_s        : 5.0                     # overlap hop (windowed mode only)
      val_fraction : 0.2                     # STARSS23 train/val split fraction
      val_seed     : 42
      num_workers  : 4

    input:
      sr           : 32000                   # model sample rate
      hop_length   : 320                     # STFT/filter hop at model_sr

    heads:
      n_tracks     : 2                       # Multi-ACCDOA track count

    training:
      batch_size   : 1

    augmentation:
      enabled      : true                    # set in dataset config YAML

Author
------
Stefano Giacomelli, Ph.D. candidate — DISIM, University of L'Aquila
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import lightning as L
from torch.utils.data import DataLoader, WeightedRandomSampler

from .augment            import SELDAugment
from .mixed_seld_dataset import MixedSELDDataset
from .seld_dataset       import SELDDataset
from .starss23           import STARSS23Dataset
from .tau_nigens         import TAUNIGENSDataset


class FinetuneDataModule(L.LightningDataModule):
    """
    LightningDataModule for Phase 4 fine-tuning on mixed STARSS23 + TAU-NIGENS.

    Args:
        cfg : full NAS configuration dict (as loaded from YAML and enriched by
              _build_finetune_cfg in main_finetune.py).
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()

        data_cfg  = cfg.get('data', {})
        in_cfg    = cfg['input']
        head_cfg  = cfg.get('heads', {})
        train_cfg = cfg.get('training', {})

        # ── Dataset roots ─────────────────────────────────────────────────────
        self.starss_root = Path(data_cfg['starss_root'])
        self.tau_root    = Path(data_cfg['tau_root'])
        self.audio_format = data_cfg.get('format', 'foa')

        # ── Model feature params ──────────────────────────────────────────────
        self.model_sr   = in_cfg['sr']
        self.hop_length = in_cfg.get('hop_length', 320)
        self.n_tracks   = head_cfg.get('n_tracks', 2)

        # ── Windowing ─────────────────────────────────────────────────────────
        self.window_s     = data_cfg.get('window_s', None)
        self.hop_s        = data_cfg.get('hop_s', None)

        # ── STARSS23 train/val split ──────────────────────────────────────────
        self.val_fraction = data_cfg.get('val_fraction', 0.2)
        self.val_seed     = data_cfg.get('val_seed', 42)

        # ── DataLoader params ─────────────────────────────────────────────────
        self.batch_size  = train_cfg.get('batch_size', 1)
        self.num_workers = data_cfg.get('num_workers', 4)

        # ── Augmentation (training split only) ───────────────────────────────
        aug_cfg = cfg.get('augmentation', {})
        if aug_cfg.get('enabled', False):
            self.augment_train: Optional[SELDAugment] = SELDAugment(
                audio_format=self.audio_format,
                **aug_cfg,
            )
        else:
            self.augment_train = None

        # ── Dataset handles (populated in setup) ─────────────────────────────
        self._train_ds: Optional[MixedSELDDataset] = None
        self._val_ds:   Optional[STARSS23Dataset]  = None
        self._test_ds:  Optional[STARSS23Dataset]  = None

    # ── Dataset construction ─────────────────────────────────────────────────

    def _common_kwargs(self, split: str, augment: bool = False) -> dict:
        """Shared keyword arguments for both dataset constructors."""
        return dict(
            split        = split,
            model_sr     = self.model_sr,
            hop_length   = self.hop_length,
            n_tracks     = self.n_tracks,
            window_s     = self.window_s,
            hop_s        = self.hop_s,
            audio_format = self.audio_format,
            augment      = self.augment_train if augment else None,
        )

    # ── Lightning hooks ──────────────────────────────────────────────────────

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ('fit', None):
            starss_train = STARSS23Dataset(
                root         = self.starss_root,
                val_fraction = self.val_fraction,
                val_seed     = self.val_seed,
                **self._common_kwargs('train', augment=True),
            )
            tau_train = TAUNIGENSDataset(
                root = self.tau_root,
                **self._common_kwargs('train', augment=True),
            )
            self._train_ds = MixedSELDDataset(starss_train, tau_train)

            self._val_ds = STARSS23Dataset(
                root         = self.starss_root,
                val_fraction = self.val_fraction,
                val_seed     = self.val_seed,
                **self._common_kwargs('val'),
            )

        if stage in ('test', None):
            self._test_ds = STARSS23Dataset(
                root         = self.starss_root,
                val_fraction = self.val_fraction,
                val_seed     = self.val_seed,
                **self._common_kwargs('test'),
            )

    def _make_loader(
        self,
        ds,
        shuffle: bool,
        drop_last: bool = False,
        use_sampler: bool = False,
    ) -> DataLoader:
        # Variable-length full clips require padding collate; windowed mode does not.
        collate = None if ds.windowed else SELDDataset.collate_fn

        sampler = None
        if use_sampler and isinstance(ds, MixedSELDDataset):
            weights = ds.clip_weights()
            sampler = WeightedRandomSampler(
                weights,
                num_samples = len(weights),
                replacement = True,
            )
            shuffle = False   # mutually exclusive with sampler

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
        # drop_last=True avoids partial batches that can crash BatchNorm with B=1
        return self._make_loader(
            self._train_ds, shuffle=True, drop_last=True, use_sampler=True
        )

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self._val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self._test_ds, shuffle=False)

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def train_dataset(self) -> Optional[MixedSELDDataset]:
        return self._train_ds

    @property
    def val_dataset(self) -> Optional[STARSS23Dataset]:
        return self._val_ds

    @property
    def test_dataset(self) -> Optional[STARSS23Dataset]:
        return self._test_ds
