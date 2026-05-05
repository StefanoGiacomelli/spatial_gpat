"""
DataModules — all PyTorch Lightning DataModules for spatial_gpat
================================================================

Catalog
-------
SELDDataModule        : Single-dataset SELD (STARSS23, TAU-NIGENS2021, or TAU2019).
MixedSELDDataModule   : Mixed STARSS23 + TAU-NIGENS2021 + TAU2019, unified 15-class schema.
AudioSetStrongDataModule : AudioSet-Strong SED fine-tuning (GP-AT / E-PANNs backbone).
STARSS23SEDDataModule : STARSS23 repackaged as SED-style batches for AT2SED training.

Note: ``semantic.dataset`` is injected automatically by all entry-point scripts
(main_seld_only.py, main_finetune.py, …) and should not be set manually in YAMLs.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import lightning as L
from torch.utils.data import DataLoader, WeightedRandomSampler

from .utils      import (SELDAugment, SELDDataset, MixedSELDDataset,
                         STARSS_TO_UNIFIED, NIGENS_TO_UNIFIED, TAU2019_TO_UNIFIED)
from .starss23   import STARSS23Dataset
from .tau_nigens import TAUNIGENSDataset
from .tau2019    import TAU2019Dataset
from .audioset_strong import AudioSetAugment, AudioSetStrongDataset


# ══════════════════════════════════════════════════════════════════════════════
# SELDDataModule
# ══════════════════════════════════════════════════════════════════════════════

class SELDDataModule(L.LightningDataModule):
    """
    LightningDataModule for a single SELD dataset.

    Supported datasets (``semantic.dataset``):
        'STARSS23'       — 13 classes, 24 kHz, room-stratified split
        'TAU-NIGENS2021' — 12 classes, 24 kHz, fold-based split
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
        elif self.dataset_name == 'TAU2019':
            return TAU2019Dataset(root=self.root, fold=self.fold, **common)
        else:
            raise ValueError(
                f"[SELDDataModule] Unknown dataset '{self.dataset_name}'. "
                "Expected 'STARSS23', 'TAU-NIGENS2021', or 'TAU2019'."
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
# MixedSELDDataModule
# ══════════════════════════════════════════════════════════════════════════════

class MixedSELDDataModule(L.LightningDataModule):
    """
    LightningDataModule for joint training on STARSS23, TAU-NIGENS2021, TAU2019.

    Produces ``MixedSELDDataset`` items with a unified 15-class schema.

    Dataset splits:
        STARSS23       — room-stratified train/val, dev-test as test
        TAU-NIGENS2021 — fold-based train/val/test (official DCASE2021 split)
        TAU2019        — mixed_mode=True: train=splits2-4, val=split1, test=eval/split0

    Configuration keys (under ``data.``)::

        starss_root   : path/to/STARSS23
        tau_root      : path/to/TAU-NIGENS2021
        tau2019_root  : path/to/TAU2019
        window_s      : seconds (null = full-clip mode)
        hop_s         : overlap hop (windowed mode only)
        val_fraction  : STARSS23 train/val split fraction (default 0.2)
        val_seed      : RNG seed for STARSS23 split (default 42)
        num_workers   : DataLoader workers (default 4)
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()

        data_cfg  = cfg.get('data', {})
        in_cfg    = cfg['input']
        head_cfg  = cfg.get('heads', {})
        train_cfg = cfg.get('training', {})

        self.starss_root  = Path(data_cfg['starss_root'])
        self.tau_root     = Path(data_cfg['tau_root'])
        self.tau2019_root = Path(data_cfg['tau2019_root'])

        self.model_sr   = in_cfg['sr']
        self.hop_length = in_cfg.get('hop_length', 320)
        self.n_tracks   = head_cfg.get('n_tracks', 3)

        self.window_s     = data_cfg.get('window_s', None)
        self.hop_s        = data_cfg.get('hop_s', None)
        self.val_fraction = data_cfg.get('val_fraction', 0.2)
        self.val_seed     = data_cfg.get('val_seed', 42)

        self.batch_size  = train_cfg.get('batch_size', 1)
        self.num_workers = data_cfg.get('num_workers', 4)

        aug_cfg = cfg.get('augmentation', {})
        if aug_cfg.get('enabled', False):
            self.augment_train: Optional[SELDAugment] = SELDAugment(**aug_cfg)
        else:
            self.augment_train = None

        self._train_ds: Optional[MixedSELDDataset] = None
        self._val_ds:   Optional[MixedSELDDataset] = None
        self._test_ds:  Optional[MixedSELDDataset] = None

    # ── Dataset construction ─────────────────────────────────────────────────

    def _make_mixed(self, split: str, augment=None) -> MixedSELDDataset:
        common = dict(
            split      = split,
            model_sr   = self.model_sr,
            hop_length = self.hop_length,
            n_tracks   = self.n_tracks,
            window_s   = self.window_s,
            hop_s      = self.hop_s,
        )
        starss_ds = STARSS23Dataset(
            root         = self.starss_root,
            val_fraction = self.val_fraction,
            val_seed     = self.val_seed,
            augment      = augment,
            **common,
        )
        tau_ds = TAUNIGENSDataset(
            root    = self.tau_root,
            augment = augment,
            **common,
        )
        tau2019_ds = TAU2019Dataset(
            root       = self.tau2019_root,
            mixed_mode = True,
            augment    = augment,
            **common,
        )
        return MixedSELDDataset(
            datasets = [starss_ds, tau_ds, tau2019_ds],
            mappings = [STARSS_TO_UNIFIED, NIGENS_TO_UNIFIED, TAU2019_TO_UNIFIED],
        )

    # ── Lightning hooks ──────────────────────────────────────────────────────

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ('fit', None):
            self._train_ds = self._make_mixed('train', augment=self.augment_train)
            self._val_ds   = self._make_mixed('val',   augment=None)
        if stage in ('test', None):
            self._test_ds  = self._make_mixed('test',  augment=None)

    def _make_loader(
        self, ds: MixedSELDDataset, shuffle: bool, drop_last: bool = False
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
    def train_dataset(self) -> Optional[MixedSELDDataset]:
        return self._train_ds

    @property
    def val_dataset(self) -> Optional[MixedSELDDataset]:
        return self._val_ds

    @property
    def test_dataset(self) -> Optional[MixedSELDDataset]:
        return self._test_ds


# ══════════════════════════════════════════════════════════════════════════════
# MixedSELDTestDataModule
# ══════════════════════════════════════════════════════════════════════════════

class MixedSELDTestDataModule(L.LightningDataModule):
    """Test-only DataModule for models trained on MixedSELD (15 unified classes).

    Provides 3 DataLoaders, each evaluating one dataset with unified 15-class targets:
        idx=0 → STARSS23  test split (dev-test-sony + dev-test-tau)
        idx=1 → TAU-NIGENS2021 test split (official fold 6)
        idx=2 → TAU2019 eval/split0 (mixed_mode=True, split='test')

    Targets are mapped to the unified 15-class schema (same as MixedSELDDataModule),
    so metrics are directly comparable between datasets and against training targets.

    Args:
        cfg : full configuration dict.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()

        data_cfg  = cfg.get('data', {})
        in_cfg    = cfg['input']
        head_cfg  = cfg.get('heads', {})
        train_cfg = cfg.get('training', {})

        self.starss_root  = Path(data_cfg['starss_root'])
        self.tau_root     = Path(data_cfg['tau_root'])
        self.tau2019_root = Path(data_cfg['tau2019_root'])

        self.model_sr   = in_cfg['sr']
        self.hop_length = in_cfg.get('hop_length', 320)
        self.n_tracks   = head_cfg.get('n_tracks', 3)

        self.window_s    = data_cfg.get('window_s', None)
        self.hop_s       = data_cfg.get('hop_s', None)
        self.batch_size  = train_cfg.get('batch_size', 1)
        self.num_workers = data_cfg.get('num_workers', 4)

        self._starss_ds:  Optional[MixedSELDDataset] = None
        self._tau_ds:     Optional[MixedSELDDataset] = None
        self._tau2019_ds: Optional[MixedSELDDataset] = None

    def _make_loader(self, ds: MixedSELDDataset) -> DataLoader:
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

    def setup(self, stage: Optional[str] = None) -> None:
        common = dict(
            split      = 'test',
            model_sr   = self.model_sr,
            hop_length = self.hop_length,
            n_tracks   = self.n_tracks,
            window_s   = self.window_s,
            hop_s      = self.hop_s,
            augment    = None,
        )
        starss_ds = STARSS23Dataset(root=self.starss_root, **common)
        tau_ds    = TAUNIGENSDataset(root=self.tau_root, **common)
        tau2019_ds = TAU2019Dataset(root=self.tau2019_root, mixed_mode=True, **common)

        self._starss_ds  = MixedSELDDataset([starss_ds],  [STARSS_TO_UNIFIED])
        self._tau_ds     = MixedSELDDataset([tau_ds],     [NIGENS_TO_UNIFIED])
        self._tau2019_ds = MixedSELDDataset([tau2019_ds], [TAU2019_TO_UNIFIED])

    def test_dataloader(self) -> List[DataLoader]:
        return [
            self._make_loader(self._starss_ds),
            self._make_loader(self._tau_ds),
            self._make_loader(self._tau2019_ds),
        ]


# ══════════════════════════════════════════════════════════════════════════════
# AudioSetStrongDataModule
# ══════════════════════════════════════════════════════════════════════════════

class AudioSetStrongDataModule(L.LightningDataModule):
    """
    LightningDataModule for AudioSet-Strong SED fine-tuning.

    Wraps :class:`AudioSetStrongDataset` and exposes train/val/test DataLoaders.
    After calling ``setup('fit')``, the ``pos_weights`` property returns the
    class-imbalance correction weights computed on the training set (post-windowing).

    Configuration (``cfg`` dict keys used)::

        data:
          audio_root_train       : path to balanced_train_segments/
          audio_root_unbalanced  : path to unbalanced_strong/ (optional extra dir)
          audio_root_test        : path to eval_segments/
          strong_tsv_train       : path to audioset_train_strong.tsv
          strong_tsv_test        : path to audioset_eval_strong.tsv
          class_map          : path to class_labels_indices.csv
          window_s           : window duration in seconds
          hop_s              : hop between windows (default = window_s)
          val_fraction       : fraction of clips for validation (default 0.2)
          val_seed           : RNG seed for train/val split (default 42)
          overlap_threshold  : fractional overlap threshold (default 0.0)
          num_workers        : DataLoader workers (default 4)

        training:
          batch_size         : batch size (default 32)
          weighted_sampler   : bool — use WeightedRandomSampler (default false)
          pos_weight_max     : cap on pos_weight values (default 20.0)
          max_negative_fraction : max fraction of negative-only windows in train
          class_balance      : bool — augment only rare classes (default false)
          rare_class_threshold : pos_rate below this → rare class (default 0.1)

        augmentation:
          enabled            : bool (default false)
          p_gain / gain_db   : random gain parameters
          p_noise / noise_snr_db : white noise parameters

        model:
          n_cls              : number of target classes (default 527)
    """

    def __init__(self, cfg: dict):
        super().__init__()

        data_cfg  = cfg.get('data', {})
        train_cfg = cfg.get('training', {})
        aug_cfg   = cfg.get('augmentation', {})

        self.audio_root_train = Path(data_cfg['audio_root_train'])
        self.audio_root_test  = Path(data_cfg['audio_root_test'])
        self.strong_tsv_train = Path(data_cfg['strong_tsv_train'])
        self.strong_tsv_test  = Path(data_cfg['strong_tsv_test'])
        self.class_map        = Path(data_cfg['class_map'])

        # Optional extra directory for unbalanced_strong clips (train+val only)
        _unbal = data_cfg.get('audio_root_unbalanced', None)
        self.audio_root_unbalanced: Optional[Path] = Path(_unbal) if _unbal else None

        self.window_s             = float(data_cfg.get('window_s', 10.0))
        self.hop_s                = float(data_cfg.get('hop_s', self.window_s))
        self.n_cls                = int(cfg.get('model', {}).get('n_cls', 527))
        self.overlap_threshold    = float(data_cfg.get('overlap_threshold', 0.0))
        self.val_fraction         = float(data_cfg.get('val_fraction', 0.2))
        self.val_seed             = int(data_cfg.get('val_seed', 42))
        self.num_workers          = int(data_cfg.get('num_workers', 4))

        self.batch_size               = int(train_cfg.get('batch_size', 32))
        self.use_weighted_sampler     = bool(train_cfg.get('weighted_sampler', False))
        self.pos_weight_max           = float(train_cfg.get('pos_weight_max', 20.0))
        self.max_neg_frac             = float(train_cfg.get('max_negative_fraction', 1.0))
        self.balanced_sampler_formula = str(train_cfg.get('balanced_sampler_formula', 'maxpos'))

        # ── HLP ───────────────────────────────────────────────────────────────
        self.hlp_enabled    = bool(data_cfg.get('hlp_enabled', False))
        _hlp_path           = data_cfg.get('hlp_table_path', None)
        self.hlp_table_path = Path(_hlp_path) if _hlp_path else None

        # ── Augmentation (train only) ─────────────────────────────────────────
        if aug_cfg.get('enabled', False):
            self._augment = AudioSetAugment(
                class_balance        = bool(train_cfg.get('class_balance', False)),
                rare_class_threshold = float(train_cfg.get('rare_class_threshold', 0.1)),
                **aug_cfg,
            )
        else:
            self._augment = None

        self._train_ds: Optional[AudioSetStrongDataset] = None
        self._val_ds:   Optional[AudioSetStrongDataset] = None
        self._test_ds:  Optional[AudioSetStrongDataset] = None
        self._pos_weights: Optional[torch.Tensor]       = None

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self, stage: Optional[str] = None) -> None:
        """Instantiate datasets for the requested stage."""
        # Guard against Lightning calling setup() a second time after we already
        # ran it manually in main_at2sed.py (avoids duplicate dataset construction
        # and duplicate prints).
        if stage in ('fit', None) and self._train_ds is not None:
            return
        if stage == 'test' and self._test_ds is not None:
            return

        common_train = dict(
            strong_tsv               = self.strong_tsv_train,
            class_map_csv            = self.class_map,
            window_s                 = self.window_s,
            hop_s                    = self.hop_s,
            n_cls                    = self.n_cls,
            overlap_threshold        = self.overlap_threshold,
            val_fraction             = self.val_fraction,
            val_seed                 = self.val_seed,
            max_negative_fraction    = self.max_neg_frac,
            hlp_enabled              = self.hlp_enabled,
            hlp_table_path           = self.hlp_table_path,
            balanced_sampler_formula = self.balanced_sampler_formula,
        )

        extra_dirs = ([self.audio_root_unbalanced]
                      if self.audio_root_unbalanced and self.audio_root_unbalanced.exists()
                      else [])

        if stage in ('fit', None):
            self._train_ds = AudioSetStrongDataset(
                split            = 'train',
                audio_dir        = self.audio_root_train,
                extra_audio_dirs = extra_dirs,
                augment          = self._augment,
                **common_train,
            )
            self._val_ds = AudioSetStrongDataset(
                split            = 'val',
                audio_dir        = self.audio_root_train,
                extra_audio_dirs = extra_dirs,
                augment          = None,
                **common_train,
            )
            # Compute and store pos_weights from training items
            pw = self._train_ds._compute_pos_weights()
            pw = pw.clamp(max=self.pos_weight_max)

            # If class_balance augmentation is enabled, identify rare classes
            if self._augment is not None and self._augment.class_balance:
                pos_rates = 1.0 / (1.0 + pw.numpy())   # approximate from weights
                thresh    = self._augment.rare_class_threshold
                rare      = set(int(c) for c in np.where(pos_rates < thresh)[0])
                self._augment.set_rare_classes(rare)
                print(
                    f'[AudioSetStrongDataModule] Class-balance: {len(rare)} rare classes '
                    f'(pos_rate < {thresh})'
                )

            self._pos_weights = pw

        if stage in ('test', None):
            self._test_ds = AudioSetStrongDataset(
                split               = 'test',
                audio_dir           = self.audio_root_test,
                strong_tsv          = self.strong_tsv_test,
                class_map_csv       = self.class_map,
                window_s            = self.window_s,
                hop_s               = self.hop_s,
                n_cls               = self.n_cls,
                overlap_threshold   = self.overlap_threshold,
                augment             = None,
                # val_fraction / val_seed not used for test split
            )

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _make_loader(
        self,
        ds: AudioSetStrongDataset,
        shuffle: bool,
        drop_last: bool = False,
        use_sampler: bool = False,
    ) -> DataLoader:
        sampler = None
        if use_sampler and self.use_weighted_sampler:
            weights = ds.item_weights()
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
            pin_memory         = False,
            persistent_workers = self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        # drop_last=True avoids B=1 batches that crash BatchNorm
        return self._make_loader(self._train_ds, shuffle=True, drop_last=True, use_sampler=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self._val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self._test_ds, shuffle=False)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def pos_weights(self) -> Optional[torch.Tensor]:
        """
        Per-class positive weights (capped at ``pos_weight_max``).

        Available after ``setup('fit')`` is called.  ``None`` before setup.
        """
        return self._pos_weights

    @property
    def train_dataset(self) -> Optional[AudioSetStrongDataset]:
        return self._train_ds

    @property
    def val_dataset(self) -> Optional[AudioSetStrongDataset]:
        return self._val_ds

    @property
    def test_dataset(self) -> Optional[AudioSetStrongDataset]:
        return self._test_ds


# ══════════════════════════════════════════════════════════════════════════════
# STARSS23SEDDataModule  +  _STARSS23SEDWrapper
# ══════════════════════════════════════════════════════════════════════════════

class STARSS23SEDDataModule(L.LightningDataModule):
    """
    LightningDataModule wrapping STARSS23Dataset for AT2SED-style training.

    Produces ``(wav_mono, label_multihot)`` batches compatible with
    ``AT2SEDLightningModule``.  The output format mirrors
    ``AudioSetStrongDataModule`` so the same training loop handles both datasets.

    Batch format
    ------------
    ``wav_mono  : (B, T_samples)``  float32, W-channel (FOA index 0) at 32 kHz
    ``label     : (B, 13)``         float32 multi-hot, class c active iff any
                                     track in any frame has a non-zero DOA vector

    Configuration keys
    ------------------
    ::

        experiment:
          dataset: starss23

        model:
          n_cls: 13

        data:
          starss_root:   data/STARSS23
          window_s:      10.0
          hop_s:         10.0
          val_fraction:  0.2
          val_seed:      42
          num_workers:   4

        training:
          batch_size:          4
          weighted_sampler:    true
          pos_weight_max:      20.0

        augmentation:
          enabled: false

    Args:
        cfg : merged config dict (from main_at2sed._build_cfg).
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()

        data_cfg  = cfg.get('data', {})
        train_cfg = cfg.get('training', {})
        aug_cfg   = cfg.get('augmentation', {})

        # ── Paths ─────────────────────────────────────────────────────────────
        self.starss_root = Path(data_cfg['starss_root'])

        # ── Windowing ─────────────────────────────────────────────────────────
        self.window_s = data_cfg.get('window_s', None)
        self.hop_s    = data_cfg.get('hop_s',    None)

        # ── Split config ──────────────────────────────────────────────────────
        self.val_fraction = float(data_cfg.get('val_fraction', 0.2))
        self.val_seed     = int(data_cfg.get('val_seed', 42))

        # ── DataLoader params ─────────────────────────────────────────────────
        self.batch_size      = int(train_cfg.get('batch_size', 4))
        self.num_workers     = int(data_cfg.get('num_workers', 4))
        self.weighted_sampler = bool(train_cfg.get('weighted_sampler', True))
        self.pos_weight_max  = float(train_cfg.get('pos_weight_max', 20.0))

        # ── Augmentation (training split only) ───────────────────────────────
        self._augment: Optional[SELDAugment] = None
        if aug_cfg.get('enabled', False):
            self._augment = SELDAugment(**aug_cfg)

        # ── Internal state ────────────────────────────────────────────────────
        self._train_ds: Optional[STARSS23Dataset] = None
        self._val_ds:   Optional[STARSS23Dataset] = None
        self._test_ds:  Optional[STARSS23Dataset] = None
        self._pos_weights: Optional[torch.Tensor] = None

    # ── Lightning hooks ──────────────────────────────────────────────────────

    def setup(self, stage: Optional[str] = None) -> None:
        common = dict(
            root         = self.starss_root,
            model_sr     = 32000,
            hop_length   = 320,
            n_tracks     = 2,
            window_s     = self.window_s,
            hop_s        = self.hop_s,
            val_fraction = self.val_fraction,
            val_seed     = self.val_seed,
        )

        if stage in ('fit', None):
            self._train_ds = STARSS23Dataset(split='train', augment=self._augment, **common)
            self._val_ds   = STARSS23Dataset(split='val',   augment=None,           **common)

        if stage in ('test', None):
            self._test_ds = STARSS23Dataset(split='test', augment=None, **common)

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self._train_ds, shuffle=True, drop_last=True, use_sampler=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self._val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self._test_ds, shuffle=False)

    # ── Convenience ──────────────────────────────────────────────────────────

    @property
    def pos_weights(self) -> Optional[torch.Tensor]:
        """(13,) pos_weight tensor; available after setup('fit')."""
        return self._pos_weights

    # ── Private helpers ───────────────────────────────────────────────────────

    def _make_loader(
        self,
        ds: STARSS23Dataset,
        shuffle: bool,
        drop_last: bool = False,
        use_sampler: bool = False,
    ) -> DataLoader:
        sampler = None
        if use_sampler and self.weighted_sampler:
            weights = self._item_weights(ds)
            sampler = WeightedRandomSampler(
                weights, num_samples=len(weights), replacement=True
            )
            shuffle = False

        return DataLoader(
            dataset     = _STARSS23SEDWrapper(ds),
            batch_size  = self.batch_size,
            shuffle     = shuffle,
            sampler     = sampler,
            drop_last   = drop_last,
            num_workers = self.num_workers,
            pin_memory  = False,
            persistent_workers = self.num_workers > 0,
        )

    def _item_weights(self, ds: STARSS23Dataset) -> torch.Tensor:
        """Per-item sampling weights (uniform fallback when pos_weights unavailable)."""
        if self._pos_weights is None:
            return torch.ones(len(ds._items), dtype=torch.float32)
        pw = self._pos_weights
        weights: List[float] = []
        for clip_idx, _ss, _forced in ds._items:   # 3-tuple: clip_idx, start, is_forced
            events = ds._events[clip_idx]
            active = {cls for (_, cls, *_) in events if cls < 13}
            w = max((pw[c].item() for c in active), default=1.0)
            weights.append(w)
        return torch.tensor(weights, dtype=torch.float32)


class _STARSS23SEDWrapper(torch.utils.data.Dataset):
    """
    Thin wrapper around ``STARSS23Dataset`` that converts the raw
    Multi-ACCDOA output to the ``(wav_mono, label_multihot)`` format
    expected by ``AT2SEDLightningModule``.

    Conversions
    -----------
    * ``wav_foa  : (4, T_samples)`` → ``wav_mono : (T_samples,)``
      W-channel (index 0) — omnidirectional pressure component.

    * ``target   : (T_mel, N, 13, 3)`` → ``label : (13,)``
      Class c is active iff any track in any frame has a non-zero DOA:
      ``label[c] = (target[:, :, c, :].norm(dim=-1) > 0).any()``
    """

    def __init__(self, ds: STARSS23Dataset) -> None:
        self._ds = ds

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        wav_foa, target = self._ds[idx]          # (4, T), (T_mel, N, 13, 3)

        # W-channel mono extraction
        wav_mono = wav_foa[0]                    # (T_samples,)

        # Multi-ACCDOA → multi-hot
        # Active iff any track in any frame has norm > 0
        norms = target.norm(dim=-1)              # (T_mel, N, 13)
        label = (norms > 0).any(dim=0).any(dim=0).float()  # (13,)

        return wav_mono, label
