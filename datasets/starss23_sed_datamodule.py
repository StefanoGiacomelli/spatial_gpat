"""
STARSS23SEDDataModule
=====================
PyTorch Lightning DataModule that wraps ``STARSS23Dataset`` to produce
``(wav_mono, label_multihot)`` batches compatible with
``AT2SEDLightningModule``.

This enables training the GP-AT (EPANNs) model directly on STARSS23 data
with a 13-class classification head.  The output format mirrors
``AudioSetStrongDataModule`` so the same training loop handles both datasets.

Batch format
------------
``wav_mono  : (B, T_samples)``  float32, W-channel (FOA index 0) at 32 kHz
``label     : (B, 13)``         float32 multi-hot, class c active iff any
                                 track in any frame has a non-zero DOA vector

Dataset splits
--------------
train → STARSS23Dataset(split='train') with optional augmentation
val   → STARSS23Dataset(split='val')   no augmentation
test  → STARSS23Dataset(split='test')  no augmentation

Sampling
--------
``WeightedRandomSampler`` (optional, maxpos formula) mirrors the AudioSet
strong pipeline. Per-item weights:
    ``w_i = max(pos_weight[c] for c in active_classes_of_clip_i)``
where ``pos_weight[c] = neg_frames[c] / pos_frames[c]`` from training CSVs.

HLP
---
Disabled unconditionally — no STARSS23 ontology table exists.

OBCE
----
Not supported with n_cls=13 (no 13×13 distance matrix).  Use ``loss: bce``.

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

Author
------
Stefano Giacomelli, Ph.D. candidate — DISIM, University of L'Aquila
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import lightning as L

from .starss23   import STARSS23Dataset, compute_pos_weights
from .augment    import SELDAugment


class STARSS23SEDDataModule(L.LightningDataModule):
    """
    LightningDataModule wrapping STARSS23Dataset for AT2SED-style training.

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
            self._augment = SELDAugment(audio_format='foa', **aug_cfg)

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
            audio_format = 'foa',
            val_fraction = self.val_fraction,
            val_seed     = self.val_seed,
        )

        if stage in ('fit', None):
            self._train_ds = STARSS23Dataset(split='train', augment=self._augment, **common)
            self._val_ds   = STARSS23Dataset(split='val',   augment=None,           **common)

            # Compute per-class pos_weights from training annotations
            pw = compute_pos_weights(self.starss_root, n_classes=13)
            pw = pw.clamp(max=self.pos_weight_max)
            self._pos_weights = pw
            print(
                f'[STARSS23SEDDataModule] pos_weight  '
                f'min={pw.min().item():.2f}  max={pw.max().item():.1f}'
                f'  (cap={self.pos_weight_max})'
            )

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
        """
        Per-item sampling weights (maxpos formula).

        ``w_i = max(pos_weight[c] for c in active_classes_of_clip_i)``

        Active classes are determined from clip-level events (same logic as
        ``MixedSELDDataset.clip_weights``), using the already-loaded
        ``ds._events`` and ``ds._items`` lists.
        """
        pw = self._pos_weights  # (13,) — clamped
        weights: List[float] = []
        for clip_idx, _ in ds._items:
            events = ds._events[clip_idx]
            active = {cls for (_, cls, *_) in events if cls < 13}
            w = max((pw[c].item() for c in active), default=1.0)
            weights.append(w)
        return torch.tensor(weights, dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Wrapper Dataset
# ══════════════════════════════════════════════════════════════════════════════

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
