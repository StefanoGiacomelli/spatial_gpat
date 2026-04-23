"""
MixedSELDDataset
================
Concatenates STARSS23Dataset and TAUNIGENSDataset under a unified 13-class
schema (STARSS23 class indices).

Class mapping
-------------
TAU-NIGENS (12 classes) and STARSS23 (13 classes) share the same DCASE
challenge class definitions for indices 0–11:

    0  Female speech       6  Walk/footsteps
    1  Male speech         7  Door
    2  Clapping            8  Music
    3  Telephone           9  Musical instrument
    4  Laughter           10  Water tap
    5  Domestic sounds    11  Bell
                          12  Knock  ← STARSS23 only

TAU-NIGENS items have class index 12 always absent (zero in the target tensor).
At __getitem__ time TAU-NIGENS targets (T, N, 12, 3) are zero-padded to
(T, N, 13, 3).  No CSV-level remapping is required.

Class balancing
---------------
``clip_weights()`` computes per-item sampling weights from the *combined*
per-class frame counts across both datasets:

    weight[item] = max over active classes c of  target_count / count[c]

where ``target_count`` defaults to the median class count.  Passing these
weights to a ``WeightedRandomSampler`` (with replacement) oversamples clips
that contain underrepresented classes.  Because the augmentation pipeline is
applied at each draw, repeated draws of the same clip receive different random
augmentations, producing genuine diversity.

Author
------
Stefano Giacomelli, Ph.D. candidate — DISIM, University of L'Aquila
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .starss23   import STARSS23Dataset
from .tau_nigens import TAUNIGENSDataset


# ── Unified class count ───────────────────────────────────────────────────────

N_CLASSES_UNIFIED = 13   # STARSS23 schema (superset)


# ══════════════════════════════════════════════════════════════════════════════
# Combined class-count utility
# ══════════════════════════════════════════════════════════════════════════════

def compute_combined_class_counts(
    starss_ds: STARSS23Dataset,
    tau_ds: TAUNIGENSDataset,
) -> Dict[int, int]:
    """
    Count unique active (clip, frame_100ms) pairs per class across both
    training datasets, using the already-parsed ``_events`` lists.

    Returns
    -------
    Dict[class_idx, frame_count]  for classes 0–12 (13-class unified schema).
    Classes absent from TAU-NIGENS (index 12) will have counts only from
    STARSS23.  The returned dict always has entries for all 13 classes (≥ 1).
    """
    counts: Dict[int, int] = {c: 0 for c in range(N_CLASSES_UNIFIED)}

    # Helper: iterate (clip_idx, events) and count unique (clip, frame) per class
    def _count(events_per_clip: List[List[Tuple]]) -> None:
        for clip_idx, events in enumerate(events_per_clip):
            # Deduplicate (frame_100ms, class_idx) pairs within each clip
            seen: set = set()
            for frame_100ms, class_idx, *_ in events:
                key = (clip_idx, frame_100ms, class_idx)
                if key not in seen:
                    seen.add(key)
                    if class_idx < N_CLASSES_UNIFIED:
                        counts[class_idx] += 1

    _count(starss_ds._events)
    _count(tau_ds._events)

    # Guard: ensure no class has a zero count (avoids division by zero)
    for c in range(N_CLASSES_UNIFIED):
        if counts[c] == 0:
            counts[c] = 1

    return counts


# ══════════════════════════════════════════════════════════════════════════════
# MixedSELDDataset
# ══════════════════════════════════════════════════════════════════════════════

class MixedSELDDataset(Dataset):
    """
    Concatenated STARSS23 + TAU-NIGENS2021 dataset with unified 13-class schema.

    Args:
        starss_ds  : STARSS23Dataset instance (any split).
        tau_ds     : TAUNIGENSDataset instance (same split).

    Both datasets must be constructed with the **same** windowing parameters
    (``window_s``, ``hop_s``, ``model_sr``, ``hop_length``, ``n_tracks``).
    """

    N_CLASSES = N_CLASSES_UNIFIED

    def __init__(
        self,
        starss_ds: STARSS23Dataset,
        tau_ds: TAUNIGENSDataset,
    ) -> None:
        self._starss   = starss_ds
        self._tau      = tau_ds
        self._n_starss = len(starss_ds)

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self._n_starss + len(self._tau)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            wav    : (n_mics, T_samples) float32 at orig_sr
            target : (T_mel, N_tracks, 13, 3) float32 Multi-ACCDOA
        """
        if idx < self._n_starss:
            # STARSS23 — target already (T, N, 13, 3)
            return self._starss[idx]

        # TAU-NIGENS — target is (T, N, 12, 3); zero-pad class dim to 13
        wav, tgt = self._tau[idx - self._n_starss]
        T, N, C, three = tgt.shape   # C == 12
        padded = torch.zeros(T, N, self.N_CLASSES, three, dtype=tgt.dtype)
        padded[:, :, :C, :] = tgt
        return wav, padded

    # ── Windowing flag (used by FinetuneDataModule to select collate_fn) ─────

    @property
    def windowed(self) -> bool:
        """True iff items are fixed-length windows (not full clips)."""
        return self._starss.windowed

    # ── Balanced sampling weights ─────────────────────────────────────────────

    def clip_weights(
        self,
        target_count: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Per-item sampling weights for WeightedRandomSampler.

        weight[item] = max over active classes c of (target_count / count[c])

        ``target_count`` defaults to the median combined class count, which
        targets a roughly uniform per-class distribution.  Clips whose rarest
        active class has a lower count receive proportionally higher weight and
        are therefore oversampled.  Combined with online augmentation, this
        produces genuinely diverse representations of rare-class clips.

        Returns
        -------
        FloatTensor of shape (len(self),).
        """
        counts = compute_combined_class_counts(self._starss, self._tau)

        if target_count is None:
            med = torch.tensor(
                list(counts.values()), dtype=torch.float32
            ).median().item()
            target_count = max(1, int(med))

        # Per-item weight = max sampling ratio over active classes in the clip
        weights: List[float] = []

        # STARSS23 items — iterate _items to map item → clip → active classes
        for clip_idx, _ in self._starss._items:
            events = self._starss._events[clip_idx]
            active = {cls for (_, cls, *_) in events if cls < self.N_CLASSES}
            if active:
                w = max(target_count / counts[c] for c in active)
            else:
                w = 1.0
            weights.append(w)

        # TAU-NIGENS items
        for clip_idx, _ in self._tau._items:
            events = self._tau._events[clip_idx]
            active = {cls for (_, cls, *_) in events if cls < self.N_CLASSES}
            if active:
                w = max(target_count / counts[c] for c in active)
            else:
                w = 1.0
            weights.append(w)

        return torch.tensor(weights, dtype=torch.float32)
