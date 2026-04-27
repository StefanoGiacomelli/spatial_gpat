"""
datasets/utils.py — Shared Utilities for the SELD Dataset Pipeline
===================================================================
Consolidates three previously separate modules:

  SELDDataset (formerly seld_dataset.py)
  ----------------------------------------
  Abstract base class for all SELD datasets.  Handles windowing,
  Multi-ACCDOA target construction, zero-padding, and collate_fn for
  variable-length clips.  Subclasses implement ``_collect_clips``,
  ``_parse_csv``, and the ``n_classes`` property.

  Target tensor convention
  ~~~~~~~~~~~~~~~~~~~~~~~~
  ``target : (T_mel, N_tracks, C_classes, 3)``
  Each entry is an activity-coupled Cartesian DOA vector:
      ``target[t, n, c, :] = [x, y, z]`` where
      ``x = sin(az)*cos(el),  y = cos(az)*cos(el),  z = sin(el)``
  and ``||[x,y,z]|| = 1`` iff the source is active, ``= 0`` iff silent.
  Track slots are filled in ascending source_idx order; excess sources
  (beyond N_tracks) are dropped.

  SELDAugment (formerly augment.py)
  ------------------------------------
  Format-aware SELD data augmentation on raw waveforms.  Supports FOA
  (azimuth rotation, global gain, elevation flip with DOA target updates)
  and MIC (global gain, per-channel gain, diffuse noise) formats.
  Instantiated by the DataModules based on cfg['augmentation'].

  MixedSELDDataset / compute_combined_class_counts (formerly mixed_seld_dataset.py)
  -----------------------------------------------------------------------------------
  Concatenated STARSS23 + TAU-NIGENS2021 dataset under a unified 13-class
  schema (STARSS23 class indices).  TAU-NIGENS items (12 classes) are
  zero-padded to (T, N, 13, 3) on-the-fly.  compute_combined_class_counts
  tallies per-class frame counts across both datasets for weighted sampling.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

# Type-only imports to avoid a circular dependency:
# starss23.py and tau_nigens.py import SELDDataset from this module, so
# importing STARSS23Dataset / TAUNIGENSDataset at runtime would create a cycle.
if TYPE_CHECKING:
    from .starss23   import STARSS23Dataset
    from .tau_nigens import TAUNIGENSDataset


# ══════════════════════════════════════════════════════════════════════════════
# SELDDataset — Abstract base class
# ══════════════════════════════════════════════════════════════════════════════

class SELDDataset(Dataset, ABC):
    """
    Abstract base class for SELD datasets.

    Args:
        split        : 'train' | 'val' | 'test'
        orig_sr      : native sample rate of audio files (e.g. 24000)
        model_sr     : model input sample rate after resampling (e.g. 32000)
        hop_length   : hop length in samples at *model_sr* (e.g. 320)
        n_tracks     : number of Multi-ACCDOA output tracks (e.g. 5)
        window_s     : window duration in seconds; None → full clip per item
        hop_s        : window stride in seconds; None → equal to window_s (no overlap)
        audio_format : 'foa' | 'mic'
        augment      : optional SELDAugment instance; applied only during train split.
    """

    # CSV annotation temporal resolution
    FRAME_DURATION_S: float = 0.1   # 100 ms

    def __init__(
        self,
        split: str,
        orig_sr: int,
        model_sr: int,
        hop_length: int,
        n_tracks: int,
        window_s: Optional[float] = None,
        hop_s: Optional[float] = None,
        audio_format: str = 'foa',
        augment=None,
    ):
        super().__init__()
        self.split        = split
        self.orig_sr      = orig_sr
        self.model_sr     = model_sr
        self.hop_length   = hop_length
        self.n_tracks     = n_tracks
        self.audio_format = audio_format
        self.augment      = augment

        # ── Windowing ────────────────────────────────────────────────────────
        self.windowed = window_s is not None
        if self.windowed:
            self.window_samples = round(window_s * orig_sr)
            self.hop_samples    = round((hop_s if hop_s is not None else window_s) * orig_sr)
            # T_mel is constant for all windows (based on fixed window duration)
            n_window_at_model   = round(self.window_samples * model_sr / orig_sr)
            self.T_mel_window   = 1 + n_window_at_model // hop_length
        else:
            self.window_samples = None
            self.hop_samples    = None
            self.T_mel_window   = None

        # ── mel frames per CSV frame (e.g. 10.0 for hop=320, sr=32000) ──────
        self.mel_per_csv_frame = (model_sr * self.FRAME_DURATION_S) / hop_length

        # ── Collect clips and build flat item index ──────────────────────────
        self.clips   = self._collect_clips(split)   # list of (audio_path, csv_path)
        self._items  = self._build_item_index()     # list of (clip_idx, start_sample)

        # Pre-parse all CSVs — metadata is small, fits comfortably in RAM
        self._events = [self._parse_csv(csv_p) for _, csv_p in self.clips]

    # ── Abstract interface ───────────────────────────────────────────────────

    @property
    @abstractmethod
    def n_classes(self) -> int:
        """Number of target sound event classes."""
        ...

    @abstractmethod
    def _collect_clips(self, split: str) -> List[Tuple[Path, Path]]:
        """Return ordered list of (audio_path, csv_path) for *split*."""
        ...

    @abstractmethod
    def _parse_csv(self, csv_path: Path) -> List[Tuple[int, int, int, float, float]]:
        """
        Parse one annotation CSV.

        Returns a list of ``(frame_100ms, class_idx, source_idx, az_deg, el_deg)``.
        The *frame_100ms* value is 0-indexed (frame 0 covers 0–100 ms).
        """
        ...

    # ── Item index ───────────────────────────────────────────────────────────

    def _build_item_index(self) -> List[Tuple[int, int]]:
        """Build flat list of ``(clip_idx, start_sample_at_orig_sr)`` pairs."""
        items: List[Tuple[int, int]] = []
        if not self.windowed:
            for i in range(len(self.clips)):
                items.append((i, 0))
        else:
            for i, (audio_path, _) in enumerate(self.clips):
                info = sf.info(str(audio_path))
                n_samples = info.frames
                start = 0
                while start < n_samples:
                    items.append((i, start))
                    start += self.hop_samples
        return items

    # ── Dataset interface ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            wav    : ``(n_mics, T_samples)`` float32 tensor at *orig_sr*
            target : ``(T_mel, N_tracks, C_classes, 3)`` float32 Multi-ACCDOA target
        """
        clip_idx, start_sample = self._items[idx]
        audio_path, _          = self.clips[clip_idx]
        events                 = self._events[clip_idx]

        # ── Load audio ───────────────────────────────────────────────────────
        if self.windowed:
            wav, _ = sf.read(
                str(audio_path),
                start    = start_sample,
                stop     = start_sample + self.window_samples,
                dtype    = 'float32',
                always_2d= True,
            )
            # Zero-pad last window if shorter
            if wav.shape[0] < self.window_samples:
                pad = self.window_samples - wav.shape[0]
                wav = np.concatenate([wav, np.zeros((pad, wav.shape[1]), dtype=np.float32)])
            n_mel = self.T_mel_window
        else:
            wav, _ = sf.read(str(audio_path), dtype='float32', always_2d=True)
            start_sample = 0
            n_samples_model = round(wav.shape[0] * self.model_sr / self.orig_sr)
            n_mel = 1 + n_samples_model // self.hop_length

        # soundfile returns (T_samples, n_mics) — transpose to (n_mics, T_samples)
        wav_t = torch.from_numpy(wav.T.copy())   # (n_mics, T_samples)

        # ── Build Multi-ACCDOA target ────────────────────────────────────────
        start_at_model = round(start_sample * self.model_sr / self.orig_sr)
        target = self._build_target(events, n_mel, start_at_model)

        # ── Data augmentation (train split only) ─────────────────────────────
        if self.augment is not None:
            wav_t, target = self.augment(wav_t, target)

        return wav_t, target

    # ── Target construction ──────────────────────────────────────────────────

    def _build_target(
        self,
        events: List[Tuple[int, int, int, float, float]],
        n_mel: int,
        start_at_model_sr: int,
    ) -> torch.Tensor:
        """
        Build ``(T_mel, N_tracks, C_classes, 3)`` Multi-ACCDOA target.

        Each mel frame is mapped to its corresponding 100ms CSV frame.
        Active sources in a frame are assigned to tracks in ascending
        ``source_idx`` order; tracks beyond ``n_tracks`` are silently dropped.
        """
        target = torch.zeros(n_mel, self.n_tracks, self.n_classes, 3,
                             dtype=torch.float32)
        if not events:
            return target

        # Index events: csv_frame → class_idx → [(source_idx, x, y, z), ...]
        csv_data: dict = {}
        for (frame_100ms, class_idx, source_idx, az_deg, el_deg) in events:
            az = math.radians(az_deg)
            el = math.radians(el_deg)
            xyz = (
                math.sin(az) * math.cos(el),
                math.cos(az) * math.cos(el),
                math.sin(el),
            )
            csv_data \
                .setdefault(frame_100ms, {}) \
                .setdefault(class_idx, []) \
                .append((source_idx, *xyz))

        # Sort sources by source_idx within each (frame, class) group
        for frame_data in csv_data.values():
            for ci in frame_data:
                frame_data[ci].sort(key=lambda s: s[0])

        # Map mel frames → csv frames using vectorised numpy
        t_s_arr = (
            start_at_model_sr + np.arange(n_mel, dtype=np.float64) * self.hop_length
        ) / self.model_sr
        t_csv_arr = (t_s_arr / self.FRAME_DURATION_S).astype(np.int64)

        # Fill target — cache per-csv-frame lookup to avoid redundant dict accesses
        prev_csv: int = -1
        frame_data: dict = {}
        for t_mel in range(n_mel):
            t_csv = int(t_csv_arr[t_mel])
            if t_csv != prev_csv:
                frame_data = csv_data.get(t_csv, {})
                prev_csv   = t_csv
            for class_idx, srcs in frame_data.items():
                for track_idx, (_, x, y, z) in enumerate(srcs[:self.n_tracks]):
                    target[t_mel, track_idx, class_idx, 0] = x
                    target[t_mel, track_idx, class_idx, 1] = y
                    target[t_mel, track_idx, class_idx, 2] = z

        return target

    # ── Weighted sampler support ─────────────────────────────────────────────

    def clip_weights(self) -> torch.Tensor:
        """
        Per-item importance weights for WeightedRandomSampler.

        Each item's weight = max(pos_weight[c]) over all classes active in the
        item's parent clip.  pos_weight is computed from training CSV annotations
        (same formula as in loss.py: neg_frames / pos_frames per class).

        Returns a FloatTensor of shape ``(len(self),)``.
        """
        pw = self._pos_weights()   # (n_classes,) FloatTensor

        # Build per-clip weights from self._events (one entry per clip)
        clip_w = []
        for events in self._events:
            active = set(cls for (_, cls, *_) in events)
            w = max((pw[c].item() for c in active), default=1.0)
            clip_w.append(w)

        # Map per-clip weights to per-item weights (items may be windows of clips)
        item_weights = torch.tensor(
            [clip_w[clip_idx] for clip_idx, _ in self._items],
            dtype=torch.float32,
        )
        return item_weights

    @abstractmethod
    def _pos_weights(self) -> torch.Tensor:
        """Return (n_classes,) FloatTensor of pos_weights for this dataset's root."""
        ...

    @staticmethod
    def collate_fn(
        batch: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad a mixed-length batch of (wav, target) pairs.

        Used automatically when ``windowed=False``; for windowed mode all
        items have the same shape and the default collate is fine.
        """
        wavs, targets = zip(*batch)
        max_wav = max(w.shape[-1] for w in wavs)
        max_t   = max(t.shape[0]  for t in targets)

        B       = len(wavs)
        n_mics  = wavs[0].shape[0]
        n_t, n_c, n_xyz = targets[0].shape[1], targets[0].shape[2], targets[0].shape[3]

        pad_wav = torch.zeros(B, n_mics, max_wav)
        pad_tgt = torch.zeros(B, max_t, n_t, n_c, n_xyz)

        for i, (w, t) in enumerate(zip(wavs, targets)):
            pad_wav[i, :, :w.shape[-1]] = w
            pad_tgt[i, :t.shape[0]]     = t

        return pad_wav, pad_tgt


# ══════════════════════════════════════════════════════════════════════════════
# SELDAugment — Format-aware SELD augmentation
# ══════════════════════════════════════════════════════════════════════════════

class SELDAugment:
    """
    Format-aware SELD data augmentation.

    Parameters
    ----------
    audio_format : 'foa' | 'mic'
    enabled      : master switch; if False, __call__ is a no-op.

    FOA parameters
    ~~~~~~~~~~~~~~
    p_rotate     : probability of applying azimuth rotation
    p_gain       : probability of applying global gain
    p_elevflip   : probability of applying elevation flip
    gain_min_db  : minimum gain in dB (negative = attenuation)
    gain_max_db  : maximum gain in dB

    MIC parameters
    ~~~~~~~~~~~~~~
    p_perchan_gain  : probability of per-channel independent gain
    perchan_gain_db : max ±dB for per-channel gain
    p_noise         : probability of adding diffuse noise
    noise_snr_db    : signal-to-noise ratio for added noise (higher = less noise)

    FOA augmentations
    ~~~~~~~~~~~~~~~~~
    A — Azimuth rotation  : rotate sound field by theta ~ U(-pi, pi).
        W unchanged; X' = X cos(θ) - Y sin(θ); Y' = X sin(θ) + Y cos(θ); Z unchanged.
        DOA target updated: x' = x cos(θ) - y sin(θ); y' = x sin(θ) + y cos(θ); z' = z.
    B — Global gain       : multiply all channels by g = exp(U(min_db, max_db) * ln10/20).
        No target modification.
    C — Elevation flip    : negate Z channel and target z coordinates.
        W, X, Y unchanged; Z_out = -Z; target: z -> -z.

    MIC augmentations (geometry-blind, no target modification)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    1 — Global gain       : applied to all channels.
    2 — Per-channel gain  : independent ±perchan_gain_db on each channel.
    3 — Diffuse noise     : add white noise at noise_snr_db SNR relative to RMS.
    """

    def __init__(
        self,
        audio_format: str = 'foa',
        enabled: bool = True,
        # FOA
        p_rotate: float = 0.5,
        p_gain: float = 0.5,
        p_elevflip: float = 0.3,
        gain_min_db: float = -6.0,
        gain_max_db: float = 6.0,
        # MIC-specific
        p_perchan_gain: float = 0.4,
        perchan_gain_db: float = 2.0,
        p_noise: float = 0.3,
        noise_snr_db: float = 35.0,
        **kwargs,  # absorb unknown keys from YAML without error
    ):
        self.audio_format    = audio_format.lower()
        self.enabled         = enabled
        self.p_rotate        = p_rotate
        self.p_gain          = p_gain
        self.p_elevflip      = p_elevflip
        self.gain_min_db     = gain_min_db
        self.gain_max_db     = gain_max_db
        self.p_perchan_gain  = p_perchan_gain
        self.perchan_gain_db = perchan_gain_db
        self.p_noise         = p_noise
        self.noise_snr_db    = noise_snr_db

        # Pre-compute log-scale gain bounds (avoids re-computation per call)
        self._gain_log_min = gain_min_db * math.log(10) / 20.0
        self._gain_log_max = gain_max_db * math.log(10) / 20.0
        self._perchan_log  = perchan_gain_db * math.log(10) / 20.0

    def __call__(
        self,
        wav: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply augmentations stochastically.

        Parameters
        ----------
        wav    : (n_mics, T_samples) — raw waveform at training SR
        target : (T_mel, N_tracks, C_classes, 3) — Multi-ACCDOA DOA targets

        Returns
        -------
        (wav_aug, target_aug) — same shapes, possibly modified in-place copy.
        """
        if not self.enabled:
            return wav, target

        if self.audio_format == 'foa':
            return self._foa(wav, target)
        return self._mic(wav, target)

    # ── FOA augmentations ────────────────────────────────────────────────────

    def _foa(
        self,
        wav: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Clone to avoid mutating the original tensors
        wav    = wav.clone()
        target = target.clone()

        if torch.rand(1).item() < self.p_rotate:
            wav, target = self._foa_rotate(wav, target)
        if torch.rand(1).item() < self.p_gain:
            wav = self._global_gain(wav)
        if torch.rand(1).item() < self.p_elevflip:
            wav, target = self._foa_elevflip(wav, target)

        return wav, target

    def _foa_rotate(
        self,
        wav: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rotate FOA sound field around vertical (Z) axis by random theta."""
        theta = (torch.rand(1).item() * 2.0 - 1.0) * math.pi   # U(-pi, pi)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        # wav channels: W=0, X=1, Y=2, Z=3
        x = wav[1].clone()
        y = wav[2].clone()
        wav[1] = cos_t * x - sin_t * y
        wav[2] = sin_t * x + cos_t * y

        # target: (T_mel, N, C, 3) where last dim = (x, y, z) cartesian DOA
        tx = target[..., 0].clone()
        ty = target[..., 1].clone()
        target[..., 0] = cos_t * tx - sin_t * ty
        target[..., 1] = sin_t * tx + cos_t * ty

        return wav, target

    def _foa_elevflip(
        self,
        wav: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Negate Z FOA channel and target z coordinates."""
        wav[3] = -wav[3]
        target[..., 2] = -target[..., 2]
        return wav, target

    # ── MIC augmentations ────────────────────────────────────────────────────

    def _mic(
        self,
        wav: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        wav = wav.clone()

        if torch.rand(1).item() < self.p_gain:
            wav = self._global_gain(wav)
        if torch.rand(1).item() < self.p_perchan_gain:
            wav = self._perchan_gain(wav)
        if torch.rand(1).item() < self.p_noise:
            wav = self._diffuse_noise(wav)

        return wav, target   # target unchanged for MIC

    def _perchan_gain(self, wav: torch.Tensor) -> torch.Tensor:
        """Apply independent gain in [-perchan_gain_db, +perchan_gain_db] per channel."""
        n_ch = wav.shape[0]
        log_gains = (torch.rand(n_ch) * 2.0 - 1.0) * self._perchan_log
        gains = torch.exp(log_gains).to(wav.dtype)
        return wav * gains.unsqueeze(1)

    def _diffuse_noise(self, wav: torch.Tensor) -> torch.Tensor:
        """Add white Gaussian noise at noise_snr_db relative to signal RMS."""
        rms = wav.pow(2).mean().sqrt().clamp(min=1e-8)
        noise_std = rms / (10.0 ** (self.noise_snr_db / 20.0))
        return wav + torch.randn_like(wav) * noise_std

    # ── Shared helpers ───────────────────────────────────────────────────────

    def _global_gain(self, wav: torch.Tensor) -> torch.Tensor:
        """Multiply all channels by exp(U(log_min, log_max))."""
        log_g = (torch.rand(1).item()
                 * (self._gain_log_max - self._gain_log_min)
                 + self._gain_log_min)
        return wav * math.exp(log_g)


# ══════════════════════════════════════════════════════════════════════════════
# MixedSELDDataset — Unified STARSS23 + TAU-NIGENS2021 dataset
# ══════════════════════════════════════════════════════════════════════════════

# Unified class count: STARSS23 schema is the superset (13 classes)
N_CLASSES_UNIFIED = 13


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

    def _count(events_per_clip: List[List[Tuple]]) -> None:
        for clip_idx, events in enumerate(events_per_clip):
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


class MixedSELDDataset(Dataset):
    """
    Concatenated STARSS23 + TAU-NIGENS2021 dataset with unified 13-class schema.

    TAU-NIGENS targets ``(T, N, 12, 3)`` are zero-padded to ``(T, N, 13, 3)``
    on-the-fly at ``__getitem__`` time.  No CSV-level remapping is required
    because both datasets share the same DCASE class indices 0–11; class 12
    (Knock) is STARSS23-only and is always absent from TAU-NIGENS items.

    Class balancing
    ---------------
    ``clip_weights()`` computes per-item sampling weights from the combined
    per-class frame counts across both datasets:

        weight[item] = max over active classes c of  target_count / count[c]

    where ``target_count`` defaults to the median class count.  Passing these
    weights to a ``WeightedRandomSampler`` (with replacement) oversamples clips
    that contain underrepresented classes.  Because the augmentation pipeline is
    applied at each draw, repeated draws of the same clip receive different
    random augmentations, producing genuine diversity.

    Args:
        starss_ds : STARSS23Dataset instance (any split).
        tau_ds    : TAUNIGENSDataset instance (same split).

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
            return self._starss[idx]

        # TAU-NIGENS — target is (T, N, 12, 3); zero-pad class dim to 13
        wav, tgt = self._tau[idx - self._n_starss]
        T, N, C, three = tgt.shape   # C == 12
        padded = torch.zeros(T, N, self.N_CLASSES, three, dtype=tgt.dtype)
        padded[:, :, :C, :] = tgt
        return wav, padded

    # ── Windowing flag ────────────────────────────────────────────────────────

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

        ``target_count`` defaults to the median combined class count, targeting
        a roughly uniform per-class distribution.  Clips whose rarest active
        class has a lower count receive proportionally higher weight and are
        therefore oversampled.  Combined with online augmentation, this produces
        genuinely diverse representations of rare-class clips.

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

        weights: List[float] = []

        # STARSS23 items
        for clip_idx, _ in self._starss._items:
            events = self._starss._events[clip_idx]
            active = {cls for (_, cls, *_) in events if cls < self.N_CLASSES}
            w = max((target_count / counts[c] for c in active), default=1.0)
            weights.append(w)

        # TAU-NIGENS items
        for clip_idx, _ in self._tau._items:
            events = self._tau._events[clip_idx]
            active = {cls for (_, cls, *_) in events if cls < self.N_CLASSES}
            w = max((target_count / counts[c] for c in active), default=1.0)
            weights.append(w)

        return torch.tensor(weights, dtype=torch.float32)
