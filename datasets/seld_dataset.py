"""
SELDDataset — Base class for SELD datasets
===========================================
Handles windowing, target construction (Multi-ACCDOA at T_mel resolution),
zero-padding, and collate_fn for variable-length clips.

Subclasses implement:
    _collect_clips(split) → List[(audio_path, csv_path)]
    _parse_csv(csv_path)  → List[(frame_100ms, class_idx, source_idx, az_deg, el_deg)]
    n_classes             → int property

Target tensor convention
------------------------
``target : (T_mel, N_tracks, C_classes, 3)``
Each entry is an activity-coupled Cartesian DOA vector:
    ``target[t, n, c, :] = [x, y, z]`` where
    ``x = sin(az)*cos(el),  y = cos(az)*cos(el),  z = sin(el)``
and ``||[x,y,z]|| = 1`` iff the source is active,  ``= 0`` iff silent.
Track slots are filled in ascending source_idx order; excess sources
(beyond N_tracks) are dropped (N_tracks=5 is generous for SELD benchmarks).

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


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

    # ── Collate function (variable-length full clips) ────────────────────────

    def clip_weights(self) -> 'torch.Tensor':
        """
        Per-item importance weights for WeightedRandomSampler.

        Each item's weight = max(pos_weight[c]) over all classes active in the
        item's parent clip.  pos_weight is computed from training CSV annotations
        (same formula as in loss.py: neg_frames / pos_frames per class).

        Returns a FloatTensor of shape ``(len(self),)``.
        """
        import torch
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
    def _pos_weights(self) -> 'torch.Tensor':
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
