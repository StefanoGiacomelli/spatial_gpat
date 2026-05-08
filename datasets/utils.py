"""
datasets/utils.py — Shared Utilities for the SELD Dataset Pipeline
===================================================================
SELDDataset
    Abstract base class.  Handles windowing, Multi-ACCDOA target
    construction, zero-padding, and collate_fn for variable-length clips.
    Subclasses implement ``_collect_clips``, ``_parse_csv``, and the
    ``n_classes`` property.

SELDAugment
    FOA-only SELD data augmentation with 3 mutually exclusive techniques:
    (A) 16 Patterns — Mazzon et al. foa_augmentations.pdf, Table 1
    (B) Gain Scaling — g ~ U(gain_min, gain_max) on all 4 channels
    (C) White Noise Injection — RMS-relative noise on all 4 channels

Target tensor convention
    ``target : (T_mel, N_tracks, C_classes, 3)``
    Active entry: ``[x, y, z]`` where
        x = sin(az)*cos(el),  y = cos(az)*cos(el),  z = sin(el)
    Silent entry: ``[0, 0, 0]``

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset


# ══════════════════════════════════════════════════════════════════════════════
# SELDDataset — Abstract base class
# ══════════════════════════════════════════════════════════════════════════════

class SELDDataset(Dataset, ABC):
    """
    Abstract base class for SELD datasets.

    Args:
        split        : 'train' | 'val' | 'test'
        orig_sr      : native sample rate of audio files (e.g. 24000)
        model_sr     : model input sample rate (e.g. 32000)
        hop_length   : hop length in samples at *model_sr* (e.g. 320)
        n_tracks     : number of Multi-ACCDOA output tracks (e.g. 5)
        window_s     : window duration in seconds; None → full clip per item
        hop_s        : window stride in seconds; None → equal to window_s
        output_sr    : if set, resample audio from orig_sr to output_sr before
                       returning (e.g. TAU2019: orig_sr=48000, output_sr=24000)
        augment      : optional SELDAugment instance
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
        output_sr: Optional[int] = None,
        augment=None,
    ):
        super().__init__()
        self.split      = split
        self.orig_sr    = orig_sr
        self.model_sr   = model_sr
        self.hop_length = hop_length
        self.n_tracks   = n_tracks
        self.output_sr  = output_sr
        self.augment    = augment

        # Effective SR seen by the model frontend (after optional resample)
        self._eff_sr = output_sr if output_sr is not None else orig_sr

        # ── Windowing ────────────────────────────────────────────────────────
        self.windowed = window_s is not None
        if self.windowed:
            self.window_samples = round(window_s * orig_sr)
            self.hop_samples    = round((hop_s if hop_s is not None else window_s) * orig_sr)
            # T_mel_window is invariant to output_sr:
            # (window_samples * output_sr/orig_sr) * model_sr/output_sr
            # = window_samples * model_sr / orig_sr
            n_window_at_model   = round(self.window_samples * model_sr / orig_sr)
            self.T_mel_window   = 1 + n_window_at_model // hop_length
        else:
            self.window_samples = None
            self.hop_samples    = None
            self.T_mel_window   = None

        self.mel_per_csv_frame = (model_sr * self.FRAME_DURATION_S) / hop_length

        # Parse CSVs before building item index (oversampling needs _events)
        self.clips   = self._collect_clips(split)
        self._events = [self._parse_csv(csv_p) for _, csv_p in self.clips]
        self._items  = self._build_item_index()

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
        """
        ...

    # ── Item index with class-based oversampling ─────────────────────────────

    def _build_item_index(self) -> List[Tuple[int, int, bool]]:
        """
        Build flat list of ``(clip_idx, start_sample, is_forced)`` triples.

        ``is_forced=True`` marks oversampled items that must receive an
        augmentation (passed as ``force=True`` to SELDAugment).
        Oversampling is applied only when ``augment is not None`` and
        ``split == 'train'``.
        """
        base_items: List[Tuple[int, int]] = []
        if not self.windowed:
            for i in range(len(self.clips)):
                base_items.append((i, 0))
        else:
            for i, (audio_path, _) in enumerate(self.clips):
                info = sf.info(str(audio_path))
                n_samples = info.frames
                start = 0
                while start < n_samples:
                    base_items.append((i, start))
                    start += self.hop_samples

        return [(ci, ss, False) for ci, ss in base_items]

    # ── Dataset interface ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            wav    : ``(4, T_samples)`` float32 FOA tensor at *output_sr*
                     (or *orig_sr* when output_sr is None)
            target : ``(T_mel, N_tracks, C_classes, 3)`` Multi-ACCDOA target
        """
        clip_idx, start_sample, is_forced = self._items[idx]
        audio_path, _                     = self.clips[clip_idx]
        events                            = self._events[clip_idx]

        # ── Load audio ───────────────────────────────────────────────────────
        if self.windowed:
            wav, _ = sf.read(
                str(audio_path),
                start     = start_sample,
                stop      = start_sample + self.window_samples,
                dtype     = 'float32',
                always_2d = True,
            )
            if wav.shape[0] < self.window_samples:
                pad = self.window_samples - wav.shape[0]
                wav = np.concatenate(
                    [wav, np.zeros((pad, wav.shape[1]), dtype=np.float32)]
                )
            n_mel = self.T_mel_window
        else:
            wav, _       = sf.read(str(audio_path), dtype='float32', always_2d=True)
            start_sample = 0   # full-clip always starts at 0

        # soundfile returns (T, n_mics) → transpose to (n_mics, T)
        wav_t = torch.from_numpy(wav.T.copy())   # (4, T_samples)

        # Optional resample (e.g. TAU2019: 48 kHz → 24 kHz)
        if self.output_sr is not None:
            wav_t = torchaudio.functional.resample(wav_t, self.orig_sr, self.output_sr)

        # n_mel for full-clip mode: computed after resample
        if not self.windowed:
            n_samples_model = round(wav_t.shape[-1] * self.model_sr / self._eff_sr)
            n_mel           = 1 + n_samples_model // self.hop_length

        # ── Build Multi-ACCDOA target ────────────────────────────────────────
        start_at_model = round(start_sample * self.model_sr / self.orig_sr)
        target = self._build_target(events, n_mel, start_at_model)

        # ── Data augmentation ────────────────────────────────────────────────
        if self.augment is not None:
            wav_t, target = self.augment(wav_t, target, force=is_forced)

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

        Active sources in a frame are assigned to tracks in ascending
        ``source_idx`` order; tracks beyond ``n_tracks`` are dropped.
        """
        target = torch.zeros(n_mel, self.n_tracks, self.n_classes, 3,
                             dtype=torch.float32)
        if not events:
            return target

        # Index events: csv_frame → class_idx → [(source_idx, x, y, z), ...]
        csv_data: dict = {}
        for (frame_100ms, class_idx, source_idx, az_deg, el_deg) in events:
            az  = math.radians(az_deg)
            el  = math.radians(el_deg)
            xyz = (
                math.sin(az) * math.cos(el),
                math.cos(az) * math.cos(el),
                math.sin(el),
            )
            (csv_data
             .setdefault(frame_100ms, {})
             .setdefault(class_idx, [])
             .append((source_idx, *xyz)))

        for frame_data in csv_data.values():
            for ci in frame_data:
                frame_data[ci].sort(key=lambda s: s[0])

        t_s_arr   = (
            start_at_model_sr + np.arange(n_mel, dtype=np.float64) * self.hop_length
        ) / self.model_sr
        t_csv_arr = (t_s_arr / self.FRAME_DURATION_S).astype(np.int64)

        prev_csv:  int  = -1
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

    # ── Collate ──────────────────────────────────────────────────────────────

    @staticmethod
    def collate_fn(
        batch: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad a mixed-length batch of (wav, target) pairs.

        Used when ``windowed=False``; for windowed mode all items have the
        same shape and the default collate is sufficient.
        """
        wavs, targets = zip(*batch)
        max_wav = max(w.shape[-1] for w in wavs)
        max_t   = max(t.shape[0]  for t in targets)

        B      = len(wavs)
        n_mics = wavs[0].shape[0]
        n_t    = targets[0].shape[1]
        n_c    = targets[0].shape[2]
        n_xyz  = targets[0].shape[3]

        pad_wav = torch.zeros(B, n_mics, max_wav)
        pad_tgt = torch.zeros(B, max_t, n_t, n_c, n_xyz)

        for i, (w, t) in enumerate(zip(wavs, targets)):
            pad_wav[i, :, :w.shape[-1]] = w
            pad_tgt[i, :t.shape[0]]     = t

        return pad_wav, pad_tgt


# ══════════════════════════════════════════════════════════════════════════════
# SELDAugment — FOA-only SELD data augmentation
# ══════════════════════════════════════════════════════════════════════════════

class SELDAugment:
    """
    FOA-only SELD data augmentation with 3 mutually exclusive techniques.

    Parameters
    ----------
    enabled       : master switch; if False, __call__ is a no-op.
    p_16patterns  : probability of applying a random FOA symmetry pattern
    p_gain        : probability of applying global amplitude scaling
    p_noise       : probability of adding white noise
    gain_min      : minimum amplitude gain (< 1.0 = attenuation)
    gain_max      : maximum amplitude gain
    noise_snr_db  : injected noise SNR in dB (higher = less noise)

    Augmentations
    ~~~~~~~~~~~~~
    A — 16 Patterns (Mazzon et al., foa_augmentations.pdf, Table 1):
        Select uniformly one of the 15 non-identity FOA symmetry transforms.
        Each pattern permutes/negates channels ch1(X), ch2(Y), ch3(Z) and
        applies the corresponding DOA transformation.  W channel (ch0) is
        always preserved.  DOA target updated consistently.

    B — Gain Scaling:
        g ~ U(gain_min, gain_max) applied to all 4 channels.
        DOA target invariant.

    C — White Noise Injection:
        noise_std = RMS_signal / 10^(noise_snr_db/20).
        Added to all 4 channels.  DOA target invariant.

    Mutual exclusion:
        p_none = 1 - (p_16patterns + p_gain + p_noise) ≥ 0.
        At most one augmentation per call (categorical sampling).
        With force=True, p_none is excluded — exactly one aug is applied.
    """

    # 15 non-identity FOA symmetry patterns from Mazzon et al., Table 1.
    # Encoding: (ch1_sign, ch1_from, ch2_sign, ch2_from, ch3_sign,
    #             cx_sign,  cx_from,  cy_sign,  cy_from,  cz_sign)
    # ch1_from / ch2_from : 1=take from X (wav[1]), 2=take from Y (wav[2])
    # cx_from  / cy_from  : 0=take from code_x (tgt[...,0]), 1=from code_y (tgt[...,1])
    _PATTERNS: List[Tuple] = [
        (-1, 2, +1, 1, +1,  +1, 1, -1, 0, +1),  # 1   φ+π/2,  θ
        (+1, 2, -1, 1, +1,  -1, 1, +1, 0, +1),  # 2   φ-π/2,  θ
        (-1, 1, -1, 2, +1,  -1, 0, -1, 1, +1),  # 3   φ+π,    θ
        (+1, 1, -1, 2, +1,  -1, 0, +1, 1, +1),  # 4  -φ,      θ
        (+1, 2, +1, 1, +1,  +1, 1, +1, 0, +1),  # 5  -φ+π/2,  θ
        (-1, 2, -1, 1, +1,  -1, 1, -1, 0, +1),  # 6  -φ-π/2,  θ
        (-1, 1, +1, 2, +1,  +1, 0, -1, 1, +1),  # 7  -φ+π,    θ
        (+1, 1, +1, 2, -1,  +1, 0, +1, 1, -1),  # 8   φ,     -θ
        (-1, 2, +1, 1, -1,  +1, 1, -1, 0, -1),  # 9   φ+π/2, -θ
        (+1, 2, -1, 1, -1,  -1, 1, +1, 0, -1),  # 10  φ-π/2, -θ
        (-1, 1, -1, 2, -1,  -1, 0, -1, 1, -1),  # 11  φ+π,   -θ
        (+1, 1, -1, 2, -1,  -1, 0, +1, 1, -1),  # 12 -φ,     -θ
        (+1, 2, +1, 1, -1,  +1, 1, +1, 0, -1),  # 13 -φ+π/2, -θ
        (-1, 2, -1, 1, -1,  -1, 1, -1, 0, -1),  # 14 -φ-π/2, -θ
        (-1, 1, +1, 2, -1,  +1, 0, -1, 1, -1),  # 15 -φ+π,   -θ
    ]

    def __init__(
        self,
        enabled: bool = True,
        p_16patterns: float = 0.20,
        p_gain: float = 0.15,
        p_noise: float = 0.10,
        gain_min: float = 0.3,
        gain_max: float = 0.9,
        noise_snr_db: float = 35.0,
        **kwargs,   # absorb unknown YAML keys
    ):
        self.enabled      = enabled
        self.p_16patterns = p_16patterns
        self.p_gain       = p_gain
        self.p_noise      = p_noise
        self.gain_min     = gain_min
        self.gain_max     = gain_max
        self.noise_snr_db = noise_snr_db

        total = p_16patterns + p_gain + p_noise
        if total > 1.0 + 1e-6:
            raise ValueError(
                f"p_16patterns + p_gain + p_noise = {total:.3f} > 1.0"
            )
        self._p_none       = max(0.0, 1.0 - total)
        self._probs        = [p_16patterns, p_gain, p_noise, self._p_none]
        self._probs_forced = [p_16patterns, p_gain, p_noise]

    def __call__(
        self,
        wav: torch.Tensor,
        target: torch.Tensor,
        force: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        wav    : (4, T_samples) FOA waveform
        target : (T_mel, N_tracks, C_classes, 3) Multi-ACCDOA target
        force  : if True, always apply one augmentation (for oversampled items)

        Returns
        -------
        (wav_aug, target_aug) — same shapes as input
        """
        if not self.enabled:
            return wav, target

        probs = self._probs_forced if force else self._probs
        total = sum(probs)
        probs_t = torch.tensor([p / total for p in probs], dtype=torch.float32)
        choice  = int(torch.multinomial(probs_t, 1).item())

        # choice 0=16patterns, 1=gain, 2=noise, [3=none (only when not forced)]
        if not force and choice == 3:
            return wav, target

        wav    = wav.clone()
        target = target.clone()

        if choice == 0:
            return self._apply_16patterns(wav, target)
        if choice == 1:
            return self._gain(wav, target)
        return self._noise(wav, target)

    # ── Augmentation methods ─────────────────────────────────────────────────

    def _apply_16patterns(
        self,
        wav: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pat = self._PATTERNS[int(torch.randint(len(self._PATTERNS), (1,)).item())]
        ch1_s, ch1_f, ch2_s, ch2_f, ch3_s, cx_s, cx_f, cy_s, cy_f, cz_s = pat

        # Read originals before in-place modification
        x_orig  = wav[1].clone()
        y_orig  = wav[2].clone()
        z_orig  = wav[3].clone()
        wav[1]  = ch1_s * (x_orig if ch1_f == 1 else y_orig)
        wav[2]  = ch2_s * (x_orig if ch2_f == 1 else y_orig)
        wav[3]  = ch3_s * z_orig

        cx_orig = target[..., 0].clone()
        cy_orig = target[..., 1].clone()
        cz_orig = target[..., 2].clone()
        target[..., 0] = cx_s * (cx_orig if cx_f == 0 else cy_orig)
        target[..., 1] = cy_s * (cx_orig if cy_f == 0 else cy_orig)
        target[..., 2] = cz_s * cz_orig

        return wav, target

    def _gain(
        self,
        wav: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        g = torch.empty(1).uniform_(self.gain_min, self.gain_max).item()
        return wav * g, target

    def _noise(
        self,
        wav: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rms       = wav.pow(2).mean().sqrt().clamp(min=1e-8)
        noise_std = rms / (10.0 ** (self.noise_snr_db / 20.0))
        return wav + torch.randn_like(wav) * noise_std, target

