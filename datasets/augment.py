"""
datasets/augment.py
====================
Format-aware SELD data augmentation applied on raw waveforms.

Supported formats
-----------------
foa (First-Order Ambisonics):
  Channels: W, X, Y, Z (indices 0-3).

  Augmentations:
    A — Azimuth rotation  : rotate sound field by theta ~ U(-pi, pi).
        W unchanged; X' = X cos(θ) - Y sin(θ); Y' = X sin(θ) + Y cos(θ); Z unchanged.
        DOA target updated: x' = x cos(θ) - y sin(θ); y' = x sin(θ) + y cos(θ); z' = z.

    B — Global gain       : multiply all channels by g = exp(U(min_db, max_db) * ln10/20).
        No target modification.

    C — Elevation flip    : negate Z channel and target z coordinates.
        W, X, Y unchanged; Z_out = -Z.
        Target: z -> -z.

mic (raw microphone array):
  Geometry-blind augmentations — no target modification required.

    1 — Global gain       : same as FOA gain, applied to all channels.
    2 — Per-channel gain  : independent ±perchan_gain_db on each channel.
    3 — Diffuse noise     : add white noise with SNR = noise_snr_db relative to RMS.

Usage
-----
Instantiated by SELDDataModule based on cfg['augmentation'] and cfg['data']['format'].
Applied in SELDDataset.__getitem__ only for the train split.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

import math
from typing import Tuple

import torch


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

    # ──────────────────────────────────────────────────────────────────────────
    # FOA augmentations
    # ──────────────────────────────────────────────────────────────────────────

    def _foa(
        self,
        wav: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Clone to avoid mutating the original tensors
        wav    = wav.clone()
        target = target.clone()

        # A — azimuth rotation
        if torch.rand(1).item() < self.p_rotate:
            wav, target = self._foa_rotate(wav, target)

        # B — global gain
        if torch.rand(1).item() < self.p_gain:
            wav = self._global_gain(wav)

        # C — elevation flip
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
        # W (channel 0) and Z (channel 3) unchanged

        # target: (T_mel, N, C, 3) where last dim = (x, y, z) cartesian DOA
        tx = target[..., 0].clone()
        ty = target[..., 1].clone()
        target[..., 0] = cos_t * tx - sin_t * ty
        target[..., 1] = sin_t * tx + cos_t * ty
        # z component (index 2) unchanged

        return wav, target

    def _foa_elevflip(
        self,
        wav: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Negate Z FOA channel and target z coordinates."""
        wav[3] = -wav[3]              # Z channel
        target[..., 2] = -target[..., 2]   # z DOA component
        return wav, target

    # ──────────────────────────────────────────────────────────────────────────
    # MIC augmentations (geometry-blind, no target change)
    # ──────────────────────────────────────────────────────────────────────────

    def _mic(
        self,
        wav: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        wav = wav.clone()

        # 1 — global gain
        if torch.rand(1).item() < self.p_gain:
            wav = self._global_gain(wav)

        # 2 — per-channel independent gain
        if torch.rand(1).item() < self.p_perchan_gain:
            wav = self._perchan_gain(wav)

        # 3 — diffuse noise
        if torch.rand(1).item() < self.p_noise:
            wav = self._diffuse_noise(wav)

        return wav, target   # target unchanged for MIC

    def _perchan_gain(self, wav: torch.Tensor) -> torch.Tensor:
        """Apply independent gain in [-perchan_gain_db, +perchan_gain_db] per channel."""
        n_ch = wav.shape[0]
        log_gains = (torch.rand(n_ch) * 2.0 - 1.0) * self._perchan_log  # (n_ch,)
        gains = torch.exp(log_gains).to(wav.dtype)
        wav = wav * gains.unsqueeze(1)
        return wav

    def _diffuse_noise(self, wav: torch.Tensor) -> torch.Tensor:
        """Add white Gaussian noise at noise_snr_db relative to signal RMS."""
        rms = wav.pow(2).mean().sqrt().clamp(min=1e-8)
        noise_std = rms / (10.0 ** (self.noise_snr_db / 20.0))
        noise = torch.randn_like(wav) * noise_std
        return wav + noise

    # ──────────────────────────────────────────────────────────────────────────
    # Shared helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _global_gain(self, wav: torch.Tensor) -> torch.Tensor:
        """Multiply all channels by exp(U(log_min, log_max))."""
        log_g = (torch.rand(1).item()
                 * (self._gain_log_max - self._gain_log_min)
                 + self._gain_log_min)
        return wav * math.exp(log_g)
