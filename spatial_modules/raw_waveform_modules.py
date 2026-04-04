"""
Raw-Waveform Front-End Modules — PyTorch
=========================================
Standalone, parametric operators for learning representations directly from
raw multichannel waveforms, ported from two Oxford papers (no public source code
available; implementations derived from equations, figures, and tables in the PDFs).

Modules
-------
LearnableGaborFilterBank
    Single-channel raw waveform → complex TF representation (real + imaginary parts).
    Gabor kernels parameterized by learnable center frequency η and bandwidth σ,
    recomputed at each forward pass.  Initialized on mel-scale.
    Source: SoundDoA Sec. 3.1, Eq. (1).

GaborFrontEnd
    Multi-channel wrapper: applies LearnableGaborFilterBank to each mic independently,
    then computes inter-channel cross-spectra for DoA encoding.
    Output shape (B, n_spatial, T_out, n_filters) is directly compatible with the
    ResNetBlock / MSCA / DoubleConv (B, C, T, F) convention.
    Source: SoundDoA Sec. 3.1, Eq. (4), Fig. 2-A.

FormantEnhancementModule
    Learnable IIR smoother along the frequency axis to extract a "formant trace",
    then split-and-combine with sigmoid/tanh branches.  Plug-in, shape-preserving.
    Source: SoundDoA Sec. 3.1, Eq. (2)–(3), Fig. 3.

MaxCorrFilterBank
    Rectangular band-pass (sinc) filter extended to multi-channel via per-channel
    learnable integer time delays.  6 learnable parameters per filter (f1, f2, t1-t4).
    Source: SoundDet Sec. 3.2, Eq. (2)–(3), Fig. 2.

References
----------
Y. He, N. Trigoni, and A. Markham,
"SoundDet: Polyphonic Sound Event Detection and Localization from Raw Waveform,"
ICML 2021.

Y. He and A. Markham,
"SoundDoA: Learn Sound Source Direction of Arrival and Semantics from Sound Raw
Waveforms," Interspeech 2022.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# HELPERS
# ============================================================

def _hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_center_freqs(n: int, fmin: float, fmax: float, sr: int) -> torch.Tensor:
    """Normalized center frequencies (0..0.5) on mel scale, shape (n,)."""
    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)
    mels    = [mel_min + i * (mel_max - mel_min) / (n - 1) for i in range(n)]
    hz      = torch.tensor([_mel_to_hz(m) for m in mels], dtype=torch.float32)
    return hz / sr   # normalized frequency in (0, 0.5)


def _sinc(x: torch.Tensor) -> torch.Tensor:
    """Element-wise sinc: sin(x)/x with sinc(0)=1."""
    return torch.where(x == 0.0, torch.ones_like(x), torch.sin(x) / x)


def _hamming(n: int, device: torch.device) -> torch.Tensor:
    """Hamming window of length n."""
    t = torch.arange(n, dtype=torch.float32, device=device)
    return 0.54 - 0.46 * torch.cos(2.0 * math.pi * t / (n - 1))


class _StraightThroughRound(torch.autograd.Function):
    """Round in forward; identity gradient in backward (STE)."""
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x.round()

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        return grad


_ste_round = _StraightThroughRound.apply


# ============================================================
# MODULE 1 — LearnableGaborFilterBank
# ============================================================

class LearnableGaborFilterBank(nn.Module):
    """
    Learnable Gabor filter bank applied to a single-channel raw waveform.

    Each of the N filters is parameterized by a learnable center frequency η_n
    (normalized in [0, 0.5]) and Gaussian bandwidth σ_n > 0:

        g_n(t) = (1/√(2πσ_n²)) · exp(-t²/2σ_n²) · exp(j2πη_n·t)

    The filter is split into real and imaginary parts and applied as two
    1-D convolutions sharing the Gaussian envelope.  Kernels are recomputed
    from (η_raw, log_σ) at every forward pass so gradients flow through them.

    A Hamming window is applied to reduce spectral leakage.

    Parameters
    ----------
    n_filters     : number of Gabor filters (= output frequency bins)
    kernel_length : kernel size in samples (odd recommended; longer → narrower band)
    stride        : hop size (temporal downsampling factor)
    sr            : sample rate in Hz (used only for mel-scale init)
    fmin, fmax    : frequency range for mel-scale initialization (Hz)

    Forward
    -------
    x : (B, T_samples)
    → (B, 2*n_filters, T_out)   first half = real, second half = imaginary
    """

    def __init__(
        self,
        n_filters    : int,
        kernel_length: int,
        stride       : int,
        sr           : int   = 24000,
        fmin         : float = 50.0,
        fmax         : float = 8000.0,
    ) -> None:
        super().__init__()
        self.n_filters     = n_filters
        self.kernel_length = kernel_length
        self.stride        = stride

        # ── Learnable parameters ──────────────────────────────
        # η_raw → sigmoid(η_raw) * 0.5  ∈ (0, 0.5)  normalized frequency
        # log_σ → exp(log_σ)  > 0                    Gaussian std in samples

        # Mel-scale initialisation
        eta_init = _mel_center_freqs(n_filters, fmin, fmax, sr)
        # Invert sigmoid: η_raw = logit(2η)
        eta_raw_init = torch.log(2.0 * eta_init / (1.0 - 2.0 * eta_init))

        # σ: inversely proportional to frequency (wider bandwidth at high freq)
        # σ_n ≈ sr / (4 * freq_n)  expressed in samples; clamp to reasonable range
        sigma_init = (sr / (4.0 * eta_init * sr)).clamp(min=4.0, max=kernel_length // 4)
        log_sigma_init = torch.log(sigma_init)

        self.eta_raw  = nn.Parameter(eta_raw_init)    # (n_filters,)
        self.log_sigma = nn.Parameter(log_sigma_init)  # (n_filters,)

        # ── Time axis (not a parameter) ───────────────────────
        # linspace gives exactly kernel_length symmetric points for any parity.
        # arange(-half, half+1) yields kernel_length+1 points when kernel_length is even.
        half = kernel_length // 2
        t = torch.linspace(-half, half, kernel_length, dtype=torch.float32)
        self.register_buffer('t', t)

    def _build_kernels(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct real and imaginary filter kernels from learnable params.

        Returns
        -------
        real_w : (n_filters, 1, kernel_length)
        imag_w : (n_filters, 1, kernel_length)
        """
        eta   = torch.sigmoid(self.eta_raw) * 0.5            # (N,) in (0, 0.5)
        sigma = torch.exp(self.log_sigma)                     # (N,) > 0

        # Gaussian envelope: (N, K) where K = kernel_length
        t    = self.t.unsqueeze(0)                            # (1, K)
        s    = sigma.unsqueeze(1)                             # (N, 1)
        e    = eta.unsqueeze(1)                               # (N, 1)
        gauss = torch.exp(-t ** 2 / (2.0 * s ** 2)) / (s * math.sqrt(2.0 * math.pi))

        # Complex modulation
        phase     = 2.0 * math.pi * e * t                    # (N, K)
        real_kern = gauss * torch.cos(phase)                  # (N, K)
        imag_kern = gauss * torch.sin(phase)                  # (N, K)

        # Hamming window (on buffer device)
        win = _hamming(self.kernel_length, self.t.device)     # (K,)
        real_kern = real_kern * win
        imag_kern = imag_kern * win

        return real_kern.unsqueeze(1), imag_kern.unsqueeze(1)  # (N,1,K) each

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T_samples)
        → (B, 2*n_filters, T_out)
        """
        real_w, imag_w = self._build_kernels()                  # (N,1,K) each
        pad = self.kernel_length // 2
        x_ = x.unsqueeze(1)                                     # (B, 1, T)
        real_out = F.conv1d(x_, real_w, stride=self.stride, padding=pad)  # (B,N,T_out)
        imag_out = F.conv1d(x_, imag_w, stride=self.stride, padding=pad)  # (B,N,T_out)
        return torch.cat([real_out, imag_out], dim=1)            # (B, 2N, T_out)


# ============================================================
# MODULE 2 — GaborFrontEnd
# ============================================================

class GaborFrontEnd(nn.Module):
    """
    Multi-channel Gabor front-end for microphone array waveforms.

    Applies one shared LearnableGaborFilterBank to each microphone channel
    independently, then computes cross-spectra between the reference channel
    and all others for inter-channel phase (DoA) encoding.

    FOA format  (format='foa'):
        Channel 0 = omni W (reference); channels 1-3 = x, y, z.
        Cross-spectra: Z_k · conj(Z_0) for k=1,2,3.
        Stack: [|Z_0|, Re(CS_1), Im(CS_1), Re(CS_2), Im(CS_2), Re(CS_3), Im(CS_3)]
        → n_spatial = 7 features per (filter, time) point.

    MIC format  (format='mic'):
        All C channels treated symmetrically; cross-spectra with channel 0.
        → n_spatial = 1 + 2*(n_mics-1).

    Output is directly compatible with 2-D CNN modules (ResNetBlock, MSCA,
    DoubleConv) via the (B, C, T, F) convention where:
        C = n_spatial  (feature/spatial channels)
        T = T_out      (time frames)
        F = n_filters  (frequency bins)

    Parameters
    ----------
    n_filters, kernel_length, stride, sr, fmin, fmax : forwarded to filter bank
    n_mics   : number of microphone channels (default 4)
    format   : 'foa' or 'mic'

    Forward
    -------
    x : (B, n_mics, T_samples)
    → (B, n_spatial, T_out, n_filters)
    """

    def __init__(
        self,
        n_filters    : int,
        kernel_length: int,
        stride       : int,
        n_mics       : int   = 4,
        format       : str   = 'foa',
        sr           : int   = 24000,
        fmin         : float = 50.0,
        fmax         : float = 8000.0,
    ) -> None:
        super().__init__()
        assert format in ('foa', 'mic'), "format must be 'foa' or 'mic'"
        self.n_filters = n_filters
        self.n_mics    = n_mics
        self.format    = format

        self.gabor = LearnableGaborFilterBank(n_filters, kernel_length, stride, sr, fmin, fmax)

    @property
    def n_spatial(self) -> int:
        """Number of output spatial/feature channels."""
        return 1 + 2 * (self.n_mics - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, n_mics, T_samples)
        → (B, n_spatial, T_out, n_filters)
        """
        B, C, T = x.shape
        assert C == self.n_mics

        # Apply Gabor to each channel: (B, T_samples) → (B, 2N, T_out)
        outs = [self.gabor(x[:, c, :]) for c in range(C)]
        # Split into real and imaginary: each (B, N, T_out)
        N = self.n_filters
        reals = [o[:, :N, :] for o in outs]   # per-channel real part
        imags = [o[:, N:, :] for o in outs]   # per-channel imaginary part

        # Reference channel (0): omni / W
        R0 = reals[0]   # (B, N, T_out)
        I0 = imags[0]   # (B, N, T_out)

        # Omni magnitude: |Z_0|
        mag0 = torch.sqrt(R0 ** 2 + I0 ** 2 + 1e-9)  # (B, N, T_out)

        # Cross-spectra with other channels: Z_k · conj(Z_0)
        features = [mag0]
        for k in range(1, C):
            Rk = reals[k]
            Ik = imags[k]
            # Z_k · Z_0* = (Rk·R0 + Ik·I0) + j(Ik·R0 - Rk·I0)
            cs_real = Rk * R0 + Ik * I0
            cs_imag = Ik * R0 - Rk * I0
            features.append(cs_real)
            features.append(cs_imag)

        # Stack: (B, n_spatial, N, T_out)
        out = torch.stack(features, dim=1)    # (B, n_spatial, N, T_out)
        # Permute to (B, n_spatial, T_out, N) = (B, C, T, F)
        return out.permute(0, 1, 3, 2).contiguous()


# ============================================================
# MODULE 3 — FormantEnhancementModule
# ============================================================

class FormantEnhancementModule(nn.Module):
    """
    Learnable formant split-and-combine enhancement module.

    A learnable IIR (infinite impulse response) filter with a single
    smoothing coefficient s = sigmoid(s_raw) ∈ (0, 1) sweeps along the
    frequency axis of a 2-D feature map to extract a "formant trace"
    fr (moving average in the frequency domain).  The complementary
    "detail" signal dt = x − fr carries fine spectral structure.

    Two independent Conv2d branches (kernel (3,1): 3-freq-bin context,
    no time context) then recombine:

        fr_n(f) = (1−s)·x(f) + s·fr(f−1)   [IIR along freq axis]
        dt(F)   = x − fr(F)
        F_out   = sigmoid(Conv_fr(fr)) + tanh(Conv_dt(dt))

    This matches the description in SoundDoA Sec. 3.1, Eq. (2)–(3) and
    the architecture table (Conv2d@3,1 for both branches).

    Parameters
    ----------
    n_channels  : number of input/output channels C
    kernel_size : Conv2d kernel size for the two branches (default (3,1))

    Forward
    -------
    x : (B, C, T, F)
    → (B, C, T, F)   shape-preserving
    """

    def __init__(
        self,
        n_channels : int,
        kernel_size: Tuple[int, int] = (3, 1),
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        pad = (kernel_size[0] // 2, kernel_size[1] // 2)

        # Learnable IIR smoothing coefficient (before sigmoid)
        self.s_raw = nn.Parameter(torch.zeros(1))

        # Two independent Conv2d branches
        self.conv_fr = nn.Conv2d(n_channels, n_channels, kernel_size, padding=pad, bias=True)
        self.conv_dt = nn.Conv2d(n_channels, n_channels, kernel_size, padding=pad, bias=True)

        nn.init.kaiming_uniform_(self.conv_fr.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.conv_dt.weight, a=math.sqrt(5))

    def _iir_formant(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute formant trace via causal IIR along the frequency axis (last dim).

        x   : (B, C, T, F)
        out : (B, C, T, F)   — same shape
        """
        s = torch.sigmoid(self.s_raw)     # scalar in (0, 1)
        B, C, T, F = x.shape
        # Run causal filter bin by bin along frequency
        fr = x[..., 0:1]                  # (B, C, T, 1)  — initialise at bin 0
        traces = [fr]
        for f in range(1, F):
            fr = (1.0 - s) * x[..., f:f+1] + s * fr
            traces.append(fr)
        return torch.cat(traces, dim=-1)   # (B, C, T, F)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, C, T, F) → (B, C, T, F)"""
        fr = self._iir_formant(x)            # formant trace
        dt = x - fr                           # detail
        return torch.sigmoid(self.conv_fr(fr)) + torch.tanh(self.conv_dt(dt))


# ============================================================
# MODULE 4 — MaxCorrFilterBank
# ============================================================

class MaxCorrFilterBank(nn.Module):
    """
    MaxCorr band-pass filter bank with learnable per-channel time shifts.

    Each of the N filters is defined by:
        k[n] = 2f2·sinc(2πf2·n) − 2f1·sinc(2πf1·n)    (band-pass sinc)

    and applied to channel c of the input with a learnable integer time delay t_c:
        g_c[n] = k[n + round(t_c)]

    The six learnable parameters per filter are f1, df (→ f2=f1+df), and t1..tc.
    - f1 = sigmoid(f1_raw) × 0.4           → f1 ∈ (0, 0.4)
    - f2 = f1 + softplus(df_raw) × 0.1     → f2 > f1, both < 0.5
    - t_i  are unconstrained; integer rounding via STE for differentiability

    A Hamming window suppresses sinc sidelobes.  The filter bank is extremely
    parameter-efficient: 2 + n_channels learnable scalars per filter vs.
    kernel_length × n_channels for a standard 1-D conv.

    Output shape (B, n_filters, T_out, n_channels) places n_channels in the
    last axis (frequency-like) for (B, C, T, F)-style compatibility with the
    rest of the toolbox.

    Parameters
    ----------
    n_filters     : number of band-pass filters
    kernel_length : sinc kernel length in samples (odd recommended; 481 in paper)
    stride        : temporal hop size
    n_channels    : number of input microphone channels (default 4)

    Forward
    -------
    x : (B, n_channels, T_samples)
    → (B, n_filters, T_out, n_channels)
    """

    def __init__(
        self,
        n_filters    : int,
        kernel_length: int,
        stride       : int,
        n_channels   : int = 4,
    ) -> None:
        super().__init__()
        self.n_filters     = n_filters
        self.kernel_length = kernel_length
        self.stride        = stride
        self.n_channels    = n_channels

        # ── Learnable frequency parameters ──────────────────
        # f1_raw: sigmoid(·)×0.4  → f1 ∈ (0, 0.4)
        # df_raw: softplus(·)×0.1 → df > 0   → f2 = f1 + df
        self.f1_raw = nn.Parameter(torch.zeros(n_filters))
        self.df_raw = nn.Parameter(torch.ones(n_filters))   # bias toward positive df

        # ── Learnable time-shift parameters ─────────────────
        # t_raw: (n_filters, n_channels) — initialized near zero (no shift)
        self.t_raw = nn.Parameter(torch.zeros(n_filters, n_channels))

        # ── Time axis buffer ─────────────────────────────────
        half = kernel_length // 2
        t = torch.arange(-half, half + 1, dtype=torch.float32)
        self.register_buffer('t', t)

    def _build_kernels(self) -> torch.Tensor:
        """
        Build per-filter, per-channel sinc kernels with time shifts.

        Returns
        -------
        kernels : (n_filters * n_channels, 1, kernel_length)
            Ready for F.conv1d with groups or loop over channels.
        """
        K   = self.kernel_length
        N   = self.n_filters
        C   = self.n_channels
        dev = self.t.device

        f1 = torch.sigmoid(self.f1_raw) * 0.4                        # (N,) ∈ (0,0.4)
        df = F.softplus(self.df_raw) * 0.1                           # (N,) > 0
        f2 = (f1 + df).clamp(max=0.499)                              # (N,) > f1, < 0.5

        # Hamming window
        win = _hamming(K, dev)                                        # (K,)

        # Base sinc band-pass kernels (before time shift)
        t   = self.t                                                  # (K,)
        two_pi_t = 2.0 * math.pi * t                                 # (K,)

        f1_ = f1.unsqueeze(1)                                        # (N,1)
        f2_ = f2.unsqueeze(1)                                        # (N,1)
        base = (2.0 * f2_ * _sinc(two_pi_t * f2_)
                - 2.0 * f1_ * _sinc(two_pi_t * f1_))                 # (N, K)
        base = base * win                                             # (N, K)

        # Integer time shifts via STE
        shifts = _ste_round(self.t_raw)                              # (N, C) integers

        # Build per-channel shifted kernels
        all_kernels = []
        for c in range(C):
            shift_c = shifts[:, c].long()                            # (N,)
            # Roll each filter kernel by its shift
            # Positive shift: kernel moves right (signal delayed)
            rolled = torch.stack([
                torch.roll(base[n], shifts[n, c].long().item())
                for n in range(N)
            ], dim=0)                                                 # (N, K)
            all_kernels.append(rolled)

        # Stack: (N, C, K) → reshape to (N*C, 1, K)
        kernels = torch.stack(all_kernels, dim=1)                    # (N, C, K)
        return kernels.view(N * C, 1, K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, n_channels, T_samples)
        → (B, n_filters, T_out, n_channels)
        """
        B, C, T = x.shape
        assert C == self.n_channels
        N = self.n_filters
        K = self.kernel_length

        kernels = self._build_kernels()                               # (N*C, 1, K)
        pad = K // 2

        # Apply each filter-channel combination independently
        # Reshape x to (B*C, 1, T) and run all N*C kernels in one conv
        # Then reshape output
        out_per_channel = []
        for c in range(C):
            # Kernels for channel c: rows c, c+C, c+2C, ... → indices [c, c+C, ...]
            # Actually layout is (N, C, K) → (N*C, 1, K) with inner dim C
            # kernel for filter n, channel c is at index n*C + c
            idx = [n * C + c for n in range(N)]
            k_c = kernels[idx]                                        # (N, 1, K)
            x_c = x[:, c:c+1, :]                                     # (B, 1, T)
            y_c = F.conv1d(x_c, k_c, stride=self.stride, padding=pad)  # (B, N, T_out)
            out_per_channel.append(y_c)

        out = torch.stack(out_per_channel, dim=-1)                    # (B, N, T_out, C)
        return out
