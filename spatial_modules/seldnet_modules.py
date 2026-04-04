"""
SELDnet Core Modules — PyTorch
==============================
PyTorch porting of the three core building blocks from SELDnet
(Adavanne et al., IEEE JSTSP 2019): a multichannel STFT front-end
(``SpectralProcessor``), a 2-D convolutional block (``ConvBlock``), and a
bidirectional GRU with multiplicative merge (``BiGRU``).

Porting notes vs. original Keras:
- STFT:         offline NumPy → online ``torchaudio.transforms.Spectrogram`` layer
- DC bin:       original keeps bin 0; this port removes it (user spec.)
- ConvBlock:    order faithful to original — Conv → BN → ReLU → MaxPool → Dropout
- BiGRU:        ``merge_mode='mul'`` implemented via explicit forward/backward GRUs;
                PyTorch has no built-in merge mode, concatenation is its default
- Rec. dropout: not supported by PyTorch fused CUDA GRU; ``dropout`` covers
                input/inter-layer dropout only

References
----------
S. Adavanne, A. Politis, J. Nikunen, and T. Virtanen,
"Sound Event Localization and Detection of Overlapping Sources Using
Convolutional Recurrent Neural Networks,"
IEEE Journal of Selected Topics in Signal Processing, 2019.
https://github.com/sharathadavanne/seld-net

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# ============================================================
# 1. INPUT HANDLER
# ============================================================

class SpectralProcessor(nn.Module):
    """
    Multichannel STFT front-end.

    Converts a raw multichannel waveform into interleaved magnitude + phase
    spectra, discarding the DC bin (index 0) and keeping only positive
    frequencies (one-sided spectrum).

    Original Keras equivalent: cls_feature_class.py :: _spectrogram()
    The original is offline NumPy; this is an online differentiable layer.

    Keras parameters replicated:
      - Window:       Hanning
      - Hop:          nfft // 2  (50 % overlap)
      - Normalisation: 'ortho'  (≈ normalized=True in torchaudio)
      - Bins kept:    nfft // 2  (positive frequencies, DC removed)

    Args:
        n_fft      (int):  FFT / window size in samples. Default: 512.
        hop_length (int):  Hop between frames in samples. Default: n_fft // 2.

    Input:
        x   : (batch, channels, samples)  — float32 waveform

    Output:
        out : (batch, 2 * channels, time_frames, freq_bins)
              Channels [0 : C]    → magnitude spectra (one per input channel)
              Channels [C : 2*C]  → phase spectra     (one per input channel)
              freq_bins = n_fft // 2  (DC bin removed from the 1-sided STFT)
    """

    def __init__(self, n_fft: int = 512, hop_length: int = None):
        super().__init__()
        self.n_fft      = n_fft
        self.hop_length = hop_length or n_fft // 2
        self.freq_bins  = n_fft // 2  # bins kept after removing DC

        # power=None → complex output; normalized≈'ortho'; onesided → n_fft//2+1 bins
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft       = n_fft,
            win_length  = n_fft,
            hop_length  = self.hop_length,
            window_fn   = torch.hann_window,
            power       = None,    # complex-valued output
            normalized  = True,    # 1/sqrt(n_fft) ≈ ortho norm
            center      = True,
            pad_mode    = 'reflect',
            onesided    = True,    # → n_fft//2 + 1 bins (0 … Nyquist)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _ = x.shape

        # Apply STFT to all channels at once via batch dimension
        x_flat = x.reshape(B * C, -1)              # (B*C, samples)
        spec   = self.stft(x_flat)                 # (B*C, n_fft//2+1, T)  complex

        # Drop DC bin (index 0); keep bins 1 … n_fft//2 → freq_bins = n_fft//2
        spec = spec[:, 1:, :]                      # (B*C, freq_bins, T)
        spec = spec.reshape(B, C, self.freq_bins, -1)  # (B, C, F, T)

        mag   = spec.abs()                         # (B, C, F, T)
        phase = spec.angle()                       # (B, C, F, T)  ∈ [-π, π]

        # Magnitude channels first, then phase channels (matches original concat)
        out = torch.cat([mag, phase], dim=1)       # (B, 2*C, F, T)

        # Permute to (B, 2*C, T, F): time = dim 2, freq = dim 3
        # Matches Keras channels_first layout: (batch, 2*nb_ch, seq_len, nfft//2)
        out = out.permute(0, 1, 3, 2).contiguous()  # (B, 2*C, T, F)
        return out


# ============================================================
# 2. CONV BLOCK
# ============================================================

class ConvBlock(nn.Module):
    """
    Single SELDnet convolutional block.

    Operation order (faithful to Keras original):
        Conv2d  →  BatchNorm2d  →  ReLU  →  MaxPool(1, pool_size)  →  Dropout

    The MaxPool kernel is ``(1, pool_size)``: pools the **frequency axis only**,
    leaving the time axis untouched — a key design choice of SELDnet that
    preserves temporal resolution throughout the convolutional stack.

    Original Keras equivalent: keras_model.py lines 22–27

    Args:
        in_channels  (int):          Input channel count.
        out_channels (int):          Number of filters. Default: 64.
        kernel_size  (tuple[int]):   Conv kernel, default (3, 3).
        pool_size    (int):          Frequency pool factor. 1 = no pooling.
        dropout      (float):        Dropout rate after pooling. Default: 0.0.
    """

    def __init__(self, 
                 in_channels:  int, 
                 out_channels: int = 64,
                 kernel_size:  Tuple[int, int]  = (3, 3),
                 pool_size:    int              = 1,
                 dropout:      float            = 0.0):
        super().__init__()

        # Padding to preserve spatial size ('same' in Keras for odd kernels)
        pad = (kernel_size[0] // 2, kernel_size[1] // 2)

        # bias=False: the following BN has its own learnable scale/shift
        self.conv      = nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad, bias=False)
        self.bn        = nn.BatchNorm2d(out_channels)
        self.pool_size = pool_size
        self.drop      = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        if self.pool_size > 1:
            x = F.max_pool2d(x, kernel_size=(1, self.pool_size))
        x = self.drop(x)
        return x


# ============================================================
# 3. BIDIRECTIONAL GRU  (merge_mode = 'mul')
# ============================================================

class BiGRU(nn.Module):
    """
    Stacked Bidirectional GRU with ``merge_mode='mul'``.

    PyTorch's ``nn.GRU(bidirectional=True)`` concatenates forward and backward
    outputs, doubling the output dimension to ``2 × hidden_size``.  The
    original SELDnet uses ``merge_mode='mul'`` (element-wise product), keeping
    the output dimension equal to ``hidden_size``.  This class reproduces that
    behaviour by running two independent GRU instances per layer.

    Recurrent dropout (``recurrent_dropout`` in Keras) is not natively
    supported by PyTorch's fused CUDA GRU kernel.  The ``dropout`` parameter
    here applies standard input / inter-layer dropout instead.  With the
    default value of 0.0 the behaviour is identical.

    Original Keras equivalent: keras_model.py lines 29–36

    Args:
        in_size  (int):   Feature dimension of the input sequence.
        hidden_size (int):   GRU units per direction. Default: 128.
        num_layers  (int):   Number of stacked Bi-GRU layers. Default: 2.
        dropout     (float): Dropout between layers (not applied after the last layer). Default: 0.0.

    Input:
        x   : (batch, time_steps, input_size)

    Output:
        out : (batch, time_steps, hidden_size)
              output_dim = hidden_size  (merge_mode='mul' preserves dim)
    """

    def __init__(self,
                 in_size:  int,
                 hidden_size: int   = 128,
                 num_layers:  int   = 2,
                 dropout:     float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.fwd_layers = nn.ModuleList()
        self.bwd_layers = nn.ModuleList()

        for i in range(num_layers):
            in_dim = in_size if i == 0 else hidden_size
            self.fwd_layers.append(nn.GRU(in_dim, hidden_size, batch_first=True))
            self.bwd_layers.append(nn.GRU(in_dim, hidden_size, batch_first=True))

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal init on recurrent weights; Kaiming on input weights; zero biases."""
        for gru in (*self.fwd_layers, *self.bwd_layers):
            for name, param in gru.named_parameters():
                if 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'weight_ih' in name:
                    nn.init.kaiming_uniform_(param, nonlinearity='sigmoid')
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, (fwd, bwd) in enumerate(zip(self.fwd_layers, self.bwd_layers)):
            fwd_out, _  = fwd(x)               # (B, T, H)  — forward pass
            bwd_out, _  = bwd(x.flip(1))       # backward: feed reversed sequence
            bwd_out     = bwd_out.flip(1)      # re-align outputs to original order
            x           = fwd_out * bwd_out    # merge_mode = 'mul'
            if i < self.num_layers - 1:
                x = self.dropout(x)
        return x
