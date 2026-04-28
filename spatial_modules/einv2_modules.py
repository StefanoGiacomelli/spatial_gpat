"""
EIN-v2 Core Modules — PyTorch
==============================
PyTorch porting of four composable building blocks from EIN-SELD (EIN-v2)
by Chang et al. / Cao et al., DCASE 2021.  These modules are extracted as
standalone operators for use in NAS-style experiments that combine them with
SELDnet and PI-RNN blocks around an E-PANNs backbone.  No dimension is
hardcoded to EIN-v2's original architecture: all channel/feature sizes are
passed at construction and adapt freely.

Porting notes
-------------
- DoubleConv:        Conv→BN→ReLU→Conv→BN→ReLU, kaiming_uniform_ (leaky_relu) init.
                     No pooling/dropout — left to the surrounding architecture.
- CrossStitching:    (C, 2, 2) learnable parameter, channel-wise (no cross-channel mix).
                     Update order preserved: branch A updated first, branch B uses new A
                     (cascade as in original, not true parallel mixing).
- PositionalEncoder: sinusoidal, scaled 0.1, supports 3-D and 4-D tensors.
- TrackTransformer:  n_tracks independent TransformerEncoders, all taking the same
                     input; optional d_input→d_model projection and optional PE.

References
----------
Y. Cao, T. Iqbal, Q. Kong, Y. Cao, W. Wang, and M. D. Plumbley,
"An improved event-independent network for polyphonic sound event localization
and detection," DCASE Workshop 2021.
https://github.com/sony/audio-visual-seld-dcase2022

Y.-H. Chang et al., "EIN-SELD system for DCASE 2021 Task 3,"
DCASE Workshop 2021.

I. Misra, A. Shrivastava, A. Gupta, and M. Hebert,
"Cross-Stitch Networks for Multi-task Learning," CVPR 2016.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


# ============================================================
# INIT HELPER
# ============================================================

def _kaiming_init(module: nn.Module) -> None:
    """kaiming_uniform_ init — faithful to original EIN-SELD model_utilities.py."""
    name = module.__class__.__name__
    if 'Conv' in name or 'Linear' in name:
        nn.init.kaiming_uniform_(module.weight, nonlinearity='leaky_relu')
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif 'BatchNorm' in name:
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.constant_(module.bias, 0.0)


# ============================================================
# 1. DOUBLE CONV
# ============================================================

class DoubleConv(nn.Module):
    """
    Two consecutive Conv2d → BatchNorm2d → ReLU blocks.

    Faithfully ported from EIN-SELD ``model_utilities.py::DoubleConv``.
    Pooling and dropout are intentionally absent — they belong to the
    surrounding architecture, not to this block.

    Original Keras equivalent: used in every CNN stage of EINV2.

    Args:
        in_channels  (int):          Input channel count.
        out_channels (int):          Output channel count (same for both Conv layers).
        kernel_size  (tuple[int]):   2-D kernel. Default (3, 3).
        bias         (bool):         Add bias to Conv layers. Default False
                                     (BatchNorm provides shift/scale anyway).

    Input:  (B, in_channels,  T, F)
    Output: (B, out_channels, T, F)  — spatial size preserved ('same' padding)
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  Tuple[int, int] = (3, 3),
        bias:         bool            = False,
    ):
        super().__init__()
        pad = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels,  out_channels, kernel_size,
                      padding=pad, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size,
                      padding=pad, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        for m in self.block:
            _kaiming_init(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ============================================================
# 2. CROSS-STITCH LAYER
# ============================================================

class CrossStitching(nn.Module):
    """
    Learnable channel-wise 2×2 coupling between two parallel feature tensors.

    Each channel ``c`` has its own 2×2 learnable matrix that mixes the two
    branch signals at that channel.  No information crosses between different
    channels — the coupling is point-wise in C.

    Faithfully ported from EIN-SELD ``seld.py::EINV2.stitch`` (lines 52-110).

    **Update order — cascade, not parallel mixing:**
    The original code overwrites branch A before computing branch B, so B uses
    the *already updated* A.  This port reproduces that behaviour exactly:

        x_a_new[c] = w[c,0,0] · x_a_old[c]  +  w[c,0,1] · x_b_old[c]
        x_b_new[c] = w[c,1,0] · x_a_NEW[c]  +  w[c,1,1] · x_b_old[c]
                                  ↑ already updated

    This is a cascade (A→B information flows in one direction per step), not
    the symmetric mixing of the original cross-stitch paper (Misra et al. 2016).

    Args:
        n_channels (int):           Channel count C of each branch tensor.
        init_range (tuple[float]):  Uniform init bounds for stitch weights.
                                    Default (0.1, 0.9) as in original.

    Input:  x_a, x_b  —  both (B, C, *any_trailing_dims)
    Output: x_a_new, x_b_new  —  same shapes as inputs
    """

    def __init__(
        self,
        n_channels: int,
        init_range: Tuple[float, float] = (0.1, 0.9),
    ):
        super().__init__()
        low, high = init_range
        self.stitch = nn.Parameter(
            torch.FloatTensor(n_channels, 2, 2).uniform_(low, high)
        )

    @staticmethod
    def _ch_scale(weight: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        """Element-wise scale along C: weight (C,) × tensor (B, C, *rest)."""
        view = (1, -1) + (1,) * (tensor.dim() - 2)
        return weight.view(view) * tensor

    def forward(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Step 1: update branch A from (old A, old B)
        x_a_new = (self._ch_scale(self.stitch[:, 0, 0], x_a) +
                   self._ch_scale(self.stitch[:, 0, 1], x_b))
        # Step 2: update branch B from (new A, old B)  ← cascade, faithful to original
        x_b_new = (self._ch_scale(self.stitch[:, 1, 0], x_a_new) +
                   self._ch_scale(self.stitch[:, 1, 1], x_b))
        return x_a_new, x_b_new


# ============================================================
# 3. POSITIONAL ENCODING
# ============================================================

class PositionalEncoder(nn.Module):
    """
    Sinusoidal positional encoding, amplitude scaled by 0.1.

    Faithfully ported from EIN-SELD ``model_utilities.py::PositionalEncoding``.

    Supports:
      - 3-D input (B, C, T):    adds PE along T (pe_type='t') or along C unused
      - 4-D input (B, C, T, F): adds PE along T (pe_type='t') or F (pe_type='f')

    The internal buffer has shape (1, d_model, max_len) so it broadcasts over B
    and can be sliced to the actual sequence length at runtime.

    Args:
        d_model (int):   Feature / channel dimension.
        max_len (int):   Maximum pre-computed length. Default 512.
        pe_type (str):   'f' = frequency axis, 't' = time axis. Default 't'.
        dropout (float): Dropout applied after adding PE. Default 0.0.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int   = 512,
        pe_type: str   = 't',
        dropout: float = 0.0,
    ):
        super().__init__()
        self.pe_type = pe_type

        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = 0.1 * torch.sin(pos * div)
        pe[:, 1::2] = 0.1 * torch.cos(pos * div)
        pe = pe.unsqueeze(0).transpose(1, 2)   # (1, d_model, max_len)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:                        # (B, C, T, F)
            if self.pe_type == 't':
                x = x + self.pe.unsqueeze(3)[:, :, :x.shape[2]]
            else:                               # 'f'
                x = x + self.pe.unsqueeze(2)[:, :, :, :x.shape[3]]
        else:                                   # (B, C, T)
            x = x + self.pe[:, :, :x.shape[2]]
        return self.dropout(x)


# ============================================================
# 4. TRACK TRANSFORMER
# ============================================================

class TrackTransformer(nn.Module):
    """
    N independent Transformer encoders — one per track — all fed the same input.

    Track separation is not enforced structurally but emerges from training
    with a Permutation-Invariant loss (PIT/tPIT): each encoder learns different
    attention patterns specialised for different concurrent sources.

    Faithfully ported from EIN-SELD ``seld.py::EINV2`` (lines 60-127),
    with two generalisations:
      - Optional ``d_input → d_model`` projection (same pattern as
        ``MHAMGU.input_proj`` in pirnn_modules.py) — allows any upstream
        feature dimension, e.g. 2048 from E-PANNs.
      - ``n_tracks`` is a free parameter (hardcoded to 2 in original).

    Args:
        d_model         (int):   Transformer hidden / output dimension.
        n_heads         (int):   Attention heads. Must divide d_model evenly.
        n_tracks        (int):   Number of independent encoder instances.
        num_layers      (int):   Encoder layers per instance. Default 2.
        dim_feedforward (int):   FFN hidden size. Default 4 × d_model.
        dropout         (float): Dropout in attention and FFN. Default 0.0.
        d_input         (int):   Input feature dim. None → same as d_model
                                 (no projection added).
        pe_enable       (bool):  Prepend sinusoidal positional encoding.
                                 Default False (disabled in original).
        pe_type         (str):   'f' or 't' for PositionalEncoding. Default 't'.
        pe_dropout      (float): Dropout after PE. Default 0.0.

    Input:  x  (B, T, d_input)
    Output:    (B, T, n_tracks, d_model)
    """

    def __init__(
        self,
        d_model:         int,
        n_heads:         int,
        n_tracks:        int,
        num_layers:      int            = 2,
        dim_feedforward: Optional[int]  = None,
        dropout:         float          = 0.0,
        d_input:         Optional[int]  = None,
        pe_enable:       bool           = False,
        pe_max_len:      int            = 2048,
        pe_type:         str            = 't',
        pe_dropout:      float          = 0.0,
    ):
        super().__init__()
        self.d_model  = d_model
        self.n_tracks = n_tracks
        dim_ff = dim_feedforward or (4 * d_model)
        d_in   = d_input or d_model

        # Optional projection: align any upstream dim to d_model
        self.input_proj = (nn.Linear(d_in, d_model, bias=False)
                           if d_in != d_model else nn.Identity())

        # Optional positional encoding (expects (B, C, T) format internally)
        self.pe = (PositionalEncoder(d_model, max_len=pe_max_len,
                                     pe_type=pe_type, dropout=pe_dropout)
                   if pe_enable else None)

        # One independent encoder per track
        self.tracks = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model         = d_model,
                    nhead           = n_heads,
                    dim_feedforward = dim_ff,
                    dropout         = dropout,
                    batch_first     = False,   # (T, B, C) — matches original
                ),
                num_layers           = num_layers,
                enable_nested_tensor = False,  # no padding masks → avoid warning
            )
            for _ in range(n_tracks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_input)
        x = self.input_proj(x)                         # (B, T, d_model)

        if self.pe is not None:
            # PE expects (B, C, T); transpose in, add PE, transpose back
            x = self.pe(x.transpose(1, 2)).transpose(1, 2)

        x_t = x.transpose(0, 1)                        # (T, B, d_model)
        out  = [trk(x_t).transpose(0, 1)               # each: (B, T, d_model)
                for trk in self.tracks]
        return torch.stack(out, dim=2)                  # (B, T, n_tracks, d_model)
