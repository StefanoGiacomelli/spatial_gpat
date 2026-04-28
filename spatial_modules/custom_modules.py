"""
custom_modules.py — Project-specific temporal modules
======================================================
Temporal Convolutional Network (TCN) for per-track sequence smoothing.

TCN replaces BiGRU in the standalone SELD smoother stage.

Advantages over BiGRU:
  - Fully parallelizable — no sequential recurrence, all frames processed at once
  - Explicit receptive field via exponential dilation
  - Fast on both CUDA and MPS (1-D depthwise conv is hardware-optimised)

Receptive field with n_layers blocks, kernel k:
    RF = k × (2^n_layers − 1)
    k=3, n=6  →  RF = 189 frames  (≈ 1.9 s at 10 ms/frame)

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

import torch
import torch.nn as nn


class TCNBlock(nn.Module):
    """Non-causal dilated depthwise-separable 1-D TCN block.

    Structure (pre-norm, residual):
        LN → transpose → DW_conv(dilation) → GELU → PW_conv → transpose → Dropout → + x

    Symmetric dilation padding ensures non-causal (full-sequence) operation,
    which maximises accuracy in offline batch training.
    """

    def __init__(
        self,
        d_model:     int,
        kernel_size: int   = 3,
        dilation:    int   = 1,
        dropout:     float = 0.0,
    ):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2   # symmetric → non-causal
        self.norm = nn.LayerNorm(d_model)
        self.dw   = nn.Conv1d(
            d_model, d_model, kernel_size,
            dilation=dilation, padding=pad,
            groups=d_model, bias=False,
        )
        self.pw   = nn.Conv1d(d_model, d_model, 1, bias=False)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        residual = x
        x = self.norm(x).transpose(1, 2)    # (B, d, T)
        x = self.act(self.dw(x))
        x = self.pw(x).transpose(1, 2)      # (B, T, d)
        return residual + self.drop(x)


class TCN(nn.Module):
    """Stack of TCNBlocks with exponential dilation.

    Input and output dim are both d_model (no projection needed when used
    as a per-track smoother receiving TrackTransformer output at d_model dim).
    """

    def __init__(
        self,
        d_model:     int,
        n_layers:    int,
        kernel_size: int   = 3,
        dropout:     float = 0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            TCNBlock(d_model, kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(n_layers)
        ])
        self.out_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        for blk in self.blocks:
            x = blk(x)
        return x   # (B, T, d_model)
