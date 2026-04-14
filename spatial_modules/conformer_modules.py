"""
Conformer-family Modules — PyTorch
====================================
Standalone, parametric ports of four building blocks used across three DCASE
challenge systems.  No source repositories exist for these papers; this
implementation is derived directly from the mathematical descriptions in the
cited reports.

Modules
-------
MultiScaleChannelAttention (MSCA)
    Dual-path (global + local) channel gating for 2-D feature maps.
    [Xue2023, Sec. II-B.a, Fig. 2, Eqs. 1-3]

ResNetBlock
    Standard residual Conv-BN-ReLU block with optional MSCA inside.
    [Du2022, Xue2023, Vo2024]

ConformerBlock
    Sandwich Conformer: ½FFN → MHSA → ConvModule → ½FFN → LN.
    Operates on frame-level 1-D sequences (B, T, d_model).
    [Xue2023, Sec. II-B.b, Fig. 3, Eqs. 4-7]

AttentiveStatisticsPooling (ASP)
    Attention-weighted mean + std pooling; collapses the time dimension.
    [Xue2023, Sec. II-B.c, Fig. 4, Eqs. 8-11]

References
----------
Q. Wang et al., "NERC-SLIP system for SELD of DCASE 2022",
DCASE Workshop 2022.

L. Xue et al., "Resnet-Conformer Network using Multi-Scale Channel Attention
for Sound Event Localization and Detection in Real Scenes", WCSP 2023.

Q. T. Vo and D. K. Han, "ResNet-Conformer with Shared Weights and Attention
Mechanism for Sound Event Localization and Detection", DCASE Workshop 2024.

M. Okabe, T. Koshinaka, and K. Shinoda, "Attentive statistics pooling for
deep speaker embedding", Interspeech 2018.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# HELPERS
# ============================================================

_ACT = {
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'silu': nn.SiLU,
}


def _get_act(name: str) -> nn.Module:
    if name not in _ACT:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(_ACT)}")
    return _ACT[name]()


# ============================================================
# 1. MULTI-SCALE CHANNEL ATTENTION (MSCA)
# ============================================================

class MultiScaleChannelAttention(nn.Module):
    """
    Dual-path channel attention that combines a global context (average-pooled
    over the entire T×F map) and a local context (1×1 conv applied pointwise)
    through a shared bottleneck, then uses the combined signal as a soft gate.

    Faithfully derived from [Xue2023] Equations (1)-(3):

        g(X)  = GAP(X)                                  # (B, C, 1, 1)
        G(X)  = BN(Conv₂(ReLU(BN(Conv₁(g(X))))))       # global path
        L(X)  = BN(Conv₂(ReLU(BN(Conv₁(X)))))          # local path (1×1 conv)
        Y     = X ⊗ Sigmoid( G(X) ⊕ L(X) )             # gate the input

    Both paths use separate Conv₁/Conv₂ pairs (C → C/r → C, 1×1 kernels)
    so their BN statistics are tracked independently.

    Args:
        n_channels      (int):  Input/output channel count C.
        reduction_ratio (int):  Bottleneck factor r.  C/r must be ≥ 1.
                                Default 4 (as used in [Xue2023]).

    Input:  (B, C, T, F)
    Output: (B, C, T, F)  — same shape, channel-attention-gated
    """

    def __init__(self, n_channels: int, reduction_ratio: int = 4):
        super().__init__()
        reduced = max(1, n_channels // reduction_ratio)

        # Global path: applied to GAP output (B, C, 1, 1)
        self.global_path = nn.Sequential(
            nn.Conv2d(n_channels, reduced, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, n_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(n_channels),
        )

        # Local path: applied to full (B, C, T, F) — same 1×1 convs, separate weights
        self.local_path = nn.Sequential(
            nn.Conv2d(n_channels, reduced, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, n_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(n_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global context: collapse T and F, pass through bottleneck
        g = x.mean(dim=(2, 3), keepdim=True)   # (B, C, 1, 1)
        G = self.global_path(g)                 # (B, C, 1, 1) — broadcasts with L below

        # Local context: per-location bottleneck
        L = self.local_path(x)                  # (B, C, T, F)

        # Gate: G broadcasts (B, C, 1, 1) → (B, C, T, F) via ⊕
        gate = torch.sigmoid(G + L)             # (B, C, T, F)
        return x * gate


# ============================================================
# 2. RESNET BLOCK
# ============================================================

class ResNetBlock(nn.Module):
    """
    Two-layer residual Conv block (ResNet-18/34 style) with optional MSCA.

    Structure:
        Conv2d(3×3) → BN → ReLU → Conv2d(3×3) → BN
        [→ MSCA]                                         ← if use_msca
        + shortcut(x)                                    ← Conv1×1+BN if shapes differ
        → ReLU

    The MSCA is applied to the second-conv output *before* the residual add,
    as described in [Xue2023]: "we employ MSCA in the Resnet block to capture
    channel-level dependencies."

    No pooling inside the block — pooling belongs to the surrounding stack.

    Args:
        in_channels    (int):           Input channel count.
        out_channels   (int):           Output channel count.
        stride         (tuple[int]):    (time_stride, freq_stride). Default (1,1).
        use_msca       (bool):          Insert MSCA before residual add. Default False.
        msca_reduction (int):           MSCA reduction ratio. Default 4.
        dropout        (float):         Dropout2d rate applied after the final ReLU.
                                        Drops entire feature-map channels (appropriate
                                        for 4-D spatial tensors). Default 0.0 (off).

    Input:  (B, in_channels,  T,  F)
    Output: (B, out_channels, T', F')   T',F' depend on stride
    """

    def __init__(
        self,
        in_channels:    int,
        out_channels:   int,
        stride:         Tuple[int, int] = (1, 1),
        use_msca:       bool            = False,
        msca_reduction: int             = 4,
        dropout:        float           = 0.0,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.msca  = (MultiScaleChannelAttention(out_channels, msca_reduction)
                      if use_msca else None)
        self.drop  = nn.Dropout2d(dropout) if dropout > 0.0 else None

        # Projection shortcut when shapes change
        needs_proj = (in_channels != out_channels) or (stride != (1, 1))
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if needs_proj else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        if self.msca is not None:
            out = self.msca(out)

        out = F.relu(out + self.shortcut(x), inplace=True)
        if self.drop is not None:
            out = self.drop(out)
        return out


# ============================================================
# 3. CONFORMER BLOCK
# ============================================================

class _FFN(nn.Module):
    """Pre-norm 2-layer FFN sub-module (used twice inside ConformerBlock)."""

    def __init__(self, d_model: int, expansion: int, dropout: float, act: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1  = nn.Linear(d_model, d_model * expansion)
        self.act  = act
        self.drop1 = nn.Dropout(dropout)
        self.fc2  = nn.Linear(d_model * expansion, d_model)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return self.drop2(x)


class _ConvModule(nn.Module):
    """
    Conformer convolution sub-module (Gulati et al. 2020 style, adapted for
    1-D temporal sequences):
        LN → Pointwise(d→2d) → GLU → DepthwiseConv1d → BN → SiLU
          → Pointwise(d→d) → Dropout
    """

    def __init__(self, d_model: int, kernel_size: int, dropout: float):
        super().__init__()
        assert kernel_size % 2 == 1, "conv_kernel must be odd for 'same' padding"
        pad = (kernel_size - 1) // 2

        self.norm   = nn.LayerNorm(d_model)
        self.pw_in  = nn.Linear(d_model, 2 * d_model)          # pointwise expand
        # GLU: F.glu splits last dim in half, no extra params needed
        self.dw     = nn.Conv1d(d_model, d_model, kernel_size,  # depthwise
                                padding=pad, groups=d_model, bias=False)
        self.bn     = nn.BatchNorm1d(d_model)
        self.act    = nn.SiLU()
        self.pw_out = nn.Linear(d_model, d_model)               # pointwise reduce
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        x = self.norm(x)
        x = self.pw_in(x)                       # (B, T, 2d)
        x = F.glu(x, dim=-1)                    # (B, T, d)  — GLU along last dim
        x = x.transpose(1, 2)                   # (B, d, T)  — Conv1d expects (B,C,L)
        x = self.dw(x)                          # (B, d, T)
        x = self.bn(x)                          # (B, d, T)
        x = self.act(x)
        x = x.transpose(1, 2)                   # (B, T, d)
        x = self.pw_out(x)                      # (B, T, d)
        return self.drop(x)


class ConformerBlock(nn.Module):
    """
    Sandwich Conformer encoder block.

    Structure (from [Xue2023] Equations 4-7):
        ẑ     = z  + ½·FFN₁(z)            pre-FFN (half residual step)
        z'    = ẑ  + MHSA(ẑ)              multi-head self-attention
        z''   = z' + ConvModule(z')        depthwise-conv module
        z_out = LayerNorm(z'' + ½·FFN₂(z''))   post-FFN + final norm

    Both FFNs use pre-normalisation (LN inside).  MHSA and ConvModule also
    use pre-norm internally.

    Args:
        d_model      (int):   Feature / hidden dimension.
        n_heads      (int):   Attention heads. Must divide d_model.
        ffn_expansion(int):   FFN hidden = d_model × ffn_expansion. Default 4.
        conv_kernel  (int):   Depthwise conv kernel size (must be odd). Default 31.
        dropout      (float): Applied in FFN, MHSA, and ConvModule. Default 0.0.
        activation   (str):   FFN non-linearity: 'silu'|'relu'|'gelu'. Default 'silu'.

    Input:  x  (B, T, d_model)
    Output:    (B, T, d_model)
    """

    def __init__(
        self,
        d_model:       int,
        n_heads:       int,
        ffn_expansion: int   = 4,
        conv_kernel:   int   = 31,
        dropout:       float = 0.0,
        activation:    str   = 'silu',
    ):
        super().__init__()
        act = _get_act(activation)

        self.ffn1       = _FFN(d_model, ffn_expansion, dropout, act)
        self.mhsa_norm  = nn.LayerNorm(d_model)
        self.mhsa       = nn.MultiheadAttention(d_model, n_heads,
                                                dropout=dropout, batch_first=True)
        self.mhsa_drop  = nn.Dropout(dropout)
        self.conv_mod   = _ConvModule(d_model, conv_kernel, dropout)
        self.ffn2       = _FFN(d_model, ffn_expansion, dropout, _get_act(activation))
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-FFN: half step
        x = x + 0.5 * self.ffn1(x)

        # MHSA with pre-norm
        x_norm = self.mhsa_norm(x)
        attn, _ = self.mhsa(x_norm, x_norm, x_norm)
        x = x + self.mhsa_drop(attn)

        # Convolution module with residual
        x = x + self.conv_mod(x)

        # Post-FFN: half step + final LayerNorm
        x = self.final_norm(x + 0.5 * self.ffn2(x))
        return x


# ============================================================
# 4. ATTENTIVE STATISTICS POOLING (ASP)
# ============================================================

class AttentiveStatisticsPooling(nn.Module):
    """
    Attention-weighted mean + standard deviation pooling.

    Aggregates a variable-length temporal sequence into a fixed-size vector
    by learning a per-frame scalar importance score, then computing the
    weighted mean μ and weighted std σ, and returning their concatenation.

    Faithfully derived from [Xue2023] Equations (8)-(11):
        eₙ = vᵀ · tanh(W·hₙ + b)              scalar attention score
        αₙ = softmax({eᵢ})                     normalised weights
        μ  = Σ αₙ · hₙ                         weighted mean
        σ  = sqrt( Σ αₙ · hₙ⊙hₙ  − μ⊙μ )     weighted std
        out = [μ ; σ]                           concatenated (2·d_input,)

    Original: Okabe et al. 2018, "Attentive statistics pooling for deep
              speaker embedding."

    Args:
        d_input    (int):          Input feature dimension d.
        hidden_dim (int, optional): Dimension of the attention MLP hidden layer
                                    (W maps d → hidden_dim). Default = d_input.

    Input:  x  (B, T, d_input)
    Output:    (B, 2 · d_input)   — time dimension is fully collapsed
    """

    def __init__(self, d_input: int, hidden_dim: Optional[int] = None):
        super().__init__()
        h = hidden_dim or d_input
        self.W    = nn.Linear(d_input, h, bias=True)   # W·hₙ + b
        self.v    = nn.Linear(h, 1,       bias=False)  # vᵀ (scalar score)

    @property
    def out_dim(self) -> int:
        """Convenience: returns 2 · d_input for downstream wiring."""
        return self.W.in_features * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        # Attention scores
        e = self.v(torch.tanh(self.W(x)))    # (B, T, 1)
        alpha = torch.softmax(e, dim=1)      # (B, T, 1)  softmax over T

        # Weighted statistics
        mu    = (alpha * x).sum(dim=1)       # (B, d)
        # Weighted variance: E[h²] − E[h]²  (clamped for numerical stability)
        var   = (alpha * x * x).sum(dim=1) - mu * mu   # (B, d)
        sigma = torch.sqrt(var.clamp(min=1e-9))         # (B, d)

        return torch.cat([mu, sigma], dim=-1)           # (B, 2d)
