"""
Spatial-Semantic Adapters
=========================
``CrossStitchBridge`` — MSCA-enhanced bidirectional bridge between a spatial
feature map and a semantic feature map at different temporal and frequency
resolutions.

The bridge resolves the shape mismatch between the two branches through a
six-step pipeline designed to preserve full spatial frequency resolution:

1. MSCA-weighted projection of the spatial branch
2. Projection of the semantic branch + temporal alignment + frequency avg
3. Joint MSCA gating at full (T_s, F_s) resolution
4. CrossStitch channel mixing at full resolution
5. Spatial output: residual addition (s + projected bridge output)
6. Semantic enrichment: s_cs frequency-averaged and fed back to e as residual

The key design choice: the spatial branch NEVER collapses the frequency axis.
DOA cues in FOA audio are encoded per-frequency bin (intensity vectors);
destroying the frequency axis destroys localisation information irreversibly.
The semantic branch (mono audio, no spatial F structure) is legitimately
frequency-averaged before broadcasting onto s.

References
----------
I. Misra, A. Shrivastava, A. Gupta, and M. Hebert,
"Cross-Stitch Networks for Multi-task Learning,"
CVPR 2016.

T.-N. Vo, K. Yatabe, and N. Harada,
"Multi-Scale Channel Attention for Sound Event Localization and Detection,"
DCASE Workshop 2024.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conformer_modules import MultiScaleChannelAttention
from .einv2_modules import CrossStitching


# ══════════════════════════════════════════════════════════════════════════════
# CrossStitchBridge
# ══════════════════════════════════════════════════════════════════════════════

class CrossStitchBridge(nn.Module):
    """MSCA-enhanced bidirectional bridge between spatial and semantic branches.

    Exchanges information between a spatial feature map ``s`` and a semantic
    feature map ``e`` that may differ in all four dimensions ``(B, C, T, F)``.

    The spatial branch retains full frequency resolution throughout — only
    the semantic branch is frequency-averaged (legitimate: it comes from mono
    audio and carries no per-frequency spatial cues).  Both outputs use additive
    residual connections so the bridge is initially neutral and gradients flow
    directly through the identity path.

    Parameters
    ----------
    c_spatial : int
        Channel count of the spatial feature map.
    c_semantic : int
        Channel count of the semantic feature map.
    c_bridge : int
        Intermediate channel dimension inside the bridge.
    dropout : float
        Dropout probability applied to the bridge outputs before residual add.
        Default 0.3.

    Inputs (``forward``)
    --------------------
    s : torch.Tensor, shape (B, c_spatial, T_s, F_s)
        Spatial features.
    e : torch.Tensor, shape (B, c_semantic, T_sem, F_sem)
        Semantic features.

    Outputs (``forward``)
    ---------------------
    s_out : torch.Tensor, shape (B, c_spatial, T_s, F_s)
        Enriched spatial features — same shape as ``s``.
    e_out : torch.Tensor, shape (B, c_semantic, T_sem, F_sem)
        Enriched semantic features — same shape as ``e``.
    """

    def __init__(
        self,
        c_spatial:  int,
        c_semantic: int,
        c_bridge:   int,
        dropout:    float = 0.3,
    ):
        """Initialize CrossStitchBridge."""
        super().__init__()
        self.c_bridge = c_bridge

        # Spatial MSCA: channel attention before projecting into bridge
        self.msca_s = MultiScaleChannelAttention(c_spatial)

        # 1×1 projections into bridge dim
        self.proj_s2b = nn.Conv2d(c_spatial,  c_bridge, kernel_size=1)
        self.proj_e2b = nn.Conv2d(c_semantic, c_bridge, kernel_size=1)

        # Joint MSCA on the concatenated features at full spatial resolution
        self.msca_joint = MultiScaleChannelAttention(2 * c_bridge)

        # CrossStitch layer (c_bridge channels, same shape for both branches)
        self.cross_stitch = CrossStitching(c_bridge)

        # 1×1 projections back to original dims
        self.proj_b2s = nn.Conv2d(c_bridge, c_spatial,  kernel_size=1)
        self.proj_b2e = nn.Conv2d(c_bridge, c_semantic, kernel_size=1)

        # Dropout applied to bridge outputs before residual addition
        self.drop = nn.Dropout(p=dropout)

    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        s: torch.Tensor,
        e: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply MSCA-enhanced CrossStitch bidirectional feature fusion.

        The spatial branch never loses its frequency axis — only the semantic
        branch is frequency-averaged (it carries no per-frequency DOA cues).
        Both outputs are additive residuals, keeping the bridge initially neutral.

        Parameters
        ----------
        s : torch.Tensor, shape (B, c_spatial, T_s, F_s)
            Spatial feature map.
        e : torch.Tensor, shape (B, c_semantic, T_sem, F_sem)
            Semantic feature map.

        Returns
        -------
        s_out : torch.Tensor, shape (B, c_spatial, T_s, F_s)
        e_out : torch.Tensor, shape (B, c_semantic, T_sem, F_sem)
        """
        T_s,   F_s   = s.shape[2], s.shape[3]
        T_sem, F_sem = e.shape[2], e.shape[3]

        # Step 1 — MSCA channel attention on s (shape-preserving)
        s_att  = self.msca_s(s)                                   # (B, C_s, T_s, F_s)

        # Step 2 — Project s into bridge space
        s_proj = self.proj_s2b(s_att)                             # (B, C_b, T_s, F_s)

        # Step 3 — Project e, align T_sem→T_s, average over F_sem, broadcast to F_s
        # Averaging F_sem is correct: e comes from mono audio with no spatial F structure.
        e_proj      = self.proj_e2b(e)                            # (B, C_b, T_sem, F_sem)
        e_t_aligned = F.interpolate(e_proj, size=(T_s, F_sem),
                                    mode='bilinear',
                                    align_corners=False)           # (B, C_b, T_s, F_sem)
        e_f_pooled  = e_t_aligned.mean(dim=3, keepdim=True)       # (B, C_b, T_s, 1)
        e_broad     = e_f_pooled.expand(-1, -1, -1, F_s)          # (B, C_b, T_s, F_s)

        # Step 4 — Joint MSCA at full (T_s, F_s) resolution
        joint  = self.msca_joint(
            torch.cat([s_proj, e_broad], dim=1)                   # (B, 2*C_b, T_s, F_s)
        )
        s_mid  = joint[:, :self.c_bridge]                         # (B, C_b, T_s, F_s)
        e_mid  = joint[:, self.c_bridge:]                         # (B, C_b, T_s, F_s)

        # Step 5 — CrossStitch at full resolution
        s_cs, e_cs = self.cross_stitch(s_mid, e_mid)             # (B, C_b, T_s, F_s) each

        # Step 6 — Spatial output: project back + dropout + residual
        s_back = self.proj_b2s(s_cs)                              # (B, C_s, T_s, F_s)
        s_out  = s + self.drop(s_back)                            # (B, C_s, T_s, F_s)

        # Step 7 — Semantic enrichment: avg F_s from s_cs, project, align T, residual
        e_from_s = s_cs.mean(dim=3, keepdim=True)                 # (B, C_b, T_s, 1)
        e_back_p = self.proj_b2e(e_from_s)                        # (B, C_e, T_s, 1)
        e_back   = F.interpolate(e_back_p, size=(T_sem, 1),
                                 mode='bilinear',
                                 align_corners=False)              # (B, C_e, T_sem, 1)
        if F_sem > 1:
            e_back = e_back.expand(-1, -1, -1, F_sem)             # (B, C_e, T_sem, F_sem)
        e_out = e + self.drop(e_back)                             # (B, C_e, T_sem, F_sem)

        return s_out, e_out
