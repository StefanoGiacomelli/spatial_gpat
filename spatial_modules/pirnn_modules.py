"""
PI-RNN Core Module — PyTorch
============================================
Multi-Head Attention Minimal Gated Unit (MHA-MGU): the stateful recurrent cell
of the Permutation-Invariant RNN proposed by Diaz-Guerra et al.

The module maintains a set of M_H track embeddings updated at each timestep via
a two-stage operation: Multi-Head Attention soft assignment followed by a
Minimal Gated Unit (MGU) state update.  The attention mechanism is
order-invariant in the detection set, making the tracker permutation-invariant
with respect to the input ordering of detections.

References
----------
D. Diaz-Guerra, A. Politis, A. Miguel, J. R. Beltran, and T. Virtanen,
"Permutation Invariant Recurrent Neural Networks for Sound Source Tracking,"
Forum Acusticum 2023 / arXiv:2306.08510.

G.-B. Zhou, J. Wu, C.-L. Zhang, and Z.-H. Zhou,
"Minimal Gated Unit for Recurrent Neural Networks,"
International Journal of Automation and Computing, 2016.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════════
# MHA-MGU cell
# ══════════════════════════════════════════════════════════════════════════════

class MHA_MGU(nn.Module):
    """Multi-Head Attention Minimal Gated Unit.

    Maintains a set of :math:`M_H` track embeddings updated at each timestep
    via:

    1. **MHA soft assignment** — each track (query) attends over the union of
       current detections and previous track states (key-value set):

       .. math::
           C(t) = \\text{MHA}\\!\\left(Q{=}H(t{-}1),\\; KV{=}X(t)\\cup H(t{-}1)\\right)

    2. **MGU gated recurrence** — context drives a simplified single-gate update:

       .. math::
           z_i(t)   &= \\sigma(W_z\\, c_i(t)) \\\\
           \\tilde h_i(t) &= \\tanh(W_h\\, c_i(t)) \\\\
           h_i(t)   &= (1 - z_i(t)) \\odot h_i(t{-}1) + \\tilde h_i(t)

    Properties
    ----------
    * **Permutation-invariant** in :math:`X(t)`: reordering detections does not
      change the set of track-specific context vectors.
    * **Permutation-equivariant** in :math:`H(t)`: reordering track states
      reorders the queries and therefore the updated states identically.

    Parameters
    ----------
    d_model : int
        Track embedding dimension (:math:`d_h` in the paper).
    n_heads : int
        Number of MHA heads.  Must divide ``d_model`` evenly.
    n_tracks : int
        Number of tracks :math:`M_H` maintained in the state.
    d_input : int, optional
        Dimension of one detection vector :math:`d_x`.
        If ``None``, assumed equal to ``d_model`` (no input projection).
    dropout : float
        Dropout rate inside MHA.  Default ``0.0``.

    Inputs (``forward``)
    --------------------
    x  : torch.Tensor, shape (B, T, M_X, d_input)
        Sequence of detection sets.
    h0 : torch.Tensor, shape (B, M_H, d_model), optional
        Initial track states.  Uses learned ``self.h0`` if ``None``.

    Outputs (``forward``)
    ---------------------
    H       : torch.Tensor, shape (B, T, M_H, d_model)
        Track states at every timestep.
    h_final : torch.Tensor, shape (B, M_H, d_model)
        Final state (pass as ``h0`` for the next chunk).
    """

    def __init__(
        self,
        d_model:  int,
        n_heads:  int,
        n_tracks: int,
        d_input:  Optional[int] = None,
        dropout:  float         = 0.0,
    ):
        """Initialize MHA_MGU."""
        super().__init__()
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.n_tracks = n_tracks
        d_in = d_input or d_model

        # Project detections to d_model if dimensions differ
        self.input_proj = (nn.Linear(d_in, d_model, bias=False)
                           if d_in != d_model else nn.Identity())

        # MHA: tracks (Q) attend over detections + previous track states (K, V)
        self.mha = nn.MultiheadAttention(
            embed_dim   = d_model,
            num_heads   = n_heads,
            dropout     = dropout,
            batch_first = True,
        )

        # MGU — forget gate and candidate (both solely from context c_i(t))
        self.W_z = nn.Linear(d_model, d_model, bias=True)
        self.W_h = nn.Linear(d_model, d_model, bias=True)

        # Learnable per-track initial state — breaks the zero-init symmetry so
        # different tracks develop different embeddings from the first step.
        self.h0 = nn.Parameter(torch.zeros(n_tracks, d_model))
        nn.init.normal_(self.h0, std=0.01)

    # ──────────────────────────────────────────────────────────────────────────

    def step(
        self,
        x_t:         torch.Tensor,
        h_prev:      torch.Tensor,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Execute a single-timestep MHA-MGU update.

        Computes one step of the two-stage update:

        .. math::
            C(t)     &= \\text{MHA}(H(t{-}1),\\; X(t) \\cup H(t{-}1)) \\\\
            h_i(t)   &= (1 - \\sigma(W_z c_i)) \\odot h_i(t{-}1)
                        + \\tanh(W_h c_i)

        Parameters
        ----------
        x_t : torch.Tensor, shape (B, M_X, d_input)
            Current detection set.
        h_prev : torch.Tensor, shape (B, M_H, d_model)
            Previous track states.
        return_attn : bool
            If ``True``, also return the MHA attention weight matrix.

        Returns
        -------
        h_new : torch.Tensor, shape (B, M_H, d_model)
            Updated track states.
        attn_weights : torch.Tensor or None
            Shape (B, M_H, M_X + M_H) when ``return_attn=True``,
            otherwise ``None``.
        """
        # 1. Project detections to d_model
        x_proj = self.input_proj(x_t)                       # (B, M_X, d_model)

        # 2. Build key-value set: X(t) ∪ H(t-1)
        kv = torch.cat([x_proj, h_prev], dim=1)             # (B, M_X+M_H, d_model)

        # 3. Soft assignment: each track (Q) attends over detections + prev tracks (K,V)
        context, attn_w = self.mha(
            h_prev, kv, kv,
            need_weights=return_attn,
        )                                                    # (B, M_H, d_model)

        # 4. Minimal Gated Unit — update each track state
        z      = torch.sigmoid(self.W_z(context))           # (B, M_H, d_model)  forget gate
        h_cand = torch.tanh(self.W_h(context))              # (B, M_H, d_model)  candidate
        h_new  = (1.0 - z) * h_prev + h_cand               # (B, M_H, d_model)

        return h_new, (attn_w if return_attn else None)

    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x:  torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a full sequence of detection sets.

        Unrolls ``step`` over ``T`` timesteps, using the learnable initial
        state ``self.h0`` when ``h0`` is not provided.

        Parameters
        ----------
        x : torch.Tensor, shape (B, T, M_X, d_input)
            Sequence of detection sets.
        h0 : torch.Tensor, shape (B, M_H, d_model), optional
            Initial track states.  Defaults to the learned ``self.h0``.

        Returns
        -------
        H : torch.Tensor, shape (B, T, M_H, d_model)
            Track states at every timestep.
        h_final : torch.Tensor, shape (B, M_H, d_model)
            Final track state (can be passed as ``h0`` for the next chunk).
        """
        B, T, _M_X, _ = x.shape

        h = (self.h0.unsqueeze(0).expand(B, -1, -1).contiguous()
             if h0 is None else h0)

        outputs = []
        for t in range(T):
            h, _ = self.step(x[:, t], h)
            outputs.append(h.unsqueeze(1))                  # (B, 1, M_H, d_model)

        H = torch.cat(outputs, dim=1)                       # (B, T, M_H, d_model)
        return H, h
