"""
Spatial SELD Loss Functions
============================

Two loss implementations, selectable via ``training.loss_type`` in the YAML:

``SpatialPITLoss``   (``loss_type: pit``, **default**)
    Exhaustive Permutation-Invariant Training over the N_tracks! permutations
    (= 2 for N_tracks=2).  Jointly minimises activity BCE + DOA MSE (active
    frames only).  No external checkpoint required — trains from scratch.

``SpatialHNetLoss``  (``loss_type: hnet``)
    Uses a pre-trained ``HNetGRU`` (see ``spatial_modules/hnet_modules.py``)
    to produce soft track-to-source assignment matrices (dMOTP + dMOTA losses).
    Requires an external ``HNetGRU`` checkpoint; the HNet is **frozen** during
    spatial model training.

    The ``HNetGRU`` checkpoint is NOT included in this repository — it must be
    trained separately using ``spatial_modules/hnet_modules.py``.  The default
    loss is ``pit``; switch to ``hnet`` only when a checkpoint is available.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-8

# ══════════════════════════════════════════════════════════════════════════════
# Permutation-Invariant Training Loss
# ══════════════════════════════════════════════════════════════════════════════

class SpatialPITLoss(nn.Module):
    """
    Exhaustive PIT over N_tracks=2 source tracks.

    Loss per permutation:

        L = λ_act · BCE(act_logits, act_target)
          + λ_doa · MSE(doa_pred, doa_target) · activity_mask / active_count

    The minimum-loss permutation is selected; both terms share the same assignment.

    Args:
        lambda_act : weight for activity BCE term (default 1.0)
        lambda_doa : weight for DOA MSE term      (default 1.0)
    """

    _PERMS = [(0, 1), (1, 0)]   # all permutations for N_tracks=2

    def __init__(
        self,
        lambda_act: float = 1.0,
        lambda_doa: float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_act = lambda_act
        self.lambda_doa = lambda_doa

    def forward(
        self,
        pred:    dict,
        targets: dict,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            pred    : ``{'doa': (B,T,N,3), 'activity': (B,T,N)}``
            targets : same keys, same shapes

        Returns:
            (total_loss, parts_dict)
            parts_dict keys: loss, loss_act, loss_doa
        """
        pred_doa = pred['doa']
        pred_act = pred['activity']
        tgt_doa  = targets['doa']
        tgt_act  = targets['activity']

        best_total, best_parts = None, None

        for pi, pj in self._PERMS:
            pd = torch.stack([pred_doa[:, :, pi], pred_doa[:, :, pj]], dim=2)
            pa = torch.stack([pred_act[:, :, pi], pred_act[:, :, pj]], dim=2)

            act_loss = F.binary_cross_entropy_with_logits(pa, tgt_act)

            mask     = tgt_act.unsqueeze(-1)                        # (B, T, N, 1)
            n_active = mask.sum().clamp(min=1)
            doa_loss = ((pd - tgt_doa).pow(2) * mask).sum() / (n_active * 3)

            total = (
                self.lambda_act * act_loss
                + self.lambda_doa * doa_loss
            )

            if best_total is None or total < best_total:
                best_total = total
                best_parts = {
                    'loss':     total,
                    'loss_act': act_loss.detach(),
                    'loss_doa': doa_loss.detach(),
                }

        return best_total, best_parts


# ══════════════════════════════════════════════════════════════════════════════
# HNetGRU-based Loss (dMOTP + dMOTA)
# ══════════════════════════════════════════════════════════════════════════════

class SpatialHNetLoss(nn.Module):
    """
    Spatial SELD loss using a pre-trained HNetGRU for soft source assignment.

    DOA + activity: dMOTP and dMOTA losses via frozen ``HNetGRU``.
    The ``HNetGRU`` is loaded from ``hnet_ckpt`` and **frozen** during training.

    Args:
        hnet_ckpt   : path to a ``HNetGRU(max_len=N_tracks)`` state-dict file.
        lambda_act  : weight for activity BCE term.
        lambda_doa  : weight for dMOTP term (named consistently with PIT loss).
        dmota_wt    : weight for dMOTA term relative to dMOTP.
        dmotp_wt    : scalar multiplier for dMOTP loss.
        ids_wt      : IDS penalty weight inside dMOTA.
        n_tracks    : number of source tracks (default 2; must match checkpoint).

    Raises:
        FileNotFoundError : if ``hnet_ckpt`` does not exist.
        RuntimeError      : if the checkpoint is incompatible with N_tracks.
    """

    def __init__(
        self,
        hnet_ckpt:  str,
        lambda_act: float = 1.0,
        lambda_doa: float = 1.0,
        dmota_wt:   float = 1.0,
        dmotp_wt:   float = 1.0,
        ids_wt:     float = 1.0,
        n_tracks:   int   = 2,
    ) -> None:
        super().__init__()

        ckpt_path = Path(hnet_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f'HNetGRU checkpoint not found: {ckpt_path}\n'
                'Train an HNetGRU first and set training.hnet_ckpt in the config.'
            )

        from spatial_modules.hnet_modules import HNetGRU   # our own port
        self.hnet = HNetGRU(max_len=n_tracks)
        self.hnet.load_state_dict(
            torch.load(str(ckpt_path), map_location='cpu')
        )
        for p in self.hnet.parameters():
            p.requires_grad = False

        self.lambda_act = lambda_act
        self.lambda_doa = lambda_doa
        self.dmota_wt   = dmota_wt
        self.dmotp_wt   = dmotp_wt
        self.ids_wt     = ids_wt
        self.n_tracks   = n_tracks

    def forward(
        self,
        pred:    dict,
        targets: dict,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            pred    : ``{'doa': (B,T,N,3), 'activity': (B,T,N)}``
            targets : same keys, same shapes

        Returns:
            (total_loss, parts_dict)
            parts_dict keys: loss, loss_dmotp, loss_dmota, loss_act
        """
        pred_doa = pred['doa']        # (B, T, N, 3)
        pred_act = pred['activity']   # (B, T, N)
        tgt_doa  = targets['doa']     # (B, T, N, 3)
        tgt_act  = targets['activity']

        B, T, N, _ = pred_doa.shape
        BT = B * T

        # L2-normalise predicted DOA unit vectors
        norm      = pred_doa.norm(dim=-1, keepdim=True).clamp(min=_EPS)
        pred_unit = pred_doa / norm   # (B, T, N, 3)

        # Reshape for HNet input: (BT, N, N) L2 distance matrix
        pred_flat = pred_unit.reshape(BT, N, 3).contiguous()
        tgt_flat  = tgt_doa.reshape(BT, N, 3).contiguous()
        dist_mat  = torch.cdist(pred_flat, tgt_flat)   # (BT, N, N)

        # HNet → soft assignment matrix
        with torch.no_grad():
            da_out, _, _ = self.hnet(dist_mat)

        da_mat      = da_out.sigmoid().view(dist_mat.shape)  # (BT, N, N)
        da_mat      = (da_mat > 0.5) * da_mat                # threshold + weight
        da_activity = da_mat.max(-1)[0]                      # (BT, N)

        # ── dMOTP: weighted angular distance ──────────────────────────────
        dmotp = (
            (dist_mat * da_mat).sum(-1).sum(-1)
            * da_mat.sum(-1).sum(-1)
            * self.dmotp_wt
        ).sum() / (da_mat.sum() + _EPS)

        # ── dMOTA: FP + FN + IDS  ─────────────────────────────────────────
        M_frame = da_activity.sum(-1)                           # (BT,)
        N_frame = tgt_act.reshape(BT, N).sum(-1).float()        # (BT,)
        FP = (M_frame - N_frame).clamp(min=0)
        FN = (N_frame - M_frame).clamp(min=0)

        # IDS computed within each sample's time axis to avoid batch-seam artefacts
        da_seq = da_mat.view(B, T, N, N)
        ids_delta = (da_seq[:, 1:] * (1 - da_seq[:, :-1])).sum(-1).sum(-1)   # (B, T-1)
        ids_full  = torch.cat(
            [torch.zeros(B, 1, device=da_mat.device), ids_delta], dim=1
        ).reshape(BT)

        dmota = (FP + FN + self.ids_wt * ids_full).sum() / (M_frame + _EPS).sum()

        # ── Activity BCE (HNet-derived activity as soft target) ────────────
        act_loss = F.binary_cross_entropy_with_logits(
            pred_act.reshape(BT, N),
            (da_activity > 0.5).float(),
        )

        total = (
            self.lambda_doa * dmotp
            + self.dmota_wt * dmota
            + self.lambda_act * act_loss
        )

        parts = {
            'loss':       total,
            'loss_dmotp': dmotp.detach(),
            'loss_dmota': dmota.detach(),
            'loss_act':   act_loss.detach(),
        }
        return total, parts
