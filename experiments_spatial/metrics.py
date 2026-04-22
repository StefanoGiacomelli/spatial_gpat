"""
SpatialMetrics — Epoch-level accumulator for spatial SELD evaluation
=====================================================================
Ported and extended from ``neural_srp/metrics.py`` (``MultiSourceMetrics``).

Changes vs. original
---------------------
* Input format is our class-agnostic tensors — no packed ACCDOA decoding.
* Added ``dist_mae_m``: distance MAE in metres on Hungarian-matched active frames.
* Lightning-compatible: stateless accumulator; caller resets between epochs.

Metrics computed
----------------
``val_le_deg``   : Mean localization error on matched pairs (degrees).  ↓ better.
``val_lf``       : Localization F-score (%) = 2·LP·LR/(LP+LR).          ↑ better.
``val_lr``       : Localization Recall  (%).                            ↑ better.
``val_lp``       : Localization Precision (%).                          ↑ better.
``val_mota``     : Multi-Object Tracking Accuracy (%).                  ↑ better.
``val_dist_mae`` : Distance MAE in metres (matched active frames).      ↓ better.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

_EPS = 1e-7


class SpatialMetrics:
    """
    Epoch-level accumulator for spatial SELD metrics.

    Call ``update()`` once per validation batch, then ``get_results()``
    at epoch end to obtain the metric dict.  Call ``reset()`` before
    each new epoch (or call ``get_results(reset=True)``).

    All inputs are expected **on any device** — they are detached and
    moved to CPU internally before numpy conversion.
    """

    def __init__(self) -> None:
        self._reset()

    # ── Public interface ─────────────────────────────────────────────────────

    def update(
        self,
        pred_doa:        torch.Tensor,
        pred_act_logits: torch.Tensor,
        pred_dist:       torch.Tensor,
        target_doa:      torch.Tensor,
        target_act:      torch.Tensor,
        target_dist:     torch.Tensor,
    ) -> None:
        """
        Accumulate one batch of predictions and targets.

        Args:
            pred_doa        : ``(B, T, N, 3)`` Cartesian DOA predictions (raw, un-normalised).
            pred_act_logits : ``(B, T, N)``    activity logits (sigmoid → > 0.5 for binary).
            pred_dist       : ``(B, T, N)``    distance predictions in cm.
            target_doa      : ``(B, T, N, 3)`` Cartesian DOA ground truth (0-vector ⟹ silent).
            target_act      : ``(B, T, N)``    binary activity targets {0, 1}.
            target_dist     : ``(B, T, N)``    distance targets in cm (0 ⟹ silent).
        """
        B, T, N, _ = pred_doa.shape
        BT = B * T

        # ── CPU/numpy conversion ────────────────────────────────────────────
        pred_doa_t  = pred_doa.detach().cpu().float()
        tgt_doa_t   = target_doa.detach().cpu().float()
        pred_act_b  = (torch.sigmoid(pred_act_logits.detach().cpu()) > 0.5).numpy().reshape(BT, N)
        tgt_act_b   = (target_act.detach().cpu() > 0.5).numpy().reshape(BT, N)
        dist_pred_n = pred_dist.detach().cpu().numpy().reshape(BT, N)
        dist_tgt_n  = target_dist.detach().cpu().numpy().reshape(BT, N)

        # L2-normalise predicted DOA unit vectors
        norm       = pred_doa_t.norm(dim=-1, keepdim=True).clamp(min=_EPS)
        pred_unit  = (pred_doa_t / norm).numpy().reshape(BT, N, 3)
        tgt_unit   = tgt_doa_t.numpy().reshape(BT, N, 3)

        # ── Frame-level counts ──────────────────────────────────────────────
        N_arr = tgt_act_b.sum(-1).astype(float)   # (BT,) ground-truth active count
        M_arr = pred_act_b.sum(-1).astype(float)  # (BT,) predicted active count

        self._total_target  += N_arr.sum()
        self._total_outputs += M_arr.sum()
        self._fp            += np.maximum(M_arr - N_arr, 0).sum()
        self._fn            += np.maximum(N_arr - M_arr, 0).sum()

        # IDS computed within each sample's time axis (not across batch seams)
        pred_act_seq = pred_act_b.reshape(B, T, N)
        self._ids += (
            pred_act_seq[:, 1:] * (1 - pred_act_seq[:, :-1])
        ).sum()

        # ── Per-frame Hungarian matching ─────────────────────────────────────
        for t in range(BT):
            n_tgt  = int(N_arr[t])
            n_pred = int(M_arr[t])
            if n_tgt == 0 or n_pred == 0:
                continue

            pred_active = pred_unit[t][pred_act_b[t] == 1]  # (M, 3)
            tgt_active  = tgt_unit[t][tgt_act_b[t]  == 1]  # (N_t, 3)

            # Angular distance matrix (M, N_t) in radians
            dot      = np.clip(pred_active @ tgt_active.T, -1 + _EPS, 1 - _EPS)
            dist_ang = np.arccos(dot)

            row_ind, col_ind = linear_sum_assignment(dist_ang)
            n_tp = len(row_ind)
            self._tp        += n_tp
            self._loc_error += dist_ang[row_ind, col_ind].sum()

            # Distance MAE on matched active pairs — cm → m
            pd = dist_pred_n[t][pred_act_b[t] == 1]   # (M,)
            td = dist_tgt_n[t][tgt_act_b[t]  == 1]   # (N_t,)
            d_err = np.abs(pd[row_ind] - td[col_ind]) / 100.0
            self._dist_error_m += d_err.sum()
            self._dist_tp      += n_tp

    def get_results(self, reset: bool = False) -> dict:
        """
        Compute and return epoch-level metrics.

        Args:
            reset : if True, reset accumulators after computing results.

        Returns:
            dict with keys: val_le_deg, val_lf, val_lr, val_lp, val_mota, val_dist_mae.
        """
        le_rad = self._loc_error / (self._tp + _EPS)
        le_deg = 180.0 * le_rad / np.pi

        lr = self._tp / (self._total_target  + _EPS)
        lp = self._tp / (self._total_outputs + _EPS)
        lf = 2.0 * lp * lr / (lp + lr + _EPS)

        mota     = 1.0 - (self._fp + self._fn + self._ids) / (self._total_target + _EPS)
        dist_mae = self._dist_error_m / (self._dist_tp + _EPS)

        out = {
            'val_le_deg':   float(le_deg),
            'val_lf':       float(100.0 * lf),
            'val_lr':       float(100.0 * lr),
            'val_lp':       float(100.0 * lp),
            'val_mota':     float(100.0 * mota),
            'val_dist_mae': float(dist_mae),
        }

        if reset:
            self._reset()
        return out

    def reset(self) -> None:
        self._reset()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _reset(self) -> None:
        self._loc_error    = 0.0
        self._tp           = 0.0
        self._total_target = _EPS
        self._total_outputs= _EPS
        self._fp           = 0.0
        self._fn           = 0.0
        self._ids          = 0.0
        self._dist_error_m = 0.0
        self._dist_tp      = 0.0
