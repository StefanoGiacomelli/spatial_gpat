"""
tPIT Loss and SELD Metrics — Separate SED/DOA Heads
====================================================
Implements four components for end-to-end SELD training:

  tPITLoss                      — Track-wise Permutation Invariant Training loss.
                                  Separate BCE (SED) + MSE (DOA) objectives on
                                  independent head outputs; standard N! permutation
                                  search (not surjections).
  SELDMetrics                   — LE_CD, LR_CD, ER_20°, F_20°, SELD_score accumulator
                                  aligned with the DCASE 2023 baseline.
  SemanticPresenceLoss          — Auxiliary BCE loss for the semantic head.
  SemanticClassificationMetrics — Torchmetrics-backed multilabel accuracy/F1/AUROC +
                                  confusion matrix for val and test.

Separation of SED and DOA heads
--------------------------------
Unlike Multi-ACCDOA / ADPIT (which encoded activity as the norm of a unified
activity-coupled direction vector), this implementation uses:

  * ``head_sed``  → raw activity logits :math:`\\in \\mathbb{R}^{B \\times T \\times N \\times C}`
  * ``head_doa``  → tanh-bounded Cartesian DOA :math:`\\in \\mathbb{R}^{B \\times T \\times N \\times C \\times 3}`

Activity detection at inference: :math:`\\sigma(\\text{act\\_logits}) > 0.5` (not vector norm).
Direction targets: unit Cartesian vectors when the source is active, zero vector when silent.

tPIT algorithm (per class :math:`c`, per frame :math:`t`)
----------------------------------------------------------
1. Derive activity target: :math:`a[n,c] = 1` iff :math:`\\|\\text{target}[n,c,:]\\|_2 > 0.5`.
2. Enumerate all :math:`N!` permutations :math:`\\pi` of :math:`N` tracks.
3. Per-permutation cost (for permutation selection, **unweighted** to avoid pos_weight bias):

   .. math::
       C(\\pi) = \\beta \\cdot \\text{BCE}_{\\text{nopw}}(\\text{act}[\\pi], a)
              + (1-\\beta) \\cdot \\text{MSE}(\\text{doa}[\\pi], d)

4. Select best permutation: :math:`\\pi^* = \\arg\\min_\\pi C(\\pi)` (non-differentiable).
5. Gradient loss (with pos_weight for class-imbalance correction):

   .. math::
       L_{\\text{tPIT}} = \\beta \\cdot \\text{BCE}_{\\text{pw}}(\\text{act}[\\pi^*], a)
                        + (1-\\beta) \\cdot \\text{MSE}(\\text{doa}[\\pi^*], d)

6. Returns ``(total, loss_sed, loss_doa)`` for per-component monitoring.

References
----------
S. Adavanne et al., "Sound Event Localization and Detection of Overlapping
Sources Using Convolutional Recurrent Neural Networks," IEEE JSTSP 2019.

Y. Cao et al., "An Improved Event-Independent Network for Polyphonic Sound
Event Localization and Detection," DCASE Workshop 2021.

A. Singh Boparai et al., "Efficient Pre-trained Audio Neural Networks via
Passive Filter Pruning," DCASE Workshop 2024.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

import itertools
import math
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelAUROC,
    MultilabelConfusionMatrix,
    MultilabelF1Score,
)

# ── Constants ─────────────────────────────────────────────────────────────────
ACTIVE_THRESH = 0.5   # sigmoid(act_logits) threshold used to detect activity at inference
TARGET_THRESH = 0.5   # ‖target‖₂ threshold used to mark active targets (norm=1 when active)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _angular_distance_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Great-circle distance in degrees between unit (or non-unit) vectors.

    Args:
        a, b : (..., 3)  — Cartesian vectors (need not be unit length)

    Returns:
        angle : (...)  degrees in [0, 180]
    """
    a_n = F.normalize(a, dim=-1, eps=1e-8)
    b_n = F.normalize(b, dim=-1, eps=1e-8)
    cos = (a_n * b_n).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.acos(cos) * (180.0 / math.pi)


# ══════════════════════════════════════════════════════════════════════════════
# tPITLoss
# ══════════════════════════════════════════════════════════════════════════════

class tPITLoss(nn.Module):
    """
    Trackwise Permutation Invariant Training loss — EIN-SELD style.

    Replaces ADPITLoss. Key differences:
      - Separate SED head (BCE) + DOA head (MSE) — activity and direction
        are decoupled, eliminating the gradient collapse on sparse data.
      - Standard N! permutation search: for M=1 sources, only ONE track is
        pushed toward the source; the others are pushed toward silence.
        (ADPITLoss surjection forced ALL tracks toward the same source.)
      - Fully vectorised over all P=N! permutations simultaneously.
      - Optional per-class ``pos_weight`` corrects for class-imbalance in
        the BCE component (rare classes receive higher gradient).
      - Configurable DOA loss mode via ``doa_loss_cfg`` dict (see below).

    The gradient flows through the *loss values* of the selected permutation;
    the argmin selection itself is non-differentiable (straight-through, as in
    EIN-SELD and standard PIT literature).

    Args:
        N             : number of tracks.
        beta          : weight for SED (BCE) vs DOA.  Default 0.5.
        pos_weight    : (C,) FloatTensor of per-class BCE weights, or None.
        focal_gamma   : Focal-loss focusing exponent γ ≥ 0.  Default 0.0.
        doa_loss_cfg  : optional dict controlling DOA loss mode. Keys:
            mode              — one of 'legacy_mse_all' (default),
                                'active_mse', 'active_cosine_norm',
                                'active_cosine_norm_smooth'
            active_eps        — threshold for norm(target) to mark a slot active
                                (default TARGET_THRESH = 0.5)
            mse_weight        — scalar weight for active MSE (default 1.0)
            direction_weight  — scalar weight for cosine loss (default 1.0)
            norm_weight       — scalar weight for norm regularisation (default 0.0)
            inactive_weight   — scalar weight for inactive norm penalty (default 0.0)
            smooth_weight     — scalar weight for temporal smoothness (default 0.0)

    Forward:
        act_logits : (B, T, N, C)     — raw SED logits from ``head_sed``
        doa_raw    : (B, T, N, C, 3)  — tanh-bounded DOA vectors from ``head_doa``
        target     : (B, T, N, C, 3)  — Multi-ACCDOA ground-truth targets

    Returns:
        (total, loss_sed, loss_doa) — scalar tensors (backward-compatible)

    Side-channels (set after each forward):
        last_components  : dict with per-component loss values and DOA diagnostics
        last_best_perm   : (BTC, N) long tensor — best permutation indices,
                           always set (used by training loop for DOA oracle alignment)
    """

    _COSINE_MODES = frozenset({'active_cosine_norm', 'active_cosine_norm_smooth'})

    def __init__(self, N: int, beta: float = 0.5,
                 pos_weight: Optional[torch.Tensor] = None,
                 focal_gamma: float = 0.0,
                 doa_loss_cfg: Optional[dict] = None):
        """Precompute all N! permutation indices as a registered buffer."""
        super().__init__()
        self.N           = N
        self.beta        = beta
        self.focal_gamma = focal_gamma

        perms = list(itertools.permutations(range(N)))
        self.register_buffer('perm_tensor', torch.tensor(perms, dtype=torch.long))
        self.register_buffer(
            'pos_weight', pos_weight.float() if pos_weight is not None else None
        )

        # ── DOA loss configuration ────────────────────────────────────────
        cfg = doa_loss_cfg or {}
        self.doa_mode             = cfg.get('mode', 'legacy_mse_all')
        # Use TARGET_THRESH as default so active detection is consistent across the codebase
        self.doa_active_eps       = float(cfg.get('active_eps', TARGET_THRESH))
        self.doa_mse_weight       = float(cfg.get('mse_weight',       1.0))
        self.doa_direction_weight = float(cfg.get('direction_weight',  1.0))
        self.doa_norm_weight      = float(cfg.get('norm_weight',       0.0))
        self.doa_inactive_weight  = float(cfg.get('inactive_weight',   0.0))
        self.doa_smooth_weight    = float(cfg.get('smooth_weight',     0.0))

        # ── Side-channels (populated on every forward pass) ───────────────
        self.last_components: dict            = {}
        self.last_best_perm:  Optional[torch.Tensor] = None

    def forward(
        self,
        act_logits: torch.Tensor,
        doa_raw:    torch.Tensor,
        target:     torch.Tensor,
    ) -> torch.Tensor:
        """Compute tPIT loss.

        Args:
            act_logits : (B, T, N, C)     — raw SED logits from head_sed
            doa_raw    : (B, T, N, C, 3)  — tanh-bounded DOA from head_doa
            target     : (B, T, N, C, 3)  — Multi-ACCDOA ground-truth

        Returns:
            (total, loss_sed, loss_doa) — all scalar tensors
        """
        B, T, N, C, _ = target.shape
        P   = self.perm_tensor.shape[0]   # N!
        BTC = B * T * C

        # ── Derive SED and DOA targets ────────────────────────────────────
        act_t = (target.norm(dim=-1) > TARGET_THRESH).float()  # (B, T, N, C)
        doa_t = target                                          # (B, T, N, C, 3)

        # ── Flatten (B, T, C) → BTC; track N stays as dim 1 ─────────────
        act_l = act_logits.permute(0, 1, 3, 2).reshape(BTC, N)      # (BTC, N)
        doa_r = doa_raw.permute(0, 1, 3, 2, 4).reshape(BTC, N, 3)   # (BTC, N, 3)
        act_b = act_t.permute(0, 1, 3, 2).reshape(BTC, N)           # (BTC, N)
        doa_b = doa_t.permute(0, 1, 3, 2, 4).reshape(BTC, N, 3)     # (BTC, N, 3)

        # ── Apply all P permutations to predictions ───────────────────────
        act_l_all = act_l[:, self.perm_tensor]     # (BTC, P, N)
        doa_r_all = doa_r[:, self.perm_tensor]     # (BTC, P, N, 3)

        # ── Per-class BCE pos_weight ──────────────────────────────────────
        if self.pos_weight is not None:
            cls_idx = torch.arange(BTC, device=act_l.device) % C
            pw = self.pos_weight[cls_idx].view(BTC, 1, 1)
        else:
            pw = None

        # ── BCE loss (with optional focal weighting) ──────────────────────
        act_b_exp = act_b.unsqueeze(1).expand_as(act_l_all)
        bce_elem  = F.binary_cross_entropy_with_logits(
            act_l_all, act_b_exp, pos_weight=pw, reduction='none',
        )  # (BTC, P, N)
        if self.focal_gamma > 0.0:
            probs_all = torch.sigmoid(act_l_all).detach()
            p_t       = torch.where(act_b_exp.bool(), probs_all, 1.0 - probs_all)
            bce_all   = ((1.0 - p_t).pow(self.focal_gamma) * bce_elem).mean(dim=-1)
        else:
            bce_all = bce_elem.mean(dim=-1)  # (BTC, P)

        # ── Active target mask (always; based on TARGET, not pred) ────────
        active_mask = (doa_b.norm(dim=-1) > self.doa_active_eps)    # (BTC, N) bool
        n_active    = active_mask.float().sum(dim=-1).clamp(min=1)  # (BTC,)
        active_exp  = active_mask.unsqueeze(1).expand(-1, P, -1)    # (BTC, P, N)

        # ── Per-slot squared-distance (shared; sum over XYZ) ─────────────
        doa_diff     = doa_r_all - doa_b.unsqueeze(1).expand_as(doa_r_all)
        mse_per_slot = doa_diff.pow(2).sum(dim=-1)                  # (BTC, P, N)

        # ── DOA PIT cost — mode-dependent ────────────────────────────────
        if self.doa_mode == 'legacy_mse_all':
            mse_all      = mse_per_slot.mean(dim=-1)                # (BTC, P)
            pit_doa_cost = mse_all

        elif self.doa_mode == 'active_mse':
            mse_active_all = (
                (mse_per_slot * active_exp.float()).sum(dim=-1)
                / n_active.unsqueeze(1)
            )  # (BTC, P) — zero when no active slots (numerator=0, denom clamped to 1)
            pit_doa_cost = mse_active_all

        else:
            # active_cosine_norm or active_cosine_norm_smooth
            pred_u_all   = F.normalize(doa_r_all, dim=-1, eps=1e-6)
            tgt_u_all    = F.normalize(
                doa_b.unsqueeze(1).expand_as(doa_r_all), dim=-1, eps=1e-6
            )
            cos_raw      = (pred_u_all * tgt_u_all).sum(dim=-1)    # (BTC, P, N)
            dir_per_slot = (1.0 - cos_raw) * active_exp.float()
            dir_active   = dir_per_slot.sum(dim=-1) / n_active.unsqueeze(1)
            norm_per_slot = (doa_r_all.norm(dim=-1) - 1.0).pow(2) * active_exp.float()
            norm_active   = norm_per_slot.sum(dim=-1) / n_active.unsqueeze(1)
            pit_doa_cost  = (
                self.doa_direction_weight * dir_active
                + self.doa_norm_weight    * norm_active
            )

        # ── Permutation selection (unweighted BCE, non-differentiable) ───
        bce_nopw = F.binary_cross_entropy_with_logits(
            act_l_all,
            act_b.unsqueeze(1).expand_as(act_l_all),
            reduction='none',
        ).mean(dim=-1)  # (BTC, P)

        pit_cost  = self.beta * bce_nopw + (1.0 - self.beta) * pit_doa_cost
        best_idx  = pit_cost.argmin(dim=-1)                    # (BTC,)
        best_mask = F.one_hot(best_idx, P).float().detach()    # (BTC, P)

        # ── Save best permutation — used by training loop for DOA oracle ──
        perm_best            = self.perm_tensor[best_idx]      # (BTC, N)
        self.last_best_perm  = perm_best.detach()

        # ── pred_best: predictions aligned to target tracks ───────────────
        # torch.gather is differentiable; gradient flows through pred values,
        # not through the (non-differentiable) index selection.
        pred_best = torch.gather(
            doa_r,
            1,
            perm_best.unsqueeze(-1).expand(-1, -1, 3),
        )  # (BTC, N, 3)

        # ── Gradient DOA loss — mode-dependent ───────────────────────────
        loss_components: dict = {
            'loss_doa_mse_active': 0.0,
            'loss_doa_direction':  0.0,
            'loss_doa_norm':       0.0,
            'loss_doa_inactive':   0.0,
            'loss_doa_smooth':     0.0,
        }

        if self.doa_mode == 'legacy_mse_all':
            loss_doa = (mse_all * best_mask).sum(dim=-1).mean()

        elif self.doa_mode == 'active_mse':
            loss_doa_active = (mse_active_all * best_mask).sum(dim=-1).mean()
            loss_doa        = self.doa_mse_weight * loss_doa_active
            loss_components['loss_doa_mse_active'] = loss_doa_active.item()

        else:
            loss_doa_dir  = (dir_active  * best_mask).sum(dim=-1).mean()
            loss_doa_norm = (norm_active * best_mask).sum(dim=-1).mean()
            loss_doa = (
                self.doa_direction_weight * loss_doa_dir
                + self.doa_norm_weight    * loss_doa_norm
            )
            loss_components['loss_doa_direction'] = loss_doa_dir.item()
            loss_components['loss_doa_norm']      = loss_doa_norm.item()

        # ── Inactive norm penalty (non-legacy, when weight > 0) ──────────
        if self.doa_mode != 'legacy_mse_all' and self.doa_inactive_weight > 0.0:
            inactive_mask = ~active_mask                                    # (BTC, N)
            n_inactive    = inactive_mask.float().sum(dim=-1).clamp(min=1) # (BTC,)
            inactive_exp  = inactive_mask.unsqueeze(1).expand(-1, P, -1)   # (BTC, P, N)
            inact_agg     = (
                (doa_r_all.norm(dim=-1).pow(2) * inactive_exp.float()).sum(dim=-1)
                / n_inactive.unsqueeze(1)
            )  # (BTC, P)
            loss_inactive = (inact_agg * best_mask).sum(dim=-1).mean()
            loss_doa      = loss_doa + self.doa_inactive_weight * loss_inactive
            loss_components['loss_doa_inactive'] = loss_inactive.item()

        # ── Temporal smoothness (only mode=active_cosine_norm_smooth, weight>0) ──
        if (self.doa_mode == 'active_cosine_norm_smooth'
                and self.doa_smooth_weight > 0.0 and T > 1):
            # Unflatten pred_best back to (B, T, N, C, 3) for temporal comparison
            pred_best_5d = pred_best.reshape(B, T, C, N, 3).permute(0, 1, 3, 2, 4)
            active_5d    = (doa_t.norm(dim=-1) > self.doa_active_eps)      # (B, T, N, C)
            active_pair  = active_5d[:, 1:] & active_5d[:, :-1]           # (B, T-1, N, C)
            pred_u_5d    = F.normalize(pred_best_5d, dim=-1, p=2, eps=1e-6)
            cos_sm       = (pred_u_5d[:, 1:] * pred_u_5d[:, :-1]).sum(dim=-1)
            n_pairs      = active_pair.float().sum().clamp(min=1)
            loss_smooth  = ((1.0 - cos_sm) * active_pair.float()).sum() / n_pairs
            loss_doa     = loss_doa + self.doa_smooth_weight * loss_smooth
            loss_components['loss_doa_smooth'] = loss_smooth.item()

        # ── SED gradient loss ─────────────────────────────────────────────
        loss_sed = (bce_all * best_mask).sum(dim=-1).mean()
        total    = self.beta * loss_sed + (1.0 - self.beta) * loss_doa

        # ── Diagnostics: norms of pred_best on active slots ───────────────
        # pred_best is aligned to target tracks — norm diagnostics are meaningful.
        with torch.no_grad():
            active_flat = active_mask.reshape(-1)  # (BTC*N,) bool
            if active_flat.any():
                pred_norms = pred_best.detach().reshape(-1, 3).norm(dim=-1)[active_flat]
                med_norm   = pred_norms.median().item()
                lt_01      = (pred_norms < 0.1).float().mean().item()
            else:
                med_norm, lt_01 = float('nan'), float('nan')

        loss_components['pred_norm_active_median']  = med_norm
        loss_components['zero_norm_active_lt_0.1'] = lt_01
        self.last_components = loss_components

        return total, loss_sed, loss_doa


# ══════════════════════════════════════════════════════════════════════════════
# SELDMetrics
# ══════════════════════════════════════════════════════════════════════════════

class SELDMetrics:
    """
    Accumulate SELD evaluation metrics over validation batches.

    Fully aligned with the DCASE 2023 baseline:
      - Hungarian optimal matching (scipy.optimize.linear_sum_assignment)
      - LE_CD and LR_CD computed per-class then macro-averaged (Class-Dependent)
      - ER_20° uses the DCASE formula: max(FP, FN) / (TP + FN)
      - F_20° computed globally: TP / (TP + 0.5*(FP+FN))

    Metrics:
        LE_CD   — macro-averaged per-class mean angular localization error (°)
        LR_CD   — macro-averaged per-class localization recall
        ER_20°  — global location-dependent error rate  (20° threshold)
        F_20°   — global location-dependent F-score     (20° threshold)
        SELD    — combined score = (ER + (1−F) + LE/180 + (1−LR)) / 4

    Usage::

        metrics = SELDMetrics(angle_thresh=20.0)
        for batch in val_loader:
            act_logits, doa_tanh, _ = model(x)
            metrics.update(act_logits, doa_tanh, batch_target)
        results = metrics.compute()   # dict with float values
        metrics.reset()

    Args:
        angle_thresh (float): angular tolerance in degrees for TP matching.
        act_thresh   (float): sigmoid(act_logits) threshold to declare a track active.
    """

    def __init__(self, angle_thresh: float = 20.0, act_thresh: float = ACTIVE_THRESH):
        """Initialize SELDMetrics accumulators."""
        self.angle_thresh = angle_thresh
        self.act_thresh   = act_thresh
        self.reset()

    def reset(self):
        """Reset all per-class accumulators."""
        self._tp        = defaultdict(int)    # class → detection TP count
        self._fp        = defaultdict(int)    # class → FP count
        self._fn        = defaultdict(int)    # class → FN count
        self._tp_loc    = defaultdict(int)    # class → localisation TP count
        self._angle_sum = defaultdict(float)  # class → sum of angular errors for TP pairs
        self._n_classes = 0

    # ------------------------------------------------------------------
    # Core matching logic
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(
        self,
        act_logits: torch.Tensor,
        doa_tanh:   torch.Tensor,
        target:     torch.Tensor,
    ):
        """
        Args:
            act_logits : (B, T, N, C)     — raw SED logits from head_sed
            doa_tanh   : (B, T, N, C, 3)  — tanh-bounded DOA from head_doa
            target     : (B, T, N, C, 3)  — Multi-ACCDOA ground-truth
        """
        B, T, N, C, _ = target.shape
        self._n_classes = C

        # Permute to (B, T, C, N, *) for class-first processing
        p = doa_tanh.permute(0, 1, 3, 2, 4)                          # (B, T, C, N, 3)
        t = target.permute(0, 1, 3, 2, 4)                            # (B, T, C, N, 3)

        # Activity: sigmoid(logit) > threshold  (not norm of combined vector)
        p_active = (torch.sigmoid(act_logits.permute(0, 1, 3, 2))
                    > self.act_thresh)                                # (B, T, C, N) bool
        t_active = t.norm(dim=-1) > TARGET_THRESH                    # (B, T, C, N) bool

        # Flatten B, T, C → one batch dim for per-frame processing
        p_flat  = p.reshape(B * T * C, N, 3)
        t_flat  = t.reshape(B * T * C, N, 3)
        pa_flat = p_active.reshape(B * T * C, N)
        ta_flat = t_active.reshape(B * T * C, N)

        # Process only frames where at least one active pred or target exists
        any_active = pa_flat.any(dim=-1) | ta_flat.any(dim=-1)
        idxs = any_active.nonzero(as_tuple=False).squeeze(-1)

        for i in idxs.tolist():
            # flat index i in B*T*C maps to class c via i % C
            # (permute gives (B,T,C,N,3), reshape is row-major → C varies fastest)
            c_idx  = i % C
            p_vecs = p_flat[i][pa_flat[i]]   # (P, 3) active pred vectors
            t_vecs = t_flat[i][ta_flat[i]]   # (Q, 3) active target vectors
            P_cnt  = p_vecs.shape[0]
            Q_cnt  = t_vecs.shape[0]

            if Q_cnt == 0:
                self._fp[c_idx] += P_cnt
                continue
            if P_cnt == 0:
                self._fn[c_idx] += Q_cnt
                continue

            # Compute all pairwise angles
            angles = _angular_distance_deg(
                p_vecs.unsqueeze(1).expand(P_cnt, Q_cnt, 3),
                t_vecs.unsqueeze(0).expand(P_cnt, Q_cnt, 3),
            )  # (P, Q)

            # Hungarian optimal matching (minimises total angular cost)
            cost        = angles.cpu().numpy()                        # (P, Q) degrees
            cost_capped = np.where(cost <= self.angle_thresh, cost, 1e9)
            row_ind, col_ind = linear_sum_assignment(cost_capped)

            tp_pairs = [
                (pi, ti) for pi, ti in zip(row_ind.tolist(), col_ind.tolist())
                if cost[pi, ti] <= self.angle_thresh
            ]

            self._tp[c_idx]       += len(tp_pairs)
            self._tp_loc[c_idx]   += len(tp_pairs)
            self._angle_sum[c_idx] += sum(cost[pi, ti] for pi, ti in tp_pairs)

            matched_p_set = {pi for pi, _ in tp_pairs}
            matched_t_set = {ti for _, ti in tp_pairs}
            self._fp[c_idx] += P_cnt - len(matched_p_set)
            self._fn[c_idx] += Q_cnt - len(matched_t_set)

    # ------------------------------------------------------------------
    # Compute final metrics
    # ------------------------------------------------------------------

    def compute(self) -> Dict[str, float]:
        C = self._n_classes

        # LE_CD: per-class mean angular error, averaged only over classes with ≥1 TP
        # (matches DCASE baseline: classes with 0 TP are excluded, not counted as 0)
        le_vals = [self._angle_sum[c] / self._tp_loc[c]
                   for c in range(C) if self._tp_loc[c] > 0]
        le_cd = sum(le_vals) / len(le_vals) if le_vals else 0.0

        # LR_CD: per-class localisation recall, averaged only over classes with ≥1 ref event
        # (matches DCASE baseline: classes with 0 reference events are excluded)
        lr_vals = [self._tp[c] / (self._tp[c] + self._fn[c])
                   for c in range(C) if (self._tp[c] + self._fn[c]) > 0]
        lr_cd = sum(lr_vals) / len(lr_vals) if lr_vals else 0.0

        # ER_20°: global (sum per-class counts), DCASE formula
        tp_g  = sum(self._tp.values())
        fp_g  = sum(self._fp.values())
        fn_g  = sum(self._fn.values())
        er_20 = max(fp_g, fn_g) / (tp_g + fn_g) if (tp_g + fn_g) > 0 else 1.0

        # F_20°: global
        f_20  = tp_g / (tp_g + 0.5 * (fp_g + fn_g)) if (tp_g + fp_g + fn_g) > 0 else 0.0

        seld = (er_20 + (1.0 - f_20) + le_cd / 180.0 + (1.0 - lr_cd)) / 4.0

        return {
            'LE_CD':      le_cd,
            'LR_CD':      lr_cd,
            'ER_20':      er_20,
            'F_20':       f_20,
            'SELD_score': seld,
        }

    def compute_perclass(self) -> list:
        """Return per-class detection F1, precision, and recall.

        Returns
        -------
        list of dict: one entry per class index, with keys
            ``class_idx``, ``tp``, ``fp``, ``fn``, ``precision``,
            ``recall``, ``f1``.
        """
        C = self._n_classes
        rows = []
        for c in range(C):
            tp = self._tp[c]
            fp = self._fp[c]
            fn = self._fn[c]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1        = (2 * precision * recall / (precision + recall)
                         if (precision + recall) > 0 else 0.0)
            rows.append({
                'class_idx': c,
                'tp':        tp,
                'fp':        fp,
                'fn':        fn,
                'precision': precision,
                'recall':    recall,
                'f1':        f1,
            })
        return rows


# ══════════════════════════════════════════════════════════════════════════════
# SemanticPresenceLoss
# ══════════════════════════════════════════════════════════════════════════════

class SemanticPresenceLoss(nn.Module):
    """
    Auxiliary BCE loss for the semantic head.

    The semantic branch processes mono audio (x_mono = channel average), which
    destroys all inter-channel phase differences.  It therefore cannot predict
    DOA reliably.  Instead of Multi-ACCDOA, the semantic head outputs raw class-
    presence logits (B, T_sem, C) trained with binary cross-entropy.

    This is aligned with E-PANNs' pre-training objective (AudioSet multi-label
    tagging with BCE) and provides a clean class-identity signal to the spatial
    branch via the CrossStitch bridges, without asking the semantic branch to
    predict spatial information it does not have.

    Args:
        pos_weight : (C,) FloatTensor of per-class BCE weights, or None.
                     Typically ``neg_frames[c] / pos_frames[c]`` computed from
                     training data.  Registered as a buffer so it is moved to
                     the correct device automatically when the module is moved.

    Forward:
        logits     : (B, T_sem, C)       — raw output of head_semantic
        tgt_spatial: (B, T_mel, N, C, 3) — Multi-ACCDOA target at mel resolution

    Returns:
        scalar BCE loss (mean over B × T_sem × C)
    """

    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        """Initialize SemanticPresenceLoss."""
        super().__init__()
        # Registered buffer → moves to correct device with .to() / Lightning
        self.register_buffer(
            'pos_weight', pos_weight.float() if pos_weight is not None else None
        )

    def forward(
        self,
        logits: torch.Tensor,
        tgt_spatial: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits      : (B, T_sem, C)        — raw class-presence logits
            tgt_spatial : (B, T_mel, N, C, 3)  — full-resolution Multi-ACCDOA target

        Returns:
            scalar BCE loss
        """
        T_sem = logits.shape[1]

        # Downsample target to semantic temporal resolution (nearest-neighbour)
        T_mel = tgt_spatial.shape[1]
        if T_mel != T_sem:
            indices  = torch.linspace(0, T_mel - 1, T_sem,
                                      dtype=torch.long, device=tgt_spatial.device)
            tgt_sem  = tgt_spatial[:, indices]          # (B, T_sem, N, C, 3)
        else:
            tgt_sem  = tgt_spatial

        # Derive binary class-presence mask:
        #   presence[b, t, c] = 1  iff at least one track of class c is active at t
        norms    = tgt_sem.norm(dim=-1)                 # (B, T_sem, N, C)
        presence = (norms > TARGET_THRESH).any(dim=2).float()   # (B, T_sem, C)

        return F.binary_cross_entropy_with_logits(
            logits, presence, pos_weight=self.pos_weight
        )


# ══════════════════════════════════════════════════════════════════════════════
# SemanticClassificationMetrics
# ══════════════════════════════════════════════════════════════════════════════

class SemanticClassificationMetrics(nn.Module):
    """
    Torchmetrics-backed accumulator for the semantic head (multi-label binary).

    Derives class-presence targets from ``tgt_spatial`` using the same
    nearest-neighbour downsample + ``TARGET_THRESH`` logic as
    ``SemanticPresenceLoss``, so metric targets exactly match training targets.

    Being an ``nn.Module`` ensures Lightning moves the internal torchmetrics
    objects to the correct device automatically.

    Args:
        n_classes : number of sound event classes (13 for STARSS23, 12 for TAU)
        threshold : sigmoid threshold for binary predictions (default 0.5)

    Metrics returned by ``compute()``:
        sem_accuracy  — multilabel subset accuracy
        sem_f1        — macro-averaged multilabel F1
        sem_auroc     — macro-averaged multilabel AUROC

    Usage::

        sem = SemanticClassificationMetrics(n_classes=13)
        # in validation_step / test_step:
        sem.update(out_e, tgt_spatial)
        # at epoch end:
        results = sem.compute()          # val — no confusion matrix
        results = sem.compute(include_confusion_matrix=True)  # test only
        sem.reset()
    """

    def __init__(self, n_classes: int, threshold: float = 0.5):
        """Initialise all torchmetrics multilabel metric objects."""
        super().__init__()
        kw = dict(num_labels=n_classes, threshold=threshold)
        self.accuracy = MultilabelAccuracy(**kw)
        self.f1       = MultilabelF1Score(average='macro', **kw)
        self.auroc    = MultilabelAUROC(num_labels=n_classes, average='macro')
        self.conf_mat = MultilabelConfusionMatrix(num_labels=n_classes)

    def reset(self) -> None:
        """Reset all internal accumulators."""
        for m in (self.accuracy, self.f1, self.auroc, self.conf_mat):
            m.reset()

    @torch.no_grad()
    def update(
        self,
        logits:      torch.Tensor,
        tgt_spatial: torch.Tensor,
    ) -> None:
        """
        Accumulate one batch.

        Args:
            logits      : (B, T_sem, C)        — raw semantic head output
            tgt_spatial : (B, T_mel, N, C, 3)  — Multi-ACCDOA spatial target
        """
        T_sem = logits.shape[1]
        T_mel = tgt_spatial.shape[1]
        if T_mel != T_sem:
            idx     = torch.linspace(0, T_mel - 1, T_sem,
                                     dtype=torch.long, device=tgt_spatial.device)
            tgt_sem = tgt_spatial[:, idx]           # (B, T_sem, N, C, 3)
        else:
            tgt_sem = tgt_spatial

        norms    = tgt_sem.norm(dim=-1)                        # (B, T_sem, N, C)
        presence = (norms > TARGET_THRESH).any(dim=2).long()   # (B, T_sem, C)
        probs    = torch.sigmoid(logits)                       # (B, T_sem, C)

        B, T, C = probs.shape
        probs_flat    = probs.reshape(B * T, C)
        presence_flat = presence.reshape(B * T, C)

        self.accuracy.update(probs_flat, presence_flat)
        self.f1.update(probs_flat, presence_flat)
        self.auroc.update(probs_flat, presence_flat)
        self.conf_mat.update(probs_flat, presence_flat)

    def compute(
        self, include_confusion_matrix: bool = False
    ) -> Dict[str, object]:
        """
        Compute and return accumulated metrics.

        Args:
            include_confusion_matrix : if True, adds ``'sem_confusion_matrix'``
                (shape C × 2 × 2, serialisable as nested list) to the dict.
                Set to True only at test time — not logged to TensorBoard.

        Returns:
            dict with keys: ``sem_accuracy``, ``sem_f1``, ``sem_auroc`` (all floats),
            and optionally ``sem_confusion_matrix`` (list[list[list[int]]]).
        """
        results: Dict[str, object] = {
            'sem_accuracy': self.accuracy.compute().item(),
            'sem_f1':       self.f1.compute().item(),
            'sem_auroc':    self.auroc.compute().item(),
        }
        if include_confusion_matrix:
            # (C, 2, 2): [[TN, FP], [FN, TP]] per class
            results['sem_confusion_matrix'] = self.conf_mat.compute().tolist()
        return results
