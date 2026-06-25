"""
utils/doa_metrics.py
====================
DOA oracle evaluation utilities for SELD training loops.

Functions extracted and adapted from analysis/diagnostics/doa_oracle_test.py so that
training scripts can compute DOA oracle metrics without duplicating logic.

Key design principle
--------------------
All functions that take ``pred_doa`` assume it is already **aligned to target
tracks** (i.e. reordered according to the PIT best permutation).  It is the
caller's responsibility to apply the permutation before calling these functions.

Usage in training loop
----------------------
1. Call ``tPITLoss.forward(...)`` — this sets ``tpit_loss.last_best_perm``.
2. Use ``align_doa_to_target(doa_raw, last_best_perm, B, T, N, C)`` to get
   the permutation-aligned DOA predictions.
3. Call ``accumulate_active_doa(pred_aligned, target_doa)`` in each
   ``validation_step`` / ``test_step`` to collect active-slot arrays on CPU.
4. At epoch end, concatenate and call ``compute_doa_oracle_metrics`` /
   ``compute_doa_oracle_per_class``.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Mirror the constant from loss.py so callers need not import it separately.
TARGET_THRESH: float = 0.5


# ══════════════════════════════════════════════════════════════════════════════
# Low-level helpers
# ══════════════════════════════════════════════════════════════════════════════

def angular_error_deg(
    pred:   np.ndarray,
    target: np.ndarray,
    eps:    float = 1e-8,
) -> np.ndarray:
    """Element-wise angular error in degrees between Cartesian vectors.

    Args:
        pred   : (..., 3) — predicted DOA vectors (need not be unit-length)
        target : (..., 3) — ground-truth DOA vectors (need not be unit-length)
        eps    : small value to avoid division by zero

    Returns:
        angles : (...) — angular error in [0, 180] degrees
    """
    pred_norm   = np.linalg.norm(pred,   axis=-1, keepdims=True)
    target_norm = np.linalg.norm(target, axis=-1, keepdims=True)
    pred_u   = pred   / (pred_norm   + eps)
    target_u = target / (target_norm + eps)
    cos_sim  = np.clip((pred_u * target_u).sum(axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(cos_sim))


def _pct(arr: np.ndarray, q: float) -> float:
    return float(np.percentile(arr, q)) if len(arr) > 0 else float('nan')


# ══════════════════════════════════════════════════════════════════════════════
# PIT alignment helper
# ══════════════════════════════════════════════════════════════════════════════

def align_doa_to_target(
    doa_raw:       torch.Tensor,
    last_best_perm: torch.Tensor,
    B: int, T: int, N: int, C: int,
) -> torch.Tensor:
    """Reorder DOA predictions to align with target track ordering.

    Applies the best permutation found by tPITLoss to ``doa_raw`` so that
    ``pred_aligned[:, :, n, c, :]`` corresponds to ``target[:, :, n, c, :]``.

    Args:
        doa_raw        : (B, T, N, C, 3) — raw DOA output from model
        last_best_perm : (BTC, N) long — ``tpit_loss.last_best_perm``
        B, T, N, C     : shape components of ``doa_raw``

    Returns:
        pred_aligned : (B, T, N, C, 3) — permutation-aligned DOA
    """
    BTC = B * T * C
    doa_r = doa_raw.permute(0, 1, 3, 2, 4).reshape(BTC, N, 3)  # (BTC, N, 3)
    pred_best = torch.gather(
        doa_r,
        1,
        last_best_perm.unsqueeze(-1).expand(-1, -1, 3),
    )  # (BTC, N, 3)
    return pred_best.reshape(B, T, C, N, 3).permute(0, 1, 3, 2, 4)  # (B, T, N, C, 3)


# ══════════════════════════════════════════════════════════════════════════════
# Accumulation helper (called in validation_step / test_step)
# ══════════════════════════════════════════════════════════════════════════════

def accumulate_active_doa(
    pred_doa:   torch.Tensor,
    target_doa: torch.Tensor,
    active_eps: float = TARGET_THRESH,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract active-slot DOA vectors from one batch.

    Filters slots where ``norm(target_doa) > active_eps`` and returns the
    corresponding predicted and target Cartesian vectors as numpy arrays.

    Args:
        pred_doa   : (B, T, N, C, 3) — PIT-aligned predicted DOA
        target_doa : (B, T, N, C, 3) — ground-truth Multi-ACCDOA
        active_eps : norm threshold for target-active detection

    Returns:
        pred_active   : (K, 3)  float32 — predicted vectors on active slots
        target_active : (K, 3)  float32 — target vectors on active slots
        class_ids     : (K,)    int32   — class index for each active slot
    """
    with torch.no_grad():
        B, T, N, C, _ = target_doa.shape
        tgt_norm  = target_doa.norm(dim=-1)                 # (B, T, N, C)
        active    = (tgt_norm > active_eps)                 # (B, T, N, C) bool

        pred_np   = pred_doa.cpu().float().numpy()          # (B, T, N, C, 3)
        tgt_np    = target_doa.cpu().float().numpy()        # (B, T, N, C, 3)
        active_np = active.cpu().numpy()                    # (B, T, N, C)

        # Class index tensor: shape (B, T, N, C) where value = c
        cls_idx = (
            np.arange(C, dtype=np.int32)[None, None, None, :]
            * np.ones((B, T, N, 1), dtype=np.int32)
        )  # (B, T, N, C)

        pred_active   = pred_np[active_np]    # (K, 3)
        target_active = tgt_np[active_np]     # (K, 3)
        class_ids     = cls_idx[active_np]    # (K,)

    return (
        pred_active.astype(np.float32),
        target_active.astype(np.float32),
        class_ids.astype(np.int32),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Global DOA oracle metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_doa_oracle_metrics(
    pred_doa:   np.ndarray,
    target_doa: np.ndarray,
    thresholds: Tuple[int, ...] = (10, 20, 30, 45, 90),
) -> Dict[str, float]:
    """Compute global DOA oracle metrics over already-filtered active slots.

    Args:
        pred_doa   : (K, 3) — predicted DOA on active slots (PIT-aligned)
        target_doa : (K, 3) — ground-truth DOA on active slots
        thresholds : angle thresholds (°) for accuracy computation

    Returns:
        dict with keys:
          evaluated_active_slots,
          angular_error_mean, angular_error_median, angular_error_p90, angular_error_p95,
          acc_{t}deg  for t in thresholds,
          pred_norm_active_mean, pred_norm_active_median, pred_norm_active_p90,
          zero_norm_active_lt_0.05, zero_norm_active_lt_0.10,
          zero_norm_active_lt_0.20, zero_norm_active_lt_0.50
    """
    K = len(pred_doa)
    if K == 0:
        result: Dict[str, float] = {'evaluated_active_slots': 0.0}
        result['angular_error_mean']   = float('nan')
        result['angular_error_median'] = float('nan')
        result['angular_error_p90']    = float('nan')
        result['angular_error_p95']    = float('nan')
        for t in thresholds:
            result[f'acc_{t}deg'] = float('nan')
        result['pred_norm_active_mean']   = float('nan')
        result['pred_norm_active_median'] = float('nan')
        result['pred_norm_active_p90']    = float('nan')
        for thr in (0.05, 0.10, 0.20, 0.50):
            result[f'zero_norm_active_lt_{thr:.2f}'] = float('nan')
        return result

    ae = angular_error_deg(pred_doa, target_doa)          # (K,)
    pred_norms = np.linalg.norm(pred_doa, axis=-1)        # (K,)

    result = {'evaluated_active_slots': float(K)}
    result['angular_error_mean']   = float(np.mean(ae))
    result['angular_error_median'] = float(np.median(ae))
    result['angular_error_p90']    = _pct(ae, 90)
    result['angular_error_p95']    = _pct(ae, 95)
    for t in thresholds:
        result[f'acc_{t}deg'] = float((ae <= t).mean())

    result['pred_norm_active_mean']   = float(np.mean(pred_norms))
    result['pred_norm_active_median'] = float(np.median(pred_norms))
    result['pred_norm_active_p90']    = _pct(pred_norms, 90)
    for thr in (0.05, 0.10, 0.20, 0.50):
        result[f'zero_norm_active_lt_{thr:.2f}'] = float((pred_norms < thr).mean())

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Per-class DOA oracle metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_doa_oracle_per_class(
    pred_doa:    np.ndarray,
    target_doa:  np.ndarray,
    class_ids:   np.ndarray,
    n_classes:   int,
    class_names: Optional[List[str]] = None,
) -> List[Dict]:
    """Compute per-class DOA oracle metrics.

    Args:
        pred_doa    : (K, 3)  — predicted DOA on active slots
        target_doa  : (K, 3)  — ground-truth DOA on active slots
        class_ids   : (K,)    — class index per active slot
        n_classes   : total number of classes
        class_names : optional list of class name strings (length n_classes)

    Returns:
        list of dicts, one per class, with keys:
          class_id, class_name, active_slots,
          median_AE, acc_20deg,
          pred_norm_active_median, zero_norm_active_lt_0.10
    """
    rows = []
    for c in range(n_classes):
        mask = class_ids == c
        name = class_names[c] if class_names and c < len(class_names) else str(c)
        if not mask.any():
            rows.append({
                'class_id':                  c,
                'class_name':                name,
                'active_slots':              0,
                'median_AE':                 float('nan'),
                'acc_20deg':                 float('nan'),
                'pred_norm_active_median':   float('nan'),
                'zero_norm_active_lt_0.10':  float('nan'),
            })
            continue

        p  = pred_doa[mask]
        t  = target_doa[mask]
        ae = angular_error_deg(p, t)
        norms = np.linalg.norm(p, axis=-1)
        rows.append({
            'class_id':                  c,
            'class_name':                name,
            'active_slots':              int(mask.sum()),
            'median_AE':                 float(np.median(ae)),
            'acc_20deg':                 float((ae <= 20.0).mean()),
            'pred_norm_active_median':   float(np.median(norms)),
            'zero_norm_active_lt_0.10':  float((norms < 0.10).mean()),
        })
    return rows
