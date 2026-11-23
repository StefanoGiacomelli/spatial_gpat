#!/usr/bin/env python3
"""
analysis/diagnostics/seld_variants.py
==============================================================================
SELD metric variants — fast kernels, clip-level stitching, segment-based
(DCASE-style) scoring.  *Re-implementation*, inference-side only.
==============================================================================

The repository's headline protocol is ``experiments_nas_seld.loss.SELDMetrics``
(frame-level, 10-ms model frames, *gated* localisation at 20 deg, windows of
10 s with hop 2.5 s pooled without stitching).  This module adds, **next to it
and never in place of it**, the machinery needed to quantify how much the
reported numbers depend on those protocol choices:

``GatedCounts``
    Numba re-implementation of ``SELDMetrics.update/compute`` (identical
    definitions, verified count-for-count against ``SELDMetrics`` in
    ``tests/test_seld_variants.py`` and on stored checkpoints).  Used to make
    threshold sweeps / bootstraps affordable.
``stitch_windows`` / ``pool_frames``
    Clip-level evaluation: overlapping window outputs are averaged (every
    frame counted once, no zero-padded tails).  Slots are aligned either with
    a reference-free rule (``align='free'``, default) or with the tPIT
    permutation w.r.t. the reference (``align='oracle'``, what
    ``main_demo.py::_infer_full_clip`` does; an upper bound).
``SegmentCounts``
    Segment-based scoring on a coarser time grid (100-ms frames, 1-s segments)
    with the structure of the DCASE-2022/2023 SELD scores (class-dependent
    LE/LR, macro-F with spatially wrong matches, per-segment S/D/I error
    rate).  **This is a re-implementation from the task description, not the
    official evaluation script** (which is not part of the repository).

Exact definitions (also recorded in ``docs/CONVENTIONS.md``)
-----------------------------------------------------------
Notation.  For frame ``f`` and class ``c``: ``R`` = reference DOA vectors of the
active slots (norm > 0.5), ``P`` = predicted DOA vectors of the slots with
``prob > tau``; the angle between two vectors is the great-circle distance of
the *normalised* vectors (``eps = 1e-8``), as in ``loss._angular_distance_deg``.

*Frame-level, gated* (``GatedCounts`` = repository protocol).  Per (frame,
class) with ``|R|+|P| > 0``: if ``|R| = 0`` all ``|P|`` count as FP; if
``|P| = 0`` all ``|R|`` count as FN; otherwise the assignment of
``min(|P|,|R|)`` pairs that first maximises the number of pairs within the
gate (<= 20 deg) and then minimises their summed angle is used (identical to
``linear_sum_assignment`` on the 1e9-capped cost); pairs within the gate are
TP (and contribute their angle to LE), unmatched predictions are FP, unmatched
references FN.  LE_CD = mean over classes with >= 1 TP of the per-class mean
angle; LR_CD = mean over classes with >= 1 reference of TP/(TP+FN);
ER = max(FP,FN)/(TP+FN) on pooled counts; F = TP/(TP+0.5(FP+FN)) pooled;
SELD = mean(ER, 1-F, LE/180, 1-LR).

*Segment-based, DCASE-style* (``SegmentCounts``).  Time is divided into segments
of ``K`` grid frames (``K = 1`` -> 100 ms, ``K = 10`` -> 1 s).  For every class and
segment: ``n_ref = max_f |R_f|`` and ``n_pred = max_f |P_f|`` (instance counts, "max"
rule); instance ``i`` of a frame is the ``i``-th active slot of that frame (ordered by
slot, i.e. by source index for the reference).

  pair_mode = 'frames' (default).  The arithmetic of the public DCASE-2022/2023 baseline
  implementation (``SELD_evaluation_metrics.py``, whose header states an MIT licence):
    in every frame in which the class is active in both reference and prediction the
    DOAs of that frame are assigned with the minimum total angle (ungated Hungarian);
    the angular distance of each matched pair is accumulated on the *reference
    instance* (its within-frame rank) and averaged over the frames in which it was
    matched.  Every reference instance with at least one matched frame is a *matched
    instance* with that average angle.
      - class-dependent localisation (ungated): every matched instance contributes its
        average angle to LE (DE_TP) and counts once in the recall;
      - matched instance within the gate (<= 20 deg): TP; above the gate: spatially
        wrong (FP_spatial, counted as an *insertion-type* error, i.e. in FP_seg only);
      - ``n_pred > n_ref``: the surplus is FP; ``n_pred < n_ref``: the shortfall is FN
        (and DE_FN); a class present only in the reference: FN, DE_FN; only in the
        prediction: FP, DE_FP;
      - a class present in both but never co-active in a frame: the public code adds
        the *predicted* count to FN / DE_FN (``official_compat=True``, default); the
        stricter ``official_compat=False`` counts all references as FN and all
        predictions as FP;
      - macro averages run over *all* classes (``official_compat=True``: F = 0, LE = 180,
        LR = 0 for a class without any event; the strict variant averages over the
        classes with references); LR_c = DE_TP / (DE_TP + DE_FN).
    ``tests`` check the kernel against a plain-Python implementation of both modes, and
    ``official_compat=True`` was verified against the public implementation itself on
    random data (K = 1, 4, 10): maximum difference 2e-8 in ER, F, LE, LR and SELD.
  pair_mode = 'mean_direction' (sensitivity): the angle of an instance pair is that of
    the normalised segment-mean directions and the ``n_ref x n_pred`` matrix is assigned
    with the minimum total angle (spatially wrong pairs counted as FP and FN).

  Common: error rate with ``FP_seg = sum_c FP_c`` (including spatially wrong
  instances) and ``FN_seg = sum_c FN_c`` per segment, ``S = min(FP_seg, FN_seg)``,
  ``D = max(0, FN_seg - FP_seg)``, ``I = max(0, FP_seg - FN_seg)``,
  ER = sum_seg (S + D + I) / sum n_ref; F_c = TP_c / (TP_c + FP_spatial_c +
  0.5 (FP_c + FN_c)), macro-averaged (or pooled with ``macro_f=False``);
  SELD = mean(ER, 1-F, LE/180, 1-LR).
``gated_loc=True`` switches LE/LR to the repository convention (only pairs within the
gate count, classes without TP are excluded from LE, gate-first assignment on the
segment-mean directions) -- which allows to move step by step from the repository
protocol to the DCASE-style one.

What remains outside this module: how model outputs become the inputs of the segment
scores (pooling of the 10-ms frames to 100-ms frames, thresholding, slot order) -- the
data preparation of ``run_metric_variants.py``; the arithmetic itself is the public one.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numba import njit

_PERM3 = np.array(
    [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]],
    dtype=np.int64,
)
_RAD2DEG = 180.0 / math.pi
TARGET_THRESH = 0.5     # ||target|| above which a slot is reference-active
DEFAULT_GATE = 20.0     # degrees


# ══════════════════════════════════════════════════════════════════════════════
# numba kernels
# ══════════════════════════════════════════════════════════════════════════════

@njit(cache=False)
def _unit_into(x, y, z, out):
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-8:
        n = 1e-8
    out[0] = x / n
    out[1] = y / n
    out[2] = z / n


@njit(cache=False)
def _angle_deg(a, b):
    c = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    if c > 1.0:
        c = 1.0
    elif c < -1.0:
        c = -1.0
    return math.acos(c) * _RAD2DEG


@njit(cache=False)
def _assign_stats(ang, P, Q, gate, gated):
    """Best assignment of min(P,Q) pairs of a (P x Q) angle matrix (P,Q <= 3).

    ``gated=True``: lexicographic (max #pairs within gate, min summed angle of
    those pairs) == ``linear_sum_assignment`` on the 1e9-capped cost.
    ``gated=False``: minimum total angle over all pairs (ungated Hungarian).

    Returns ``(n_within, sum_within, sum_all)`` of the chosen assignment.
    """
    best_nf = -1
    best_sc = 0.0
    best_all = 0.0
    best_total = 1e30
    first = P <= Q
    m = P if first else Q
    for k in range(6):
        ok = True
        nf = 0
        sc = 0.0
        tot = 0.0
        for i in range(m):
            j = _PERM3[k, i]
            if first:
                if j >= Q:
                    ok = False
                    break
                a = ang[i, j]
            else:
                if j >= P:
                    ok = False
                    break
                a = ang[j, i]
            tot += a
            if a <= gate:
                nf += 1
                sc += a
        if not ok:
            continue
        if gated:
            if nf > best_nf or (nf == best_nf and sc < best_sc):
                best_nf = nf
                best_sc = sc
                best_all = tot
        else:
            if tot < best_total:
                best_total = tot
                best_nf = nf
                best_sc = sc
                best_all = tot
    return best_nf, best_sc, best_all


@njit(cache=False)
def _assign_pairs(ang, P, Q, out_p, out_q):
    """Ungated minimum-total-angle assignment of min(P,Q) pairs (P,Q <= 3).

    Fills ``out_p[t]`` (row index into the P predicted vectors) and ``out_q[t]`` (column
    index into the Q references) for t < m and returns m = min(P, Q).
    """
    first = P <= Q
    m = P if first else Q
    best_total = 1e30
    bk = 0
    for k in range(6):
        ok = True
        tot = 0.0
        for i in range(m):
            j = _PERM3[k, i]
            if first:
                if j >= Q:
                    ok = False
                    break
                tot += ang[i, j]
            else:
                if j >= P:
                    ok = False
                    break
                tot += ang[j, i]
        if ok and tot < best_total:
            best_total = tot
            bk = k
    for i in range(m):
        j = _PERM3[bk, i]
        if first:
            out_p[i] = i
            out_q[i] = j
        else:
            out_p[i] = j
            out_q[i] = i
    return m


@njit(cache=False)
def gated_frame_counts(pa, pv, ra, rv, gate, tp, fp, fn, asum):
    """Accumulate repository-protocol counts over ``F`` frames.

    Args:
        pa (F,N,C) bool       predicted-active slots
        pv (F,N,C,3) float32  predicted DOA vectors
        ra (F,N,C) bool       reference-active slots
        rv (F,N,C,3) float32  reference vectors
        gate                  angular gate (deg)
        tp, fp, fn, asum (C,) accumulators (int64 / float64), updated in place
    """
    F, N, C = pa.shape
    pu = np.empty((3, 3))
    ru = np.empty((3, 3))
    ang = np.empty((3, 3))
    for f in range(F):
        for c in range(C):
            P = 0
            Q = 0
            for n in range(N):
                if pa[f, n, c]:
                    _unit_into(pv[f, n, c, 0], pv[f, n, c, 1], pv[f, n, c, 2], pu[P])
                    P += 1
                if ra[f, n, c]:
                    _unit_into(rv[f, n, c, 0], rv[f, n, c, 1], rv[f, n, c, 2], ru[Q])
                    Q += 1
            if P == 0 and Q == 0:
                continue
            if Q == 0:
                fp[c] += P
                continue
            if P == 0:
                fn[c] += Q
                continue
            for i in range(P):
                for j in range(Q):
                    ang[i, j] = _angle_deg(pu[i], ru[j])
            nf, sc, _ = _assign_stats(ang, P, Q, gate, True)
            tp[c] += nf
            asum[c] += sc
            fp[c] += P - nf
            fn[c] += Q - nf


@njit(cache=False)
def _align_best(acc_p, acc_v, g, c, cnt, prob_f, vec_f):
    """Index (into _PERM3) of the slot permutation of the new set that best
    matches the running mean of ``acc`` at (g, c).  Reference-free."""
    mu = np.empty(3)
    vu = np.empty(3)
    best = -1e30
    bk = 0
    for k in range(6):
        s = 0.0
        for n in range(3):
            j = _PERM3[k, n]
            mp = acc_p[g, n, c] / cnt
            pn = prob_f[j, c]
            w = mp if mp < pn else pn
            _unit_into(acc_v[g, n, c, 0], acc_v[g, n, c, 1], acc_v[g, n, c, 2], mu)
            _unit_into(vec_f[j, c, 0], vec_f[j, c, 1], vec_f[j, c, 2], vu)
            cs = mu[0] * vu[0] + mu[1] * vu[1] + mu[2] * vu[2]
            d = mp - pn
            if d < 0:
                d = -d
            s += w * cs - 0.5 * d
        if s > best:
            best = s
            bk = k
    return bk


@njit(cache=False)
def _stitch_add(acc_p, acc_v, cnt, prob, vec, dest, perm, use_perm):
    """Add ``F`` frames of slot sets to the accumulators.

    ``dest[f]`` is the destination frame of input frame ``f`` (``-1`` = skip).
    With ``use_perm`` the slots are re-ordered with ``perm[f, c, n]`` (the slot
    of the prediction assigned to target slot ``n``); otherwise each frame is
    aligned to the running mean of its destination (reference-free).
    """
    F, N, C = prob.shape
    pf = np.empty((3, C))
    vf = np.empty((3, C, 3))
    for f in range(F):
        g = dest[f]
        if g < 0:
            continue
        for c in range(C):
            if use_perm:
                for n in range(N):
                    j = perm[f, c, n]
                    pf[n, c] = prob[f, j, c]
                    for d in range(3):
                        vf[n, c, d] = vec[f, j, c, d]
            else:
                for n in range(N):
                    pf[n, c] = prob[f, n, c]
                    for d in range(3):
                        vf[n, c, d] = vec[f, n, c, d]
        if cnt[g] > 0 and not use_perm:
            for c in range(C):
                k = _align_best(acc_p, acc_v, g, c, float(cnt[g]), pf, vf)
                for n in range(N):
                    j = _PERM3[k, n]
                    acc_p[g, n, c] += pf[j, c]
                    for d in range(3):
                        acc_v[g, n, c, d] += vf[j, c, d]
        else:
            for c in range(C):
                for n in range(N):
                    acc_p[g, n, c] += pf[n, c]
                    for d in range(3):
                        acc_v[g, n, c, d] += vf[n, c, d]
        cnt[g] += 1


@njit(cache=False)
def segment_kernel(pa, pv, ra, rv, K, gate, gated_loc, pair_mode, official,
                   de_sum, de_cnt, de_fn, tp, fps, fp, fn, nref, npred, seg_stats):
    """Segment-based (DCASE-style) counts, see the module docstring.

    Arrays are on the grid of the segments: ``pa/ra`` (G,N,C) bool,
    ``pv/rv`` (G,N,C,3).  ``pair_mode``: 1 = per-frame assignment averaged per reference
    instance, 0 = segment-mean directions.  ``official`` (pair_mode 1 only): reproduce the
    bookkeeping of the public DCASE implementation exactly, including its treatment of a class
    that is active in the segment in both reference and prediction but never co-active in a
    frame (the predicted count is added to FN) and its localisation-recall denominator
    (``de_fn``).  ``seg_stats`` = [sum max(FP_seg,FN_seg), S, D, I].
    """
    G, N, C = pa.shape
    nseg = (G + K - 1) // K
    pm = np.zeros((3, 3))
    rm = np.zeros((3, 3))
    pc = np.zeros(3)
    rc = np.zeros(3)
    u = np.empty(3)
    pu = np.empty((3, 3))
    ru = np.empty((3, 3))
    ang = np.empty((3, 3))
    fpu = np.empty((3, 3))
    fru = np.empty((3, 3))
    fang = np.empty((3, 3))
    dsum = np.zeros(3)
    dcnt = np.zeros(3)
    op = np.zeros(3, np.int64)
    oq = np.zeros(3, np.int64)
    for b in range(nseg):
        f0 = b * K
        f1 = f0 + K
        if f1 > G:
            f1 = G
        loc_fp = 0
        loc_fn = 0
        for c in range(C):
            nr = 0
            npd = 0
            for i in range(3):
                pc[i] = 0.0
                rc[i] = 0.0
                for d in range(3):
                    pm[i, d] = 0.0
                    rm[i, d] = 0.0
            for f in range(f0, f1):
                P = 0
                Q = 0
                for n in range(N):
                    if pa[f, n, c]:
                        _unit_into(pv[f, n, c, 0], pv[f, n, c, 1], pv[f, n, c, 2], u)
                        for d in range(3):
                            pm[P, d] += u[d]
                        pc[P] += 1.0
                        P += 1
                    if ra[f, n, c]:
                        _unit_into(rv[f, n, c, 0], rv[f, n, c, 1], rv[f, n, c, 2], u)
                        for d in range(3):
                            rm[Q, d] += u[d]
                        rc[Q] += 1.0
                        Q += 1
                if P > npd:
                    npd = P
                if Q > nr:
                    nr = Q
            if nr == 0 and npd == 0:
                continue
            nref[c] += nr
            npred[c] += npd
            if nr == 0:
                fp[c] += npd
                loc_fp += npd
                continue
            if npd == 0:
                fn[c] += nr
                de_fn[c] += nr
                loc_fn += nr
                continue
            if pair_mode == 1 and not gated_loc:
                # per-frame ungated assignment, distance accumulated per reference instance
                for i in range(3):
                    dsum[i] = 0.0
                    dcnt[i] = 0.0
                for f in range(f0, f1):
                    P = 0
                    Q = 0
                    for n in range(N):
                        if pa[f, n, c]:
                            _unit_into(pv[f, n, c, 0], pv[f, n, c, 1], pv[f, n, c, 2], fpu[P])
                            P += 1
                        if ra[f, n, c]:
                            _unit_into(rv[f, n, c, 0], rv[f, n, c, 1], rv[f, n, c, 2], fru[Q])
                            Q += 1
                    if P == 0 or Q == 0:
                        continue
                    for i in range(P):
                        for j in range(Q):
                            fang[i, j] = _angle_deg(fpu[i], fru[j])
                    mf = _assign_pairs(fang, P, Q, op, oq)
                    for t in range(mf):
                        dsum[oq[t]] += fang[op[t], oq[t]]
                        dcnt[oq[t]] += 1.0
                n_matched = 0
                wrong = 0
                for r in range(nr):
                    if dcnt[r] > 0.0:
                        avg = dsum[r] / dcnt[r]
                        de_sum[c] += avg
                        de_cnt[c] += 1
                        n_matched += 1
                        if avg <= gate:
                            tp[c] += 1
                        else:
                            wrong += 1
                if n_matched == 0:
                    if official:
                        # public implementation: the predicted count is added to FN
                        fn[c] += npd
                        de_fn[c] += npd
                        loc_fn += npd
                    else:
                        # never co-active: every reference missed, every prediction spurious
                        fn[c] += nr
                        fp[c] += npd
                        de_fn[c] += nr
                        loc_fn += nr
                        loc_fp += npd
                    continue
                fps[c] += wrong
                loc_fp += wrong                      # spatial error: insertion-type only
                if npd > nr:
                    fp[c] += npd - nr
                    loc_fp += npd - nr
                elif npd < nr:
                    fn[c] += nr - npd
                    de_fn[c] += nr - npd
                    loc_fn += nr - npd
                continue
            # pair_mode 0 (and the gated repository components): segment-mean directions
            for i in range(npd):
                _unit_into(pm[i, 0] / pc[i], pm[i, 1] / pc[i], pm[i, 2] / pc[i], pu[i])
            for j in range(nr):
                _unit_into(rm[j, 0] / rc[j], rm[j, 1] / rc[j], rm[j, 2] / rc[j], ru[j])
            for i in range(npd):
                for j in range(nr):
                    ang[i, j] = _angle_deg(pu[i], ru[j])
            nf, sc, sall = _assign_stats(ang, npd, nr, gate, gated_loc)
            m = npd if npd < nr else nr
            wrong = m - nf
            if gated_loc:
                de_sum[c] += sc
                de_cnt[c] += nf
            else:
                de_sum[c] += sall
                de_cnt[c] += m
            tp[c] += nf
            fps[c] += wrong
            fp[c] += npd - m
            fn[c] += nr - m
            de_fn[c] += nr - m
            loc_fp += (npd - m) + wrong
            loc_fn += nr - m
            if gated_loc:
                loc_fn += wrong                      # gated pairs beyond the gate: unmatched both ways
        s_ = loc_fp if loc_fp < loc_fn else loc_fn
        dd = loc_fn - loc_fp
        if dd < 0:
            dd = 0
        ii = loc_fp - loc_fn
        if ii < 0:
            ii = 0
        seg_stats[0] += s_ + dd + ii
        seg_stats[1] += s_
        seg_stats[2] += dd
        seg_stats[3] += ii


# ══════════════════════════════════════════════════════════════════════════════
# Python glue
# ══════════════════════════════════════════════════════════════════════════════

def seld_from_counts(tp, fp, fn, asum, n_classes: Optional[int] = None) -> Dict[str, float]:
    """Repository formulas (``SELDMetrics.compute``) applied to per-class counts."""
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    fn = np.asarray(fn, dtype=np.float64)
    asum = np.asarray(asum, dtype=np.float64)
    le_vals = [asum[c] / tp[c] for c in range(len(tp)) if tp[c] > 0]
    le_cd = sum(le_vals) / len(le_vals) if le_vals else 0.0
    lr_vals = [tp[c] / (tp[c] + fn[c]) for c in range(len(tp)) if tp[c] + fn[c] > 0]
    lr_cd = sum(lr_vals) / len(lr_vals) if lr_vals else 0.0
    tp_g, fp_g, fn_g = tp.sum(), fp.sum(), fn.sum()
    er = max(fp_g, fn_g) / (tp_g + fn_g) if (tp_g + fn_g) > 0 else 1.0
    f20 = tp_g / (tp_g + 0.5 * (fp_g + fn_g)) if (tp_g + fp_g + fn_g) > 0 else 0.0
    seld = (er + (1.0 - f20) + le_cd / 180.0 + (1.0 - lr_cd)) / 4.0
    return {"LE_CD": float(le_cd), "LR_CD": float(lr_cd), "ER_20": float(er),
            "F_20": float(f20), "SELD_score": float(seld)}


class GatedCounts:
    """Accumulator with the semantics of ``SELDMetrics`` (gated, frame-level)."""

    def __init__(self, n_classes: int, gate: float = DEFAULT_GATE):
        self.C = int(n_classes)
        self.gate = float(gate)
        self.tp = np.zeros(self.C, np.int64)
        self.fp = np.zeros(self.C, np.int64)
        self.fn = np.zeros(self.C, np.int64)
        self.asum = np.zeros(self.C, np.float64)

    def update(self, pa, pv, ra, rv) -> None:
        """``pa/ra`` (F,N,C) bool, ``pv/rv`` (F,N,C,3) float32 (frames first)."""
        if pa.shape[1] != 3:
            raise ValueError("kernel supports exactly N = 3 tracks")
        gated_frame_counts(
            np.ascontiguousarray(pa), np.ascontiguousarray(pv, dtype=np.float32),
            np.ascontiguousarray(ra), np.ascontiguousarray(rv, dtype=np.float32),
            self.gate, self.tp, self.fp, self.fn, self.asum)

    def compute(self) -> Dict[str, float]:
        return seld_from_counts(self.tp, self.fp, self.fn, self.asum)

    def counts(self) -> Dict[str, np.ndarray]:
        return {"tp": self.tp.copy(), "fp": self.fp.copy(), "fn": self.fn.copy(),
                "angle_sum": self.asum.copy()}


class SegmentCounts:
    """Accumulator of segment-based (DCASE-style) counts."""

    def __init__(self, n_classes: int, seg_frames: int, gate: float = DEFAULT_GATE,
                 gated_loc: bool = False, macro_f: bool = True,
                 pair_mode: str = "frames", official_compat: Optional[bool] = None):
        if pair_mode not in ("frames", "mean_direction"):
            raise ValueError("pair_mode must be 'frames' or 'mean_direction'")
        self.C = int(n_classes)
        self.K = int(seg_frames)
        self.gate = float(gate)
        self.gated_loc = bool(gated_loc)
        self.macro_f = bool(macro_f)
        self.pair_mode = 1 if pair_mode == "frames" else 0
        # default: exact public-implementation bookkeeping for the DCASE-style mode
        self.official = bool(official_compat) if official_compat is not None else (
            pair_mode == "frames" and not gated_loc)
        if self.official and (pair_mode != "frames" or gated_loc):
            raise ValueError("official_compat requires pair_mode='frames' and gated_loc=False")
        z = lambda dt: np.zeros(self.C, dt)
        self.de_sum, self.de_cnt, self.de_fn = z(np.float64), z(np.int64), z(np.int64)
        self.tp, self.fps, self.fp, self.fn = z(np.int64), z(np.int64), z(np.int64), z(np.int64)
        self.nref, self.npred = z(np.int64), z(np.int64)
        self.seg_stats = np.zeros(4, np.float64)

    def update(self, pa, pv, ra, rv) -> None:
        """Arrays on the segment grid of ONE clip: (G,N,C[,3])."""
        if pa.shape[1] != 3:
            raise ValueError("kernel supports exactly N = 3 tracks")
        segment_kernel(
            np.ascontiguousarray(pa), np.ascontiguousarray(pv, dtype=np.float32),
            np.ascontiguousarray(ra), np.ascontiguousarray(rv, dtype=np.float32),
            self.K, self.gate, self.gated_loc, self.pair_mode, self.official,
            self.de_sum, self.de_cnt, self.de_fn, self.tp, self.fps, self.fp, self.fn,
            self.nref, self.npred, self.seg_stats)

    def compute(self) -> Dict[str, float]:
        # official bookkeeping averages over *all* classes (F = 0, LE = 180, LR = 0 for classes
        # without any event); the strict variant averages over the classes with references
        present = list(range(self.C)) if self.official else [c for c in range(self.C) if self.nref[c] > 0]
        nref_tot = float(self.nref.sum())
        er = float(self.seg_stats[0]) / nref_tot if nref_tot > 0 else 1.0
        # class-wise
        f_c, le_c, lr_c = [], [], []
        for c in present:
            denom = self.tp[c] + self.fps[c] + 0.5 * (self.fp[c] + self.fn[c])
            f_c.append(self.tp[c] / denom if denom > 0 else 0.0)
            if self.gated_loc:
                if self.de_cnt[c] > 0:
                    le_c.append(self.de_sum[c] / self.de_cnt[c])
                lr_c.append(self.tp[c] / self.nref[c])
            else:
                le_c.append(self.de_sum[c] / self.de_cnt[c] if self.de_cnt[c] > 0 else 180.0)
                if self.official:
                    den = self.de_cnt[c] + self.de_fn[c]
                    lr_c.append(self.de_cnt[c] / den if den > 0 else 0.0)
                else:
                    lr_c.append(self.de_cnt[c] / self.nref[c] if self.nref[c] > 0 else 0.0)
        if self.macro_f:
            f = float(np.mean(f_c)) if f_c else 0.0
        else:
            tp, fps, fp, fn = (float(x.sum()) for x in (self.tp, self.fps, self.fp, self.fn))
            denom = tp + fps + 0.5 * (fp + fn)
            f = tp / denom if denom > 0 else 0.0
        le = float(np.mean(le_c)) if le_c else (0.0 if self.gated_loc else 180.0)
        lr = float(np.mean(lr_c)) if lr_c else 0.0
        seld = (er + (1.0 - f) + le / 180.0 + (1.0 - lr)) / 4.0
        return {
            "LE_CD": le, "LR_CD": lr, "ER": er, "F": f, "SELD_score": float(seld),
            "S_rate": float(self.seg_stats[1]) / nref_tot if nref_tot else float("nan"),
            "D_rate": float(self.seg_stats[2]) / nref_tot if nref_tot else float("nan"),
            "I_rate": float(self.seg_stats[3]) / nref_tot if nref_tot else float("nan"),
            "n_ref_instances": int(nref_tot),
            "n_pred_instances": int(self.npred.sum()),
            "n_classes_scored": len(present),
        }


# ── stitching / pooling ───────────────────────────────────────────────────────

def stitch_windows(probs: np.ndarray, vecs: np.ndarray, offsets: Sequence[int],
                   n_frames: int, align: str = "free",
                   perm: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average overlapping window outputs into a clip-level sequence.

    Args:
        probs    (K,T,N,C) sigmoid probabilities of the K windows of one clip
        vecs     (K,T,N,C,3) DOA vectors
        offsets  global frame index of the first frame of every window
        n_frames number of frames of the clip (frames beyond are dropped)
        align    ``'free'`` (reference-free slot alignment) or ``'oracle'``
                 (use ``perm`` = tPIT permutation w.r.t. the reference)
        perm     (K,T,C,N) int8, required for ``align='oracle'``

    Returns:
        prob (n_frames,N,C), vec (n_frames,N,C,3), count (n_frames,) number of
        windows covering each frame.
    """
    K, T, N, C = probs.shape
    if N != 3:
        raise ValueError("stitching kernel supports exactly N = 3 tracks")
    acc_p = np.zeros((n_frames, N, C), np.float64)
    acc_v = np.zeros((n_frames, N, C, 3), np.float64)
    cnt = np.zeros(n_frames, np.int32)
    use_perm = align == "oracle"
    if use_perm and perm is None:
        raise ValueError("align='oracle' needs the stored tPIT permutations")
    dummy = np.zeros((1, 1, 1), np.int8)
    for k in range(K):
        dest = np.arange(T, dtype=np.int64) + int(offsets[k])
        dest[dest >= n_frames] = -1
        _stitch_add(acc_p, acc_v, cnt,
                    np.ascontiguousarray(probs[k], dtype=np.float32),
                    np.ascontiguousarray(vecs[k], dtype=np.float32), dest,
                    np.ascontiguousarray(perm[k]) if use_perm else dummy, use_perm)
    valid = np.maximum(cnt, 1)
    return ((acc_p / valid[:, None, None]).astype(np.float32),
            (acc_v / valid[:, None, None, None]).astype(np.float32), cnt)


def pool_frames(prob: np.ndarray, vec: np.ndarray, k: int
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Average ``k`` consecutive frames (slot-aligned, reference-free) -> coarser grid."""
    F, N, C = prob.shape
    G = (F + k - 1) // k
    acc_p = np.zeros((G, N, C), np.float64)
    acc_v = np.zeros((G, N, C, 3), np.float64)
    cnt = np.zeros(G, np.int32)
    dest = np.arange(F, dtype=np.int64) // k
    _stitch_add(acc_p, acc_v, cnt, np.ascontiguousarray(prob, dtype=np.float32),
                np.ascontiguousarray(vec, dtype=np.float32), dest,
                np.zeros((1, 1, 1), np.int8), False)
    valid = np.maximum(cnt, 1)
    return ((acc_p / valid[:, None, None]).astype(np.float32),
            (acc_v / valid[:, None, None, None]).astype(np.float32))


def reference_from_events(events: Sequence[Tuple[int, int, int, float, float]],
                          n_frames: int, n_tracks: int, n_classes: int
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Reference slots on the 100-ms label grid, built exactly like the dataset
    (label frame ``x = sin az cos el, y = cos az cos el, z = sin el``; active
    sources of a (frame, class) sorted by source index, truncated to N)."""
    ra = np.zeros((n_frames, n_tracks, n_classes), bool)
    rv = np.zeros((n_frames, n_tracks, n_classes, 3), np.float32)
    table: Dict[Tuple[int, int], list] = {}
    for frame, cls, src, az, el in events:
        if 0 <= frame < n_frames:
            azr, elr = math.radians(az), math.radians(el)
            table.setdefault((frame, cls), []).append(
                (src, math.sin(azr) * math.cos(elr), math.cos(azr) * math.cos(elr),
                 math.sin(elr)))
    for (frame, cls), lst in table.items():
        lst.sort(key=lambda s: s[0])
        for n, (_, x, y, z) in enumerate(lst[:n_tracks]):
            ra[frame, n, cls] = True
            rv[frame, n, cls] = (x, y, z)
    return ra, rv


# ── oracle DOA statistics (tPIT-aligned) ──────────────────────────────────────

def angular_error_array(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise angle in degrees between two (...,3) arrays."""
    an = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)
    bn = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-8)
    return np.degrees(np.arccos(np.clip((an * bn).sum(-1), -1.0, 1.0)))
