"""Data-free tests of ``analysis/diagnostics/seld_variants.py``.

* the numba kernel reproduces ``SELDMetrics`` count-for-count on random data;
* toy cases with hand-computed answers for the segment-based (DCASE-style)
  re-implementation (gate boundary, temporal tolerance, S/D/I, 180-degree rule);
* stitching: aligned/permuted windows, coverage counts, 100-ms pooling;
* (optional, skipped by default) the segment scorer against the public DCASE baseline implementation
  when ``DCASE_SELD_METRICS_DIR`` points to a local copy of ``SELD_evaluation_metrics.py``.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest
import torch

from analysis.diagnostics.seld_variants import (
    GatedCounts, SegmentCounts, pool_frames, reference_from_events,
    seld_from_counts, stitch_windows,
)
from experiments_nas_seld.loss import SELDMetrics

N, C = 3, 4


def unit(az_deg: float, el_deg: float = 0.0) -> np.ndarray:
    a, e = math.radians(az_deg), math.radians(el_deg)
    return np.array([math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e)],
                    dtype=np.float32)


def empty(F: int, C_: int = C):
    return (np.zeros((F, N, C_), bool), np.zeros((F, N, C_, 3), np.float32))


def put(arrs, f, n, c, vec):
    a, v = arrs
    a[f, n, c] = True
    v[f, n, c] = vec


# ── kernel == SELDMetrics ─────────────────────────────────────────────────────

@pytest.mark.parametrize("gate", [10.0, 20.0, 45.0])
@pytest.mark.parametrize("thr", [0.3, 0.5])
def test_gated_kernel_matches_seldmetrics(gate, thr):
    rng = np.random.default_rng(3)
    B, T = 2, 120
    ref_act = rng.random((B, T, N, C)) < 0.15
    ref = rng.normal(size=(B, T, N, C, 3)).astype(np.float32)
    ref /= np.linalg.norm(ref, axis=-1, keepdims=True)
    ref *= ref_act[..., None]
    logits = np.where(ref_act, rng.normal(1.0, 1.5, ref_act.shape),
                      rng.normal(-2.0, 1.5, ref_act.shape)).astype(np.float32)
    doa = np.where(ref_act[..., None], ref + rng.normal(0, 0.25, ref.shape),
                   rng.normal(0, 0.3, ref.shape)).astype(np.float32)
    doa = np.tanh(doa)
    m = SELDMetrics(angle_thresh=gate, act_thresh=thr)
    m.update(torch.tensor(logits), torch.tensor(doa), torch.tensor(ref))
    r_ref = m.compute()

    g = GatedCounts(C, gate)
    pa = (1.0 / (1.0 + np.exp(-logits)) > thr).reshape(B * T, N, C)
    g.update(pa, doa.reshape(-1, N, C, 3), (np.linalg.norm(ref, axis=-1) > 0.5).reshape(-1, N, C),
             ref.reshape(-1, N, C, 3))
    r = g.compute()
    for c in range(C):
        assert (g.tp[c], g.fp[c], g.fn[c]) == (m._tp[c], m._fp[c], m._fn[c])
    for k in r_ref:
        assert r[k] == pytest.approx(r_ref[k], abs=1e-4)


def test_gate_boundary_and_er_above_one():
    # reference at azimuth 0; predictions 19.99 deg / 20.01 deg away (gate: <= 20)
    ra, rv = empty(2)
    put((ra, rv), 0, 0, 0, unit(0))
    put((ra, rv), 1, 0, 0, unit(0))
    pa, pv = empty(2)
    put((pa, pv), 0, 0, 0, unit(19.99))
    put((pa, pv), 1, 0, 0, unit(20.01))
    g = GatedCounts(C, 20.0)
    g.update(pa, pv, ra, rv)
    assert (g.tp[0], g.fp[0], g.fn[0]) == (1, 1, 1)
    assert g.asum[0] == pytest.approx(19.99, abs=1e-3)
    # ER above 1: one reference, three predictions (1 TP + 2 FP) -> ER = 2
    ra, rv = empty(1)
    put((ra, rv), 0, 0, 1, unit(10))
    pa, pv = empty(1)
    for n, az in enumerate((10, 100, 200)):
        put((pa, pv), 0, n, 1, unit(az))
    g = GatedCounts(C, 20.0)
    g.update(pa, pv, ra, rv)
    res = g.compute()
    assert (g.tp[1], g.fp[1], g.fn[1]) == (1, 2, 0)
    assert res["ER_20"] == pytest.approx(2.0)
    assert res["F_20"] == pytest.approx(1.0 / (1.0 + 0.5 * 2))
    assert res["SELD_score"] > 0.5


def test_assignment_prefers_more_pairs_within_gate():
    # two references 10 and 30 deg; predictions at 18 and 40 deg.  Greedy nearest would
    # pair (18,10) and (40,30) = 2 TP (angles 8, 10); a wrong pairing would give fewer.
    ra, rv = empty(1)
    put((ra, rv), 0, 0, 0, unit(10))
    put((ra, rv), 0, 1, 0, unit(30))
    pa, pv = empty(1)
    put((pa, pv), 0, 0, 0, unit(40))
    put((pa, pv), 0, 1, 0, unit(18))
    g = GatedCounts(C, 20.0)
    g.update(pa, pv, ra, rv)
    assert (g.tp[0], g.fp[0], g.fn[0]) == (2, 0, 0)
    assert g.asum[0] == pytest.approx(8.0 + 10.0, abs=1e-3)


def test_seld_from_counts_matches_repository_formula():
    tp, fp, fn = np.array([5, 0]), np.array([2, 1]), np.array([3, 4])
    asum = np.array([50.0, 0.0])
    r = seld_from_counts(tp, fp, fn, asum)
    le, lr = 10.0, (5 / 8 + 0 / 4) / 2
    er, f = max(3, 7) / (5 + 7), 5 / (5 + 0.5 * 10)
    assert r["LE_CD"] == pytest.approx(le)
    assert r["LR_CD"] == pytest.approx(lr)
    assert r["SELD_score"] == pytest.approx((er + 1 - f + le / 180 + 1 - lr) / 4)


# ── segment-based (DCASE-style) toy cases ─────────────────────────────────────

def seg_counts(pred, ref, K, **kw):
    """Toy cases encode the *strict* definitions (macro over classes with references, a class that is
    never co-active inside a segment = miss + spurious) unless ``official_compat=True`` is passed."""
    if "official_compat" not in kw and not kw.get("gated_loc") and kw.get("pair_mode", "frames") == "frames":
        kw["official_compat"] = False
    s = SegmentCounts(C, K, **kw)
    s.update(pred[0], pred[1], ref[0], ref[1])
    return s


def fill(arrs, frames, n, c, vec):
    for f in frames:
        put(arrs, f, n, c, vec)


def test_segment_single_instance_within_gate():
    ref, pred = empty(10), empty(10)
    fill(ref, range(10), 0, 0, unit(0))
    fill(pred, range(10), 0, 0, unit(10))
    r = seg_counts(pred, ref, 10).compute()
    assert r["ER"] == pytest.approx(0.0)
    assert r["F"] == pytest.approx(1.0)
    assert r["LE_CD"] == pytest.approx(10.0, abs=1e-3)
    assert r["LR_CD"] == pytest.approx(1.0)
    assert r["SELD_score"] == pytest.approx((0 + 0 + 10 / 180 + 0) / 4, abs=1e-5)


def test_segment_spatially_wrong_pair_dcase_vs_gated():
    ref, pred = empty(10), empty(10)
    fill(ref, range(10), 0, 0, unit(0))
    fill(pred, range(10), 0, 0, unit(30))          # 30 deg > gate
    r = seg_counts(pred, ref, 10).compute()        # DCASE-style (frames mode)
    assert (r["ER"], r["F"]) == (pytest.approx(1.0), pytest.approx(0.0))
    # spatially wrong match is an insertion-type error (FP only), not a substitution
    assert r["I_rate"] == pytest.approx(1.0) and r["S_rate"] == 0 and r["D_rate"] == 0
    assert r["LE_CD"] == pytest.approx(30.0, abs=1e-3)      # ungated class-dependent LE
    assert r["LR_CD"] == pytest.approx(1.0)                  # class detected, ungated
    assert r["SELD_score"] == pytest.approx((1 + 1 + 30 / 180 + 0) / 4, abs=1e-5)
    g = seg_counts(pred, ref, 10, gated_loc=True).compute()  # repository components
    assert g["LE_CD"] == 0.0 and g["LR_CD"] == 0.0
    assert g["S_rate"] == pytest.approx(1.0)                 # unmatched both ways = substitution
    assert g["SELD_score"] == pytest.approx((1 + 1 + 0 + 1) / 4)


def test_segment_partial_overlap_and_temporal_tolerance():
    ref, pred = empty(10), empty(10)
    fill(ref, range(0, 7), 0, 0, unit(0))      # reference frames 0-6
    fill(pred, range(3, 10), 0, 0, unit(0))    # prediction frames 3-9 (4 frames overlap)
    r1 = seg_counts(pred, ref, 10).compute()   # 1-s segment: matched on the co-active frames
    assert r1["ER"] == pytest.approx(0.0) and r1["F"] == pytest.approx(1.0)
    r2 = seg_counts(pred, ref, 1).compute()    # 100-ms segments: 3 FN frames, 3 FP frames
    assert r2["ER"] == pytest.approx(6 / 7)
    assert r2["F"] == pytest.approx(4 / (4 + 0.5 * 6))


def test_segment_disjoint_activity_in_segment():
    ref, pred = empty(10), empty(10)
    fill(ref, range(0, 5), 0, 0, unit(0))      # never co-active inside the segment
    fill(pred, range(5, 10), 0, 0, unit(0))
    r = seg_counts(pred, ref, 10).compute()    # frames mode: no co-active frame -> miss + spurious
    assert r["ER"] == pytest.approx(1.0) and r["F"] == pytest.approx(0.0)
    assert r["S_rate"] == pytest.approx(1.0)
    m = seg_counts(pred, ref, 10, pair_mode="mean_direction").compute()   # sensitivity variant
    assert m["ER"] == pytest.approx(0.0) and m["F"] == pytest.approx(1.0)


def test_segment_pair_distance_modes_differ_for_moving_source():
    ref, pred = empty(10), empty(10)
    for f in range(10):
        put(ref, f, 0, 0, unit(3.0 * f))       # reference moves 0 -> 27 deg
        put(pred, f, 0, 0, unit(15.0))         # static prediction
    fr = seg_counts(pred, ref, 10).compute()
    md = seg_counts(pred, ref, 10, pair_mode="mean_direction").compute()
    assert fr["LE_CD"] == pytest.approx(np.mean([abs(15 - 3 * f) for f in range(10)]), abs=1e-2)
    assert md["LE_CD"] == pytest.approx(1.5, abs=0.05)


def test_segment_sdi_substitution_across_classes():
    ref, pred = empty(10), empty(10)
    fill(ref, range(10), 0, 0, unit(0))        # class 0: reference only
    fill(pred, range(10), 0, 1, unit(0))       # class 1: prediction only
    r = seg_counts(pred, ref, 10).compute()
    # segment: FN = 1 (class 0), FP = 1 (class 1) -> one substitution
    assert r["S_rate"] == pytest.approx(1.0)
    assert r["D_rate"] == 0 and r["I_rate"] == 0 and r["ER"] == pytest.approx(1.0)
    # deletions / insertions
    ref2, pred2 = empty(10), empty(10)
    fill(ref2, range(10), 0, 0, unit(0))
    fill(ref2, range(10), 1, 0, unit(90))      # two instances of class 0
    fill(pred2, range(10), 0, 0, unit(0))      # one detected
    r2 = seg_counts(pred2, ref2, 10).compute()
    assert r2["D_rate"] == pytest.approx(1 / 2) and r2["S_rate"] == 0 and r2["I_rate"] == 0
    assert r2["ER"] == pytest.approx(0.5)
    r3 = seg_counts(ref2, pred2, 10).compute()   # roles swapped: insertion
    assert r3["I_rate"] == pytest.approx(1.0)


def test_segment_180_degree_convention_and_macro_over_present_classes():
    ref, pred = empty(10), empty(10)
    fill(ref, range(10), 0, 0, unit(0))        # class 0 detected well
    fill(pred, range(10), 0, 0, unit(0))
    fill(ref, range(10), 0, 1, unit(0))        # class 1 present in reference, never predicted
    r = seg_counts(pred, ref, 10).compute()
    assert r["n_classes_scored"] == 2
    assert r["LE_CD"] == pytest.approx((0.0 + 180.0) / 2)
    assert r["LR_CD"] == pytest.approx((1.0 + 0.0) / 2)
    assert r["F"] == pytest.approx((1.0 + 0.0) / 2)
    # a class without reference does not enter the macro average
    pred2 = empty(10)
    fill(pred2, range(10), 0, 2, unit(0))      # class 2: only false positives
    r2 = seg_counts(pred2, ref, 10).compute()
    assert r2["n_classes_scored"] == 2


def test_segment_instances_use_max_rule_and_mean_direction():
    # two simultaneous reference instances, prediction detects both with 5-deg offset
    ref, pred = empty(10), empty(10)
    fill(ref, range(10), 0, 0, unit(0))
    fill(ref, range(10), 1, 0, unit(90))
    fill(pred, range(10), 0, 0, unit(95))      # slots deliberately swapped
    fill(pred, range(10), 1, 0, unit(5))
    r = seg_counts(pred, ref, 10).compute()
    assert r["ER"] == pytest.approx(0.0) and r["F"] == pytest.approx(1.0)
    assert r["LE_CD"] == pytest.approx(5.0, abs=1e-3)
    assert r["n_ref_instances"] == 2


def test_official_compat_averages_over_all_classes_and_counts_never_coactive_as_deletion():
    ref, pred = empty(10), empty(10)
    fill(ref, range(10), 0, 0, unit(0))
    fill(pred, range(10), 0, 0, unit(10))
    r = seg_counts(pred, ref, 10, official_compat=True).compute()     # C = 4 classes, one with events
    assert r["F"] == pytest.approx(1.0 / 4) and r["LR_CD"] == pytest.approx(1.0 / 4)
    assert r["LE_CD"] == pytest.approx((10.0 + 3 * 180.0) / 4, abs=1e-3)
    assert r["n_classes_scored"] == C
    # never co-active inside the segment: the public implementation adds the predicted count to FN
    ref, pred = empty(10), empty(10)
    fill(ref, range(0, 5), 0, 0, unit(0))
    fill(pred, range(5, 10), 0, 0, unit(0))
    fill(pred, range(5, 10), 1, 0, unit(90))                           # two predicted instances
    o = seg_counts(pred, ref, 10, official_compat=True)
    res = o.compute()
    assert o.fn[0] == 2 and o.fp[0] == 0 and o.de_fn[0] == 2           # FN += n_pred (2), not n_ref (1)
    assert res["D_rate"] == pytest.approx(2.0) and res["S_rate"] == 0 and res["I_rate"] == 0
    st = seg_counts(pred, ref, 10, official_compat=False)
    assert st.fn[0] == 1 and st.fp[0] == 2                              # strict: reference missed, predictions spurious


# ── stitching / pooling ───────────────────────────────────────────────────────

def test_stitch_identity_and_counts():
    T, K = 12, 3
    rng = np.random.default_rng(1)
    base_p = rng.random((40, N, C)).astype(np.float32)
    base_v = rng.normal(size=(40, N, C, 3)).astype(np.float32)
    offs = [0, 4, 8]
    probs = np.stack([base_p[o:o + T] for o in offs])
    vecs = np.stack([base_v[o:o + T] for o in offs])
    p, v, cnt = stitch_windows(probs, vecs, offs, 20)
    assert cnt.tolist() == [1] * 4 + [2] * 4 + [3] * 4 + [2] * 4 + [1] * 4
    np.testing.assert_allclose(p, base_p[:20], atol=1e-6)
    np.testing.assert_allclose(v, base_v[:20], atol=1e-6)


def test_stitch_reference_free_alignment_recovers_permuted_slots():
    T = 8
    p0 = np.zeros((2, T, N, C), np.float32)
    v0 = np.zeros((2, T, N, C, 3), np.float32)
    # one source per frame in class 0; slot differs between the two windows
    p0[0, :, 0, 0] = 0.9
    v0[0, :, 0, 0] = unit(40)
    p0[1, :, 2, 0] = 0.8
    v0[1, :, 2, 0] = unit(42)
    p, v, cnt = stitch_windows(p0, v0, [0, 0], T)
    assert cnt.tolist() == [2] * T
    # the two detections end up in the same slot: mean probability 0.85
    slot = int(np.argmax(p[0, :, 0]))
    assert p[0, slot, 0] == pytest.approx(0.85, abs=1e-5)
    ang = math.degrees(math.acos(float(np.clip(
        np.dot(v[0, slot, 0] / np.linalg.norm(v[0, slot, 0]), unit(41)), -1, 1))))
    assert ang < 0.5
    assert np.sort(p[0, :, 0])[-2] == pytest.approx(0.0, abs=1e-6)


def test_stitch_oracle_uses_given_permutation():
    T = 4
    probs = np.zeros((2, T, N, C), np.float32)
    vecs = np.zeros((2, T, N, C, 3), np.float32)
    probs[0, :, 1, 0] = 1.0
    vecs[0, :, 1, 0] = unit(10)
    probs[1, :, 2, 0] = 1.0
    vecs[1, :, 2, 0] = unit(10)
    perm = np.zeros((2, T, C, N), np.int8)
    for c in range(C):
        perm[:, :, c] = [0, 1, 2]
    perm[0, :, 0] = [1, 0, 2]      # window 0: target slot 0 <- prediction slot 1
    perm[1, :, 0] = [2, 0, 1]      # window 1: target slot 0 <- prediction slot 2
    p, v, _ = stitch_windows(probs, vecs, [0, 0], T, align="oracle", perm=perm)
    assert np.allclose(p[:, 0, 0], 1.0) and np.allclose(p[:, 1, 0], 0.0)


def test_pool_frames_mean_of_aligned_frames():
    F = 20
    prob = np.zeros((F, N, C), np.float32)
    vec = np.zeros((F, N, C, 3), np.float32)
    prob[:, 0, 0] = 1.0
    vec[:, 0, 0] = unit(0)
    p, v = pool_frames(prob, vec, 10)
    assert p.shape == (2, N, C)
    assert np.allclose(p[:, 0, 0], 1.0)


def test_reference_from_events_slot_order_and_frame_convention():
    events = [(0, 1, 7, 90.0, 0.0), (0, 1, 3, 0.0, 0.0), (1, 2, 5, 0.0, 90.0)]
    ra, rv = reference_from_events(events, 3, 2, 4)
    # sorted by source index: source 3 (az 0) first, then source 7 (az 90)
    np.testing.assert_allclose(rv[0, 0, 1], [0.0, 1.0, 0.0], atol=1e-6)   # x = sin az, y = cos az
    np.testing.assert_allclose(rv[0, 1, 1], [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(rv[1, 0, 2], [0.0, 0.0, 1.0], atol=1e-6)
    assert ra.sum() == 3


# ── differential test of the segment kernel against a plain-Python implementation ────────

def _unit_np(v):
    n = max(float(np.linalg.norm(v)), 1e-8)
    return np.asarray(v, dtype=np.float64) / n


def _ang(a, b):
    return math.degrees(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0))))


def reference_segment_counts(pa, pv, ra, rv, K, gate, official=False):
    """Plain-Python version of the 'frames' pair mode (scipy Hungarian per frame)."""
    from scipy.optimize import linear_sum_assignment
    G, Nn, Cc = pa.shape
    acc = dict(de_sum=np.zeros(Cc), de_cnt=np.zeros(Cc, int), de_fn=np.zeros(Cc, int), tp=np.zeros(Cc, int),
               fps=np.zeros(Cc, int), fp=np.zeros(Cc, int), fn=np.zeros(Cc, int),
               nref=np.zeros(Cc, int), seg=np.zeros(4))
    for b in range((G + K - 1) // K):
        frames = range(b * K, min(G, (b + 1) * K))
        loc_fp = loc_fn = 0
        for c in range(Cc):
            P = [[_unit_np(pv[f, n, c]) for n in range(Nn) if pa[f, n, c]] for f in frames]
            R = [[_unit_np(rv[f, n, c]) for n in range(Nn) if ra[f, n, c]] for f in frames]
            nr, npd = max(len(x) for x in R), max(len(x) for x in P)
            if nr == 0 and npd == 0:
                continue
            acc["nref"][c] += nr
            if nr == 0:
                acc["fp"][c] += npd
                loc_fp += npd
                continue
            if npd == 0:
                acc["fn"][c] += nr
                acc["de_fn"][c] += nr
                loc_fn += nr
                continue
            dsum, dcnt = np.zeros(3), np.zeros(3)
            for p_f, r_f in zip(P, R):
                if not p_f or not r_f:
                    continue
                cost = np.array([[_ang(p, r) for r in r_f] for p in p_f])
                pi, ri = linear_sum_assignment(cost)
                for i, j in zip(pi, ri):
                    dsum[j] += cost[i, j]
                    dcnt[j] += 1
            matched = [r for r in range(nr) if dcnt[r] > 0]
            if not matched:
                if official:
                    acc["fn"][c] += npd
                    acc["de_fn"][c] += npd
                    loc_fn += npd
                else:
                    acc["fn"][c] += nr
                    acc["fp"][c] += npd
                    acc["de_fn"][c] += nr
                    loc_fn += nr
                    loc_fp += npd
                continue
            wrong = 0
            for r in matched:
                avg = dsum[r] / dcnt[r]
                acc["de_sum"][c] += avg
                acc["de_cnt"][c] += 1
                if avg <= gate:
                    acc["tp"][c] += 1
                else:
                    wrong += 1
            acc["fps"][c] += wrong
            loc_fp += wrong
            if npd > nr:
                acc["fp"][c] += npd - nr
                loc_fp += npd - nr
            elif npd < nr:
                acc["fn"][c] += nr - npd
                acc["de_fn"][c] += nr - npd
                loc_fn += nr - npd
        s, d, i = min(loc_fp, loc_fn), max(0, loc_fn - loc_fp), max(0, loc_fp - loc_fn)
        acc["seg"] += [s + d + i, s, d, i]
    return acc


@pytest.mark.parametrize("official", [False, True])
@pytest.mark.parametrize("K", [1, 4, 10])
def test_segment_kernel_matches_plain_python_reference(K, official):
    rng = np.random.default_rng(11 + K)
    G = 40
    ref_act = rng.random((G, N, C)) < 0.18
    pred_act = np.where(ref_act & (rng.random(ref_act.shape) < 0.7), True, rng.random(ref_act.shape) < 0.05)
    # compact slots like the models do not: random slot positions on purpose
    rv_ = rng.normal(size=(G, N, C, 3)).astype(np.float32)
    rv_ /= np.linalg.norm(rv_, axis=-1, keepdims=True)
    pv_ = (rv_ + rng.normal(0, 0.35, rv_.shape)).astype(np.float32)
    s = SegmentCounts(C, K, official_compat=official)
    s.update(pred_act, pv_, ref_act, rv_)
    ref = reference_segment_counts(pred_act, pv_, ref_act, rv_, K, 20.0, official=official)
    np.testing.assert_allclose(s.de_sum, ref["de_sum"], atol=1e-3)
    for name in ("de_cnt", "de_fn", "tp", "fps", "fp", "fn", "nref"):
        np.testing.assert_array_equal(getattr(s, name), ref[name], err_msg=name)
    np.testing.assert_allclose(s.seg_stats, ref["seg"])


@pytest.mark.skipif(
    not os.environ.get("DCASE_SELD_METRICS_DIR"),
    reason="set DCASE_SELD_METRICS_DIR to a directory holding the public SELD_evaluation_metrics.py "
           "(not redistributed; see analysis/diagnostics/check_against_public_dcase.py)")
def test_segment_scorer_matches_public_dcase_implementation():
    from analysis.diagnostics.check_against_public_dcase import run_check
    worst, cases = run_check(os.environ["DCASE_SELD_METRICS_DIR"])
    assert cases == 18
    assert worst < 1e-5, f"max difference to the public implementation: {worst:.3e}"
