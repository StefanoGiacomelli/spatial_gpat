"""Unit tests of ``experiments_nas_seld.loss.SELDMetrics`` (toy cases, known answers).

Covers the definitions documented in the class docstring: gated Hungarian matching,
the ``<=`` gate, error rate above 1, class macro-averaging rules and the optional
``gated=False`` (class-dependent, ungated localisation) mode.
"""
from __future__ import annotations

import math

import pytest
import torch

from experiments_nas_seld.loss import SELDMetrics

N, C = 3, 3


def vec(az_deg: float, el_deg: float = 0.0) -> torch.Tensor:
    a, e = math.radians(az_deg), math.radians(el_deg)
    return torch.tensor([math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e)])


def frame():
    """One frame: logits (very negative = inactive), DOA, target."""
    return (torch.full((1, 1, N, C), -10.0), torch.zeros(1, 1, N, C, 3),
            torch.zeros(1, 1, N, C, 3))


def set_pred(f, slot, cls, az, logit=10.0):
    f[0][0, 0, slot, cls] = logit
    f[1][0, 0, slot, cls] = vec(az)


def set_ref(f, slot, cls, az):
    f[2][0, 0, slot, cls] = vec(az)


def run(f, **kw):
    m = SELDMetrics(angle_thresh=20.0, act_thresh=0.5, **kw)
    m.update(*f)
    return m, m.compute()


def test_single_true_positive():
    f = frame()
    set_ref(f, 0, 0, 0.0)
    set_pred(f, 1, 0, 10.0)                     # slot index differs on purpose (PIT-free metric)
    m, r = run(f)
    assert (m._tp[0], m._fp[0], m._fn[0]) == (1, 0, 0)
    assert r["ER_20"] == pytest.approx(0.0)
    assert r["F_20"] == pytest.approx(1.0)
    assert r["LE_CD"] == pytest.approx(10.0, abs=1e-3)
    assert r["LR_CD"] == pytest.approx(1.0)
    assert r["SELD_score"] == pytest.approx((0 + 0 + 10 / 180 + 0) / 4, abs=1e-5)


def test_gate_is_inclusive_and_pair_beyond_gate_is_fp_plus_fn():
    f = frame()
    set_ref(f, 0, 0, 0.0)
    set_pred(f, 0, 0, 19.9)
    m, r = run(f)
    assert (m._tp[0], m._fp[0], m._fn[0]) == (1, 0, 0)
    f = frame()
    set_ref(f, 0, 0, 0.0)
    set_pred(f, 0, 0, 20.1)
    m, r = run(f)
    assert (m._tp[0], m._fp[0], m._fn[0]) == (0, 1, 1)
    assert r["ER_20"] == pytest.approx(1.0) and r["F_20"] == pytest.approx(0.0)
    assert r["LE_CD"] == 0.0 and r["LR_CD"] == pytest.approx(0.0)     # classes without TP excluded from LE
    assert r["SELD_score"] == pytest.approx((1 + 1 + 0 + 1) / 4)


def test_error_rate_can_exceed_one():
    f = frame()
    set_ref(f, 0, 0, 0.0)
    for slot, az in enumerate((5.0, 120.0, -120.0)):
        set_pred(f, slot, 0, az)
    m, r = run(f)
    assert (m._tp[0], m._fp[0], m._fn[0]) == (1, 2, 0)
    assert r["ER_20"] == pytest.approx(2.0)          # max(FP, FN) / (TP + FN) = 2 / 1
    assert r["SELD_score"] > 0.5


def test_pooled_counts_and_macro_rules():
    # class 0: TP at 10 deg; class 1: reference never detected; class 2 absent from the reference
    f = frame()
    set_ref(f, 0, 0, 0.0)
    set_pred(f, 0, 0, 10.0)
    set_ref(f, 0, 1, 90.0)
    set_pred(f, 1, 2, 45.0)                         # false positive in class 2
    m, r = run(f)
    assert [(m._tp[c], m._fp[c], m._fn[c]) for c in range(3)] == [(1, 0, 0), (0, 0, 1), (0, 1, 0)]
    assert r["ER_20"] == pytest.approx(max(1, 1) / (1 + 1))
    assert r["F_20"] == pytest.approx(1 / (1 + 0.5 * 2))
    assert r["LE_CD"] == pytest.approx(10.0, abs=1e-3)           # only classes with a TP
    assert r["LR_CD"] == pytest.approx((1.0 + 0.0) / 2)          # classes with references
    rows = m.compute_perclass()
    assert rows[0]["f1"] == pytest.approx(1.0) and rows[1]["recall"] == 0.0 and rows[2]["precision"] == 0.0


def test_hungarian_assignment_pairs_both_references():
    f = frame()
    set_ref(f, 0, 0, 10.0)
    set_ref(f, 1, 0, 30.0)
    set_pred(f, 0, 0, 40.0)
    set_pred(f, 1, 0, 18.0)
    m, r = run(f)
    assert (m._tp[0], m._fp[0], m._fn[0]) == (2, 0, 0)
    assert r["LE_CD"] == pytest.approx(9.0, abs=1e-3)            # (8 + 10) / 2


def test_ungated_mode_class_dependent_localisation():
    f = frame()
    set_ref(f, 0, 0, 0.0)
    set_pred(f, 0, 0, 40.0)                          # detected, 40 deg off
    set_ref(f, 0, 1, 0.0)                            # class 1: reference, no prediction
    _, g = run(f)                                    # default (gated)
    _, u = run(f, gated=False)
    # ER/F use the same gate in both modes
    assert u["ER_20"] == pytest.approx(g["ER_20"]) and u["F_20"] == pytest.approx(g["F_20"])
    assert g["LE_CD"] == 0.0 and g["LR_CD"] == 0.0
    assert u["LE_CD"] == pytest.approx((40.0 + 180.0) / 2, abs=1e-3)   # 180 deg convention
    assert u["LR_CD"] == pytest.approx((1.0 + 0.0) / 2)
    assert u["SELD_score"] == pytest.approx(
        (u["ER_20"] + 1 - u["F_20"] + u["LE_CD"] / 180 + 1 - u["LR_CD"]) / 4)


def test_default_mode_keys_unchanged():
    f = frame()
    set_ref(f, 0, 0, 0.0)
    set_pred(f, 0, 0, 0.0)
    _, r = run(f)
    assert sorted(r) == ["ER_20", "F_20", "LE_CD", "LR_CD", "SELD_score"]
    m = SELDMetrics()
    assert m.gated is True
