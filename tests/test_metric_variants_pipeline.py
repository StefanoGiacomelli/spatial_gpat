"""End-to-end test of the window cache -> metric-variant pipeline on a synthetic dataset.

A synthetic 12.3-s clip (4-channel wav written to a temporary directory) with moving sources is
windowed exactly like the real datasets (10 s, hop 2.5 s).  A *perfect predictor* (logits and DOA
copied from the targets) must obtain ER 0, F 1, LE 0, LR 1 under **every** variant: this checks the
window offsets, the frame counts of the stitching, the 100-ms grid and the segment bookkeeping
together.  A degraded predictor checks that the scores move in the expected direction.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from analysis.diagnostics.run_metric_variants import VARIANTS, clip_n_mel, evaluate_variants
from analysis.diagnostics.window_outputs import WindowCache, window_start_frame, window_target
from datasets.utils import SELDDataset

SR, DUR_S = 24000, 12.3
N_TR, N_CLS = 3, 4


class SynthDataset(SELDDataset):
    EVENTS = [
        # (frame_100ms, class, source, az, el): a moving class-0 source and a static class-2 source,
        # plus two simultaneous class-1 sources
        *[(f, 0, 0, 20.0 + 1.5 * f, 10.0) for f in range(5, 100)],
        *[(f, 2, 0, -70.0, -20.0) for f in range(30, 120)],
        *[(f, 1, 0, 120.0, 0.0) for f in range(50, 90)],
        *[(f, 1, 1, -120.0, 15.0) for f in range(60, 100)],
        *[(f, 3, 0, 150.0, -10.0) for f in range(70, 110)],     # every class present (the DCASE-style rows average over all classes)
    ]

    def __init__(self, tmp_path, **kw):
        self._tmp = tmp_path
        super().__init__(split="test", orig_sr=SR, model_sr=32000, hop_length=320,
                         n_tracks=N_TR, window_s=10.0, hop_s=2.5, **kw)

    @property
    def n_classes(self):
        return N_CLS

    def _collect_clips(self, split):
        wav = self._tmp / "clip.wav"
        csv_p = self._tmp / "clip.csv"
        sf.write(str(wav), np.zeros((int(DUR_S * SR), 4), np.float32), SR, subtype="PCM_16")
        csv_p.write_text("")
        return [(wav, csv_p)]

    def _parse_csv(self, csv_path):
        return list(self.EVENTS)


def build_cache(tmp_path, ds, predictor):
    W = len(ds)
    T = ds.T_mel_window
    logits = np.zeros((W, T, N_TR, N_CLS), np.float32)
    doa = np.zeros((W, T, N_TR, N_CLS, 3), np.float32)
    for i in range(W):
        tgt = window_target(ds, i).numpy()
        active = np.linalg.norm(tgt, axis=-1) > 0.5
        lg, dv = predictor(tgt, active, window_start_frame(ds, i))
        logits[i], doa[i] = lg, dv
    d = tmp_path / "cache"
    d.mkdir()
    np.save(d / "logits.npy", logits)
    np.save(d / "doa.npy", doa)
    perm = np.broadcast_to(np.arange(N_TR, dtype=np.int8), (W, T, N_CLS, N_TR)).copy()
    np.save(d / "perm.npy", perm)
    (d / "meta.json").write_text(json.dumps({
        "n_windows": W, "n_tracks": N_TR, "n_classes": N_CLS,
        "items": [[int(c), int(s)] for c, s, _ in ds._items]}))
    return WindowCache(d)


def perfect(tgt, active, offset):
    return np.where(active, 8.0, -8.0).astype(np.float32), tgt.astype(np.float32)


_NOISE = np.random.default_rng(3).normal(0, 0.45, (256, N_TR, N_CLS, 3))


def noisy(tgt, active, offset):
    """Direction noise that is a fixed function of the *global* 100-ms bin, so that window
    stitching and 100-ms pooling do not average it away (all variants see the same errors)."""
    bins = (offset + np.arange(tgt.shape[0])) // 10
    v = tgt + _NOISE[bins]
    return np.where(active, 8.0, -8.0).astype(np.float32), (v * active[..., None]).astype(np.float32)


def test_window_geometry_helpers(tmp_path):
    ds = SynthDataset(tmp_path)
    assert len(ds) == 5 and ds.T_mel_window == 1001
    assert [window_start_frame(ds, i) for i in range(5)] == [0, 250, 500, 750, 1000]
    assert clip_n_mel(ds, 0) == 1 + round(int(DUR_S * SR) * 32000 / SR) // 320
    full = SynthDataset.__new__(SynthDataset)   # target of the full clip has the same frame count
    t = ds._build_target(ds._events[0], clip_n_mel(ds, 0), 0)
    assert t.shape[0] == clip_n_mel(ds, 0)


def test_perfect_predictor_is_perfect_under_every_variant(tmp_path):
    ds = SynthDataset(tmp_path)
    cache = build_cache(tmp_path, ds, perfect)
    res = evaluate_variants(cache, ds, [0.5], log=lambda *_: None)
    assert set(res) == set(VARIANTS)
    for name, per_thr in res.items():
        m = per_thr[0.5]
        er = m.get("ER_20", m.get("ER"))
        f = m.get("F_20", m.get("F"))
        assert er == pytest.approx(0.0, abs=1e-9), name
        assert f == pytest.approx(1.0, abs=1e-9), name
        # the 100-ms-grid rows (B2, D, S*) use the exact label bins while the predictor copies the
        # dataset labels, which differ by one bin on ~3 % of the frames (float artefact); for a moving
        # source this leaves a mean angular error far below 0.1 deg
        tol = 0.1
        assert m["LE_CD"] == pytest.approx(0.0, abs=tol), name
        assert m["LR_CD"] == pytest.approx(1.0, abs=1e-9), name
        assert m["SELD_score"] == pytest.approx(0.0, abs=1e-3), name


def test_degraded_predictor_scores_worse_and_ungated_le_exceeds_gated(tmp_path):
    ds = SynthDataset(tmp_path)
    good = evaluate_variants(build_cache(tmp_path, ds, perfect), ds, [0.5], log=lambda *_: None)
    (tmp_path / "cache").rename(tmp_path / "cache_perfect")
    bad = evaluate_variants(build_cache(tmp_path, ds, noisy), ds, [0.5], log=lambda *_: None)
    for name in VARIANTS:
        assert bad[name][0.5]["SELD_score"] > good[name][0.5]["SELD_score"] + 1e-3, name
    # gated LE is capped by the gate; the ungated class-dependent LE of the DCASE-style rows is not
    assert bad["A_repo_pooled_windows"][0.5]["LE_CD"] <= 20.0 + 1e-6
    assert bad["S4_seg1s_dcase"][0.5]["LE_CD"] > bad["A_repo_pooled_windows"][0.5]["LE_CD"]
