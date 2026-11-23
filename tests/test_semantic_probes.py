"""Inference-time probes of the semantic prior: intervention code and split-forward equivalence (CPU, no data)."""
from __future__ import annotations

import types

import numpy as np
import pytest
import torch
import torchaudio.functional as AF

from analysis.diagnostics.semantic_probes import (
    check_equivalence, derangement, head_from, semantic_input, semantic_late, spatial_trunk)
from analysis.profiling.param_accounting import build_model, load_cfg
from conftest import ROOT

CFG = ROOT / "configs" / "loss_experiments" / "expA_nas_seld_balanced_starss23.yaml"
needs_ckpt = pytest.mark.skipif(
    not (ROOT / "semantic_backbone" / "epanns" / "checkpoint_closeto_.44.pt").exists(),
    reason="tracked E-PANNs checkpoint not found")


@pytest.mark.parametrize("n", [2, 3, 5, 10])
def test_derangement_has_no_fixed_point(n):
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = derangement(n, rng)
        assert sorted(p.tolist()) == list(range(n)) and not np.any(p == np.arange(n))


def test_semantic_input_interventions():
    x = torch.randn(2, 4, 32000 * 5)
    partner = torch.randn(2, 4, 32000 * 5)
    sr = 32000
    assert torch.equal(semantic_input("baseline", x, None, sr, [0, 1]), x.mean(dim=1))
    assert torch.equal(semantic_input("bridge_zeroed", x, None, sr, [0, 1]), x.mean(dim=1))
    assert torch.count_nonzero(semantic_input("zero_semantic", x, None, sr, [0, 1])) == 0
    assert torch.equal(semantic_input("w_only", x, None, sr, [0, 1]), x[:, 0])
    assert torch.equal(semantic_input("other_recording", x, partner, sr, [0, 1]), partner.mean(dim=1))
    mono = x.mean(dim=1)
    shuf = semantic_input("time_shuffled", x, None, sr, [7, 8])
    assert shuf.shape == mono.shape and not torch.equal(shuf, mono)
    blocks = lambda t: sorted(tuple(t[i * sr:(i + 1) * sr].tolist()[:3]) for i in range(5))
    assert blocks(shuf[0]) == blocks(mono[0])                    # same blocks, different order
    assert not any(torch.equal(shuf[0, i * sr:(i + 1) * sr], mono[0, i * sr:(i + 1) * sr]) for i in range(5))
    again = semantic_input("time_shuffled", x, None, sr, [7, 8])
    assert torch.equal(shuf, again)                              # deterministic per window id


@needs_ckpt
def test_split_forward_equals_nasmodel_forward_and_probes_change_output():
    model = build_model(load_cfg(CFG))
    module = types.SimpleNamespace(model=model, orig_sr=24000)
    torch.manual_seed(0)
    x = torch.randn(1, 4, 24000 * 4)
    assert check_equivalence(module, x, torch.device("cpu")) == 0.0
    with torch.inference_mode():
        xr = AF.resample(x, 24000, model.src_sr)
        S = spatial_trunk(model, xr)
        h = semantic_late(model, xr.mean(dim=1))
        a0, d0, rel0 = head_from(model, S, h, False)
        a1, d1, rel1 = head_from(model, S, None, True)           # bridge bypassed
        h0 = semantic_late(model, torch.zeros_like(xr[:, 0]))
        a2, _, _ = head_from(model, S, h0, False)                # zeroed semantic input
    assert float(rel0) > 0.0 and float(rel1) == 0.0
    assert not torch.allclose(a0, a1) and not torch.allclose(a0, a2)
