"""Parameter accounting by instantiation (CPU, no data; needs only the tracked E-PANNs checkpoint).

Locks the arithmetic of the final-model profile reported for the model:
30,986,080 = 27,475,182 (E-PANNs 24,289,295 + copied heads 3,179,023 + head 6,864)
             + 2,932,082 spatial + 578,816 bridge; non-trainable 1,083,456; and the bridge size
difference between N = 3 and N = 2 tracks (115,456).
"""
from __future__ import annotations

import copy

import pytest

from analysis.profiling.param_accounting import breakdown, build_model, load_cfg
from conftest import ROOT

CFG = ROOT / "configs" / "loss_experiments" / "expA_nas_seld_balanced_starss23.yaml"

pytestmark = pytest.mark.skipif(
    not (ROOT / "semantic_backbone" / "epanns" / "checkpoint_closeto_.44.pt").exists(),
    reason="tracked E-PANNs checkpoint semantic_backbone/epanns/checkpoint_closeto_.44.pt not found")


@pytest.fixture(scope="module")
def expa():
    return breakdown(build_model(load_cfg(CFG)))


def test_expa_total_and_branches(expa):
    assert expa["total"] == 30_986_080
    assert expa["trainable"] == 29_902_624 and expa["non_trainable"] == 1_083_456
    assert expa["semantic_total"] == 27_475_182
    assert expa["semantic_epanns_backbone"] == 24_289_295
    assert expa["semantic_sem_fc1"] == 1024 * 2048 + 2048
    assert expa["semantic_sem_fc_audioset"] == 2048 * 527 + 527
    assert expa["semantic_head"] == 527 * 13 + 13
    assert expa["spatial_seld"] == 2_932_082
    assert expa["bridge"] == 578_816
    assert expa["other"] == 0
    assert expa["task_facing_seld"] == 3_510_898
    assert expa["task_facing_share_percent"] == pytest.approx(11.33, abs=0.005)
    assert expa["total"] == expa["semantic_total"] + expa["spatial_seld"] + expa["bridge"]


def test_non_trainable_are_the_torchlibrosa_bases():
    # STFT bases 2 * 513 * 1024 + mel filterbank 513 * 64 (counted as parameters by PyTorch)
    assert 2 * 513 * 1024 + 513 * 64 == 1_083_456


def test_bridge_size_n3_vs_n2_is_115456(expa):
    cfg = copy.deepcopy(load_cfg(CFG))
    cfg["heads"]["n_tracks"] = 2
    cfg["spatial_late"]["n_tracks"] = 2
    n2 = breakdown(build_model(cfg))
    assert expa["bridge"] - n2["bridge"] == 115_456
    assert n2["bridge"] == 463_360
