"""Dataset-protocol tests (data-free unless marked; data-dependent tests skip with a reason).

* label-frame conversion, slot order and the 10-ms <-> 100-ms mapping of ``SELDDataset``;
* window geometry (10 s / hop 2.5 s -> 1001 model frames, 250-frame hop);
* TAU2019 fold tables and the fold-cycling rule;
* cross-dataset class-mapping tables against the tracked ``mapping_coverage_matrix.csv``;
* STARSS23 split reproducibility (72 / 18 clips, seed 42; 78 test clips), window counts
  (4,674 test windows) -- these need the local datasets and skip otherwise.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from conftest import ROOT, requires_data
from datasets.utils import SELDDataset


class ToyDataset(SELDDataset):
    """Minimal concrete dataset: no files, only the target construction is exercised."""

    @property
    def n_classes(self) -> int:
        return 4

    def _collect_clips(self, split):
        return []

    def _parse_csv(self, csv_path):
        return []


def toy(n_tracks=2, **kw):
    return ToyDataset(split="test", orig_sr=24000, model_sr=32000, hop_length=320,
                      n_tracks=n_tracks, window_s=None, **kw)


# ── label frame ───────────────────────────────────────────────────────────────

def test_label_frame_slot_order_and_truncation():
    ds = toy(n_tracks=2)
    # frame 0, class 1: three sources (indices 9, 2, 5); slots keep ascending source index
    events = [(0, 1, 9, 90.0, 0.0), (0, 1, 2, 0.0, 0.0), (0, 1, 5, 180.0, 0.0),
              (1, 2, 4, 0.0, 90.0)]
    t = ds._build_target(events, n_mel=40, start_at_model_sr=0).numpy()
    # model frames 0..9 carry csv frame 0 (apart from the float artefact tested below)
    np.testing.assert_allclose(t[5, 0, 1], [0.0, 1.0, 0.0], atol=1e-6)     # az 0: x = sin az, y = cos az
    np.testing.assert_allclose(t[5, 1, 1], [0.0, -1.0, 0.0], atol=1e-6)    # source 5, az 180
    assert np.linalg.norm(t[5, :, 1], axis=-1).tolist() == pytest.approx([1.0, 1.0])  # source 9 dropped
    np.testing.assert_allclose(t[15, 0, 2], [0.0, 0.0, 1.0], atol=1e-6)    # el 90 -> +z
    assert np.abs(t[25]).sum() == 0                                          # frame 2: silent


def test_label_frame_is_xy_swapped_with_respect_to_foa_frame():
    """Labels are (sin az cos el, cos az cos el, sin el); the FOA/Ch.2 frame has x = cos az cos el."""
    ds = toy(n_tracks=1)
    t = ds._build_target([(0, 0, 0, 90.0, 0.0)], 10, 0).numpy()[5, 0, 0]    # source to the left (+90)
    np.testing.assert_allclose(t, [1.0, 0.0, 0.0], atol=1e-6)               # label x = +1
    # in the FOA frame (x forward, y left) the same direction is (0, 1, 0)


def test_float_artifact_of_label_bin_mapping_is_characterised():
    """``t_csv = int((t_s / 0.1))`` assigns ~3 % of the model frames to the preceding 100-ms bin
    (``t_s / 0.1`` evaluates to e.g. 2.9999999999999996).  Characterisation test: documents
    docs/CONVENTIONS.md section 2; the fraction is small and no frame is off by more than one bin."""
    hop, sr = 320, 32000
    m = np.arange(6001)
    t_csv = (((m * hop) / sr) / 0.1).astype(np.int64)
    exact = m // 10
    off = t_csv != exact
    assert 0.02 < off.mean() < 0.05
    assert np.all((exact - t_csv)[off] == 1)


# ── window geometry ───────────────────────────────────────────────────────────

def test_window_geometry_10s_hop_2p5s():
    ds = ToyDataset(split="test", orig_sr=24000, model_sr=32000, hop_length=320,
                    n_tracks=3, window_s=10.0, hop_s=2.5)
    assert ds.T_mel_window == 1001
    assert ds.hop_samples == 60000 and ds.window_samples == 240000
    assert round(ds.hop_samples * 32000 / 24000) % 320 == 0
    assert round(ds.hop_samples * 32000 / 24000) // 320 == 250
    tau = ToyDataset(split="test", orig_sr=48000, model_sr=32000, hop_length=320,
                     n_tracks=3, window_s=10.0, hop_s=2.5)
    assert tau.T_mel_window == 1001 and round(tau.hop_samples * 32000 / 48000) // 320 == 250


# ── TAU2019 ───────────────────────────────────────────────────────────────────

def test_tau2019_fold_tables_and_cycling_rule():
    from datasets.tau2019 import TAU2019Dataset
    table = TAU2019Dataset.FOLD_TABLE
    for fold, parts in table.items():
        used = parts["train"] + parts["val"] + parts["test"]
        assert sorted(used) == [1, 2, 3, 4], f"fold {fold}"
    assert TAU2019Dataset.MIXED_SPLITS == {"train": [2, 3, 4], "val": [1]}
    # FoldCyclingDataModule: epoch e uses fold (e % 4) + 1 -> train/val schedule of its docstring
    sched = [(table[(e % 4) + 1]["train"], table[(e % 4) + 1]["val"]) for e in range(5)]
    assert sched == [([3, 4], [2]), ([4, 1], [3]), ([1, 2], [4]), ([2, 3], [1]), ([3, 4], [2])]
    # every dev split is training data in 3 of the 4 epochs: validation is not held out
    for split_no in (1, 2, 3, 4):
        assert sum(split_no in table[f]["train"] for f in table) == 2
        assert sum(split_no in table[f]["val"] for f in table) == 1


# ── class-mapping tables ──────────────────────────────────────────────────────

def test_cross_dataset_mapping_tables_match_tracked_coverage_matrix():
    from analysis.diagnostics.dataset_generalization_analysis import (
        DATASET_ALIASES, build_mapping, mapping_report)
    path = ROOT / "results" / "diagnostics" / "dataset_generalization" / "aggregate" / \
        "mapping_coverage_matrix.csv"
    if not path.exists():
        pytest.skip(f"tracked file absent: {path.relative_to(ROOT)}")
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 16
    for r in rows:
        report = mapping_report(r["test_dataset"], r["train_space"],
                                build_mapping(r["test_dataset"], r["train_space"]))
        assert report["mapped_source_classes"] == int(r["mapped_source_classes"]), r
        assert report["source_classes"] == int(r["source_classes"]), r
        assert report["covered_destination_classes"] == int(r["covered_destination_classes"]), r
        assert report["destination_classes"] == int(r["destination_classes"]), r


def test_explicit_mapping_examples():
    from analysis.diagnostics.dataset_generalization_analysis import LABELS, build_mapping
    m = build_mapping("tau_nigens2021", "starss23")           # dataset classes -> model classes
    names = LABELS["tau_nigens2021"]
    dest = {names[i]: [LABELS["starss23"][j] for j in d] for i, d in enumerate(m)}
    assert dest["piano"] == ["music", "musical_instrument"]
    assert dest["female_scream"] == ["female_speech"]
    assert dest["alarm"] == []                                # no STARSS23 counterpart


# ── STARSS23 (needs the local dataset) ────────────────────────────────────────

def starss(split, seed=42, window_s=None, hop_s=None):
    from datasets.starss23 import STARSS23Dataset
    return STARSS23Dataset(root=ROOT / "data" / "STARSS23", split=split, model_sr=32000,
                           hop_length=320, n_tracks=3, window_s=window_s, hop_s=hop_s,
                           val_fraction=0.2, val_seed=seed)


@requires_data("STARSS23", "metadata_dev")
def test_starss23_split_72_18_seed42_is_reproducible_and_room_stratified():
    tr, va, te = starss("train"), starss("val"), starss("test")
    assert (len(tr.clips), len(va.clips), len(te.clips)) == (72, 18, 78)
    names = lambda ds: [c[1].stem for c in ds.clips]
    assert not set(names(tr)) & set(names(va)) and not set(names(tr)) & set(names(te))
    assert names(tr) == names(starss("train")) and names(va) == names(starss("val"))
    # room-by-room (site, room) stratification: n_val = max(1, round(0.2 * n_group))
    groups = {}
    for ds, key in ((tr, "train"), (va, "val")):
        for c in ds.clips:
            site = "sony" if "sony" in c[1].parent.name else "tau"
            g = groups.setdefault((site, c[1].stem.split("_")[1]), {"train": 0, "val": 0})
            g[key] += 1
    assert len(groups) == 9
    for g in groups.values():
        assert g["val"] == max(1, round(0.2 * (g["train"] + g["val"])))


@requires_data("STARSS23", "metadata_dev")
def test_starss23_seed2_validation_pool_overlaps_the_seed42_training_clips():
    """The post-hoc diagnostics build their validation pool with seed 2 whereas training used
    seed 42: the two 18-clip pools differ, so part of the diagnostic 'validation' set was seen
    in training (docs/REPRODUCIBILITY.md).  Characterisation of the repository state."""
    names = lambda ds: {c[1].stem for c in ds.clips}
    train42, val42, val2 = names(starss("train", 42)), names(starss("val", 42)), names(starss("val", 2))
    assert len(val2) == 18 and val2 != val42
    overlap = len(val2 & train42)
    assert 1 <= overlap <= 18 and overlap == 18 - len(val2 & val42)


@requires_data("STARSS23", "foa_dev", "dev-test-sony")
def test_starss23_test_window_count_is_4674():
    te = starss("test", window_s=10.0, hop_s=2.5)
    assert len(te) == 4674


@requires_data("TAU2019", "eval")
def test_tau2019_eval_window_count():
    from datasets.tau2019 import TAU2019Dataset
    ds = TAU2019Dataset(root=ROOT / "data" / "TAU2019", split="test", model_sr=32000,
                        hop_length=320, n_tracks=3, window_s=10.0, hop_s=2.5, fold=1,
                        mixed_mode=True)
    assert len(ds) == 2405
