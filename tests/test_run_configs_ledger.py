"""The effective-configuration ledger (``results/experiments/run_configs.csv``).

Data-free checks on the tracked CSV (counts, ExpA / reference rows of the reported results) and, when the
``runs/`` archive is restored, a regeneration check against the stored ``hparams.yaml`` files.
"""
from __future__ import annotations

import csv

import pytest

from conftest import ROOT

LEDGER = ROOT / "results" / "experiments" / "run_configs.csv"


@pytest.fixture(scope="module")
def rows():
    with LEDGER.open() as fh:
        return {r["run_id"]: r for r in csv.DictReader(fh)}


def by_exp(rows, name):
    hits = [r for r in rows.values() if r["exp_name"] == name]
    assert len(hits) == 1, name
    return hits[0]


def test_counts_and_uniqueness(rows):
    assert len(rows) == 69
    fam = [r["family"] for r in rows.values()]
    assert fam.count("at2seld_nas") == 44 and fam.count("standalone_seld") == 25
    assert all(r["seed"] == "42" for r in rows.values())


def test_stage123_executed_values(rows):
    stage123 = [r for r in rows.values()
                if r["group"] in ("at2seld_nas/shallow_spec", "at2seld_nas/shallow_gab",
                                  "at2seld_nas/deep_spec", "at2seld_nas/deep_reg")]
    assert len(stage123) == 27
    assert {r["n_tracks"] for r in stage123} == {"2"}
    assert {(r["lambda_spatial"], r["lambda_semantic"]) for r in stage123} == {("0.9", "0.1")}
    assert {r["aug_enabled"] for r in stage123} == {"False"}
    beta = {r["group"]: r["beta_sed"] for r in stage123}
    assert beta["at2seld_nas/shallow_spec"] == "0.3" and beta["at2seld_nas/deep_spec"] == "0.6"


def test_headline_model_provenance(rows):
    a = by_exp(rows, "expA_nas_seld_balanced_starss23")
    assert (a["dataset"], a["aug_enabled"], a["n_tracks"], a["beta_sed"]) == ("STARSS23", "False", "3", "0.5")
    assert a["doa_loss_mode"] == "active_mse" and a["inactive_weight"] == "0.005"
    assert (a["best_epoch"], a["best_checkpoint"]) == ("34", "34_0.3375.ckpt")
    assert float(a["test_SELD"]) == pytest.approx(0.6355, abs=1e-4)
    ref = by_exp(rows, "exp_ft_mixed_focal")
    assert (ref["dataset"], ref["aug_enabled"], ref["beta_sed"]) == ("BalancedSTARSS23", "True", "0.85")


def test_augmentation_scope(rows):
    aug_nas = sorted(r["exp_name"] for r in rows.values()
                     if r["family"] == "at2seld_nas" and r["aug_enabled"] == "True")
    assert len(aug_nas) == 11 and "expA_nas_seld_balanced_starss23" not in aug_nas
    assert all(r["aug_enabled"] == "True" for r in rows.values() if r["family"] == "standalone_seld")


@pytest.mark.skipif(not (ROOT / "runs" / "at2seld_nas").exists(), reason="runs/ archive not restored")
def test_ledger_regenerates_identically_from_runs():
    from analysis.experiments.export_run_configs import build_rows, write_csv
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        out = pathlib.Path(d) / "x.csv"
        write_csv(build_rows(ROOT / "runs"), out)
        assert out.read_text() == LEDGER.read_text()
