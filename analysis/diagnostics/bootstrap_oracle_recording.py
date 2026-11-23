#!/usr/bin/env python3
"""
analysis/diagnostics/bootstrap_oracle_recording.py
==============================================================================
Recording-level bootstrap of the oracle-DOA accuracy (test-set sampling uncertainty)
==============================================================================

Input : the tracked per-recording tables written by ``doa_oracle_test.py``
        (``results/diagnostics/doa_oracle_test/<model>/<run>/doa_recording_summary.csv``,
        78 STARSS23 test recordings per run; tPIT-aligned active-slot statistics).
Method: resample the 78 recordings with replacement (10,000 resamples, seed 0,
        the *same* resampled indices for every run, so that paired differences
        are meaningful); the statistic is the slot-weighted accuracy
        ``sum(acc_20deg_i * active_slots_i) / sum(active_slots_i)``.
Output: ``results/diagnostics/bootstrap_oracle_recording_phase2/``
        ``bootstrap_oracle_accuracy.csv`` (point estimate and 95 % percentile CI per run),
        ``paired_differences_vs_reference.csv`` and ``provenance.json``.

What it measures.  This is the sampling uncertainty of the *test set* (which
recordings were drawn).  It is **not** training-seed variability: every run
exists with a single seed (42) and no repeated runs exist in the repository.
Say "recording-level bootstrap" in text and captions.

Standalone controls.  ``doa_oracle_test.py`` aligns the DOA slots with the checkpoint's tPITLoss for
NAS-SELD models only; the stored oracle logs of the standalone Conformer/ResNet baselines (which are trained
with the same tPITLoss) were written without alignment (raw slot order) and are not comparable.  The
standalone rows of the main table therefore come from the offline re-inference with ``--align-standalone``
(``results/diagnostics/doa_oracle_test_phase2/*_expA_standalone_tpit_aligned/``); the historical raw-slot
Conformer row is kept in the accuracy table, labelled, and excluded from every paired difference.

Origin.  The computation was first run from a scratch location; this version
uses repository-relative paths and includes the standalone rows, so that the
numbers are reproducible with one command.

Usage
-----
    python analysis/diagnostics/bootstrap_oracle_recording.py

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.paths import DIAGNOSTIC_RESULTS_DIR  # noqa: E402

ORACLE_DIR = DIAGNOSTIC_RESULTS_DIR / "doa_oracle_test"
DEFAULT_OUT = DIAGNOSTIC_RESULTS_DIR / "bootstrap_oracle_recording_phase2"

RUNS: Dict[str, str] = {
    "mixed_bce": "nas-seld/exp_ft_mixed_bce",
    "mixed_focal (reference)": "nas-seld/exp_ft_mixed_focal",
    "ExpA (native STARSS23, no augmentation)": "nas-seld/expA_balanced_starss23",
    "ExpB": "nas-seld/balanced_starss23_expB",
    "ExpB-beta.7": "nas-seld/balanced_starss23_expB_beta07_thr03",
    "ExpC": "nas-seld/balanced_starss23_expC",
    "ExpA-posw": "nas-seld/posw_balanced_expA",
    "Conformer ExpA (standalone, historical raw-slot log, NOT PIT-aligned: not comparable)": "conformer/conformer_loss_experiments",
}
# offline re-inference with --align-standalone (tPIT-aligned, same pipeline as the NAS-SELD rows)
PHASE2_DIR = DIAGNOSTIC_RESULTS_DIR / "doa_oracle_test_phase2"
PHASE2_RUNS: Dict[str, str] = {
    "Conformer ExpA (standalone, Balanced + augmentation; tPIT-aligned)": "conformer_expA_standalone_tpit_aligned",
    "ResNet ExpA (standalone, Balanced + augmentation; tPIT-aligned)": "resnet_expA_standalone_tpit_aligned",
}
REFERENCE = "mixed_focal (reference)"
RAW_STANDALONE = "Conformer ExpA (standalone, historical raw-slot log, NOT PIT-aligned: not comparable)"
STANDALONES = list(PHASE2_RUNS)
# AT2SELD variants trained on Balanced + augmentation with an active-only loss: the
# closest AT2SELD counterparts of the standalone ExpA controls (ExpA itself is native STARSS23 without
# augmentation, i.e. not data-matched; it is listed last for information).
VS_STANDALONE = ["ExpB", "ExpC", "ExpA-posw", "ExpB-beta.7", "ExpA (native STARSS23, no augmentation)"]


def load_run(run_dir: Path) -> Dict[str, Tuple[float, float]]:
    path = run_dir / "doa_recording_summary.csv"
    with path.open() as fh:
        return {r["recording_id"]: (float(r["active_slots"]), float(r["acc_20deg"]))
                for r in csv.DictReader(fh)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("Usage")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oracle-dir", type=Path, default=ORACLE_DIR)
    ap.add_argument("--phase2-dir", type=Path, default=PHASE2_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for label, rel in RUNS.items():
        p = args.oracle_dir / rel
        if (p / "doa_recording_summary.csv").exists():
            data[label] = load_run(p)
        else:
            print(f"missing: {label} ({p})")
    for label, rel in PHASE2_RUNS.items():
        p = args.phase2_dir / rel
        if (p / "doa_recording_summary.csv").exists():
            data[label] = load_run(p)
        else:
            print(f"missing (run the offline re-inference with --align-standalone): {label} ({p})")
    common = None
    for v in data.values():
        common = set(v) if common is None else common & set(v)
    common = sorted(common or [])
    n = len(common)
    print(f"recordings in common: {n}")
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, n, size=(args.resamples, n))

    stats: Dict[str, Tuple[float, np.ndarray]] = {}
    rows: List[dict] = []
    for label, d in data.items():
        w = np.array([d[r][0] for r in common])
        a = np.array([d[r][1] for r in common])
        point = float((w * a).sum() / w.sum())
        boots = np.array([(w[i] * a[i]).sum() / w[i].sum() for i in idx])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        stats[label] = (point, boots)
        rows.append({"run": label, "n_recordings": n, "active_slots": int(w.sum()),
                     "acc_20deg_percent": round(100 * point, 2),
                     "ci95_low_percent": round(100 * lo, 2),
                     "ci95_high_percent": round(100 * hi, 2)})
        print(f"{label:55s} acc@20 = {100 * point:5.1f} %  95% CI [{100 * lo:5.1f}, {100 * hi:5.1f}]")

    paired: List[dict] = []

    def add_pair(label: str, ref_label: str) -> None:
        point, boots = stats[label]
        ref_p, ref_b = stats[ref_label]
        diff = boots - ref_b
        lo, hi = np.percentile(diff, [2.5, 97.5])
        paired.append({"run": label, "reference": ref_label,
                       "delta_pp": round(100 * (point - ref_p), 2),
                       "ci95_low_pp": round(100 * lo, 2), "ci95_high_pp": round(100 * hi, 2),
                       "p_delta_gt_0": round(float(np.mean(diff > 0)), 4)})
        print(f"  paired vs {ref_label[:22]:22s}: {label:55s} {100 * (point - ref_p):+5.1f} pp "
              f"[{100 * lo:+5.1f}, {100 * hi:+5.1f}]")

    if REFERENCE in stats:
        for label in stats:
            if label not in (REFERENCE, RAW_STANDALONE):
                add_pair(label, REFERENCE)
    for standalone in STANDALONES:
        if standalone in stats:
            for label in VS_STANDALONE:
                if label in stats:
                    add_pair(label, standalone)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in (("bootstrap_oracle_accuracy.csv", rows),
                        ("paired_differences_vs_reference.csv", paired)):
        if table:
            with (args.out_dir / name).open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(table[0]), lineterminator="\n")
                w.writeheader()
                w.writerows(table)
    (args.out_dir / "provenance.json").write_text(json.dumps({
        "script": "analysis/diagnostics/bootstrap_oracle_recording.py",
        "inputs": {**{k: str((args.oracle_dir / v / "doa_recording_summary.csv").relative_to(ROOT))
                      for k, v in RUNS.items() if k in data},
                   **{k: str((args.phase2_dir / v / "doa_recording_summary.csv").relative_to(ROOT))
                      for k, v in PHASE2_RUNS.items() if k in data}},
        "note": "standalone rows: re-inference with doa_oracle_test.py --align-standalone (tPIT-aligned); "
                "the historical standalone log is raw-slot (not PIT-aligned) and excluded from paired differences",
        "resamples": args.resamples, "seed": args.seed, "n_recordings": n,
        "statistic": "slot-weighted acc_20deg (tPIT-aligned oracle activity); 95% percentile CI",
        "interpretation": "test-set sampling (recording bootstrap), not training-seed variance",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=1))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
