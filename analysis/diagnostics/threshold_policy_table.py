#!/usr/bin/env python3
"""
analysis/diagnostics/threshold_policy_table.py
==============================================================================
T3 — Threshold-policy table assembled from the stored diagnostic CSVs
==============================================================================

Collects, for every Stage-4 NAS-SELD checkpoint with a stored threshold sweep and
for the policies that were calibrated on validation, the *same* test-set metrics
under explicitly labelled threshold policies, so that no "Test SELD" number is
quoted without its policy:

  headline            fixed / configured threshold (0.5 unless the run was configured otherwise)
  validation-selected global or class-specific thresholds chosen on the validation
                      split (macro-F1 criterion of ``class_specific_threshold_calibration.py``;
                      SELD-at-20-deg criterion of ``dataset_generalization_analysis.py``)
  post-hoc            test-optimised thresholds (global minimum of the test SELD over
                      the stored sweep; class-wise test oracle) -- diagnostics, *not*
                      deployable operating points

No model is run: every number is read from files under ``results/diagnostics`` and the
source file and row are recorded in the output.

Outputs: ``results/diagnostics/threshold_policy_phase2/threshold_policy_table.csv`` and
``threshold_policy_table.md``.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.paths import DIAGNOSTIC_RESULTS_DIR as DIAG  # noqa: E402

OUT = DIAG / "threshold_policy_phase2"

# run folder (under sed_test/nas-seld) -> (label used in the reported tables, configured threshold)
RUNS = {
    "exp_ft_mixed_bce": ("mixed_bce", 0.5),
    "exp_ft_mixed_focal": ("mixed_focal (reference)", 0.5),
    "expA_balanced_starss23": ("ExpA", 0.5),
    "balanced_starss23_expB": ("ExpB", 0.5),
    "balanced_starss23_expB_beta07_thr03": ("ExpB-beta.7", 0.3),
    "balanced_starss23_expC": ("ExpC", 0.5),
    "posw_balanced_expA": ("ExpA-posw", 0.3),
    "exp_ft_tau2019_focal": ("TAU2019 focal (native)", 0.5),
    "tau2019_expA": ("TAU2019-ExpA (native)", 0.5),
    "exp_ft_tau_nigens_focal": ("NIGENS2021 focal (native)", 0.5),
    "tau_nigens2021_expA": ("NIGENS2021-ExpA (native)", 0.5),
    "exp_ictx_balanced_w20": ("ictx w20", 0.5),
    "exp_topt_BiGRU_n2_h512_merge_sum": ("topt BiGRU", 0.5),
    "exp_topt_TT_n1_ffn8_pe_4tracks": ("topt TT-4", 0.5),
}
FIELDS = ["model", "test_set", "policy", "label", "threshold", "SELD", "ER", "F", "LE", "LR",
          "source_file", "source_row"]


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def read_csv(p: Path) -> List[dict]:
    with p.open() as fh:
        return list(csv.DictReader(fh))


def f(x) -> float:
    return float(x)


def sweep_rows(run: str, label: str, configured: float) -> List[dict]:
    p = DIAG / "sed_test" / "nas-seld" / run / "threshold_summary.csv"
    if not p.exists():
        return []
    rows = read_csv(p)
    # the diagnostic scripts name the evaluation mode after the label space; the
    # test split of "balanced_starss23" runs is the STARSS23 dev-test split
    test_set = {"balanced_starss23": "starss23", "tau_nigens": "tau_nigens2021"}.get(
        rows[0]["dataset"], rows[0]["dataset"])
    out = []
    best = min(rows, key=lambda r: f(r["SELD_score"]))
    for r in rows:
        if abs(f(r["threshold"]) - configured) < 1e-9:
            out.append(_row(label, test_set, "fixed_configured_threshold", "headline", f(r["threshold"]),
                            r, p, f"threshold={f(r['threshold']):.2f}"))
    out.append(_row(label, test_set, "test_optimal_global_threshold", "post-hoc (test-optimised)",
                    f(best["threshold"]), best, p, f"min SELD over {len(rows)} thresholds"))
    return out


def _row(model, test_set, policy, label, thr, r, p, row_desc, keys=None) -> dict:
    k = keys or {"SELD": "SELD_score", "ER": "ER_20", "F": "F_20", "LE": "LE_CD", "LR": "LR_CD"}
    return {"model": model, "test_set": test_set, "policy": policy, "label": label,
            "threshold": thr if isinstance(thr, str) else round(thr, 4),
            "SELD": round(f(r[k["SELD"]]), 4), "ER": round(f(r[k["ER"]]), 4),
            "F": round(f(r[k["F"]]), 4), "LE": round(f(r[k["LE"]]), 3), "LR": round(f(r[k["LR"]]), 4),
            "source_file": rel(p), "source_row": row_desc}


def calibration_rows() -> List[dict]:
    out = []
    for preset, model in (("nas_expa", "ExpA"), ("nas_mixed_focal_legacy", "mixed_focal (reference)")):
        p = DIAG / "class_threshold_calibration" / preset / "strategy_summary.csv"
        if not p.exists():
            continue
        thr_file = DIAG / "class_threshold_calibration" / preset / "class_thresholds.json"
        glob_thr = None
        if thr_file.exists():
            js = json.loads(thr_file.read_text())
            glob_thr = js.get("global_threshold", js.get("global"))
        for r in read_csv(p):
            name = r["strategy"]
            mapping = {
                "fixed_0.5": ("fixed_0.5_recomputed", "headline", 0.5),
                "global_validation": ("val_selected_global_macroF1", "validation-selected",
                                      glob_thr if glob_thr is not None else "validation global"),
                "class_validation": ("val_selected_class_macroF1", "validation-selected",
                                     "class-specific (validation)"),
                "class_test_oracle": ("class_wise_test_oracle", "post-hoc (test-optimised)",
                                      "class-specific (test oracle)"),
            }
            if name in mapping:
                pol, lab, thr = mapping[name]
                out.append(_row(model, "starss23", pol, lab, thr, r, p, f"strategy={name}"))
    return out


def matrix_rows() -> List[dict]:
    p = DIAG / "dataset_generalization" / "aggregate" / "all_test_metrics.csv"
    names = {"nas_expa_starss23": "ExpA", "nas_expa_tau2019": "TAU2019-ExpA (native)",
             "nas_expa_tau_nigens2021": "NIGENS2021-ExpA (native)",
             "nas_expa_tau_nigens2020": "NIGENS2020-ExpA (native)"}
    out = []
    for r in read_csv(p):
        if r["model"] not in names or r["train_space"] != r["test_dataset"]:
            continue
        proto = r["protocol"]
        if proto in ("val_selected_global_seld_score_at_20deg",
                     "val_selected_class_specific_seld_at_20deg",
                     "val_selected_class_specific_sed",
                     "val_selected_global_sed_f1_at_20deg"):
            thr = r["threshold"] if r["threshold"].startswith("[") is False else "class-specific vector"
            out.append(_row(names[r["model"]], r["test_dataset"], f"matrix:{proto}",
                            "validation-selected", thr, r, p, f"protocol={proto}",
                            keys={"SELD": "seld_score", "ER": "er", "F": "seld_f1",
                                  "LE": "le", "LR": "lr"}))
    return out


def main() -> None:
    rows: List[dict] = []
    for run, (label, conf) in RUNS.items():
        rows += sweep_rows(run, label, conf)
    rows += calibration_rows()
    rows += matrix_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "threshold_policy_table.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    lines = ["| model | test set | policy | label | threshold | SELD | ER | F | LE (deg) | LR |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['test_set']} | {r['policy']} | {r['label']} | "
                     f"{r['threshold']} | {r['SELD']:.4f} | {r['ER']:.4f} | {r['F']:.4f} | "
                     f"{r['LE']:.2f} | {r['LR']:.4f} |")
    (OUT / "threshold_policy_table.md").write_text("\n".join(lines) + "\n")
    print(f"{len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
