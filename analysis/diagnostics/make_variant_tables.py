#!/usr/bin/env python3
"""
analysis/diagnostics/make_variant_tables.py
==============================================================================
Compact "metric-variant sensitivity" tables from ``run_metric_variants.py`` outputs
==============================================================================

Reads ``results/diagnostics/metric_variants_phase2/<run>/variants_long.csv`` and writes
``results/diagnostics/metric_variants_phase2/sensitivity_tables.md`` (+ ``.csv``):

* Table A -- one checkpoint (default ExpA, STARSS23 test): every variant, at the fixed
  threshold 0.5 and at the post-hoc (test-optimal) threshold of that variant;
* Table B -- the SELD score of every checkpoint under every variant at the fixed threshold.

The first row of each table is the repository protocol (the headline); the segment-based
rows (``S*``) are a re-implementation of DCASE-style scoring (arithmetic of the public baseline code, see
``check_against_public_dcase.py``), not the official evaluation pipeline.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.paths import DIAGNOSTIC_RESULTS_DIR  # noqa: E402

BASE = DIAGNOSTIC_RESULTS_DIR / "metric_variants_phase2"
RUNS = [("expA_starss23_test", "ExpA (STARSS23 test)"),
        ("mixedfocal_starss23_test", "mixed_focal reference (STARSS23 test)"),
        ("expB_starss23_test", "ExpB (STARSS23 test)"),
        ("tau2019_test", "TAU2019-ExpA (TAU2019 eval)"),
        ("nigens2021_test", "NIGENS2021-ExpA (NIGENS2021 test)"),
        ("nigens2020_test", "NIGENS2020-ExpA (NIGENS2020 test)")]
LABELS = {
    "A_repo_pooled_windows": "A  repository protocol: 10-ms frames, windows pooled (4x overlap), gated 20 deg  [headline]",
    "B_stitched_10ms": "B  windows stitched (reference-free), each frame once, same gated metric",
    "B2_stitched_10ms_exact_labels": "B2 as B, exact 100-ms label bins (no float artefact)",
    "C_stitched_oracle_10ms": "C  as B, tPIT-aligned slots (upper bound, as main_demo.py)",
    "D_stitched_100ms_frame": "D  stitched, 100-ms frames, gated frame-level metric",
    "S1_seg100ms_dcase": "S1 100-ms segments, DCASE-style (ungated LE/LR, macro-F, S/D/I; public-code arithmetic) [re-impl.]",
    "S2_seg1s_repo_components": "S2 1-s segments, repository components (gated LE/LR, pooled F)",
    "S3_seg1s_gated_macroF": "S3 1-s segments, gated LE/LR, macro-F",
    "S4_seg1s_dcase": "S4 1-s segments, DCASE-style (ungated LE/LR, macro-F, S/D/I; public-code arithmetic) [re-impl.]",
    "S4s_seg1s_dcase_strict": "S4s S4 with stricter bookkeeping (never-co-active class = miss + spurious; macro over classes with references)",
    "S4m_seg1s_dcase_meandir": "S4m S4 with segment-mean-direction instance angles (sensitivity)",
}


def load(run: str) -> Dict[str, Dict[float, dict]]:
    by: Dict[str, Dict[float, dict]] = {}
    with (BASE / run / "variants_long.csv").open() as fh:
        for r in csv.DictReader(fh):
            by.setdefault(r["variant"], {})[float(r["threshold"])] = r
    return by


def fmt(x, nd=3) -> str:
    return "" if x in (None, "") else f"{float(x):.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("Reads")[0])
    ap.add_argument("--table-a-run", default="expA_starss23_test")
    args = ap.parse_args()
    out_rows: List[dict] = []
    md: List[str] = []

    by = load(args.table_a_run)
    title = dict(RUNS).get(args.table_a_run, args.table_a_run)
    md += [f"**Table A -- metric-variant sensitivity, {title}.** Same predictions, different evaluation protocols "
           "(first row = headline). `tau` = 0.5 (fixed) and the post-hoc test-optimal tau of each row "
           "(diagnostic, not an operating point). S/D/I = per-segment substitution/deletion/insertion rates "
           "of the DCASE-style rows.", "",
           "| variant | ER | F | LE (deg) | LR | SELD | S / D / I | best tau (post-hoc) | SELD at best tau |",
           "|---|---|---|---|---|---|---|---|---|"]
    for v, label in LABELS.items():
        if v not in by:
            continue
        d = by[v]
        r = d[0.5]
        bt = min(d, key=lambda t: float(d[t]["SELD_score"]))
        sdi = " / ".join(fmt(r.get(k), 2) for k in ("S_rate", "D_rate", "I_rate")) if r.get("S_rate") else ""
        md.append(f"| {label} | {fmt(r['ER'])} | {fmt(r['F'])} | {fmt(r['LE_CD'], 2)} | {fmt(r['LR_CD'])} | "
                  f"{fmt(r['SELD_score'])} | {sdi} | {bt:.2f} | {fmt(d[bt]['SELD_score'])} |")
        out_rows.append({"table": "A", "run": args.table_a_run, "variant": v, "tau": 0.5,
                         "ER": r["ER"], "F": r["F"], "LE": r["LE_CD"], "LR": r["LR_CD"],
                         "SELD": r["SELD_score"], "best_tau": bt, "SELD_best_tau": d[bt]["SELD_score"]})
    md += ["", "**Table B -- SELD score at the fixed threshold 0.5 under each protocol.**", "",
           "| variant | " + " | ".join(name for _, name in RUNS) + " |",
           "|---|" + "---|" * len(RUNS)]
    data = {run: load(run) for run, _ in RUNS if (BASE / run / "variants_long.csv").exists()}
    for v, label in LABELS.items():
        cells = []
        for run, _ in RUNS:
            if run in data and v in data[run]:
                cells.append(fmt(data[run][v][0.5]["SELD_score"]))
                out_rows.append({"table": "B", "run": run, "variant": v, "tau": 0.5,
                                 "SELD": data[run][v][0.5]["SELD_score"]})
            else:
                cells.append("")
        md.append(f"| {label} | " + " | ".join(cells) + " |")
    md += ["", "Segment-based rows (S*) are a re-implementation of DCASE-style scoring: the arithmetic of S1/S4 was verified "
               "against the public DCASE-2022/2023 baseline implementation on random inputs (maximum difference 2e-8; "
               "`analysis/diagnostics/check_against_public_dcase.py`), the frame preparation (10-ms outputs pooled to 100-ms frames, "
               "thresholding, slot order) is this repository's; they are not the official evaluation pipeline "
               "(definitions: `analysis/diagnostics/seld_variants.py`, `docs/CONVENTIONS.md` section 4.2)."]
    (BASE / "sensitivity_tables.md").write_text("\n".join(md) + "\n")
    with (BASE / "sensitivity_tables.csv").open("w", newline="") as fh:
        fields: List[str] = []
        for r in out_rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {BASE / 'sensitivity_tables.md'}")


if __name__ == "__main__":
    main()
