#!/usr/bin/env python3
"""
summarize_class_threshold_calibration.py — Threshold Calibration Aggregates
==========================================================================
Aggregate class-specific SED threshold-calibration experiments.

The script reads outputs produced by ``class_specific_threshold_calibration.py``
and writes machine-readable CSV summaries only. Threshold optima are always
selected on the validation split; test curves are never used for selection.

Outputs
-------
  - threshold_objective_long.csv
  - threshold_objective_matrix.csv
  - threshold_objective_metrics.csv
  - threshold_objective_global_metrics.csv
  - strategy_comparison_all.csv

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable


PRESET_LABELS = {
    "nas_expa": "NAS-SELD ExpA",
    "conformer_expa": "Conformer ExpA",
    "resnet_expa": "ResNet ExpA",
    "nas_mixed_focal_legacy": "NAS-SELD mixed focal",
    "conformer_native_cse": "Conformer native CSE",
    "resnet_native": "ResNet native",
}

CLASS_LABELS = [
    ("Female speech", "Fem."),
    ("Male speech", "Male"),
    ("Clapping", "Clap."),
    ("Telephone", "Tel."),
    ("Laughter", "Laugh."),
    ("Domestic sounds", "Dom."),
    ("Walk, footsteps", "Foot."),
    ("Door open/close", "Door"),
    ("Music", "Music"),
    ("Musical instrument", "Instr."),
    ("Water tap", "Water"),
    ("Bell", "Bell"),
    ("Knock", "Knock"),
]

OBJECTIVES = ("precision", "recall", "f1")


def parse_args() -> argparse.Namespace:
    """Parse aggregate-calibration CLI options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "results"
        / "diagnostics"
        / "class_threshold_calibration",
        help="Directory containing one result directory per preset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Aggregate output directory (default: RESULTS_ROOT/aggregate).",
    )
    parser.add_argument(
        "--presets",
        nargs="+",
        choices=list(PRESET_LABELS),
        default=list(PRESET_LABELS),
        help="Preset result directories to include, in presentation order.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file as a list of dictionaries."""

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str) -> float:
    """Convert CSV string fields to floats, preserving missing values as NaN."""

    return float(value) if value not in ("", None) else math.nan


def select_best(rows: Iterable[dict[str, str]], metric: str) -> dict[str, str]:
    """Match the calibration script's deterministic tie-breaking policy."""
    candidates = list(rows)
    if not candidates:
        raise ValueError(f"No candidates available for metric {metric}")
    return max(
        candidates,
        key=lambda row: (
            as_float(row[metric]),
            -abs(as_float(row["threshold"]) - 0.5),
            as_float(row["threshold"]),
        ),
    )


def load_class_names(result_dir: Path) -> list[str]:
    """Read class names from one preset's threshold summary."""

    rows = read_csv(result_dir / "class_thresholds.csv")
    return [row["class_name"] for row in rows]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """Write rows to CSV with an explicit field order."""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_threshold(value: str | float) -> str:
    """Format thresholds for compact matrix fields."""

    if value in ("", None):
        return "--"
    return f"{float(value):.2f}"


def find_threshold_row(
    rows: Iterable[dict[str, str]], threshold: float
) -> dict[str, str]:
    """Find the metric row corresponding to a selected threshold."""

    for row in rows:
        if math.isclose(as_float(row["threshold"]), threshold, abs_tol=1e-9):
            return row
    raise ValueError(f"Threshold {threshold} is not available in the curve")


def select_global_objective_threshold(
    rows: list[dict[str, str]], metric: str
) -> float:
    """Select the best global threshold for one aggregate metric."""

    thresholds = sorted({as_float(row["threshold"]) for row in rows})
    candidates = []
    for threshold in thresholds:
        threshold_rows = [
            row
            for row in rows
            if math.isclose(as_float(row["threshold"]), threshold, abs_tol=1e-9)
        ]
        candidates.append(
            {
                "threshold": threshold,
                metric: sum(as_float(row[metric]) for row in threshold_rows)
                / len(threshold_rows),
            }
        )
    return as_float(select_best(candidates, metric)["threshold"])


def format_triplet(row: dict, prefix: str) -> list[str]:
    """Return precision, recall, and F1 strings for a metric prefix."""

    return [
        f'{float(row[f"{prefix}_precision"]):.3f}',
        f'{float(row[f"{prefix}_recall"]):.3f}',
        f'{float(row[f"{prefix}_f1"]):.3f}',
    ]


def main() -> None:
    """Aggregate all requested calibration preset directories."""

    args = parse_args()
    results_root = args.results_root.resolve()
    output_dir = (args.output_dir or results_root / "aggregate").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    preset_dirs = [results_root / preset for preset in args.presets]
    missing = [
        path for path in preset_dirs
        if not (path / "class_threshold_curves.csv").exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing calibration outputs: " + ", ".join(str(path) for path in missing)
        )

    long_rows: list[dict] = []
    matrix_rows: list[dict] = []
    detailed_rows: list[dict] = []
    objective_global_rows: list[dict] = []
    strategy_rows: list[dict] = []

    for result_dir in preset_dirs:
        preset = result_dir.name
        class_names = load_class_names(result_dir)
        expected_names = [name for name, _ in CLASS_LABELS]
        if class_names != expected_names:
            raise ValueError(
                f"{preset}: unexpected class order {class_names}; "
                f"expected {expected_names}"
            )

        all_curves = read_csv(result_dir / "class_threshold_curves.csv")
        curves = [
            row
            for row in all_curves
            if row.get("selection_split") == "val"
        ]
        test_curves = [
            row for row in all_curves if row.get("selection_split") == "test"
        ]
        by_class = {
            class_name: [
                row for row in curves if int(row["class_id"]) == class_id
            ]
            for class_id, class_name in enumerate(class_names)
        }
        test_by_class = {
            class_name: [
                row for row in test_curves if int(row["class_id"]) == class_id
            ]
            for class_id, class_name in enumerate(class_names)
        }
        global_fallbacks = {
            objective: select_global_objective_threshold(curves, objective)
            for objective in OBJECTIVES
        }

        best_by_class: dict[str, dict[str, dict[str, str]]] = {}
        for class_name in class_names:
            support = int(float(by_class[class_name][0]["support"]))
            best_by_class[class_name] = (
                {
                    objective: select_best(by_class[class_name], objective)
                    for objective in OBJECTIVES
                }
                if support > 0
                else {}
            )
            row = {
                "preset": preset,
                "model_label": PRESET_LABELS[preset],
                "class_name": class_name,
                "support": support,
            }
            for objective in OBJECTIVES:
                best = best_by_class[class_name].get(objective)
                threshold = (
                    as_float(best["threshold"])
                    if best
                    else global_fallbacks[objective]
                )
                val_result = find_threshold_row(by_class[class_name], threshold)
                test_result = find_threshold_row(test_by_class[class_name], threshold)
                row[f"{objective}_threshold"] = threshold
                row[f"{objective}_value"] = val_result[objective]
                detailed_rows.append(
                    {
                        "preset": preset,
                        "model_label": PRESET_LABELS[preset],
                        "objective": objective,
                        "class_name": class_name,
                        "threshold": threshold,
                        "used_fallback": not bool(best),
                        "val_precision": val_result["precision"],
                        "val_recall": val_result["recall"],
                        "val_f1": val_result["f1"],
                        "test_precision": test_result["precision"],
                        "test_recall": test_result["recall"],
                        "test_f1": test_result["f1"],
                    }
                )
            long_rows.append(row)

        for objective in OBJECTIVES:
            row = {
                "preset": preset,
                "model_label": PRESET_LABELS[preset],
                "objective": objective,
            }
            for class_name in class_names:
                best = best_by_class[class_name].get(objective)
                row[class_name] = (
                    best["threshold"] if best else global_fallbacks[objective]
                )
            matrix_rows.append(row)
            selected = [
                row
                for row in detailed_rows
                if row["preset"] == preset and row["objective"] == objective
            ]
            objective_global_rows.append(
                {
                    "preset": preset,
                    "model_label": PRESET_LABELS[preset],
                    "objective": objective,
                    **{
                        f"{split}_{metric}": sum(
                            float(row[f"{split}_{metric}"]) for row in selected
                        )
                        / len(selected)
                        for split in ("val", "test")
                        for metric in ("precision", "recall", "f1")
                    },
                }
            )

        summary = {
            row["strategy"]: row
            for row in read_csv(result_dir / "strategy_summary.csv")
        }
        diagnostic = json.loads(
            (result_dir / "diagnostic_summary.json").read_text(encoding="utf-8")
        )
        fixed = summary["fixed_0.5"]
        global_validation = summary["global_validation"]
        class_validation = summary["class_validation"]
        strategy_rows.append(
            {
                "preset": preset,
                "model_label": PRESET_LABELS[preset],
                "global_threshold": diagnostic["global_threshold"],
                "fixed_precision": fixed["macro_precision"],
                "fixed_recall": fixed["macro_recall"],
                "fixed_f1": fixed["macro_f1"],
                "fixed_seld": fixed["SELD_score"],
                "global_precision": global_validation["macro_precision"],
                "global_recall": global_validation["macro_recall"],
                "global_f1": global_validation["macro_f1"],
                "global_seld": global_validation["SELD_score"],
                "class_precision": class_validation["macro_precision"],
                "class_recall": class_validation["macro_recall"],
                "class_f1": class_validation["macro_f1"],
                "class_seld": class_validation["SELD_score"],
            }
        )

    long_fields = [
        "preset",
        "model_label",
        "class_name",
        "support",
        "precision_threshold",
        "precision_value",
        "recall_threshold",
        "recall_value",
        "f1_threshold",
        "f1_value",
    ]
    matrix_fields = [
        "preset",
        "model_label",
        "objective",
        *[name for name, _ in CLASS_LABELS],
    ]
    strategy_fields = list(strategy_rows[0])
    write_csv(output_dir / "threshold_objective_long.csv", long_fields, long_rows)
    write_csv(output_dir / "threshold_objective_matrix.csv", matrix_fields, matrix_rows)
    write_csv(
        output_dir / "threshold_objective_metrics.csv",
        list(detailed_rows[0]),
        detailed_rows,
    )
    write_csv(
        output_dir / "threshold_objective_global_metrics.csv",
        list(objective_global_rows[0]),
        objective_global_rows,
    )
    write_csv(output_dir / "strategy_comparison_all.csv", strategy_fields, strategy_rows)

    print(f"Aggregated {len(preset_dirs)} presets into {output_dir}")


if __name__ == "__main__":
    main()
