#!/usr/bin/env python3
"""
summarize_dataset_generalization.py — Cross-Dataset Aggregate Assets
====================================================================
Aggregate generalization runs into CSV summaries and diagnostic heatmaps.

The script consumes completed per-pair outputs from
``dataset_generalization_analysis.py`` and rebuilds compact machine-readable
matrices plus publication-quality diagnostic heatmaps.  It does not run model
inference or mutate training outputs.

Outputs
-------
  - all_test_metrics.csv / all_test_per_class_metrics.csv
  - cross_domain_*_matrix.csv files
  - native_domain_summary.csv / threshold_tolerance_summary.csv
  - figures/diagnostics/dataset_generalization/aggregate/*.png|*.svg

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(__file__).with_name(
    "dataset_generalization_manifest.json"
)


def load_runs(root: Path, allowed_models: set[str],
              allowed_datasets: set[str]):
    """Load completed model/dataset run directories into aggregate frames."""

    metric_frames = []
    class_frames = []
    coverage_rows = []
    for run_dir in sorted(root.glob("*__on__*")):
        metadata_path = run_dir / "run_metadata.json"
        metrics_path = run_dir / "test_val_selected_metrics.csv"
        if not (metadata_path.exists() and metrics_path.exists()):
            continue
        metadata = json.loads(metadata_path.read_text())
        if (
            metadata["model_name"] not in allowed_models
            or metadata["dataset_space"] not in allowed_datasets
        ):
            continue
        metrics = pd.read_csv(metrics_path)
        metrics.insert(0, "model", metadata["model_name"])
        metrics.insert(1, "train_space", metadata["model_space"])
        metrics.insert(2, "test_dataset", metadata["dataset_space"])
        metric_frames.append(metrics)

        class_path = run_dir / "test_val_selected_per_class.csv"
        if class_path.exists():
            frame = pd.read_csv(class_path)
            frame.insert(0, "model", metadata["model_name"])
            frame.insert(1, "train_space", metadata["model_space"])
            frame.insert(2, "test_dataset", metadata["dataset_space"])
            class_frames.append(frame)

        coverage = json.loads(
            (run_dir / "mapping_coverage.json").read_text()
        )
        coverage_rows.append({
            "model": metadata["model_name"],
            "train_space": metadata["model_space"],
            "test_dataset": metadata["dataset_space"],
            "mapped_source_classes": coverage["mapped_source_classes"],
            "source_classes": coverage["source_classes"],
            "covered_destination_classes":
                coverage["covered_destination_classes"],
            "destination_classes": coverage["destination_classes"],
        })
    return (
        pd.concat(metric_frames, ignore_index=True)
        if metric_frames else pd.DataFrame(),
        pd.concat(class_frames, ignore_index=True)
        if class_frames else pd.DataFrame(),
        pd.DataFrame(coverage_rows),
    )


DATASET_ORDER = [
    "starss23",
    "tau2019",
    "tau_nigens2020",
    "tau_nigens2021",
]

DATASET_PLOT_LABELS = {
    "starss23": "STARSS23",
    "tau2019": "TAU2019",
    "tau_nigens2020": "TAU-NIGENS2020",
    "tau_nigens2021": "TAU-NIGENS2021",
}


def heatmap(frame: pd.DataFrame, protocol: str, metric: str,
            colorbar_label: str, path: Path) -> None:
    """Render a protocol-specific train-by-test heatmap as PNG and SVG."""

    subset = frame[frame["protocol"] == protocol]
    if subset.empty:
        return
    pivot = subset.pivot(
        index="train_space", columns="test_dataset", values=metric
    )
    row_order = [dataset for dataset in DATASET_ORDER if dataset in pivot.index]
    column_order = [
        dataset for dataset in DATASET_ORDER if dataset in pivot.columns
    ]
    pivot = pivot.reindex(index=row_order, columns=column_order)
    pivot.index = [DATASET_PLOT_LABELS[name] for name in pivot.index]
    pivot.columns = [DATASET_PLOT_LABELS[name] for name in pivot.columns]

    figure_width = 10.5
    figure_height = 5.4
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="Greys",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="0.75",
        cbar_kws={"label": colorbar_label},
        annot_kws={"fontsize": 10},
        ax=ax,
    )
    for annotation, value in zip(ax.texts, pivot.to_numpy().ravel()):
        annotation.set_color("white" if value >= 0.55 else "black")

    ax.set_xlabel("Test Dataset")
    ax.set_ylabel("Training Dataset")
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=20, ha="right", rotation_mode="anchor"
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.tick_params(axis="both", labelsize=10)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def save_compact_assets(frame: pd.DataFrame, output: Path) -> None:
    """Write compact native-domain, cross-domain, and tolerance CSV assets."""

    standard = frame[
        frame["protocol"] == "val_selected_global_seld_score_at_20deg"
    ].copy()
    native = standard[
        standard["train_space"] == standard["test_dataset"]
    ].copy()
    native_columns = [
        "model", "test_dataset", "threshold", "sed_precision", "sed_recall",
        "sed_f1", "seld_f1", "le", "lr", "seld_score",
    ]
    native[native_columns].to_csv(
        output / "native_domain_summary.csv", index=False
    )

    for metric in ("seld_score", "sed_f1", "seld_f1", "le", "lr"):
        matrix = standard.pivot(
            index="model", columns="test_dataset", values=metric
        )
        matrix.to_csv(output / f"cross_domain_{metric}_matrix.csv")

    joint = frame[
        frame["protocol"] == "val_selected_global_joint_seld_score"
    ][
        ["model", "test_dataset", "threshold", "angle_deg", "sed_f1",
         "seld_f1", "le", "lr", "seld_score"]
    ].copy()
    standard_for_merge = standard[
        ["model", "test_dataset", "threshold", "seld_f1", "le", "lr",
         "seld_score"]
    ].copy()
    tolerance = standard_for_merge.merge(
        joint, on=["model", "test_dataset"], suffixes=("_20deg", "_joint")
    )
    tolerance["delta_seld_f1"] = (
        tolerance["seld_f1_joint"] - tolerance["seld_f1_20deg"]
    )
    tolerance["delta_seld_score"] = (
        tolerance["seld_score_joint"] - tolerance["seld_score_20deg"]
    )
    tolerance.to_csv(
        output / "threshold_tolerance_summary.csv", index=False
    )


def save_filtered_profiles(input_root: Path, output: Path,
                           allowed_models: set[str]) -> None:
    """Copy profiling rows that correspond to the aggregated model set."""

    profile_dir = input_root / "profiling"
    csv_path = profile_dir / "nas_seld_branch_profiles.csv"
    json_path = profile_dir / "nas_seld_branch_profiles.json"
    if csv_path.exists():
        profile = pd.read_csv(csv_path)
        profile = profile[profile["model"].isin(allowed_models)].copy()
        profile.to_csv(output / csv_path.name, index=False)
    if json_path.exists():
        rows = json.loads(json_path.read_text())
        rows = [row for row in rows if row.get("model") in allowed_models]
        (output / json_path.name).write_text(json.dumps(rows, indent=2))


def main() -> None:
    """Aggregate completed generalization runs and regenerate heatmaps."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root", type=Path,
        default=ROOT / "results" / "diagnostics"
        / "dataset_generalization",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=ROOT / "figures" / "diagnostics"
        / "dataset_generalization" / "aggregate",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--allow-incomplete", action="store_true",
        help="Aggregate available ExpA pairs instead of requiring the full matrix.",
    )
    args = parser.parse_args()
    output = args.output_dir or args.input_root / "aggregate"
    output.mkdir(parents=True, exist_ok=True)
    figures = args.figures_dir
    figures.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text())
    allowed_models = set(manifest["models"])
    allowed_datasets = set(manifest["datasets"])
    expected_pairs = {
        (model, dataset)
        for model in allowed_models
        for dataset in allowed_datasets
    }

    metrics, classes, coverage = load_runs(
        args.input_root, allowed_models, allowed_datasets
    )
    if metrics.empty:
        raise SystemExit(f"No completed runs found under {args.input_root}")
    actual_pairs = set(zip(metrics["model"], metrics["test_dataset"]))
    missing_pairs = sorted(expected_pairs - actual_pairs)
    if missing_pairs and not args.allow_incomplete:
        missing = "\n".join(
            f"  - {model} on {dataset}" for model, dataset in missing_pairs
        )
        raise SystemExit(
            "The ExpA matrix is incomplete. Missing combinations:\n" + missing
        )
    metrics.to_csv(output / "all_test_metrics.csv", index=False)
    classes.to_csv(output / "all_test_per_class_metrics.csv", index=False)
    coverage.to_csv(output / "mapping_coverage_matrix.csv", index=False)
    save_compact_assets(metrics, output)
    save_filtered_profiles(args.input_root, output, allowed_models)

    heatmap(
        metrics, "val_selected_global_seld_score_at_20deg", "seld_score",
        "SELD Score", figures / "seld_score_heatmap",
    )
    heatmap(
        metrics, "val_selected_global_sed_f1_at_20deg", "sed_f1",
        "SED F1", figures / "sed_f1_heatmap",
    )
    heatmap(
        metrics, "val_selected_global_seld_f1_at_20deg", "seld_f1",
        r"$F_{20^\circ}$", figures / "seld_f1_heatmap",
    )
    print(f"Saved aggregate CSV/JSON assets to {output}")
    print(f"Saved aggregate figures to {figures}")


if __name__ == "__main__":
    main()
