#!/usr/bin/env python3
"""
class_specific_threshold_calibration.py — Class-Specific SED Calibration
========================================================================
Validation-to-test threshold calibration for SELD checkpoints.

Thresholds are selected on a calibration split (normally STARSS23 validation)
and then frozen before evaluation on a separate split (normally test). The
script supports the NAS-SELD, standalone Conformer, and standalone ResNet
families through explicit repository presets.

Outputs
-------
  - class_thresholds.json / class_thresholds.csv
  - class_threshold_curves.csv / global_threshold_curve.csv
  - strategy_summary.csv
  - per_class_metrics.csv
  - diagnostic_summary.json
  - figures/diagnostics/class_threshold_calibration/<preset>/*.png|*.svg

The optional test-oracle strategy selects class thresholds directly on the
evaluation split. It is an optimistic diagnostic upper bound and must not be
reported as a calibrated test result.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.diagnostics.activation_test import (  # noqa: E402
    CLASS_NAMES,
    _ANGLE_THRESH,
    _TGT_THRESH,
    build_dataset,
    load_nas_model,
    load_standalone_model,
    load_yaml_cfg,
    run_inference,
    select_pool_indices,
)
from datasets.utils import SELDDataset  # noqa: E402
from experiments_nas_seld.loss import SELDMetrics  # noqa: E402


@dataclass(frozen=True)
class ModelPreset:
    """Static checkpoint/configuration preset for a supported model family."""

    family: str
    config: str
    checkpoint: str
    description: str


PRESETS: Dict[str, ModelPreset] = {
    "nas_expa": ModelPreset(
        family="nas",
        config="configs/loss_experiments/expA_nas_seld_balanced_starss23.yaml",
        checkpoint=(
            "runs/at2seld_nas/loss_experiments/balanced_starss23_expA/"
            "checkpoints/expA_nas_seld_balanced_starss23/34_0.3375.ckpt"
        ),
        description="NAS-SELD ExpA, BalancedSTARSS23, active-only MSE",
    ),
    "conformer_expa": ModelPreset(
        family="standalone",
        config="configs/loss_experiments/expA_standalone_conformer_balanced_starss23.yaml",
        checkpoint=(
            "runs/standalone_seld/loss_experiments_conformer/checkpoints/"
            "expA_standalone_conformer_balanced_starss23/22_0.4209.ckpt"
        ),
        description="Standalone Conformer ExpA, BalancedSTARSS23",
    ),
    "resnet_expa": ModelPreset(
        family="standalone",
        config="configs/loss_experiments/expA_standalone_resnet_balanced_starss23.yaml",
        checkpoint=(
            "runs/standalone_seld/loss_experiments_resnet/checkpoints/"
            "expA_standalone_resnet_balanced_starss23/14_0.4618.ckpt"
        ),
        description="Standalone ResNet ExpA, BalancedSTARSS23",
    ),
    "nas_mixed_focal_legacy": ModelPreset(
        family="nas",
        config="configs/experiments_nas-seld/foa/finetune/exp_ft_mixed_focal.yaml",
        checkpoint=(
            "runs/at2seld_nas/finetune/exp_ft_mixed_focal/checkpoints/"
            "exp_ft_mixed_focal/30_0.4702.ckpt"
        ),
        description="NAS-SELD mixed focal legacy reference",
    ),
    "conformer_native_cse": ModelPreset(
        family="standalone",
        config=(
            "configs/experiments_standalone_seld/foa/preliminary_conformer/"
            "conformer_starss23_cse.yaml"
        ),
        checkpoint=(
            "runs/standalone_seld/conformer_starss23_cse/checkpoints/"
            "conformer_starss23_cse/44_0.3329.ckpt"
        ),
        description="Best native STARSS23 Conformer (CSE)",
    ),
    "resnet_native": ModelPreset(
        family="standalone",
        config=(
            "configs/experiments_standalone_seld/foa/resnet_standalone/"
            "resnet_starss23.yaml"
        ),
        checkpoint=(
            "runs/standalone_seld/resnet_starss23/checkpoints/"
            "resnet_starss23/42_0.4671.ckpt"
        ),
        description="Best native STARSS23 standalone ResNet",
    ),
}


@dataclass
class BinaryMetrics:
    """Binary SED counts and derived metrics for one class or macro setting."""

    support: int
    predicted: int
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    balanced_accuracy: float
    pred_target_ratio: float


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _metric_from_counts(tp: int, fp: int, fn: int, tn: int) -> BinaryMetrics:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    return BinaryMetrics(
        support=tp + fn,
        predicted=tp + fp,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=precision,
        recall=recall,
        f1=f1,
        balanced_accuracy=0.5 * (recall + specificity),
        pred_target_ratio=_safe_div(tp + fp, tp + fn),
    )


def _collapse_activity(cache) -> Tuple[np.ndarray, np.ndarray]:
    """Return probabilities and targets with tracks collapsed by any()."""
    probs, targets = [], []
    for logits, target in zip(cache.act_logits_list, cache.targets_list):
        frames = min(logits.shape[1], target.shape[1])
        probs.append(torch.sigmoid(logits[:, :frames]).amax(dim=2).numpy())
        targets.append(
            (target[:, :frames].norm(dim=-1) > _TGT_THRESH).any(dim=2).numpy()
        )
    return (
        np.concatenate(probs, axis=0).reshape(-1, cache.n_classes),
        np.concatenate(targets, axis=0).reshape(-1, cache.n_classes).astype(bool),
    )


def _binary_metrics(
    probs: np.ndarray, targets: np.ndarray, thresholds: Sequence[float]
) -> List[BinaryMetrics]:
    threshold_array = np.asarray(thresholds, dtype=np.float32).reshape(1, -1)
    predictions = probs > threshold_array
    rows = []
    for class_idx in range(probs.shape[1]):
        pred = predictions[:, class_idx]
        tgt = targets[:, class_idx]
        tp = int(np.logical_and(pred, tgt).sum())
        fp = int(np.logical_and(pred, ~tgt).sum())
        fn = int(np.logical_and(~pred, tgt).sum())
        tn = int(np.logical_and(~pred, ~tgt).sum())
        rows.append(_metric_from_counts(tp, fp, fn, tn))
    return rows


def _score(metric: BinaryMetrics, objective: str) -> float:
    if objective == "f1":
        return metric.f1
    if objective == "precision":
        return metric.precision
    if objective == "recall":
        return metric.recall
    if objective == "balanced_accuracy":
        return metric.balanced_accuracy
    raise ValueError(f"Unsupported objective: {objective}")


def _select_class_thresholds(
    probs: np.ndarray,
    targets: np.ndarray,
    candidates: Sequence[float],
    objective: str,
    fallback_threshold: float,
    min_positive_frames: int,
) -> Tuple[List[float], List[dict]]:
    selected, curves = [], []
    for class_idx in range(probs.shape[1]):
        support = int(targets[:, class_idx].sum())
        candidates_with_metrics = []
        for threshold in candidates:
            metric = _binary_metrics(
                probs[:, [class_idx]], targets[:, [class_idx]], [threshold]
            )[0]
            curves.append(
                {
                    "class_id": class_idx,
                    "threshold": threshold,
                    **asdict(metric),
                    "objective": objective,
                    "objective_value": _score(metric, objective),
                }
            )
            candidates_with_metrics.append((threshold, metric))

        if support < min_positive_frames:
            selected.append(float(fallback_threshold))
            continue

        # Prefer the threshold closest to the fallback when objective values tie.
        best_threshold, _ = max(
            candidates_with_metrics,
            key=lambda item: (
                _score(item[1], objective),
                -abs(item[0] - fallback_threshold),
                item[0],
            ),
        )
        selected.append(float(best_threshold))
    return selected, curves


def _select_global_threshold(
    probs: np.ndarray,
    targets: np.ndarray,
    candidates: Sequence[float],
    objective: str,
    fallback_threshold: float,
) -> Tuple[float, List[dict]]:
    rows = []
    for threshold in candidates:
        metrics = _binary_metrics(
            probs, targets, [threshold] * probs.shape[1]
        )
        macro_value = float(np.mean([_score(metric, objective) for metric in metrics]))
        rows.append(
            {
                "threshold": threshold,
                "macro_objective": macro_value,
                "macro_f1": float(np.mean([metric.f1 for metric in metrics])),
                "macro_precision": float(
                    np.mean([metric.precision for metric in metrics])
                ),
                "macro_recall": float(np.mean([metric.recall for metric in metrics])),
            }
        )
    best = max(
        rows,
        key=lambda row: (
            row["macro_objective"],
            -abs(row["threshold"] - fallback_threshold),
            row["threshold"],
        ),
    )
    return float(best["threshold"]), rows


def _adjust_logits_for_thresholds(
    logits: torch.Tensor, thresholds: Sequence[float]
) -> torch.Tensor:
    """Map per-class thresholds to an equivalent scalar threshold of 0.5."""
    threshold_tensor = torch.as_tensor(
        thresholds, dtype=logits.dtype, device=logits.device
    ).clamp(1e-6, 1.0 - 1e-6)
    offsets = torch.logit(threshold_tensor).view(1, 1, 1, -1)
    return logits - offsets


def _evaluate_seld(cache, thresholds: Sequence[float]) -> Dict[str, float]:
    metrics = SELDMetrics(angle_thresh=_ANGLE_THRESH, act_thresh=0.5)
    for logits, doa, target in zip(
        cache.act_logits_list, cache.doa_tanh_list, cache.targets_list
    ):
        metrics.update(_adjust_logits_for_thresholds(logits, thresholds), doa, target)
    return metrics.compute()


def _build_loader(
    dataset, indices: Sequence[int], batch_size: int, num_workers: int
) -> DataLoader:
    subset = Subset(dataset, list(indices))
    windowed = getattr(dataset, "windowed", False)
    collate = None if windowed else SELDDataset.collate_fn
    return DataLoader(
        subset,
        batch_size=batch_size if windowed else 1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
    )


def _run_split(
    model,
    cfg: dict,
    model_family: str,
    model_name: str,
    dataset_name: str,
    split: str,
    root: Path,
    pool_size: int,
    pool_policy: str,
    pool_seed: int,
    val_seed: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    orig_sr: int,
):
    dataset = build_dataset(dataset_name, split, root, cfg, val_seed=val_seed)
    indices = select_pool_indices(dataset, pool_size, pool_policy, pool_seed)
    loader = _build_loader(dataset, indices, batch_size, num_workers)
    hop_length = int(cfg.get("input", {}).get("hop_length", 320))
    cache = run_inference(
        model,
        loader,
        device,
        orig_sr,
        hop_length,
        model_family == "nas",
        f"{model_name}:{split}",
    )
    return cache, len(dataset), len(indices)


def _strategy_rows(
    cache,
    strategies: Dict[str, Sequence[float]],
    class_names: Sequence[str],
) -> Tuple[List[dict], List[dict]]:
    probs, targets = _collapse_activity(cache)
    summary_rows, per_class_rows = [], []
    for strategy, thresholds in strategies.items():
        binary = _binary_metrics(probs, targets, thresholds)
        seld = _evaluate_seld(cache, thresholds)
        summary_rows.append(
            {
                "strategy": strategy,
                "macro_f1": float(np.mean([row.f1 for row in binary])),
                "macro_precision": float(
                    np.mean([row.precision for row in binary])
                ),
                "macro_recall": float(np.mean([row.recall for row in binary])),
                "macro_balanced_accuracy": float(
                    np.mean([row.balanced_accuracy for row in binary])
                ),
                "micro_f1": _metric_from_counts(
                    sum(row.tp for row in binary),
                    sum(row.fp for row in binary),
                    sum(row.fn for row in binary),
                    sum(row.tn for row in binary),
                ).f1,
                **seld,
            }
        )
        for class_idx, (threshold, row) in enumerate(zip(thresholds, binary)):
            per_class_rows.append(
                {
                    "strategy": strategy,
                    "class_id": class_idx,
                    "class_name": class_names[class_idx],
                    "threshold": threshold,
                    **asdict(row),
                }
            )
    return summary_rows, per_class_rows


def _write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    import csv

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _save_plots(
    figures_dir: Path,
    class_names: Sequence[str],
    thresholds: Sequence[float],
    summary_rows: Sequence[dict],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib unavailable: plots were not generated.")
        return

    figures_dir.mkdir(parents=True, exist_ok=True)
    short_names = [name.replace(" ", "\n") for name in class_names]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    positions = np.arange(len(class_names))
    ax.bar(positions, thresholds, color="#35618f")
    ax.axhline(0.5, color="#b34a4a", linestyle="--", linewidth=1.2, label="fixed 0.5")
    ax.set_xticks(positions, short_names, fontsize=8)
    ax.set_ylim(0, 0.55)
    ax.set_ylabel("Validation-selected threshold")
    ax.set_title("Class-specific SED activation thresholds")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(figures_dir / f"threshold_profile.{suffix}", dpi=200)
    plt.close(fig)

    strategies = [row["strategy"] for row in summary_rows]
    metrics = ["macro_f1", "F_20", "SELD_score"]
    labels = ["SED macro-F1", r"$F_{20^\circ}$", "SELD score"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, metric, label in zip(axes, metrics, labels):
        values = [float(row.get(metric, math.nan)) for row in summary_rows]
        ax.bar(strategies, values, color=["#777777", "#4f83b8", "#35618f", "#b36b3c"][:len(values)])
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(figures_dir / f"strategy_comparison.{suffix}", dpi=200)
    plt.close(fig)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA unavailable; using CPU.")
        requested = "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        warnings.warn("MPS unavailable; using CPU.")
        requested = "cpu"
    return torch.device(requested)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a calibration/evaluation run."""

    parser = argparse.ArgumentParser(
        description="Validation-to-test class-specific SED threshold calibration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preset", required=True, choices=sorted(PRESETS))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", default="balanced_starss23",
                        choices=["starss23", "balanced_starss23"])
    parser.add_argument("--calibration-split", default="val", choices=["train", "val"])
    parser.add_argument("--evaluation-split", default="test", choices=["val", "test"])
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[float(round(value, 2)) for value in np.arange(0.05, 1.0, 0.05)],
        help="Candidate thresholds used for global and per-class calibration",
    )
    parser.add_argument(
        "--objective",
        default="f1",
        choices=["f1", "precision", "recall", "balanced_accuracy"],
    )
    parser.add_argument("--fallback-threshold", type=float, default=0.5)
    parser.add_argument("--min-positive-frames", type=int, default=100)
    parser.add_argument("--calibration-pool-size", type=int, default=0)
    parser.add_argument("--evaluation-pool-size", type=int, default=0)
    parser.add_argument("--sample-policy", default="random",
                        choices=["first", "random", "balanced-active"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--include-test-oracle", action="store_true")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip PNG/SVG generation")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "diagnostics" / "class_threshold_calibration",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=ROOT / "figures" / "diagnostics" / "class_threshold_calibration",
    )
    return parser.parse_args()


def main() -> None:
    """Run threshold selection, frozen evaluation, and artifact serialization."""

    args = parse_args()
    args.thresholds = sorted(set(args.thresholds))
    preset = PRESETS[args.preset]
    config_path = ROOT / preset.config
    checkpoint_path = ROOT / preset.checkpoint
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    output_dir = args.output_dir / args.preset
    figures_dir = args.figures_dir / args.preset
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    print(f"Preset: {args.preset} - {preset.description}")
    print(f"Device: {device}")
    print(f"Calibration: {args.calibration_split}; evaluation: {args.evaluation_split}")

    cfg = load_yaml_cfg(config_path)
    if preset.family == "nas":
        model, cfg, orig_sr = load_nas_model(config_path, checkpoint_path, device)
    else:
        model, cfg, orig_sr = load_standalone_model(config_path, checkpoint_path, device)

    calibration_cache, calibration_size, calibration_used = _run_split(
        model, cfg, preset.family, args.preset, args.dataset,
        args.calibration_split, args.root, args.calibration_pool_size,
        args.sample_policy, args.seed, args.val_seed, args.batch_size,
        args.num_workers, device, orig_sr,
    )
    evaluation_cache, evaluation_size, evaluation_used = _run_split(
        model, cfg, preset.family, args.preset, args.dataset,
        args.evaluation_split, args.root, args.evaluation_pool_size,
        args.sample_policy, args.seed + 1, args.val_seed, args.batch_size,
        args.num_workers, device, orig_sr,
    )

    class_names = CLASS_NAMES[args.dataset]
    calibration_probs, calibration_targets = _collapse_activity(calibration_cache)
    global_threshold, global_curve = _select_global_threshold(
        calibration_probs, calibration_targets, args.thresholds,
        args.objective, args.fallback_threshold,
    )
    class_thresholds, class_curves = _select_class_thresholds(
        calibration_probs, calibration_targets, args.thresholds,
        args.objective, global_threshold, args.min_positive_frames,
    )

    strategies: Dict[str, Sequence[float]] = {
        "fixed_0.5": [0.5] * len(class_names),
        "global_validation": [global_threshold] * len(class_names),
        "class_validation": class_thresholds,
    }
    oracle_thresholds = None
    if args.include_test_oracle:
        evaluation_probs, evaluation_targets = _collapse_activity(evaluation_cache)
        oracle_thresholds, oracle_curves = _select_class_thresholds(
            evaluation_probs, evaluation_targets, args.thresholds,
            args.objective, global_threshold, args.min_positive_frames,
        )
        strategies["class_test_oracle"] = oracle_thresholds
        for row in oracle_curves:
            row["selection_split"] = args.evaluation_split
        class_curves.extend(oracle_curves)

    for row in class_curves:
        row.setdefault("selection_split", args.calibration_split)
    for row in global_curve:
        row["selection_split"] = args.calibration_split

    summary_rows, per_class_rows = _strategy_rows(
        evaluation_cache, strategies, class_names
    )

    threshold_rows = []
    calibration_support = calibration_targets.sum(axis=0)
    for class_idx, threshold in enumerate(class_thresholds):
        threshold_rows.append(
            {
                "class_id": class_idx,
                "class_name": class_names[class_idx],
                "threshold": threshold,
                "calibration_positive_frames": int(calibration_support[class_idx]),
                "used_fallback": bool(
                    calibration_support[class_idx] < args.min_positive_frames
                ),
            }
        )

    _write_csv(output_dir / "class_thresholds.csv", threshold_rows)
    _write_csv(output_dir / "class_threshold_curves.csv", class_curves)
    _write_csv(output_dir / "global_threshold_curve.csv", global_curve)
    _write_csv(output_dir / "strategy_summary.csv", summary_rows)
    _write_csv(output_dir / "per_class_metrics.csv", per_class_rows)

    threshold_payload = {
        "preset": args.preset,
        "objective": args.objective,
        "selection_split": args.calibration_split,
        "global_threshold": global_threshold,
        "class_thresholds": {
            name: threshold for name, threshold in zip(class_names, class_thresholds)
        },
    }
    if oracle_thresholds is not None:
        threshold_payload["test_oracle_thresholds"] = {
            name: threshold for name, threshold in zip(class_names, oracle_thresholds)
        }
    with (output_dir / "class_thresholds.json").open("w") as handle:
        json.dump(threshold_payload, handle, indent=2, default=_json_default)

    diagnostic = {
        "preset": args.preset,
        "preset_spec": asdict(preset),
        "run_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "calibration_dataset_size": calibration_size,
        "calibration_items_used": calibration_used,
        "evaluation_dataset_size": evaluation_size,
        "evaluation_items_used": evaluation_used,
        "global_threshold": global_threshold,
        "class_thresholds": class_thresholds,
        "strategy_results": summary_rows,
        "test_oracle_is_diagnostic_only": args.include_test_oracle,
    }
    with (output_dir / "diagnostic_summary.json").open("w") as handle:
        json.dump(diagnostic, handle, indent=2, default=_json_default)

    if not args.no_plots:
        _save_plots(figures_dir, class_names, class_thresholds, summary_rows)

    print("\nSelected validation thresholds:")
    for row in threshold_rows:
        marker = " (fallback)" if row["used_fallback"] else ""
        print(f"  {row['class_name']:<22} {row['threshold']:.2f}{marker}")
    print("\nEvaluation strategies:")
    for row in summary_rows:
        print(
            f"  {row['strategy']:<20} "
            f"macro-F1={row['macro_f1']:.4f} "
            f"F20={row['F_20']:.4f} SELD={row['SELD_score']:.4f}"
        )
    print(f"\nOutputs: {output_dir}")
    if not args.no_plots:
        print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    main()
