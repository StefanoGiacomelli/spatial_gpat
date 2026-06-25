#!/usr/bin/env python3
"""
dataset_generalization_doa_oracle.py — Coverage-Aware DOA Oracle
=================================================================
Coverage-aware DOA-oracle analysis for the NAS-SELD ExpA matrix.

The script evaluates all requested model/dataset combinations without training.
Ground-truth activity gates the DOA analysis. The primary alignment selects the
track permutation that minimizes angular error only on active target slots,
while a secondary tPIT-aligned result preserves comparability with the training
pipeline and earlier native diagnostics. Dataset targets are projected into the
checkpoint label space with the mappings defined by
``dataset_generalization_analysis.py``.

Each completed pair is cached as a compressed NPZ file. Aggregate CSV,
per-class CSV, metric matrices, and diagnostic heatmaps are rebuilt after every
pair, so interrupted matrix runs can be resumed without losing results.

Outputs
-------
  - compressed pair caches under results/diagnostics/.../doa_oracle/cache
  - aggregate, per-class, coverage, and matrix CSV files
  - diagnostic heatmaps under figures/diagnostics/.../doa_oracle

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.diagnostics.activation_test import (  # noqa: E402
    load_yaml_cfg,
    select_pool_indices,
)
from analysis.diagnostics.dataset_generalization_analysis import (  # noqa: E402
    LABELS,
    build_dataset,
    build_mapping,
    mapping_report,
    normalize_space,
    project_target,
    resolve_device,
)

DEFAULT_MANIFEST = Path(__file__).with_name(
    "dataset_generalization_manifest.json"
)
DEFAULT_OUTPUT = (
    ROOT / "results" / "diagnostics"
    / "dataset_generalization" / "aggregate" / "doa_oracle"
)
DEFAULT_FIGURES = (
    ROOT / "figures" / "diagnostics"
    / "dataset_generalization" / "aggregate" / "doa_oracle"
)

DATASET_ORDER = [
    "starss23",
    "tau2019",
    "tau_nigens2020",
    "tau_nigens2021",
]
DATASET_LABELS = {
    "starss23": "STARSS23",
    "tau2019": "TAU2019",
    "tau_nigens2020": "TAU-NIGENS2020",
    "tau_nigens2021": "TAU-NIGENS2021",
}
MODEL_TO_SPACE = {
    "nas_expa_starss23": "starss23",
    "nas_expa_tau2019": "tau2019",
    "nas_expa_tau_nigens2020": "tau_nigens2020",
    "nas_expa_tau_nigens2021": "tau_nigens2021",
}
MODEL_LABELS = {
    key: f"ExpA--{DATASET_LABELS[value]}"
    for key, value in MODEL_TO_SPACE.items()
}

ANGLE_THRESHOLDS = (5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 90.0)
ZERO_NORM_THRESHOLDS = (0.05, 0.10, 0.20, 0.50)
KNOWN_NATIVE_ACC20 = {
    ("nas_expa_starss23", "starss23"): 0.336703,
    ("nas_expa_tau2019", "tau2019"): 0.937122,
    ("nas_expa_tau_nigens2021", "tau_nigens2021"): 0.543059,
}


@dataclass
class PairCache:
    """Compressed per-pair DOA-oracle arrays and coverage metadata."""

    angular_error: np.ndarray
    pred_norm: np.ndarray
    tpit_angular_error: np.ndarray
    tpit_pred_norm: np.ndarray
    class_idx: np.ndarray
    track_idx: np.ndarray
    n_total_frames: int
    mapped_source_classes: int
    source_classes: int
    covered_destination_classes: int
    destination_classes: int


def absolute(path: str | Path) -> Path:
    """Resolve a path relative to the repository root when needed."""

    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically write a text file by replacing a temporary sibling."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def atomic_write_csv(path: Path, rows: Sequence[dict],
                     fieldnames: Sequence[str]) -> None:
    """Atomically write rows to a CSV file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_write_dataframe(path: Path, frame: pd.DataFrame) -> None:
    """Atomically write a pandas DataFrame to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def angular_error_deg(prediction: np.ndarray, target: np.ndarray,
                      eps: float = 1e-8) -> np.ndarray:
    """Return vector angular error in degrees for predicted and target DOAs."""

    pred_norm = np.linalg.norm(prediction, axis=-1, keepdims=True)
    target_norm = np.linalg.norm(target, axis=-1, keepdims=True)
    pred_unit = prediction / (pred_norm + eps)
    target_unit = target / (target_norm + eps)
    cosine = np.clip((pred_unit * target_unit).sum(axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(cosine)).astype(np.float32)


def align_doa_to_target(doa: torch.Tensor, permutation: torch.Tensor
                        ) -> torch.Tensor:
    """Apply a flattened track permutation to DOA predictions."""

    batch, frames, tracks, classes, _ = doa.shape
    flattened = doa.permute(0, 1, 3, 2, 4).reshape(
        batch * frames * classes, tracks, 3
    )
    aligned = torch.gather(
        flattened,
        1,
        permutation.unsqueeze(-1).expand(-1, -1, 3),
    )
    return aligned.reshape(
        batch, frames, classes, tracks, 3
    ).permute(0, 1, 3, 2, 4)


def oracle_doa_permutation(doa: torch.Tensor, target: torch.Tensor,
                           permutations: torch.Tensor,
                           active_eps: float) -> torch.Tensor:
    """Select the track permutation using only active-target angular error."""
    batch, frames, tracks, classes, _ = doa.shape
    flattened_doa = doa.permute(0, 1, 3, 2, 4).reshape(
        batch * frames * classes, tracks, 3
    )
    flattened_target = target.permute(0, 1, 3, 2, 4).reshape(
        batch * frames * classes, tracks, 3
    )
    candidate = flattened_doa[:, permutations]
    candidate = torch.nn.functional.normalize(candidate, dim=-1, eps=1e-8)
    reference = torch.nn.functional.normalize(
        flattened_target, dim=-1, eps=1e-8
    )
    cosine = torch.clamp(
        (candidate * reference.unsqueeze(1)).sum(dim=-1), -1.0, 1.0
    )
    angular_cost = torch.acos(cosine)
    active = flattened_target.norm(dim=-1) > active_eps
    active_expanded = active.unsqueeze(1).expand_as(angular_cost)
    denominator = active.sum(dim=-1).clamp(min=1).unsqueeze(1)
    cost = (
        (angular_cost * active_expanded.float()).sum(dim=-1)
        / denominator
    )
    best_index = cost.argmin(dim=-1)
    return permutations[best_index]


def load_module(config: Path, checkpoint: Path, device: torch.device):
    """Load a NAS-SELD Lightning checkpoint and prepare it for inference."""

    from experiments_nas_seld.train import NASLightningModule

    cfg = load_yaml_cfg(config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module = NASLightningModule.load_from_checkpoint(
            str(checkpoint), cfg=cfg, map_location="cpu"
        )
    module.model.eval().to(device)
    module.tpit_loss.eval().to(device)
    orig_sr = getattr(
        module, "orig_sr", cfg.get("input", {}).get("sr", 24000)
    )
    return module, cfg, orig_sr


def infer_pair(
    model_name: str,
    model_entry: dict,
    dataset_name: str,
    dataset_root: Path,
    split: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    pool_size: int,
    seed: int,
) -> tuple[PairCache, dict]:
    """Infer one model/dataset pair and return cache arrays plus metadata."""

    module, cfg, orig_sr = load_module(
        absolute(model_entry["config"]),
        absolute(model_entry["checkpoint"]),
        device,
    )
    model_space = normalize_space(cfg["semantic"]["dataset"])
    expected_space = MODEL_TO_SPACE.get(model_name)
    if expected_space and model_space != expected_space:
        raise RuntimeError(
            f"{model_name} resolves to {model_space}, expected {expected_space}"
        )

    dataset_space = normalize_space(dataset_name)
    mapping = build_mapping(dataset_space, model_space)
    coverage = mapping_report(dataset_space, model_space, mapping)
    dataset = build_dataset(
        dataset_space, split, dataset_root, cfg, val_seed=seed
    )
    indices = select_pool_indices(dataset, pool_size, "random", seed)
    subset = Subset(dataset, indices)
    from datasets.utils import SELDDataset

    collate = (
        None if getattr(dataset, "windowed", False)
        else SELDDataset.collate_fn
    )
    loader = DataLoader(
        subset,
        batch_size=batch_size if getattr(dataset, "windowed", False) else 1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
    )

    error_parts: list[np.ndarray] = []
    norm_parts: list[np.ndarray] = []
    tpit_error_parts: list[np.ndarray] = []
    tpit_norm_parts: list[np.ndarray] = []
    class_parts: list[np.ndarray] = []
    track_parts: list[np.ndarray] = []
    n_total_frames = 0
    model_classes = len(LABELS[model_space])
    model_tracks = int(cfg.get("heads", {}).get("n_tracks", 3))

    description = f"{model_name} on {dataset_space}"
    with torch.inference_mode(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for batch in tqdm(loader, desc=description, unit="batch"):
            audio = batch[0].to(device)
            source_target = batch[1].to(device)
            act_logits, doa, _ = module.model(audio, orig_sr=orig_sr)
            target = project_target(
                source_target, mapping, model_classes, model_tracks
            )
            frames = min(
                act_logits.shape[1], doa.shape[1], target.shape[1]
            )
            act_logits = act_logits[:, :frames]
            doa = doa[:, :frames]
            target = target[:, :frames]
            n_total_frames += int(target.shape[0] * frames)

            module.tpit_loss(act_logits, doa, target)
            tpit_permutation = module.tpit_loss.last_best_perm
            if tpit_permutation is None:
                raise RuntimeError("tPIT did not expose last_best_perm")
            tpit_aligned = align_doa_to_target(doa, tpit_permutation)
            doa_permutation = oracle_doa_permutation(
                doa,
                target,
                module.tpit_loss.perm_tensor,
                module.tpit_loss.doa_active_eps,
            )
            aligned = align_doa_to_target(doa, doa_permutation)
            active = target.norm(dim=-1) > module.tpit_loss.doa_active_eps
            if not active.any():
                continue

            active_indices = active.nonzero(as_tuple=False)
            predicted = aligned[active].detach().cpu().numpy()
            tpit_predicted = tpit_aligned[active].detach().cpu().numpy()
            reference = target[active].detach().cpu().numpy()
            error_parts.append(angular_error_deg(predicted, reference))
            tpit_error_parts.append(
                angular_error_deg(tpit_predicted, reference)
            )
            norm_parts.append(
                np.linalg.norm(predicted, axis=-1).astype(np.float32)
            )
            tpit_norm_parts.append(
                np.linalg.norm(tpit_predicted, axis=-1).astype(np.float32)
            )
            track_parts.append(
                active_indices[:, 2].detach().cpu().numpy().astype(np.int16)
            )
            class_parts.append(
                active_indices[:, 3].detach().cpu().numpy().astype(np.int16)
            )

    del module
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    if error_parts:
        angular_error = np.concatenate(error_parts)
        pred_norm = np.concatenate(norm_parts)
        tpit_angular_error = np.concatenate(tpit_error_parts)
        tpit_pred_norm = np.concatenate(tpit_norm_parts)
        class_idx = np.concatenate(class_parts)
        track_idx = np.concatenate(track_parts)
    else:
        angular_error = np.empty(0, dtype=np.float32)
        pred_norm = np.empty(0, dtype=np.float32)
        tpit_angular_error = np.empty(0, dtype=np.float32)
        tpit_pred_norm = np.empty(0, dtype=np.float32)
        class_idx = np.empty(0, dtype=np.int16)
        track_idx = np.empty(0, dtype=np.int16)

    cache = PairCache(
        angular_error=angular_error,
        pred_norm=pred_norm,
        tpit_angular_error=tpit_angular_error,
        tpit_pred_norm=tpit_pred_norm,
        class_idx=class_idx,
        track_idx=track_idx,
        n_total_frames=n_total_frames,
        mapped_source_classes=coverage["mapped_source_classes"],
        source_classes=coverage["source_classes"],
        covered_destination_classes=coverage[
            "covered_destination_classes"
        ],
        destination_classes=coverage["destination_classes"],
    )
    metadata = {
        "model_name": model_name,
        "model_space": model_space,
        "test_dataset": dataset_space,
        "split": split,
        "seed": seed,
        "pool_size": pool_size,
        "evaluated_items": len(indices),
        "config": str(absolute(model_entry["config"])),
        "checkpoint": str(absolute(model_entry["checkpoint"])),
        "dataset_root": str(dataset_root),
        "primary_alignment": "DOA-only angular-cost oracle permutation",
        "secondary_alignment": "checkpoint tPIT permutation",
        "activity_policy": "ground-truth oracle activity",
        "coverage_aware": True,
        "mapping": coverage,
    }
    return cache, metadata


def save_pair_cache(path: Path, cache: PairCache, metadata: dict) -> None:
    """Persist one pair cache as a compressed NPZ file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        angular_error=cache.angular_error,
        pred_norm=cache.pred_norm,
        tpit_angular_error=cache.tpit_angular_error,
        tpit_pred_norm=cache.tpit_pred_norm,
        class_idx=cache.class_idx,
        track_idx=cache.track_idx,
        n_total_frames=np.array(cache.n_total_frames, dtype=np.int64),
        mapped_source_classes=np.array(
            cache.mapped_source_classes, dtype=np.int16
        ),
        source_classes=np.array(cache.source_classes, dtype=np.int16),
        covered_destination_classes=np.array(
            cache.covered_destination_classes, dtype=np.int16
        ),
        destination_classes=np.array(
            cache.destination_classes, dtype=np.int16
        ),
        metadata=np.array(json.dumps(metadata)),
    )
    os.replace(temporary, path)


def load_pair_cache(path: Path) -> tuple[PairCache, dict]:
    """Restore one compressed pair cache and its JSON metadata."""

    with np.load(path, allow_pickle=False) as payload:
        cache = PairCache(
            angular_error=payload["angular_error"],
            pred_norm=payload["pred_norm"],
            tpit_angular_error=payload["tpit_angular_error"],
            tpit_pred_norm=payload["tpit_pred_norm"],
            class_idx=payload["class_idx"],
            track_idx=payload["track_idx"],
            n_total_frames=int(payload["n_total_frames"]),
            mapped_source_classes=int(payload["mapped_source_classes"]),
            source_classes=int(payload["source_classes"]),
            covered_destination_classes=int(
                payload["covered_destination_classes"]
            ),
            destination_classes=int(payload["destination_classes"]),
        )
        metadata = json.loads(str(payload["metadata"]))
    return cache, metadata


def percentile(values: np.ndarray, quantile: float) -> float:
    """Return a percentile or NaN for empty arrays."""

    return (
        float(np.percentile(values, quantile))
        if values.size else float("nan")
    )


def metric_values(errors: np.ndarray, norms: np.ndarray) -> dict:
    """Compute aggregate angular-error, accuracy, and norm diagnostics."""

    values = {
        "evaluated_slots": int(errors.size),
        "mean_angular_error": (
            float(errors.mean()) if errors.size else float("nan")
        ),
        "median_angular_error": percentile(errors, 50),
        "std_angular_error": (
            float(errors.std()) if errors.size else float("nan")
        ),
        "p75_angular_error": percentile(errors, 75),
        "p90_angular_error": percentile(errors, 90),
        "p95_angular_error": percentile(errors, 95),
        "pred_norm_median": percentile(norms, 50),
    }
    for angle in ANGLE_THRESHOLDS:
        values[f"doa_acc_{int(angle)}deg"] = (
            float((errors <= angle).mean())
            if errors.size else float("nan")
        )
    for threshold in ZERO_NORM_THRESHOLDS:
        key = f"zero_norm_lt_{threshold:.2f}"
        values[key] = (
            float((norms < threshold).mean())
            if norms.size else float("nan")
        )
    return values


def pair_rows(cache_root: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Build aggregate, per-class, and coverage rows from all pair caches."""

    aggregate_rows: list[dict] = []
    class_rows: list[dict] = []
    coverage_rows: list[dict] = []
    for path in sorted(cache_root.glob("*__on__*.npz")):
        cache, metadata = load_pair_cache(path)
        model_name = metadata["model_name"]
        model_space = metadata["model_space"]
        dataset_name = metadata["test_dataset"]
        domain = "native" if model_space == dataset_name else "cross_domain"
        common = {
            "model": model_name,
            "training_dataset": model_space,
            "test_dataset": dataset_name,
            "domain": domain,
            "split": metadata["split"],
            "seed": metadata["seed"],
            "pool_size": metadata["pool_size"],
            "evaluated_items": metadata["evaluated_items"],
        }
        aggregate_rows.append({
            **common,
            "n_total_frames": cache.n_total_frames,
            "mapped_source_classes": cache.mapped_source_classes,
            "source_classes": cache.source_classes,
            "covered_destination_classes":
                cache.covered_destination_classes,
            "destination_classes": cache.destination_classes,
            "source_coverage_ratio":
                cache.mapped_source_classes / cache.source_classes,
            "destination_coverage_ratio":
                cache.covered_destination_classes
                / cache.destination_classes,
            **metric_values(cache.angular_error, cache.pred_norm),
            **{
                f"tpit_{key}": value
                for key, value in metric_values(
                    cache.tpit_angular_error, cache.tpit_pred_norm
                ).items()
            },
        })
        coverage_rows.append({
            **common,
            "mapped_source_classes": cache.mapped_source_classes,
            "source_classes": cache.source_classes,
            "source_coverage_ratio":
                cache.mapped_source_classes / cache.source_classes,
            "covered_destination_classes":
                cache.covered_destination_classes,
            "destination_classes": cache.destination_classes,
            "destination_coverage_ratio":
                cache.covered_destination_classes
                / cache.destination_classes,
            "mapped_classes": "; ".join(
                row["source_class"]
                for row in metadata["mapping"]["rows"]
                if row["destination_indices"]
            ),
        })
        destination_sources: dict[int, list[str]] = {
            index: [] for index in range(len(LABELS[model_space]))
        }
        for mapping_row in metadata["mapping"]["rows"]:
            for destination_index in mapping_row["destination_indices"]:
                destination_sources[destination_index].append(
                    mapping_row["source_class"]
                )
        for class_index, class_name in enumerate(LABELS[model_space]):
            mask = cache.class_idx == class_index
            if not mask.any():
                continue
            class_rows.append({
                **common,
                "class_idx": class_index,
                "class_name": class_name,
                "mapped_source_classes": "; ".join(
                    destination_sources[class_index]
                ),
                **metric_values(
                    cache.angular_error[mask], cache.pred_norm[mask]
                ),
                **{
                    f"tpit_{key}": value
                    for key, value in metric_values(
                        cache.tpit_angular_error[mask],
                        cache.tpit_pred_norm[mask],
                    ).items()
                },
            })
    return aggregate_rows, class_rows, coverage_rows


def write_matrix(frame: pd.DataFrame, metric: str, path: Path) -> pd.DataFrame:
    """Pivot a metric into a train-by-test dataset matrix and save it."""

    matrix = frame.pivot(
        index="training_dataset", columns="test_dataset", values=metric
    )
    rows = [name for name in DATASET_ORDER if name in matrix.index]
    columns = [name for name in DATASET_ORDER if name in matrix.columns]
    matrix = matrix.reindex(index=rows, columns=columns)
    matrix.to_csv(path)
    return matrix


def save_heatmap(matrix: pd.DataFrame, colorbar_label: str, path: Path,
                 value_range: tuple[float, float] | None = None) -> None:
    """Render one diagnostic matrix as PNG and SVG heatmaps."""

    plot = matrix.copy()
    plot.index = [DATASET_LABELS[name] for name in plot.index]
    plot.columns = [DATASET_LABELS[name] for name in plot.columns]
    figure, axis = plt.subplots(figsize=(10.5, 5.4))
    vmin, vmax = value_range if value_range else (None, None)
    sns.heatmap(
        plot,
        annot=True,
        fmt=".3f",
        cmap="Greys",
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor="0.75",
        cbar_kws={"label": colorbar_label},
        annot_kws={"fontsize": 10},
        ax=axis,
    )
    finite = plot.to_numpy().ravel()
    finite = finite[np.isfinite(finite)]
    midpoint = (
        (vmin + vmax) / 2
        if vmin is not None and vmax is not None
        else (float(np.median(finite)) if finite.size else 0.5)
    )
    for annotation in axis.texts:
        try:
            value = float(annotation.get_text())
        except ValueError:
            value = float("nan")
        annotation.set_color(
            "white" if np.isfinite(value) and value >= midpoint else "black"
        )
    axis.set_xlabel("Test Dataset")
    axis.set_ylabel("Training Dataset")
    axis.set_xticklabels(
        axis.get_xticklabels(),
        rotation=20,
        ha="right",
        rotation_mode="anchor",
    )
    axis.set_yticklabels(axis.get_yticklabels(), rotation=0)
    axis.tick_params(axis="both", labelsize=10)
    figure.tight_layout()
    figure.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def rebuild_outputs(output: Path, figures: Path) -> None:
    """Regenerate CSV summaries, matrices, heatmaps, and run metadata."""

    aggregate_rows, class_rows, coverage_rows = pair_rows(output / "cache")
    if not aggregate_rows:
        return
    figures.mkdir(parents=True, exist_ok=True)
    aggregate = pd.DataFrame(aggregate_rows)
    classes = pd.DataFrame(class_rows)
    coverage = pd.DataFrame(coverage_rows)
    atomic_write_dataframe(
        output / "doa_oracle_aggregate.csv", aggregate
    )
    atomic_write_dataframe(
        output / "doa_oracle_per_class.csv", classes
    )
    atomic_write_dataframe(
        output / "doa_oracle_mapping_coverage.csv", coverage
    )

    acc20 = write_matrix(
        aggregate,
        "doa_acc_20deg",
        output / "doa_oracle_accuracy_20deg_matrix.csv",
    )
    median_error = write_matrix(
        aggregate,
        "median_angular_error",
        output / "doa_oracle_median_angular_error_matrix.csv",
    )
    tpit_acc20 = write_matrix(
        aggregate,
        "tpit_doa_acc_20deg",
        output / "doa_oracle_tpit_accuracy_20deg_matrix.csv",
    )
    save_heatmap(
        acc20,
        r"$\mathrm{Oracle\ DOAAcc}_{20^\circ}$",
        figures / "doa_oracle_accuracy_20deg_heatmap",
        value_range=(0.0, 1.0),
    )
    save_heatmap(
        median_error,
        r"Median Angular Error ($^\circ$)",
        figures / "doa_oracle_median_angular_error_heatmap",
    )
    save_heatmap(
        tpit_acc20,
        r"$\mathrm{tPIT\ DOAAcc}_{20^\circ}$",
        figures / "doa_oracle_tpit_accuracy_20deg_heatmap",
        value_range=(0.0, 1.0),
    )
    summary = {
        "completed_pairs": int(len(aggregate)),
        "expected_pairs": 16,
        "angle_thresholds_deg": list(ANGLE_THRESHOLDS),
        "zero_norm_thresholds": list(ZERO_NORM_THRESHOLDS),
        "activity_policy": "ground-truth oracle activity",
        "primary_alignment": "DOA-only angular-cost oracle permutation",
        "secondary_alignment": "checkpoint tPIT permutation",
        "selection_policy": "no threshold calibration or test optimization",
    }
    atomic_write_text(
        output / "doa_oracle_run_metadata.json",
        json.dumps(summary, indent=2) + "\n",
    )


def validate_known_native(output: Path, tolerance: float) -> None:
    """Warn when known native full-test reference metrics drift unexpectedly."""

    path = output / "doa_oracle_aggregate.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    for pair, expected in KNOWN_NATIVE_ACC20.items():
        model_name, dataset_name = pair
        match = frame[
            (frame["model"] == model_name)
            & (frame["test_dataset"] == dataset_name)
            & (frame["pool_size"] == 0)
            & (frame["split"] == "test")
        ]
        if match.empty:
            continue
        actual = float(match.iloc[0]["tpit_doa_acc_20deg"])
        difference = abs(actual - expected)
        status = "OK" if difference <= tolerance else "WARNING"
        print(
            f"[{status}] native regression {model_name} on {dataset_name}: "
            f"actual={actual:.6f}, reference={expected:.6f}, "
            f"|delta|={difference:.6f}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line options for pair inference or aggregate rebuilds."""

    parser = argparse.ArgumentParser(
        description=(
            "Compute coverage-aware DOA-oracle metrics for the NAS-SELD "
            "ExpA model-by-dataset matrix."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pool-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute pair caches that already exist.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from completed pair caches. This is already the default "
            "behavior and the flag is provided for explicit workflows."
        ),
    )
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help=(
            "Reuse completed pair caches. This is already the default "
            "behavior; use --force to recompute them."
        ),
    )
    parser.add_argument(
        "--discard-cache",
        action="store_true",
        help="Delete pair NPZ caches after all requested runs complete.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Rebuild CSV and figures from existing NPZ caches.",
    )
    parser.add_argument(
        "--native-regression-tolerance",
        type=float,
        default=0.005,
        help="Warning tolerance for known full-test native Acc@20 values.",
    )
    return parser.parse_args()


def main() -> None:
    """Run requested DOA-oracle pairs or rebuild aggregate outputs from cache."""

    args = parse_args()
    if args.seed != 2:
        warnings.warn(
            "This study is standardized on seed=2; a different seed was "
            "requested explicitly.",
            stacklevel=2,
        )
    output = args.output_dir
    figures = args.figures_dir
    cache_root = output / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        rebuild_outputs(output, figures)
        validate_known_native(output, args.native_regression_tolerance)
        return

    manifest = json.loads(args.manifest.read_text())
    model_names = args.models or list(manifest["models"])
    dataset_names = args.datasets or list(manifest["datasets"])
    unknown_models = sorted(set(model_names) - set(manifest["models"]))
    unknown_datasets = sorted(set(dataset_names) - set(manifest["datasets"]))
    if unknown_models or unknown_datasets:
        raise SystemExit(
            f"Unknown models={unknown_models}, datasets={unknown_datasets}"
        )

    device = resolve_device(args.device)
    for model_name in model_names:
        model_entry = manifest["models"][model_name]
        for dataset_name in dataset_names:
            dataset_space = normalize_space(dataset_name)
            cache_path = (
                cache_root / f"{model_name}__on__{dataset_space}.npz"
            )
            if cache_path.exists() and not args.force:
                print(
                    f"[skip completed] {model_name} on {dataset_space}",
                    flush=True,
                )
                continue
            print(
                f"[run] {model_name} on {dataset_space} "
                f"(split={args.split}, seed={args.seed})",
                flush=True,
            )
            cache, metadata = infer_pair(
                model_name=model_name,
                model_entry=model_entry,
                dataset_name=dataset_space,
                dataset_root=absolute(
                    manifest["datasets"][dataset_name]
                ),
                split=args.split,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pool_size=args.pool_size,
                seed=args.seed,
            )
            save_pair_cache(cache_path, cache, metadata)
            rebuild_outputs(output, figures)
            validate_known_native(
                output, args.native_regression_tolerance
            )

    rebuild_outputs(output, figures)
    validate_known_native(output, args.native_regression_tolerance)
    if args.discard_cache:
        for path in cache_root.glob("*__on__*.npz"):
            path.unlink()
        print(f"Discarded pair caches under {cache_root}", flush=True)
    print(f"Saved DOA-oracle CSV/JSON/NPZ assets to {output}", flush=True)
    print(f"Saved DOA-oracle figures to {figures}", flush=True)


if __name__ == "__main__":
    main()
