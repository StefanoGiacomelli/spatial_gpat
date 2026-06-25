#!/usr/bin/env python3
"""
dataset_generalization_analysis.py — Cross-Dataset NAS-SELD Evaluation
======================================================================
Cross-domain NAS-SELD evaluation and deployment-tolerance analysis.

One invocation evaluates one trained NAS-SELD checkpoint on one dataset.  The
dataset targets are projected into the model label space using conservative,
explicit mappings.  Validation selects global and class-specific thresholds;
the frozen settings are then evaluated on test.  Optional test-oracle sweeps
are saved separately and must be interpreted as diagnostics, not calibration.

The script never trains or updates model weights.

Outputs
-------
  - mapping_coverage.json
  - validation_global_grid.csv / validation_class_optima.csv
  - test_val_selected_metrics.csv / test_val_selected_per_class.csv
  - optional diagnostic_test_oracle_*.csv files
  - reusable inference caches under the selected output directory

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
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.diagnostics.activation_test import (  # noqa: E402
    InferenceCache,
    load_nas_model,
    load_yaml_cfg,
    select_pool_indices,
)
from experiments_nas_seld.loss import SELDMetrics  # noqa: E402


LABELS: Dict[str, List[str]] = {
    "starss23": [
        "female_speech", "male_speech", "clapping", "telephone", "laughter",
        "domestic_sounds", "footsteps", "door", "music",
        "musical_instrument", "water_tap", "bell", "knock",
    ],
    "tau_nigens2021": [
        "alarm", "baby_cry", "crash", "dog_bark", "female_scream",
        "female_speech", "footsteps", "knock", "male_scream", "male_speech",
        "telephone", "piano",
    ],
    "tau_nigens2020": [
        "alarm", "baby_cry", "crash", "dog_bark", "running_engine",
        "female_scream", "female_speech", "burning_fire", "footsteps", "knock",
        "male_scream", "male_speech", "telephone", "piano",
    ],
    "tau2019": [
        "clearthroat", "cough", "door", "drawer", "keyboard", "keys_drop",
        "knock", "laughter", "page_turn", "telephone", "speech",
    ],
}

DATASET_ALIASES = {
    "STARSS23": "starss23",
    "BalancedSTARSS23": "starss23",
    "TAU-NIGENS2021": "tau_nigens2021",
    "TAU-NIGENS2020": "tau_nigens2020",
    "TAU2019": "tau2019",
}

# Additional mappings beyond exact canonical-name matches.
# Each source label maps to zero, one, or multiple destination labels.
EXPLICIT_MAPS: Dict[Tuple[str, str], Dict[str, List[str]]] = {
    ("tau_nigens2021", "starss23"): {
        "female_scream": ["female_speech"],
        "male_scream": ["male_speech"],
        "piano": ["music", "musical_instrument"],
    },
    ("tau_nigens2020", "starss23"): {
        "female_scream": ["female_speech"],
        "male_scream": ["male_speech"],
        "piano": ["music", "musical_instrument"],
    },
    ("tau2019", "starss23"): {
        "drawer": ["domestic_sounds"],
        "keyboard": ["domestic_sounds"],
        "keys_drop": ["domestic_sounds"],
        "page_turn": ["domestic_sounds"],
    },
    ("starss23", "tau2019"): {
        "female_speech": ["speech"],
        "male_speech": ["speech"],
    },
    ("tau_nigens2021", "tau2019"): {
        "female_speech": ["speech"],
        "male_speech": ["speech"],
    },
    ("tau_nigens2020", "tau2019"): {
        "female_speech": ["speech"],
        "male_speech": ["speech"],
    },
    ("starss23", "tau_nigens2021"): {
        "musical_instrument": ["piano"],
    },
    ("starss23", "tau_nigens2020"): {
        "musical_instrument": ["piano"],
    },
}


@dataclass
class MetricRow:
    """Scalar metrics for one split, protocol, threshold, and DOA tolerance."""

    split: str
    protocol: str
    threshold: float | str
    angle_deg: float
    seld_score: float
    er: float
    seld_f1: float
    le: float
    lr: float
    sed_precision: float
    sed_recall: float
    sed_f1: float
    target_events: int
    predicted_events: int


def normalize_space(value: str) -> str:
    """Normalize a dataset name or alias to the canonical label-space key."""

    try:
        return DATASET_ALIASES[value]
    except KeyError:
        key = value.lower().replace("-", "_")
        if key == "balanced_starss23":
            key = "starss23"
        if key not in LABELS:
            raise ValueError(f"Unknown label space: {value}")
        return key


def build_mapping(source: str, destination: str) -> List[List[int]]:
    """Build a source-to-destination class-index projection."""

    source, destination = normalize_space(source), normalize_space(destination)
    destination_index = {name: idx for idx, name in enumerate(LABELS[destination])}
    additions = EXPLICIT_MAPS.get((source, destination), {})
    mapping: List[List[int]] = []
    for label in LABELS[source]:
        destinations: List[str] = []
        if label in destination_index:
            destinations.append(label)
        destinations.extend(additions.get(label, []))
        unique = []
        for name in destinations:
            if name in destination_index and name not in unique:
                unique.append(name)
        mapping.append([destination_index[name] for name in unique])
    return mapping


def mapping_report(source: str, destination: str,
                   mapping: Sequence[Sequence[int]]) -> dict:
    """Summarize label-space coverage for a class projection."""

    source, destination = normalize_space(source), normalize_space(destination)
    rows = []
    mapped_source = 0
    covered_destination = set()
    for source_idx, destinations in enumerate(mapping):
        if destinations:
            mapped_source += 1
        covered_destination.update(destinations)
        rows.append({
            "source_index": source_idx,
            "source_class": LABELS[source][source_idx],
            "destination_indices": list(destinations),
            "destination_classes": [
                LABELS[destination][idx] for idx in destinations
            ],
        })
    return {
        "source_space": source,
        "destination_space": destination,
        "mapped_source_classes": mapped_source,
        "source_classes": len(LABELS[source]),
        "covered_destination_classes": len(covered_destination),
        "destination_classes": len(LABELS[destination]),
        "rows": rows,
    }


def project_target(target: torch.Tensor, mapping: Sequence[Sequence[int]],
                   destination_classes: int, destination_tracks: int
                   ) -> torch.Tensor:
    """Project ``(B,T,N,C,3)`` targets and repack colliding tracks."""
    batch, frames, source_tracks, _, _ = target.shape
    result = target.new_zeros(
        batch, frames, destination_tracks, destination_classes, 3
    )
    occupied = torch.zeros(
        batch, frames, destination_tracks, destination_classes,
        dtype=torch.bool, device=target.device,
    )
    for source_class, destinations in enumerate(mapping):
        if not destinations:
            continue
        for source_track in range(source_tracks):
            vector = target[:, :, source_track, source_class]
            active = vector.norm(dim=-1) > 0.5
            if not active.any():
                continue
            for destination_class in destinations:
                remaining = active.clone()
                for destination_track in range(destination_tracks):
                    place = remaining & ~occupied[
                        :, :, destination_track, destination_class
                    ]
                    if place.any():
                        result[
                            :, :, destination_track, destination_class
                        ][place] = vector[place]
                        occupied[
                            :, :, destination_track, destination_class
                        ][place] = True
                        remaining &= ~place
                    if not remaining.any():
                        break
    return result


def build_dataset(space: str, split: str, root: Path, cfg: dict,
                  val_seed: int):
    """Instantiate the dataset implementation matching ``space`` and ``split``."""

    space = normalize_space(space)
    input_cfg = cfg["input"]
    data_cfg = cfg.get("data", {})
    common = {
        "split": split,
        "model_sr": input_cfg["sr"],
        "hop_length": input_cfg.get("hop_length", 320),
        "n_tracks": cfg.get("heads", {}).get("n_tracks", 3),
        "window_s": data_cfg.get("window_s", 10.0),
        "hop_s": data_cfg.get("hop_s", 2.5),
        "augment": None,
    }
    if space == "starss23":
        from datasets.starss23 import STARSS23Dataset
        return STARSS23Dataset(
            root=root, val_fraction=data_cfg.get("val_fraction", 0.2),
            val_seed=val_seed, **common,
        )
    if space == "tau_nigens2021":
        from datasets.tau_nigens import TAUNIGENSDataset
        return TAUNIGENSDataset(root=root, **common)
    if space == "tau_nigens2020":
        from datasets.tau_nigens2020 import TAUNIGENS2020Dataset
        return TAUNIGENS2020Dataset(root=root, **common)
    if space == "tau2019":
        from datasets.tau2019 import TAU2019Dataset
        return TAU2019Dataset(
            root=root, fold=data_cfg.get("fold", 1),
            mixed_mode=(split == "test"), **common,
        )
    raise AssertionError(space)


def cache_path(output_dir: Path, split: str) -> Path:
    """Return the prediction-cache path for a split-specific inference run."""

    return output_dir / "cache" / f"{split}.pt"


def run_inference(model: torch.nn.Module, dataset, mapping, model_classes: int,
                  model_tracks: int, device: torch.device, orig_sr: int,
                  hop_length: int, batch_size: int, workers: int,
                  pool_size: int, seed: int, split: str,
                  output_dir: Path, reuse_cache: bool,
                  include_uncovered_outputs: bool) -> InferenceCache:
    """Run checkpoint inference or restore a compatible cached prediction set."""

    path = cache_path(output_dir, split)
    if reuse_cache and path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        protocol = payload.pop("_cache_protocol", {})
        expected = {
            "mapping": [list(row) for row in mapping],
            "include_uncovered_outputs": include_uncovered_outputs,
            "seed": seed,
            "pool_size": pool_size,
        }
        if protocol != expected:
            raise RuntimeError(
                f"Cache protocol mismatch for {path}. Remove the cache or "
                "rerun without --reuse-cache."
            )
        return InferenceCache(**payload)

    indices = select_pool_indices(dataset, pool_size, "random", seed)
    subset = Subset(dataset, indices)
    from datasets.utils import SELDDataset
    collate = None if getattr(dataset, "windowed", False) else SELDDataset.collate_fn
    loader = DataLoader(
        subset,
        batch_size=batch_size if getattr(dataset, "windowed", False) else 1,
        shuffle=False, num_workers=workers, collate_fn=collate,
    )

    logits_list, doa_list, target_list = [], [], []
    probability_parts, target_parts = [], []
    frame_count = 0
    covered_classes = sorted({
        destination for destinations in mapping for destination in destinations
    })
    uncovered_classes = sorted(set(range(model_classes)) - set(covered_classes))
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"inference:{split}", unit="batch"):
            audio, source_target = batch[0].to(device), batch[1].to(device)
            logits, doa, _ = model(audio, orig_sr=orig_sr)
            if uncovered_classes and not include_uncovered_outputs:
                logits[..., uncovered_classes] = -30.0
                doa[..., uncovered_classes, :] = 0.0
            target = project_target(
                source_target, mapping, model_classes, model_tracks
            )
            frames = min(logits.shape[1], target.shape[1])
            logits, doa, target = (
                logits[:, :frames], doa[:, :frames], target[:, :frames]
            )
            logits_cpu = logits.cpu()
            doa_cpu = doa.cpu()
            target_cpu = target.cpu()
            logits_list.append(logits_cpu)
            doa_list.append(doa_cpu)
            target_list.append(target_cpu)
            probability_parts.append(
                torch.sigmoid(logits_cpu).numpy().ravel().astype(np.float32)
            )
            target_parts.append(
                (target_cpu.norm(dim=-1) > 0.5).numpy().ravel()
            )
            frame_count += logits.shape[0] * frames

    cache = InferenceCache(
        act_logits_list=logits_list,
        doa_tanh_list=doa_list,
        targets_list=target_list,
        n_classes=model_classes,
        n_tracks=model_tracks,
        n_frames=frame_count,
        orig_sr=orig_sr,
        hop_length=hop_length,
        model_name=output_dir.name,
        probs_flat=np.concatenate(probability_parts),
        tgt_flat=np.concatenate(target_parts).astype(bool),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "act_logits_list": cache.act_logits_list,
        "doa_tanh_list": cache.doa_tanh_list,
        "targets_list": cache.targets_list,
        "n_classes": cache.n_classes,
        "n_tracks": cache.n_tracks,
        "n_frames": cache.n_frames,
        "orig_sr": cache.orig_sr,
        "hop_length": cache.hop_length,
        "model_name": cache.model_name,
        "probs_flat": cache.probs_flat,
        "tgt_flat": cache.tgt_flat,
        "_cache_protocol": {
            "mapping": [list(row) for row in mapping],
            "include_uncovered_outputs": include_uncovered_outputs,
            "seed": seed,
            "pool_size": pool_size,
        },
    }, path)
    return cache


def threshold_logits(logits: torch.Tensor,
                     thresholds: float | Sequence[float]) -> torch.Tensor:
    """Convert per-class probability thresholds into equivalent logit offsets."""

    if isinstance(thresholds, (float, int)):
        return logits
    values = torch.as_tensor(
        thresholds, dtype=logits.dtype, device=logits.device
    ).clamp(1e-6, 1 - 1e-6)
    offsets = torch.log(values / (1.0 - values))
    return logits - offsets.view(1, 1, 1, -1)


def sed_counts(cache: InferenceCache, thresholds: float | Sequence[float],
               class_idx: int | None = None) -> Tuple[int, int, int]:
    """Count SED true positives, false positives, and false negatives."""

    tp = fp = fn = 0
    for logits, target in zip(cache.act_logits_list, cache.targets_list):
        frames = min(logits.shape[1], target.shape[1])
        probabilities = torch.sigmoid(logits[:, :frames])
        if isinstance(thresholds, (float, int)):
            predicted = probabilities > float(thresholds)
        else:
            vector = torch.as_tensor(thresholds).view(1, 1, 1, -1)
            predicted = probabilities > vector
        reference = target[:, :frames].norm(dim=-1) > 0.5
        predicted = predicted.any(dim=2)
        reference = reference.any(dim=2)
        if class_idx is not None:
            predicted = predicted[..., class_idx]
            reference = reference[..., class_idx]
        tp += int((predicted & reference).sum())
        fp += int((predicted & ~reference).sum())
        fn += int((~predicted & reference).sum())
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1 from detection counts."""

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return precision, recall, f1


def evaluate(cache: InferenceCache, thresholds: float | Sequence[float],
             angle: float, split: str, protocol: str) -> MetricRow:
    """Evaluate cached predictions for one threshold protocol and angle limit."""

    metrics = SELDMetrics(
        angle_thresh=angle,
        act_thresh=float(thresholds) if isinstance(thresholds, (float, int)) else 0.5,
    )
    for logits, doa, target in zip(
        cache.act_logits_list, cache.doa_tanh_list, cache.targets_list
    ):
        metrics.update(threshold_logits(logits, thresholds), doa, target)
    values = metrics.compute()
    tp, fp, fn = sed_counts(cache, thresholds)
    precision, recall, f1 = prf(tp, fp, fn)
    return MetricRow(
        split=split, protocol=protocol,
        threshold=float(thresholds) if isinstance(thresholds, (float, int))
        else json.dumps([round(float(v), 6) for v in thresholds]),
        angle_deg=angle,
        seld_score=values["SELD_score"], er=values["ER_20"],
        seld_f1=values["F_20"], le=values["LE_CD"], lr=values["LR_CD"],
        sed_precision=precision, sed_recall=recall, sed_f1=f1,
        target_events=tp + fn, predicted_events=tp + fp,
    )


def global_grid(cache: InferenceCache, thresholds: Sequence[float],
                angles: Sequence[float], split: str) -> List[MetricRow]:
    """Sweep global SED thresholds and DOA tolerances on one cached split."""

    rows = []
    for angle in angles:
        for threshold in thresholds:
            rows.append(evaluate(
                cache, threshold, angle, split, "global_grid"
            ))
    return rows


def best_global(rows: Sequence[MetricRow]) -> Dict[str, MetricRow]:
    """Select validation-optimal global settings for the reported protocols."""

    standard = [row for row in rows if abs(row.angle_deg - 20.0) < 1e-6]
    if not standard:
        standard = list(rows)
    return {
        "sed_f1_at_20deg": max(
            standard, key=lambda row: (row.sed_f1, -row.seld_score)
        ),
        "seld_f1_at_20deg": max(
            standard, key=lambda row: (row.seld_f1, -row.seld_score)
        ),
        "seld_score_at_20deg": min(
            standard, key=lambda row: row.seld_score
        ),
        "joint_seld_f1": max(
            rows, key=lambda row: (row.seld_f1, -row.seld_score)
        ),
        "joint_seld_score": min(rows, key=lambda row: row.seld_score),
    }


def class_optima(cache: InferenceCache, thresholds: Sequence[float],
                 angles: Sequence[float], class_names: Sequence[str],
                 split: str) -> List[dict]:
    """Find class-specific SED and SELD optima on a calibration split."""

    rows = []
    for class_idx, class_name in enumerate(class_names):
        support_counts = sed_counts(cache, 0.5, class_idx)
        support = support_counts[0] + support_counts[2]
        if support == 0:
            rows.append({
                "split": split, "class_idx": class_idx,
                "class_name": class_name, "target_support": 0,
                "best_sed_threshold": float("nan"),
                "best_sed_f1": float("nan"),
                "best_seld_threshold": float("nan"),
                "best_seld_angle_deg": float("nan"),
                "best_seld_f1": float("nan"),
            })
            continue

        sed_candidates = []
        seld_candidates = []
        for threshold in thresholds:
            tp, fp, fn = sed_counts(cache, threshold, class_idx)
            precision, recall, f1 = prf(tp, fp, fn)
            sed_candidates.append((f1, precision, recall, threshold))
            for angle in angles:
                metrics = SELDMetrics(angle_thresh=angle, act_thresh=threshold)
                for logits, doa, target in zip(
                    cache.act_logits_list,
                    cache.doa_tanh_list,
                    cache.targets_list,
                ):
                    metrics.update(
                        logits[..., class_idx:class_idx + 1],
                        doa[..., class_idx:class_idx + 1, :],
                        target[..., class_idx:class_idx + 1, :],
                    )
                result = metrics.compute()
                seld_candidates.append((
                    result["F_20"], -result["SELD_score"], threshold, angle,
                    result,
                ))
        best_sed = max(sed_candidates)
        best_seld = max(seld_candidates, key=lambda item: item[:2])
        rows.append({
            "split": split, "class_idx": class_idx,
            "class_name": class_name, "target_support": support,
            "best_sed_threshold": best_sed[3],
            "best_sed_precision": best_sed[1],
            "best_sed_recall": best_sed[2],
            "best_sed_f1": best_sed[0],
            "best_seld_threshold": best_seld[2],
            "best_seld_angle_deg": best_seld[3],
            "best_seld_f1": best_seld[0],
            "best_seld_score": -best_seld[1],
            "best_seld_le": best_seld[4]["LE_CD"],
            "best_seld_lr": best_seld[4]["LR_CD"],
        })
    return rows


def evaluate_per_class_settings(cache: InferenceCache, settings: Sequence[dict],
                                split: str) -> Tuple[List[MetricRow], List[dict]]:
    """Evaluate frozen class-specific settings on a target split."""

    sed_thresholds = [
        row["best_sed_threshold"] if math.isfinite(row["best_sed_threshold"])
        else 0.5 for row in settings
    ]
    seld_thresholds = [
        row["best_seld_threshold"] if math.isfinite(row["best_seld_threshold"])
        else 0.5 for row in settings
    ]
    aggregate = [
        evaluate(cache, sed_thresholds, 20.0, split,
                 "val_selected_class_specific_sed"),
        evaluate(cache, seld_thresholds, 20.0, split,
                 "val_selected_class_specific_seld_at_20deg"),
    ]
    per_class = []
    for row in settings:
        class_idx = row["class_idx"]
        threshold = row["best_seld_threshold"]
        angle = row["best_seld_angle_deg"]
        if not (math.isfinite(threshold) and math.isfinite(angle)):
            continue
        metrics = SELDMetrics(angle_thresh=angle, act_thresh=threshold)
        for logits, doa, target in zip(
            cache.act_logits_list, cache.doa_tanh_list, cache.targets_list
        ):
            metrics.update(
                logits[..., class_idx:class_idx + 1],
                doa[..., class_idx:class_idx + 1, :],
                target[..., class_idx:class_idx + 1, :],
            )
        result = metrics.compute()
        tp, fp, fn = sed_counts(cache, threshold, class_idx)
        precision, recall, sed_f1 = prf(tp, fp, fn)
        per_class.append({
            "split": split, "class_idx": class_idx,
            "class_name": row["class_name"], "threshold": threshold,
            "angle_deg": angle, "seld_score": result["SELD_score"],
            "seld_f1": result["F_20"], "le": result["LE_CD"],
            "lr": result["LR_CD"], "sed_precision": precision,
            "sed_recall": recall, "sed_f1": sed_f1,
            "target_support": tp + fn,
        })
    return aggregate, per_class


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    """Write a sequence of dictionaries to CSV with stable unioned fields."""

    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def resolve_device(value: str) -> torch.device:
    """Resolve the requested accelerator string into a torch device."""

    if value == "auto":
        if torch.cuda.is_available():
            value = "cuda"
        elif torch.backends.mps.is_available():
            value = "mps"
        else:
            value = "cpu"
    return torch.device(value)


def main() -> None:
    """Run one model/dataset generalization job and serialize diagnostics."""

    parser = argparse.ArgumentParser(
        description="Dataset-dependent NAS-SELD generalization analysis."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", required=True,
                        choices=list(LABELS))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[round(v, 2) for v in np.arange(0.05, 1.0, 0.05)])
    parser.add_argument("--angles", type=float, nargs="+",
                        default=[5, 10, 15, 20, 30, 45])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pool-size", type=int, default=0,
                        help="<=0 evaluates the complete split")
    parser.add_argument(
        "--seed", type=int, default=2,
        help=(
            "Validation-split and pool-selection seed. Seed 2 is the project "
            "standard because it preserves class representation in validation."
        ),
    )
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument(
        "--discard-cache", action="store_true",
        help="Delete raw val/test prediction caches after all analyses finish.",
    )
    parser.add_argument("--test-oracle-sweep", action="store_true")
    parser.add_argument(
        "--include-uncovered-output-classes", action="store_true",
        help=(
            "Diagnostic only: retain predictions for model classes that the "
            "evaluation dataset cannot map. The default coverage-aware "
            "protocol suppresses those unscorable outputs."
        ),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "results" / "diagnostics"
        / "dataset_generalization",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, cfg, orig_sr = load_nas_model(
        args.config, args.checkpoint, device
    )
    model_space = normalize_space(cfg["semantic"]["dataset"])
    dataset_space = normalize_space(args.dataset)
    mapping = build_mapping(dataset_space, model_space)
    report = mapping_report(dataset_space, model_space, mapping)

    slug = f"{args.model_name}__on__{dataset_space}"
    output_dir = args.output_root / slug
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mapping_coverage.json").write_text(
        json.dumps(report, indent=2)
    )

    caches = {}
    for split in ("val", "test"):
        dataset = build_dataset(
            dataset_space, split, args.data_root, cfg, val_seed=args.seed
        )
        caches[split] = run_inference(
            model, dataset, mapping, len(LABELS[model_space]),
            int(cfg.get("heads", {}).get("n_tracks", 3)),
            device, orig_sr, int(cfg["input"].get("hop_length", 320)),
            args.batch_size, args.num_workers, args.pool_size, args.seed,
            split, output_dir, args.reuse_cache,
            args.include_uncovered_output_classes,
        )

    validation_grid = global_grid(
        caches["val"], args.thresholds, args.angles, "val"
    )
    selected = best_global(validation_grid)
    class_settings = class_optima(
        caches["val"], args.thresholds, args.angles,
        LABELS[model_space], "val",
    )

    selected_test = []
    for objective, row in selected.items():
        selected_test.append(evaluate(
            caches["test"], float(row.threshold), row.angle_deg, "test",
            f"val_selected_global_{objective}",
        ))
    class_aggregate, class_test = evaluate_per_class_settings(
        caches["test"], class_settings, "test"
    )
    selected_test.extend(class_aggregate)

    write_csv(
        output_dir / "validation_global_grid.csv",
        [asdict(row) for row in validation_grid],
    )
    write_csv(output_dir / "validation_class_optima.csv", class_settings)
    write_csv(
        output_dir / "test_val_selected_metrics.csv",
        [asdict(row) for row in selected_test],
    )
    write_csv(
        output_dir / "test_val_selected_per_class.csv", class_test
    )

    if args.test_oracle_sweep:
        test_grid = global_grid(
            caches["test"], args.thresholds, args.angles, "test"
        )
        test_class = class_optima(
            caches["test"], args.thresholds, args.angles,
            LABELS[model_space], "test",
        )
        write_csv(
            output_dir / "diagnostic_test_oracle_global_grid.csv",
            [asdict(row) for row in test_grid],
        )
        write_csv(
            output_dir / "diagnostic_test_oracle_class_optima.csv",
            test_class,
        )

    metadata = {
        "model_name": args.model_name,
        "model_space": model_space,
        "dataset_space": dataset_space,
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "thresholds": args.thresholds,
        "angles": args.angles,
        "pool_size": args.pool_size,
        "seed": args.seed,
        "selection_rule": "validation-selected and frozen on test",
        "diagnostic_test_oracle_sweep": args.test_oracle_sweep,
        "coverage_aware": not args.include_uncovered_output_classes,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )
    if args.discard_cache:
        for split in ("val", "test"):
            path = cache_path(output_dir, split)
            if path.exists():
                path.unlink()
        cache_dir = output_dir / "cache"
        if cache_dir.exists() and not any(cache_dir.iterdir()):
            cache_dir.rmdir()
    print(f"Saved analysis to {output_dir}")


if __name__ == "__main__":
    main()
