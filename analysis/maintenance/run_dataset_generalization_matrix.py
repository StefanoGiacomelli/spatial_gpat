#!/usr/bin/env python3
"""
run_dataset_generalization_matrix.py — ExpA Generalization Matrix Runner
========================================================================
Run the NAS-SELD ExpA model-by-dataset generalization matrix.

The script is a thin orchestration layer over
``analysis/diagnostics/dataset_generalization_analysis.py``.  It expands the
manifest into model/dataset pairs, skips completed combinations unless forced,
and preserves the per-pair cache/output layout under ``results/diagnostics``.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "diagnostics"
    / "dataset_generalization_manifest.json"
)


def absolute(path: str) -> Path:
    """Resolve a manifest path relative to the repository root."""

    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def main() -> None:
    """Run or dry-run the requested model/dataset matrix combinations."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pool-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--discard-cache", action="store_true")
    parser.add_argument("--test-oracle-sweep", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute combinations whose final result files already exist.",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "results" / "diagnostics"
        / "dataset_generalization",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    model_names = args.models or list(manifest["models"])
    dataset_names = args.datasets or list(manifest["datasets"])
    analysis = ROOT / "analysis" / "diagnostics" / "dataset_generalization_analysis.py"

    failures = []
    for model_name in model_names:
        model = manifest["models"][model_name]
        for dataset_name in dataset_names:
            run_dir = (
                args.output_root / f"{model_name}__on__{dataset_name}"
            )
            required_outputs = (
                "run_metadata.json",
                "mapping_coverage.json",
                "validation_global_grid.csv",
                "validation_class_optima.csv",
                "test_val_selected_metrics.csv",
                "test_val_selected_per_class.csv",
            )
            if (
                not args.force
                and all((run_dir / name).exists() for name in required_outputs)
            ):
                print(
                    f"[skip completed] {model_name} on {dataset_name}",
                    flush=True,
                )
                continue
            command = [
                sys.executable, str(analysis),
                "--model-name", model_name,
                "--config", str(absolute(model["config"])),
                "--checkpoint", str(absolute(model["checkpoint"])),
                "--dataset", dataset_name,
                "--data-root", str(absolute(
                    manifest["datasets"][dataset_name]
                )),
                "--device", args.device,
                "--batch-size", str(args.batch_size),
                "--num-workers", str(args.num_workers),
                "--pool-size", str(args.pool_size),
                "--seed", str(args.seed),
                "--output-root", str(args.output_root),
            ]
            if args.reuse_cache:
                command.append("--reuse-cache")
            if args.discard_cache:
                command.append("--discard-cache")
            if args.test_oracle_sweep:
                command.append("--test-oracle-sweep")
            print(" ".join(command), flush=True)
            if args.dry_run:
                continue
            result = subprocess.run(command, cwd=ROOT, check=False)
            if result.returncode:
                failures.append((model_name, dataset_name, result.returncode))

    if failures:
        for model, dataset, code in failures:
            print(f"FAILED: {model} on {dataset} (exit {code})")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
