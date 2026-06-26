#!/usr/bin/env python3
"""
profile_nas_seld_branches.py — NAS-SELD Branch Profiling
=========================================================
Profile the task-facing SELD branch of trained NAS-SELD checkpoints.

The full NAS model contains a large E-PANNs semantic branch and a much smaller
spatial SELD path.  This script reports their parameter ownership separately
and profiles a spatial-only forward pass where semantic bridge tensors are
precomputed once and then held fixed.  The latter measures the incremental
cost of producing ``head_sed`` and ``head_doa`` outputs, including enabled
cross-stitch bridges but excluding E-PANNs feature extraction.

Outputs
-------
  - nas_seld_branch_profiles.csv
  - nas_seld_branch_profiles.json

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SPATIAL_PREFIXES = (
    "input_proc", "input_norm_mag", "input_norm_phase", "input_norm_iv",
    "input_norm_gcc", "spatial_early", "spatial_late", "smoother_spatial",
    "mhamgu_expand", "head_sed", "head_doa",
)
SEMANTIC_PREFIXES = (
    "split_epanns", "sem_fc1", "sem_fc_audioset", "head_semantic",
)
BRIDGE_PREFIXES = ("bridge_early", "bridge_late")


def load_cfg(path: Path) -> dict:
    """Load a YAML configuration and unwrap an optional top-level ``cfg`` key."""

    with path.open() as fh:
        raw = yaml.safe_load(fh)
    return raw.get("cfg", raw)


def classify_parameter(name: str) -> str:
    """Assign a model parameter to spatial, semantic, bridge, or other ownership."""

    if name.startswith(SPATIAL_PREFIXES):
        return "spatial_seld"
    if name.startswith(SEMANTIC_PREFIXES):
        return "semantic_epanns"
    if name.startswith(BRIDGE_PREFIXES):
        return "cross_stitch"
    return "other"


def parameter_breakdown(model: nn.Module) -> Dict[str, int]:
    """Count unique parameters by ownership group for one NAS-SELD model."""

    result = {
        "spatial_seld": 0,
        "semantic_epanns": 0,
        "cross_stitch": 0,
        "other": 0,
        "total": 0,
    }
    seen = set()
    for name, parameter in model.named_parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        count = parameter.numel()
        result[classify_parameter(name)] += count
        result["total"] += count
    result["effective_seld"] = result["spatial_seld"] + result["cross_stitch"]
    return result


class SpatialSELDWrapper(nn.Module):
    """Run only the spatial path with cached semantic bridge activations."""

    def __init__(self, model: nn.Module, h_early: torch.Tensor,
                 h_late: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("h_early_cached", h_early.detach(), persistent=False)
        self.register_buffer("h_late_cached", h_late.detach(), persistent=False)

    def forward(self, x: torch.Tensor, orig_sr: int | None = None):
        m = self.model
        if orig_sr is not None and orig_sr != m.src_sr:
            x = AF.resample(x, orig_sr, m.src_sr)

        spatial = m._run_input_processor(x)
        if m.itype == "spectral":
            channels = m.n_mics
            parts = [
                m.input_norm_mag(spatial[:, :channels]),
                m.input_norm_phase(spatial[:, channels:2 * channels]),
            ]
            offset = 2 * channels
            if hasattr(m, "input_norm_iv"):
                parts.append(m.input_norm_iv(spatial[:, offset:offset + 3]))
                offset += 3
            if hasattr(m, "input_norm_gcc"):
                parts.append(m.input_norm_gcc(spatial[:, offset:]))
            spatial = torch.cat(parts, dim=1)

        for module in m.spatial_early:
            spatial = module(spatial)

        h_early = self.h_early_cached.expand(x.shape[0], -1, -1, -1)
        if m.cs_early_on:
            spatial, h_early = m.bridge_early(spatial, h_early)

        spatial_late = m._run_spatial_late(spatial, spatial.shape[1])
        h_late = self.h_late_cached.expand(x.shape[0], -1, -1, -1)
        if m.cs_late_on:
            if m._sl_is_transformer:
                spatial_4d = spatial_late.permute(0, 2, 1).unsqueeze(3)
                spatial_4d, _ = m.bridge_late(spatial_4d, h_late)
                spatial_late = spatial_4d.squeeze(3).permute(0, 2, 1)
            else:
                spatial_late, _ = m.bridge_late(spatial_late, h_late)

        if m._sl_is_transformer:
            sequence = spatial_late
        elif spatial_late.dim() == 4:
            sequence = spatial_late.mean(dim=3).permute(0, 2, 1)
        else:
            sequence = spatial_late

        hidden = m._smoother_forward(
            m.smoother_spatial, m._tss_type, sequence
        )
        batch = x.shape[0]
        n_tracks, n_classes = m.n_tracks, m.n_cls
        act = m.head_sed(hidden).view(batch, -1, n_tracks, n_classes)
        doa = torch.tanh(
            m.head_doa(hidden).view(batch, -1, n_tracks, n_classes, 3)
        )
        return act, doa


def cached_semantic_features(model: nn.Module, x: torch.Tensor
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the semantic branch once and return cached bridge activations."""

    mono = x.mean(dim=1)
    h_early = model.split_epanns.forward_early(mono)
    h_late = model.split_epanns.forward_late(h_early)
    return h_early[:1], h_late[:1]


def memory_mb(module: nn.Module) -> float:
    """Estimate module parameter and buffer memory footprint in megabytes."""

    size = sum(p.numel() * p.element_size() for p in module.parameters())
    size += sum(b.numel() * b.element_size() for b in module.buffers())
    return size / (1024 ** 2)


def gflops(module: nn.Module, x: torch.Tensor, orig_sr: int) -> float:
    """Estimate multiply-add complexity through torchinfo when available."""

    try:
        from torchinfo import summary
        info = summary(
            module, input_data=(x, orig_sr), verbose=0,
            device=str(x.device),
        )
        return float(info.total_mult_adds) / 1e9
    except Exception as exc:
        warnings.warn(f"GFLOP profiling failed: {exc}", stacklevel=2)
        return float("nan")


def synchronize(device: torch.device) -> None:
    """Synchronize CUDA or MPS work queues before timing."""

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def latency_ms(module: nn.Module, x: torch.Tensor, orig_sr: int,
               warmup: int, repeats: int) -> Tuple[float, float]:
    """Measure median and mean inference latency in milliseconds."""

    with torch.inference_mode():
        for _ in range(warmup):
            module(x, orig_sr)
        synchronize(x.device)
        values = []
        for _ in range(repeats):
            start = time.perf_counter()
            module(x, orig_sr)
            synchronize(x.device)
            values.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(values), statistics.mean(values)


def resolve_device(name: str) -> torch.device:
    """Resolve ``auto`` to the best available torch accelerator."""

    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    return torch.device(name)


def profile_one(name: str, cfg_path: Path, checkpoint: Path,
                device: torch.device, duration: float, batch_size: int,
                warmup: int, repeats: int) -> dict:
    """Profile one full checkpoint and its cached-semantic spatial path."""

    from analysis.diagnostics.activation_test import load_nas_model

    model, cfg, orig_sr = load_nas_model(cfg_path, checkpoint, device)
    model.eval()
    model_sr = int(cfg["input"]["sr"])
    x = torch.randn(
        batch_size, int(cfg["input"].get("n_mics", 4)),
        int(duration * model_sr), device=device,
    )

    with torch.inference_mode():
        h_early, h_late = cached_semantic_features(model, x)
    spatial = SpatialSELDWrapper(model, h_early, h_late).to(device).eval()

    breakdown = parameter_breakdown(model)
    with torch.inference_mode():
        full_outputs = model(x, orig_sr=model_sr)
        spatial_outputs = spatial(x, model_sr)

    full_gflops = gflops(model, x, model_sr)
    spatial_gflops = gflops(spatial, x, model_sr)
    full_median, full_mean = latency_ms(
        model, x, model_sr, warmup, repeats
    )
    spatial_median, spatial_mean = latency_ms(
        spatial, x, model_sr, warmup, repeats
    )

    row = {
        "model": name,
        "config": str(cfg_path),
        "checkpoint": str(checkpoint),
        "device": str(device),
        "duration_s": duration,
        "batch_size": batch_size,
        **{f"params_{key}": value for key, value in breakdown.items()},
        "spatial_fraction_params": (
            breakdown["effective_seld"] / breakdown["total"]
        ),
        "semantic_fraction_params": (
            breakdown["semantic_epanns"] / breakdown["total"]
        ),
        "full_parameter_memory_mb": memory_mb(model),
        "full_gflops": full_gflops,
        "spatial_incremental_gflops": spatial_gflops,
        "spatial_fraction_gflops": spatial_gflops / full_gflops
        if full_gflops > 0 else float("nan"),
        "full_latency_median_ms": full_median,
        "full_latency_mean_ms": full_mean,
        "spatial_latency_median_ms": spatial_median,
        "spatial_latency_mean_ms": spatial_mean,
        "spatial_fraction_latency": spatial_median / full_median,
        "act_shape": list(full_outputs[0].shape),
        "doa_shape": list(full_outputs[1].shape),
        "semantic_shape": list(full_outputs[2].shape),
        "spatial_act_shape": list(spatial_outputs[0].shape),
        "spatial_doa_shape": list(spatial_outputs[1].shape),
        "bridge_early_enabled": bool(model.cs_early_on),
        "bridge_late_enabled": bool(model.cs_late_on),
    }
    del spatial, model, x
    return row


def parse_spec(value: str) -> Tuple[str, Path, Path]:
    """Parse ``NAME::CONFIG::CHECKPOINT`` CLI model specifications."""

    parts = value.split("::")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "Expected NAME::CONFIG::CHECKPOINT"
        )
    return parts[0], Path(parts[1]), Path(parts[2])


def main() -> None:
    """Profile all requested NAS-SELD model specifications."""

    parser = argparse.ArgumentParser(
        description="Profile full NAS-SELD and its spatial SELD branch."
    )
    parser.add_argument(
        "--model", action="append", type=parse_spec,
        metavar="NAME::CONFIG::CHECKPOINT",
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=Path(__file__).with_name(
            "dataset_generalization_manifest.json"
        ),
        help="Model manifest used when --model is omitted.",
    )
    parser.add_argument(
        "--models", nargs="+",
        help="Optional subset of manifest model names.",
    )
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results" / "diagnostics"
        / "dataset_generalization" / "profiling",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    model_specs = args.model
    if not model_specs:
        manifest = json.loads(args.manifest.read_text())
        names = args.models or list(manifest["models"])
        model_specs = []
        for name in names:
            spec = manifest["models"][name]
            config = Path(spec["config"])
            checkpoint = Path(spec["checkpoint"])
            if not config.is_absolute():
                config = ROOT / config
            if not checkpoint.is_absolute():
                checkpoint = ROOT / checkpoint
            model_specs.append((name, config, checkpoint))

    rows = []
    for name, cfg_path, checkpoint in model_specs:
        print(f"[profile] {name}")
        rows.append(profile_one(
            name, cfg_path, checkpoint, device, args.duration_s,
            args.batch_size, args.warmup, args.repeats,
        ))

    json_path = args.output_dir / "nas_seld_branch_profiles.json"
    csv_path = args.output_dir / "nas_seld_branch_profiles.csv"
    json_path.write_text(json.dumps(rows, indent=2))
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {csv_path}")
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
