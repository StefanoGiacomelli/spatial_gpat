#!/usr/bin/env python3
"""
analysis/profiling/reprofile_effective_configs.py
==============================================================================
T11 — Re-profile the Stage 1-3 architectures with the *executed* configuration (10-s input)
==============================================================================

The NAS-table profiles (``results/profiling/nas_configs_profile_foa_*.csv``) were produced by
``test_seld_nas_configs.py`` from the template YAMLs with a 30-s input on CPU (batch 1, one timed pass).
This script instantiates ``NASModel`` from the **stored hparams** of the 27 Stage 1-3 runs (N = 2 tracks,
the executed block types, ``use_msca``) and reports, for a 10-s window (the window of training and of the
final profile) with batch 1:

* total and trainable parameters (both, so that the Stage-2 / Stage-3 column mismatch disappears);
* multiply-accumulate operations as counted by ``torchinfo`` (``total_mult_adds``; GMACs) and the
  corresponding FLOPs convention (2 x MACs); the model is called as the pipeline calls it
  (``orig_sr = 24000`` -> internal resampling to 32 kHz);
* median forward latency on CPU (2 warm-up passes, 5 timed) and on MPS (3 warm-up, 10 timed,
  ``torch.mps.synchronize``), with the machine load at start and end.

``--legacy-check NAME [NAME ...]`` additionally reproduces the call of the original profiler for the
named runs: ``torchinfo`` was invoked as ``summary(model, input_data=x)`` *without* ``orig_sr``, i.e.
the 24-kHz samples were treated as 32-kHz audio (fewer model frames than the timed forward, which did
pass ``orig_sr``).  Both MAC counts are written next to each other.

``--legacy-check-all`` does this for every run and compares the result with the ``gflops`` column of the stored
NAS-table CSVs (``results/profiling/nas_configs_profile_foa_*.csv``): the stored value equals the legacy-call MAC count
for 30 s of 24-kHz samples (2,251 model frames), whereas the stored latency was timed on 3,001 frames (30 s at 32 kHz).

``torchinfo`` counts the multiply-adds of convolution / linear / recurrent layers; it is not a complete
operation count (attention matrix products, normalisation, element-wise operations are not included).

Output: ``results/profiling/nas_effective_configs_profile_phase2.csv``.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import List

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.paths import PROFILING_RESULTS_DIR, RUNS_DIR  # noqa: E402

GROUPS = ["shallow_spec", "shallow_gab", "deep_spec", "deep_reg"]
SR = 24000


def load() -> str:
    try:
        return " ".join(f"{v:.2f}" for v in os.getloadavg())
    except OSError:  # pragma: no cover
        return "n/a"


def count_params(model: torch.nn.Module):
    seen, total, trainable = set(), 0, 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
        trainable += p.numel() if p.requires_grad else 0
    return total, trainable


def torchinfo_gmacs(model, x, device: str, with_orig_sr: bool) -> float:
    from torchinfo import summary
    kwargs = {"orig_sr": SR} if with_orig_sr else {}
    stats = summary(model, input_data=x, verbose=0, device=device, **kwargs)
    return stats.total_mult_adds / 1e9


def latency_ms(model, x, device: str, warmup: int, repeats: int) -> float:
    times: List[float] = []
    with torch.inference_mode():
        for i in range(warmup + repeats):
            if device == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            model(x, orig_sr=SR)
            if device == "mps":
                torch.mps.synchronize()
            if i >= warmup:
                times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("Output:")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR / "at2seld_nas")
    ap.add_argument("--groups", nargs="+", default=GROUPS, choices=GROUPS)
    ap.add_argument("--duration-s", type=float, default=10.0)
    ap.add_argument("--legacy-check", nargs="*", default=[],
                    help="run names for which the original torchinfo call (no orig_sr) is also evaluated")
    ap.add_argument("--legacy-check-all", action="store_true",
                    help="evaluate the original torchinfo call for every run and compare with the stored NAS-table CSVs")
    ap.add_argument("--legacy-duration-s", type=float, default=30.0)
    ap.add_argument("--no-mps", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="only the first N runs (smoke test)")
    ap.add_argument("--out", type=Path, default=PROFILING_RESULTS_DIR / "nas_effective_configs_profile_phase2.csv")
    args = ap.parse_args()

    from experiments_nas_seld.nas_model import NASModel
    stored = {}
    for csv_path in sorted(PROFILING_RESULTS_DIR.glob("nas_configs_profile_foa_*.csv")):
        with csv_path.open() as fh:
            for r in csv.DictReader(fh):
                if r.get("gflops") not in (None, "", "nan"):
                    stored.setdefault(r["config_name"], []).append(float(r["gflops"]))
    use_mps = (not args.no_mps) and torch.backends.mps.is_available()
    hps = []
    for g in args.groups:
        hps += sorted((args.runs_dir / g / "checkpoints").glob("*/version_0/hparams.yaml"))
    if args.limit:
        hps = hps[: args.limit]
    fields = ["group", "run", "n_tracks", "early", "late", "smoother", "total_params", "trainable_params",
              "input_s", "model_frames", "gmacs_torchinfo", "flops_equiv_2xMACs_G", "latency_cpu_ms",
              "latency_mps_ms", "legacy_input_s", "gmacs_legacy_call_no_orig_sr", "gmacs_same_input_with_orig_sr",
              "legacy_model_frames_no_orig_sr", "stored_nas_table_gflops", "legacy_call_matches_stored", "load_start", "load_end", "status"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd()
    os.chdir(ROOT)
    rows = []
    try:
        for hp in hps:
            cfg = yaml.safe_load(hp.read_text())["cfg"]
            group, name = hp.parts[-5], hp.parts[-3]
            row = {k: "" for k in fields}
            row.update(group=group, run=name, load_start=load(), input_s=args.duration_s,
                       n_tracks=cfg["heads"]["n_tracks"], early=cfg["spatial_early"]["type"],
                       late=cfg["spatial_late"]["type"],
                       smoother=cfg.get("track_smoothing_spatial", {}).get("type", ""))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = NASModel(cfg).eval()
                total, trainable = count_params(model)
                row["total_params"], row["trainable_params"] = total, trainable
                x = torch.randn(1, 4, int(SR * args.duration_s))
                with torch.inference_mode():
                    row["model_frames"] = model(x, orig_sr=SR)[0].shape[1]
                gm = torchinfo_gmacs(model, x, "cpu", True)
                row["gmacs_torchinfo"] = f"{gm:.3f}"
                row["flops_equiv_2xMACs_G"] = f"{2 * gm:.3f}"
                row["latency_cpu_ms"] = f"{latency_ms(model, x, 'cpu', 2, 5):.1f}"
                if args.legacy_check_all or name in args.legacy_check:
                    xl = torch.randn(1, 4, int(SR * args.legacy_duration_s))
                    row["legacy_input_s"] = args.legacy_duration_s
                    row["gmacs_legacy_call_no_orig_sr"] = f"{torchinfo_gmacs(model, xl, 'cpu', False):.3f}"
                    row["gmacs_same_input_with_orig_sr"] = f"{torchinfo_gmacs(model, xl, 'cpu', True):.3f}"
                    row["legacy_model_frames_no_orig_sr"] = int(
                        1 + xl.shape[-1] // cfg["input"].get("hop_length", 320))
                    ref = stored.get(name)
                    if ref:
                        row["stored_nas_table_gflops"] = ";".join(f"{v:.3f}" for v in ref)
                        legacy = float(row["gmacs_legacy_call_no_orig_sr"])
                        row["legacy_call_matches_stored"] = any(abs(legacy - v) < 0.01 * max(1.0, v / 100) for v in ref)
                if use_mps:
                    model = model.to("mps")
                    row["latency_mps_ms"] = f"{latency_ms(model, x.to('mps'), 'mps', 3, 10):.1f}"
                row["status"] = "OK"
            except Exception as exc:  # keep going, record the failure
                row["status"] = f"FAIL: {type(exc).__name__}: {exc}"
                traceback.print_exc()
            row["load_end"] = load()
            rows.append(row)
            print(f"{group:13s} {name:28s} {row['status'][:4]} params {row['total_params']} GMACs {row['gmacs_torchinfo']} "
                  f"cpu {row['latency_cpu_ms']} ms mps {row['latency_mps_ms']} ms (load {row['load_end']})", flush=True)
            with args.out.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
                w.writeheader()
                w.writerows(rows)
    finally:
        os.chdir(cwd)
    print(f"{len(rows)} runs -> {args.out}")


if __name__ == "__main__":
    main()
