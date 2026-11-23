#!/usr/bin/env python3
"""
analysis/diagnostics/validation_seed_check.py
==============================================================================
Do the validation-selected thresholds of the STARSS23 diagonal cell depend on a
validation pool that overlaps the training clips?
==============================================================================

Training splits STARSS23 ``dev-train`` with ``val_seed = 42`` (72 training / 18 validation
clips, room-stratified).  The cross-dataset matrix (``dataset_generalization_analysis.py``)
and the activation / oracle diagnostics build their STARSS23 validation pool with
``--seed 2`` ("project standard"), which yields a *different* 18-clip pool: 15 of its 18
clips are training clips of the AT2SELD checkpoints.  (The class-threshold calibration of
``tab:nas_expa_calibration`` uses ``--val-seed 42`` and is not affected.)

This script re-selects the five global protocols of the matrix (``best_global``) on the
**held-out** seed-42 validation pool and on the seed-2 pool, freezes each choice and
evaluates it on the STARSS23 test split, for ExpA (native STARSS23, the checkpoint of the
diagonal cell).  Inputs are window caches written by ``window_outputs.py``.

Outputs (``--out-dir``): ``validation_seed_comparison.csv``, ``provenance.json``.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.diagnostics.native_subset_references import (  # noqa: E402
    class_mask, run_split, select_protocols,
)
from analysis.diagnostics.window_outputs import (  # noqa: E402
    WindowCache, build_split_dataset, machine_load,
)


def clip_names(ds) -> set:
    return {Path(c[1]).stem for c in ds.clips}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("Outputs")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--val-seed2-cache", default="expA_starss23_val")
    ap.add_argument("--val-seed42-cache", default="expA_starss23_val_seed42")
    ap.add_argument("--test-cache", default="expA_starss23_test")
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "results" / "diagnostics" / "validation_seed_phase2")
    ap.add_argument("--stored", type=Path, default=ROOT / "results" / "diagnostics" /
                    "dataset_generalization" / "aggregate" / "all_test_metrics.csv")
    args = ap.parse_args()
    t0, load0 = time.time(), machine_load()

    test_cache = WindowCache(args.cache_root / args.test_cache)
    cfg_path = Path(test_cache.meta["config"])
    cfg = yaml.safe_load(open(cfg_path if cfg_path.is_absolute() else ROOT / cfg_path))["cfg"]
    root = ROOT / "data" / "STARSS23"
    n_cls = test_cache.meta["n_classes"]
    masks = {"full": class_mask(n_cls, range(n_cls))}

    test_ds = build_split_dataset("starss23", "test", root, cfg, 2)
    print("[val-seed] evaluating test grid", flush=True)
    test_ev = run_split(test_cache, test_ds, masks, n_cls)

    pools = {"seed2_matrix_pool": (args.val_seed2_cache, 2), "seed42_heldout": (args.val_seed42_cache, 42)}
    train42 = clip_names(build_split_dataset("starss23", "train", root, cfg, 42))
    stored = {}
    with args.stored.open() as fh:
        for r in csv.DictReader(fh):
            if r["model"] == "nas_expa_starss23" and r["test_dataset"] == "starss23":
                stored[r["protocol"]] = r
    rows = []
    overlap = {}
    for pool, (cache_name, seed) in pools.items():
        cache = WindowCache(args.cache_root / cache_name)
        ds = build_split_dataset("starss23", "val", root, cfg, seed)
        names = clip_names(ds)
        overlap[pool] = {"n_clips": len(names), "n_training_clips_seed42": len(names & train42)}
        print(f"[val-seed] {pool}: {len(names)} clips, {len(names & train42)} are training clips", flush=True)
        val_ev = run_split(cache, ds, masks, n_cls)
        sel = select_protocols(val_ev, "full")
        for proto, (t, a) in sel.items():
            m = test_ev.metrics("full", t, a)
            v = val_ev.metrics("full", t, a)
            st = stored.get(f"val_selected_global_{proto}")
            rows.append({
                "validation_pool": pool, "val_seed": seed, "protocol": proto,
                "threshold": t, "angle_deg": a,
                "val_seld": v["seld_score"], "val_f1": v["sed_f1"],
                "test_seld": m["seld_score"], "test_er": m["er"], "test_f": m["seld_f1"],
                "test_le": m["le"], "test_lr": m["lr"], "test_sed_f1": m["sed_f1"],
                "stored_matrix_threshold": st["threshold"] if st else "",
                "stored_matrix_test_seld": float(st["seld_score"]) if st else "",
            })
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "validation_seed_comparison.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    (args.out_dir / "provenance.json").write_text(json.dumps({
        "script": "analysis/diagnostics/validation_seed_check.py", "argv": sys.argv[1:],
        "pools": overlap, "elapsed_s": round(time.time() - t0, 1), "load_at_start": load0,
        "load_at_end": machine_load(), "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "seed 42 = the validation split of the training runs (held out); seed 2 = pool of "
                "the matrix/diagnostic scripts (15 of 18 clips are training clips)",
    }, indent=1))
    for r in rows:
        print(f"  {r['validation_pool']:18s} {r['protocol']:22s} thr {r['threshold']:.2f} "
              f"angle {r['angle_deg']:.0f}  test SELD {r['test_seld']:.4f}")


if __name__ == "__main__":
    main()
