#!/usr/bin/env python3
"""
analysis/diagnostics/native_subset_oracle.py
==============================================================================
T5 (oracle DOA part) — subset-matched native references for the DOA-oracle matrix
==============================================================================

Companion of ``native_subset_references.py``.  The DOA-oracle matrix
(``dataset_generalization_doa_oracle.py``) scores, for every cell (model A on dataset B), the
angular error of the predicted slots on the *reference-active* slots of the classes of B that
the mapping projects into A's label space.  This script computes the same statistics for the
**native** checkpoint of B restricted to the *same class subset*, for both alignments of the
matrix:

* ``primary``: per (frame, class) the slot permutation that minimises the angular error on the
  active target slots only (``oracle_doa_permutation``; DOA-only, ignores the activity logits);
* ``tpit``: the permutation chosen by the checkpoint's tPIT loss (stored in the window cache).

The per-slot errors are computed once for all classes and then filtered by class mask, so the
cost is one pass over the cache per dataset.  The full-class result is compared with the stored
diagonal cell as a self-check.

Outputs (``--out-dir``): ``native_subset_oracle.csv`` (one row per dataset x mask x alignment),
``pairwise_oracle_comparison.csv`` (stored cell vs native subset) and ``provenance.json``.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.diagnostics.dataset_generalization_analysis import LABELS, build_mapping  # noqa: E402
from analysis.diagnostics.native_subset_references import SPACES, MODEL_OF  # noqa: E402
from analysis.diagnostics.window_outputs import WindowCache, build_split_dataset, machine_load, window_target  # noqa: E402

PERMS = np.array(list(itertools.permutations(range(3))), dtype=np.int64)     # (6, 3)
ACC = (5, 10, 15, 20, 30, 45, 90)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


def slot_errors(doa: np.ndarray, tgt: np.ndarray, perm: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per reference-active slot: class index, primary error, tPIT error (degrees).

    doa/tgt (w,T,N,C,3), perm (w,T,C,N) tPIT permutation (predicted slot for each target slot).
    """
    w, T, N, C, _ = tgt.shape
    act = np.linalg.norm(tgt, axis=-1) > 0.5                       # (w,T,N,C)
    rows = np.nonzero(act.any(axis=2))                              # (w,T,C) rows with >= 1 active slot
    iw, it, ic = rows
    pred = doa[iw, it, :, ic, :]                                    # (R,N,3)
    ref = tgt[iw, it, :, ic, :]                                     # (R,N,3)
    a = act[iw, it, :, ic]                                          # (R,N)
    cand = _unit(pred[:, PERMS, :])                                 # (R,6,N,3)  slot n <- pred[perm[n]]
    ang = np.degrees(np.arccos(np.clip((cand * _unit(ref)[:, None]).sum(-1), -1, 1)))   # (R,6,N)
    cost = (ang * a[:, None, :]).sum(-1) / np.maximum(a.sum(-1), 1)[:, None]
    best = cost.argmin(1)
    err_primary = ang[np.arange(len(best)), best]                   # (R,N)
    pp = perm[iw, it, ic].astype(np.int64)                          # (R,N): predicted slot for target slot n
    aligned = np.take_along_axis(pred, pp[:, :, None], axis=1)
    err_tpit = np.degrees(np.arccos(np.clip((_unit(aligned) * _unit(ref)).sum(-1), -1, 1)))
    cls = np.broadcast_to(ic[:, None], a.shape)
    return cls[a], err_primary[a].astype(np.float32), err_tpit[a].astype(np.float32)


def stats(err: np.ndarray) -> Dict[str, float]:
    out = {"slots": int(err.size), "mean": float(err.mean()), "median": float(np.median(err)),
           "p90": float(np.percentile(err, 90))}
    for t in ACC:
        out[f"acc_{t}deg"] = float((err <= t).mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("Outputs")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "results" / "diagnostics" / "native_subset_references_phase2")
    ap.add_argument("--stored", type=Path, default=ROOT / "results" / "diagnostics" /
                    "dataset_generalization" / "aggregate" / "doa_oracle" / "doa_oracle_aggregate.csv")
    ap.add_argument("--spaces", nargs="+", default=SPACES, choices=SPACES)
    args = ap.parse_args()
    t0, load0 = time.time(), machine_load()
    cache_names = {"starss23": "expA_starss23_test", "tau2019": "tau2019_test",
                   "tau_nigens2021": "nigens2021_test", "tau_nigens2020": "nigens2020_test"}
    data_roots = {"starss23": "data/STARSS23", "tau2019": "data/TAU2019",
                  "tau_nigens2021": "data/TAU-NIGENS2021", "tau_nigens2020": "data/TAU-NIGENS2020"}
    stored = {}
    with args.stored.open() as fh:
        for r in csv.DictReader(fh):
            stored[(r["model"], r["test_dataset"])] = r
    import yaml
    rows: List[dict] = []
    pairs: List[dict] = []
    for B in args.spaces:
        cache = WindowCache(args.cache_root / cache_names[B])
        if cache.perm is None:
            raise SystemExit(f"{cache_names[B]} has no tPIT permutations (--tpit-perm)")
        cfg_path = Path(cache.meta["config"])
        cfg = yaml.safe_load(open(cfg_path if cfg_path.is_absolute() else ROOT / cfg_path))["cfg"]
        ds = build_split_dataset(B, "test", ROOT / data_roots[B], cfg, 2)
        cls_l, ep_l, et_l = [], [], []
        chunk = 24
        for w0 in range(0, cache.n_windows, chunk):
            w1 = min(w0 + chunk, cache.n_windows)
            tgt = np.stack([window_target(ds, i).numpy() for i in range(w0, w1)])
            c, e1, e2 = slot_errors(np.asarray(cache.doa[w0:w1]), tgt, np.asarray(cache.perm[w0:w1]))
            cls_l.append(c); ep_l.append(e1); et_l.append(e2)
        cls = np.concatenate(cls_l); ep = np.concatenate(ep_l); et = np.concatenate(et_l)
        n_cls = len(LABELS[B])
        masks = {"full": list(range(n_cls))}
        for A in SPACES:
            if A != B:
                cov = [i for i, d in enumerate(build_mapping(B, A)) if d]
                if cov:
                    masks[f"as_cell_of_{A}"] = cov
        for key, idx in masks.items():
            sel = np.isin(cls, idx)
            for name, err in (("primary", ep), ("tpit", et)):
                s = stats(err[sel])
                rows.append({"model": MODEL_OF[B], "dataset": B, "mask": key, "alignment": name,
                             "n_classes": len(idx), **s})
        for key in masks:
            if key == "full":
                continue
            A = key.replace("as_cell_of_", "")
            cell = stored.get((MODEL_OF[A], B))
            if cell is None:
                continue
            for name, col_acc, col_med in (("primary", "doa_acc_20deg", "median_angular_error"),
                                           ("tpit", "tpit_doa_acc_20deg", "tpit_median_angular_error")):
                nat = next(r for r in rows if r["dataset"] == B and r["mask"] == key and r["alignment"] == name)
                full = next(r for r in rows if r["dataset"] == B and r["mask"] == "full" and r["alignment"] == name)
                pairs.append({
                    "target_dataset": B, "cross_model": MODEL_OF[A], "alignment": name,
                    "n_classes_scored": nat["n_classes"],
                    "cell_slots": cell["evaluated_slots"], "native_subset_slots": nat["slots"],
                    "cell_acc20": float(cell[col_acc]), "native_subset_acc20": nat["acc_20deg"],
                    "native_full_acc20": full["acc_20deg"],
                    "delta_acc20_cell_minus_native_subset": float(cell[col_acc]) - nat["acc_20deg"],
                    "cell_median_deg": float(cell[col_med]), "native_subset_median_deg": nat["median"],
                    "native_full_median_deg": full["median"],
                })
        print(f"[T5-oracle] {B} done ({time.time() - t0:.0f} s)", flush=True)
    # diagonal self-check
    chk = []
    for B in args.spaces:
        cell = stored[(MODEL_OF[B], B)]
        for name, col in (("primary", "doa_acc_20deg"), ("tpit", "tpit_doa_acc_20deg")):
            nat = next(r for r in rows if r["dataset"] == B and r["mask"] == "full" and r["alignment"] == name)
            chk.append({"dataset": B, "alignment": name, "acc20_new": nat["acc_20deg"],
                        "acc20_stored": float(cell[col]), "abs_diff": abs(nat["acc_20deg"] - float(cell[col])),
                        "slots_new": nat["slots"], "slots_stored": cell["evaluated_slots"]})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in (("native_subset_oracle.csv", rows), ("pairwise_oracle_comparison.csv", pairs),
                        ("oracle_diagonal_reproduction.csv", chk)):
        with (args.out_dir / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(table[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(table)
    (args.out_dir / "provenance_oracle.json").write_text(json.dumps({
        "script": "analysis/diagnostics/native_subset_oracle.py", "argv": sys.argv[1:],
        "caches": cache_names, "elapsed_s": round(time.time() - t0, 1), "load_at_start": load0,
        "load_at_end": machine_load(), "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stored": str(args.stored.relative_to(ROOT)),
    }, indent=1))
    print(f"[T5-oracle] diagonal max |diff| acc@20 = {max(c['abs_diff'] for c in chk):.2e}")


if __name__ == "__main__":
    main()
