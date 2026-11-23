#!/usr/bin/env python3
"""
analysis/diagnostics/native_subset_references.py
==============================================================================
T5 — Subset-matched native references for the cross-dataset matrix
==============================================================================

An off-diagonal cell of the cross-dataset matrix (model A on dataset B) scores only
the classes of B that the explicit mapping projects into the label space of A
(``dataset_generalization_analysis.build_mapping(B, A)``; e.g. NIGENS2021 -> NIGENS2020
covers 12 of the 14 NIGENS2020 classes).  The diagonal "native" cell (model B on B)
scores *all* classes of B.  Statements such as "model A outperforms the native model
on B" therefore compare different class subsets.  This script evaluates the **native
checkpoint of B with exactly the class mask of every off-diagonal cell** (outputs of the
unmapped classes suppressed, their targets removed -- the coverage-aware convention of the
matrix) and compares it with the stored off-diagonal cell, like for like.

Protocol (identical to ``dataset_generalization_analysis.py``): global thresholds
0.05..0.95 and angular tolerances {5,10,15,20,30,45} deg are swept on the *validation*
split (STARSS23: validation pool of seed 2, as in the matrix), the five global protocols of
``best_global`` select a setting, which is then frozen and evaluated on the test split.
Computation uses the count-exact kernel of ``seld_variants.py`` on window caches written
by ``window_outputs.py``; the *diagonal* cells of the stored matrix are reproduced first
as a self-check (``diagonal_reproduction.csv``).

Caveats recorded in the output (``granularity_note``): where the mapping merges several
classes of B into one class of A (many-to-one) or duplicates one class into several
(one-to-many), the stored cell is scored at A's class granularity while the native
reference is scored at B's; those pairs are flagged and are not strictly like-for-like.

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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.diagnostics.dataset_generalization_analysis import LABELS, build_mapping  # noqa: E402
from analysis.diagnostics.seld_variants import GatedCounts  # noqa: E402
from analysis.diagnostics.window_outputs import (  # noqa: E402
    WindowCache, build_split_dataset, machine_load, window_target,
)

THRESHOLDS = [round(v, 2) for v in np.arange(0.05, 1.0, 0.05)]
ANGLES = [5.0, 10.0, 15.0, 20.0, 30.0, 45.0]
SPACES = ["starss23", "tau2019", "tau_nigens2020", "tau_nigens2021"]
MODEL_OF = {s: f"nas_expa_{s}" for s in SPACES}
PROTOCOLS = ["sed_f1_at_20deg", "seld_f1_at_20deg", "seld_score_at_20deg",
             "joint_seld_f1", "joint_seld_score"]


def sigmoid32(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)


def class_mask(n_classes: int, idx: Sequence[int]) -> np.ndarray:
    m = np.zeros(n_classes, bool)
    m[list(idx)] = True
    return m


class SplitEvaluator:
    """Accumulates (threshold x angle) SELD counts and SED counts for several masks."""

    def __init__(self, masks: Dict[str, np.ndarray], thresholds, angles, n_classes: int):
        self.masks, self.th, self.an = masks, list(thresholds), list(angles)
        self.seld = {(k, t, a): GatedCounts(n_classes, a)
                     for k in masks for t in self.th for a in self.an}
        self.sed = {(k, t): np.zeros(3, np.int64) for k in masks for t in self.th}

    def update(self, probs, doa, tgt):
        """probs (w,T,N,C) float32 probabilities, doa (w,T,N,C,3), tgt (w,T,N,C,3)."""
        N, C = probs.shape[2], probs.shape[3]
        ra_full = (np.linalg.norm(tgt, axis=-1) > 0.5)
        doa_f, tgt_f = doa.reshape(-1, N, C, 3), tgt.reshape(-1, N, C, 3)
        for key, m in self.masks.items():
            ra = (ra_full & m[None, None, None, :]).reshape(-1, N, C)
            tgt_m = tgt_f * ra[..., None]
            for t in self.th:
                pa = ((probs > t) & m[None, None, None, :]).reshape(-1, N, C)
                for a in self.an:
                    self.seld[(key, t, a)].update(pa, doa_f, ra, tgt_m)
                pr_any, rf_any = pa.any(axis=1), ra.any(axis=1)
                self.sed[(key, t)] += [int((pr_any & rf_any).sum()), int((pr_any & ~rf_any).sum()),
                                       int((~pr_any & rf_any).sum())]

    def metrics(self, key: str, t: float, a: float) -> dict:
        r = self.seld[(key, t, a)].compute()
        tp, fp, fn = (int(v) for v in self.sed[(key, t)])
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return {"threshold": t, "angle_deg": a, "seld_score": r["SELD_score"], "er": r["ER_20"],
                "seld_f1": r["F_20"], "le": r["LE_CD"], "lr": r["LR_CD"],
                "sed_precision": prec, "sed_recall": rec, "sed_f1": f1,
                "target_events": tp + fn, "predicted_events": tp + fp}


def run_split(cache: WindowCache, ds, masks, n_classes, log=print) -> SplitEvaluator:
    ev = SplitEvaluator(masks, THRESHOLDS, ANGLES, n_classes)
    t0 = time.time()
    chunk = 24
    for w0 in range(0, cache.n_windows, chunk):
        w1 = min(w0 + chunk, cache.n_windows)
        probs = sigmoid32(np.asarray(cache.logits[w0:w1]))
        doa = np.asarray(cache.doa[w0:w1])
        tgt = np.stack([window_target(ds, i).numpy() for i in range(w0, w1)])
        ev.update(probs, doa, tgt)
        if (w0 // chunk) % 40 == 0:
            log(f"    windows {w1}/{cache.n_windows} ({time.time() - t0:.0f} s)")
    return ev


def select_protocols(val: SplitEvaluator, key: str) -> Dict[str, Tuple[float, float]]:
    """Validation-optimal (threshold, angle) of the five ``best_global`` protocols
    (first minimum / maximum in iteration order: angle outer, threshold inner)."""
    rows = [val.metrics(key, t, a) for a in ANGLES for t in THRESHOLDS]
    std = [r for r in rows if abs(r["angle_deg"] - 20.0) < 1e-6]
    pick = {
        "sed_f1_at_20deg": max(std, key=lambda r: (r["sed_f1"], -r["seld_score"])),
        "seld_f1_at_20deg": max(std, key=lambda r: (r["seld_f1"], -r["seld_score"])),
        "seld_score_at_20deg": min(std, key=lambda r: r["seld_score"]),
        "joint_seld_f1": max(rows, key=lambda r: (r["seld_f1"], -r["seld_score"])),
        "joint_seld_score": min(rows, key=lambda r: r["seld_score"]),
    }
    return {k: (v["threshold"], v["angle_deg"]) for k, v in pick.items()}


def read_stored(path: Path) -> Dict[Tuple[str, str, str], dict]:
    out = {}
    with path.open() as fh:
        for r in csv.DictReader(fh):
            out[(r["model"], r["test_dataset"], r["protocol"])] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("Protocol")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-root", type=Path, required=True,
                    help="folder holding the window caches (expA_starss23_{val,test}, tau2019_*, ...)")
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "results" / "diagnostics" / "native_subset_references_phase2")
    ap.add_argument("--stored", type=Path, default=ROOT / "results" / "diagnostics" /
                    "dataset_generalization" / "aggregate" / "all_test_metrics.csv")
    ap.add_argument("--spaces", nargs="+", default=SPACES, choices=SPACES)
    args = ap.parse_args()

    cache_names = {"starss23": ("expA_starss23_val", "expA_starss23_test"),
                   "tau2019": ("tau2019_val", "tau2019_test"),
                   "tau_nigens2021": ("nigens2021_val", "nigens2021_test"),
                   "tau_nigens2020": ("nigens2020_val", "nigens2020_test")}
    data_roots = {"starss23": "data/STARSS23", "tau2019": "data/TAU2019",
                  "tau_nigens2021": "data/TAU-NIGENS2021", "tau_nigens2020": "data/TAU-NIGENS2020"}
    stored = read_stored(args.stored)
    out_cells: List[dict] = []
    pairwise: List[dict] = []
    diag: List[dict] = []
    t_start, load0 = time.time(), machine_load()

    for B in args.spaces:
        n_cls = len(LABELS[B])
        val_cache = WindowCache(args.cache_root / cache_names[B][0])
        test_cache = WindowCache(args.cache_root / cache_names[B][1])
        cfg_path = Path(val_cache.meta["config"])
        cfg = yaml.safe_load(open(cfg_path if cfg_path.is_absolute() else ROOT / cfg_path))["cfg"]
        val_ds = build_split_dataset(B, "val", ROOT / data_roots[B], cfg, 2)
        test_ds = build_split_dataset(B, "test", ROOT / data_roots[B], cfg, 2)
        # masks: the full native class set + the subset of every off-diagonal cell
        masks = {"full": class_mask(n_cls, range(n_cls))}
        notes: Dict[str, dict] = {}
        for A in SPACES:
            if A == B:
                continue
            mapping = build_mapping(B, A)             # rows: classes of B -> destination classes of A
            covered_B = [i for i, d in enumerate(mapping) if d]
            if not covered_B:
                continue
            dest = [tuple(mapping[i]) for i in covered_B]
            flat = [x for d in dest for x in d]
            many_to_one = len(set(flat)) < len(covered_B)
            one_to_many = any(len(d) > 1 for d in dest)
            masks[f"as_cell_of_{A}"] = class_mask(n_cls, covered_B)
            notes[f"as_cell_of_{A}"] = {
                "source_model_space": A, "n_classes_scored": len(covered_B),
                "classes_scored": [LABELS[B][i] for i in covered_B],
                "granularity_note": ("+".join(
                    n for n, f in (("many-to-one merge in the cell", many_to_one),
                                   ("one-to-many duplication in the cell", one_to_many)) if f)
                                     or "one-to-one class mapping (like for like)")}
        print(f"[T5] {B}: masks {list(masks)}  (load {machine_load()})", flush=True)
        val_ev = run_split(val_cache, val_ds, masks, n_cls)
        test_ev = run_split(test_cache, test_ds, masks, n_cls)
        for key in masks:
            sel = select_protocols(val_ev, key)
            for proto, (t, a) in sel.items():
                m = test_ev.metrics(key, t, a)
                row = {"model": MODEL_OF[B], "train_space": B, "test_dataset": B, "mask": key,
                       "split": "test", "protocol": f"val_selected_global_{proto}", **m}
                if key == "full":
                    st = stored.get((MODEL_OF[B], B, f"val_selected_global_{proto}"))
                    if st:
                        diag.append({"model": MODEL_OF[B], "dataset": B, "protocol": proto,
                                     "threshold_new": t, "threshold_stored": st["threshold"],
                                     **{f"{c}_new": m[c] for c in ("seld_score", "er", "seld_f1", "le", "lr")},
                                     **{f"{c}_stored": float(st[c]) for c in
                                        ("seld_score", "er", "seld_f1", "le", "lr")},
                                     "max_abs_diff": max(abs(m[c] - float(st[c])) for c in
                                                         ("seld_score", "er", "seld_f1", "le", "lr"))})
                out_cells.append(row)
            if key == "full":
                continue
            A = notes[key]["source_model_space"]
            for proto in ("seld_score_at_20deg", "sed_f1_at_20deg", "joint_seld_score"):
                cell = stored.get((MODEL_OF[A], B, f"val_selected_global_{proto}"))
                if cell is None:
                    continue
                t, a = sel[proto]
                ref = test_ev.metrics(key, t, a)
                # native reference frozen at the threshold of the FULL native cell (no re-selection)
                st_full = stored.get((MODEL_OF[B], B, f"val_selected_global_{proto}"))
                at_full = (test_ev.metrics(key, float(st_full["threshold"]), float(st_full["angle_deg"]))
                           if st_full else None)
                pairwise.append({
                    "target_dataset": B, "cross_model": MODEL_OF[A], "native_model": MODEL_OF[B],
                    "protocol": proto, **{k: v for k, v in notes[key].items()
                                          if k != "source_model_space"},
                    "classes_scored": ";".join(notes[key]["classes_scored"]),
                    "cell_threshold": cell["threshold"], "cell_angle": cell["angle_deg"],
                    "cell_seld": float(cell["seld_score"]), "cell_er": float(cell["er"]),
                    "cell_f": float(cell["seld_f1"]), "cell_le": float(cell["le"]),
                    "cell_lr": float(cell["lr"]), "cell_sed_f1": float(cell["sed_f1"]),
                    "native_subset_threshold": ref["threshold"], "native_subset_angle": ref["angle_deg"],
                    "native_subset_seld": ref["seld_score"], "native_subset_er": ref["er"],
                    "native_subset_f": ref["seld_f1"], "native_subset_le": ref["le"],
                    "native_subset_lr": ref["lr"], "native_subset_sed_f1": ref["sed_f1"],
                    "native_at_full_cell_threshold_seld": at_full["seld_score"] if at_full else "",
                    "native_full_cell_seld": float(st_full["seld_score"]) if st_full else "",
                    "delta_seld_cell_minus_native_subset": float(cell["seld_score"]) - ref["seld_score"],
                    "cross_better_than_native_subset": float(cell["seld_score"]) < ref["seld_score"],
                    "cross_better_than_native_full": (float(cell["seld_score"]) < float(st_full["seld_score"])
                                                      if st_full else ""),
                })
        print(f"[T5] {B} done ({time.time() - t_start:.0f} s)", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    def write(name: str, rows: List[dict]) -> None:
        if not rows:
            return
        fields: List[str] = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
        with (args.out_dir / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)

    write("native_reference_cells.csv", out_cells)
    write("pairwise_comparison.csv", pairwise)
    write("diagonal_reproduction.csv", diag)
    (args.out_dir / "provenance.json").write_text(json.dumps({
        "script": "analysis/diagnostics/native_subset_references.py", "argv": sys.argv[1:],
        "caches": {s: list(cache_names[s]) for s in args.spaces},
        "thresholds": THRESHOLDS, "angles": ANGLES,
        "validation": "val split (STARSS23 validation pool of seed 2, as in the stored matrix)",
        "stored_matrix": str(args.stored.relative_to(ROOT)),
        "kernel": "analysis/diagnostics/seld_variants.GatedCounts (count-exact vs SELDMetrics)",
        "elapsed_s": round(time.time() - t_start, 1), "load_at_start": load0,
        "load_at_end": machine_load(), "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=1))
    print(f"[T5] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
