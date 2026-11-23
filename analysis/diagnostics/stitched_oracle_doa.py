#!/usr/bin/env python3
"""
analysis/diagnostics/stitched_oracle_doa.py
==============================================================================
Oracle-activity DOA statistics on stitched (clip-level) predictions vs pooled windows
==============================================================================

``doa_oracle_test.py`` evaluates the angular error of the predicted slots on the reference-active slots
of every 10-s window and pools the windows (interior frames counted up to four times, zero-padded tails
included).  This script evaluates the *same predictions* (a window cache from ``window_outputs.py``)
two ways:

* ``pooled``   -- windows pooled, as the stored diagnostics (reproduces them; self-check);
* ``stitched`` -- overlapping windows averaged per frame first (every frame counted once), then the
  error is measured on the reference-active slots of the clip.

For both modes the two alignments of the matrix are reported: ``tpit`` (the slots are aligned with the
checkpoint's tPIT permutation w.r.t. the reference *before* averaging) and ``primary`` (reference-free
stitching, then the DOA-only angular-cost permutation per frame and class).

Outputs (``--out-dir``): ``stitched_oracle_doa.csv``, ``provenance.json``.

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
from typing import Dict, List

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.diagnostics.native_subset_oracle import slot_errors, stats  # noqa: E402
from analysis.diagnostics.run_metric_variants import clip_n_mel, sigmoid  # noqa: E402
from analysis.diagnostics.seld_variants import stitch_windows  # noqa: E402
from analysis.diagnostics.window_outputs import (  # noqa: E402
    WindowCache, build_split_dataset, machine_load, window_start_frame, window_target,
)

DEFAULT_RUNS = ["expA_starss23_test", "mixedfocal_starss23_test", "expB_starss23_test"]


def evaluate(cache: WindowCache, ds, dataset: str) -> List[dict]:
    N, C = cache.meta["n_tracks"], cache.meta["n_classes"]
    items = cache.meta["items"]
    # pooled windows
    e_pri, e_tpit = [], []
    chunk = 24
    for w0 in range(0, cache.n_windows, chunk):
        w1 = min(w0 + chunk, cache.n_windows)
        tgt = np.stack([window_target(ds, i).numpy() for i in range(w0, w1)])
        _, ep, et = slot_errors(np.asarray(cache.doa[w0:w1]), tgt, np.asarray(cache.perm[w0:w1]))
        e_pri.append(ep)
        e_tpit.append(et)
    out = [("pooled", "primary", np.concatenate(e_pri)), ("pooled", "tpit", np.concatenate(e_tpit))]
    # stitched clips
    by_clip: Dict[int, List[int]] = {}
    for w, (ci, _) in enumerate(items):
        by_clip.setdefault(int(ci), []).append(w)
    s_pri, s_tpit = [], []
    for ci, widx in sorted(by_clip.items()):
        lo, hi = widx[0], widx[-1] + 1
        n_mel = clip_n_mel(ds, ci)
        offs = [window_start_frame(ds, w) for w in widx]
        probs = sigmoid(np.asarray(cache.logits[lo:hi])).astype(np.float32)
        doa = np.asarray(cache.doa[lo:hi])
        tgt = ds._build_target(ds._events[ci], n_mel, 0).numpy()[None]
        ident = np.broadcast_to(np.arange(N, dtype=np.int8), (1, n_mel, C, N)).copy()
        _, vo, _ = stitch_windows(probs, doa, offs, n_mel, align="oracle", perm=np.asarray(cache.perm[lo:hi]))
        _, vf, _ = stitch_windows(probs, doa, offs, n_mel, align="free")
        _, _, et = slot_errors(vo[None], tgt, ident)
        _, ep, _ = slot_errors(vf[None], tgt, ident)
        s_tpit.append(et)
        s_pri.append(ep)
    out += [("stitched", "primary", np.concatenate(s_pri)), ("stitched", "tpit", np.concatenate(s_tpit))]
    rows = []
    for mode, align, err in out:
        rows.append({"mode": mode, "alignment": align, **stats(err)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("Outputs")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--dataset", default="starss23")
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "results" / "diagnostics" / "stitched_oracle_doa_phase2")
    args = ap.parse_args()
    t0, load0 = time.time(), machine_load()
    rows: List[dict] = []
    for name in args.runs:
        cache = WindowCache(args.cache_root / name)
        cfg_path = Path(cache.meta["config"])
        cfg = yaml.safe_load(open(cfg_path if cfg_path.is_absolute() else ROOT / cfg_path))["cfg"]
        ds = build_split_dataset(args.dataset, "test", ROOT / "data" / "STARSS23", cfg, 2)
        for r in evaluate(cache, ds, args.dataset):
            rows.append({"run": name, **r})
        print(f"[stitched-oracle] {name} done ({time.time() - t0:.0f} s)", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "stitched_oracle_doa.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    (args.out_dir / "provenance.json").write_text(json.dumps({
        "script": "analysis/diagnostics/stitched_oracle_doa.py", "argv": sys.argv[1:],
        "runs": args.runs, "elapsed_s": round(time.time() - t0, 1), "load_at_start": load0,
        "load_at_end": machine_load(), "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "pooled = windows pooled as the stored diagnostics; stitched = overlapping windows averaged first",
    }, indent=1))
    for r in rows:
        print(f"  {r['run']:26s} {r['mode']:8s} {r['alignment']:8s} slots {r['slots']:>9,} acc@20 {r['acc_20deg']:.4f} median {r['median']:.2f}")


if __name__ == "__main__":
    main()
