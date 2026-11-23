#!/usr/bin/env python3
"""
analysis/diagnostics/run_metric_variants.py
==============================================================================
T4 — Stitched, clip-level and DCASE-style metric variants from a window cache
==============================================================================

Reads a window-output cache produced by ``window_outputs.py`` (one checkpoint,
one split) and evaluates the *same predictions* under several protocols.  The
repository protocol (variant ``A_repo_pooled_windows``) is the headline and is
reproduced first; every other row is an additional, labelled variant:

  A_repo_pooled_windows         windows pooled (4x overlap), 10-ms frames, gated 20 deg
                                (``SELDMetrics`` semantics)                  [headline]
  B_stitched_10ms               overlapping windows averaged (reference-free slot
                                alignment), every frame counted once, same metric
  B2_stitched_10ms_exact_labels B with the 100-ms label bins mapped to model frames by exact
                                integer division (the dataset's ``floor(t/0.1)`` assigns ~3 %
                                of the frames to the preceding bin because of float rounding)
  C_stitched_oracle_10ms        same with the tPIT permutation w.r.t. the reference
                                (what ``main_demo.py::_infer_full_clip`` does; upper bound)
  D_stitched_100ms_frame        B pooled to 100-ms frames, gated frame-level metric
  S1_seg100ms_dcase             100-ms segments, DCASE-style (ungated class-dependent LE/LR,
                                macro-F, per-segment S/D/I); arithmetic of the public DCASE
                                implementation (official_compat) -- re-implementation
  S2_seg1s_repo_components      1-s segments, repository components (gated LE/LR, pooled F)
  S3_seg1s_gated_macroF         1-s segments, gated LE/LR, macro-F
  S4_seg1s_dcase                1-s segments, DCASE-style (full), official_compat -- re-implementation
  S4s_seg1s_dcase_strict        S4 with the stricter bookkeeping (a class never co-active in a
                                segment = missed reference + spurious prediction; macro over the
                                classes with references; recall over reference instances)
  S4m_seg1s_dcase_meandir       S4 with instance angles taken between segment-mean directions
                                (sensitivity of the re-implementation to that definition)

Outputs (``--out-dir``): ``variants_long.csv`` (every variant x threshold),
``variants_summary.json``, ``provenance.json``.

Nothing is trained; inference lives in ``window_outputs.py``.

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
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.diagnostics.seld_variants import (  # noqa: E402
    GatedCounts, SegmentCounts, pool_frames, reference_from_events, stitch_windows,
)
from analysis.diagnostics.window_outputs import (  # noqa: E402
    WindowCache, build_split_dataset, machine_load, window_start_frame, window_target,
)

DEFAULT_THRESHOLDS = [round(v, 2) for v in np.arange(0.05, 1.0, 0.05)]
GRID = 10  # model frames (10 ms) per 100-ms label frame

VARIANTS = [
    "A_repo_pooled_windows", "B_stitched_10ms", "B2_stitched_10ms_exact_labels",
    "C_stitched_oracle_10ms",
    "D_stitched_100ms_frame", "S1_seg100ms_dcase", "S2_seg1s_repo_components",
    "S3_seg1s_gated_macroF", "S4_seg1s_dcase", "S4s_seg1s_dcase_strict", "S4m_seg1s_dcase_meandir",
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


def clip_n_mel(ds, clip_idx: int) -> int:
    """Number of model frames of the full clip (as ``SELDDataset.__getitem__``)."""
    path, _ = ds.clips[clip_idx]
    n = sf.info(str(path)).frames
    if ds.output_sr is not None:
        n = math.ceil(n * ds.output_sr / ds.orig_sr)
    n_model = round(n * ds.model_sr / ds._eff_sr)
    return 1 + n_model // ds.hop_length


def class_mask_arrays(mask: Optional[Sequence[int]], n_classes: int) -> Optional[np.ndarray]:
    if mask is None:
        return None
    m = np.zeros(n_classes, bool)
    m[list(mask)] = True
    return m


def evaluate_variants(cache: WindowCache, ds, thresholds: Sequence[float],
                      limit_clips: Optional[int] = None,
                      class_mask: Optional[Sequence[int]] = None,
                      variants: Optional[Sequence[str]] = None,
                      log=print) -> Dict[str, Dict[float, dict]]:
    """Evaluate all variants at all thresholds.  Returns ``{variant: {thr: metrics}}``."""
    meta = cache.meta
    N, C = meta["n_tracks"], meta["n_classes"]
    items = meta["items"]
    n_win = cache.n_windows
    cmask = class_mask_arrays(class_mask, C)
    use = set(variants or VARIANTS)
    do_oracle = "C_stitched_oracle_10ms" in use and cache.perm is not None

    accA = {t: GatedCounts(C) for t in thresholds}
    accB = {t: GatedCounts(C) for t in thresholds}
    accB2 = {t: GatedCounts(C) for t in thresholds}
    accC = {t: GatedCounts(C) for t in thresholds}
    accD = {t: GatedCounts(C) for t in thresholds}
    accS1 = {t: SegmentCounts(C, 1) for t in thresholds}
    accS2 = {t: SegmentCounts(C, GRID, gated_loc=True, macro_f=False) for t in thresholds}
    accS3 = {t: SegmentCounts(C, GRID, gated_loc=True, macro_f=True) for t in thresholds}
    accS4 = {t: SegmentCounts(C, GRID) for t in thresholds}
    accS4s = {t: SegmentCounts(C, GRID, official_compat=False) for t in thresholds}
    accS4m = {t: SegmentCounts(C, GRID, pair_mode="mean_direction") for t in thresholds}

    # ── A: windows pooled (repository protocol) ────────────────────────────
    if "A_repo_pooled_windows" in use:
        t0 = time.time()
        chunk = 32
        for w0 in range(0, n_win, chunk):
            w1 = min(w0 + chunk, n_win)
            probs = sigmoid(np.asarray(cache.logits[w0:w1])).astype(np.float32)   # (w,T,N,C)
            doa = np.asarray(cache.doa[w0:w1])
            tgt = np.stack([window_target(ds, i).numpy() for i in range(w0, w1)])   # (w,T,N,C,3)
            ra = np.linalg.norm(tgt, axis=-1) > 0.5
            if cmask is not None:
                ra &= cmask[None, None, None, :]
                tgt = tgt * ra[..., None]
            shp = (-1, N, C)
            doaf, tgtf, raf = doa.reshape(-1, N, C, 3), tgt.reshape(-1, N, C, 3), ra.reshape(shp)
            for t in thresholds:
                pa = (probs > t)
                if cmask is not None:
                    pa &= cmask[None, None, None, :]
                accA[t].update(pa.reshape(shp), doaf, raf, tgtf)
        log(f"[variants] A done in {time.time() - t0:.0f} s")

    # ── per clip: stitched variants ────────────────────────────────────────
    by_clip: Dict[int, List[int]] = {}
    for w, (ci, _) in enumerate(items):
        by_clip.setdefault(int(ci), []).append(w)
    clip_ids = sorted(by_clip)[: (limit_clips or len(by_clip))]
    t0 = time.time()
    need_stitch = use & {"B_stitched_10ms", "B2_stitched_10ms_exact_labels",
                         "D_stitched_100ms_frame", "S1_seg100ms_dcase",
                         "S2_seg1s_repo_components", "S3_seg1s_gated_macroF", "S4_seg1s_dcase",
                         "S4s_seg1s_dcase_strict", "S4m_seg1s_dcase_meandir"}
    for n_done, ci in enumerate(clip_ids):
        widx = by_clip[ci]
        n_mel = clip_n_mel(ds, ci)
        offsets = [window_start_frame(ds, w) for w in widx]
        w_lo, w_hi = widx[0], widx[-1] + 1
        probs = sigmoid(np.asarray(cache.logits[w_lo:w_hi])).astype(np.float32)
        doa = np.asarray(cache.doa[w_lo:w_hi])
        events = ds._events[ci]
        tgt10 = ds._build_target(events, n_mel, 0).numpy()                      # (n_mel,N,C,3)
        ra10 = np.linalg.norm(tgt10, axis=-1) > 0.5
        if cmask is not None:
            ra10 &= cmask[None, None, :]
            tgt10 = tgt10 * ra10[..., None]
        if need_stitch:
            pf, vf, _ = stitch_windows(probs, doa, offsets, n_mel, align="free")
        if do_oracle:
            po, vo, _ = stitch_windows(probs, doa, offsets, n_mel, align="oracle",
                                       perm=np.asarray(cache.perm[w_lo:w_hi]))
        # 100-ms grid
        n_grid_frames = n_mel - 1
        G = math.ceil(n_grid_frames / GRID)
        ra100, rv100 = reference_from_events(events, G, N, C)
        if cmask is not None:
            ra100 &= cmask[None, None, :]
            rv100 = rv100 * ra100[..., None]
        if need_stitch:
            p100, v100 = pool_frames(pf[:n_grid_frames], vf[:n_grid_frames], GRID)
        if "B2_stitched_10ms_exact_labels" in use:
            ra10x = np.zeros((n_mel, N, C), bool)
            rv10x = np.zeros((n_mel, N, C, 3), np.float32)
            rep_a = np.repeat(ra100, GRID, axis=0)[:n_mel]
            rep_v = np.repeat(rv100, GRID, axis=0)[:n_mel]
            ra10x[: rep_a.shape[0]] = rep_a
            rv10x[: rep_v.shape[0]] = rep_v
        for t in thresholds:
            if "B_stitched_10ms" in use:
                pa = pf > t
                if cmask is not None:
                    pa &= cmask[None, None, :]
                accB[t].update(pa, vf, ra10, tgt10)
            if "B2_stitched_10ms_exact_labels" in use:
                pa = pf > t
                if cmask is not None:
                    pa &= cmask[None, None, :]
                accB2[t].update(pa, vf, ra10x, rv10x)
            if do_oracle:
                pa = po > t
                if cmask is not None:
                    pa &= cmask[None, None, :]
                accC[t].update(pa, vo, ra10, tgt10)
            if need_stitch:
                pa100 = p100 > t
                if cmask is not None:
                    pa100 &= cmask[None, None, :]
                if "D_stitched_100ms_frame" in use:
                    accD[t].update(pa100, v100, ra100, rv100)
                if "S1_seg100ms_dcase" in use:
                    accS1[t].update(pa100, v100, ra100, rv100)
                if "S2_seg1s_repo_components" in use:
                    accS2[t].update(pa100, v100, ra100, rv100)
                if "S3_seg1s_gated_macroF" in use:
                    accS3[t].update(pa100, v100, ra100, rv100)
                if "S4_seg1s_dcase" in use:
                    accS4[t].update(pa100, v100, ra100, rv100)
                if "S4s_seg1s_dcase_strict" in use:
                    accS4s[t].update(pa100, v100, ra100, rv100)
                if "S4m_seg1s_dcase_meandir" in use:
                    accS4m[t].update(pa100, v100, ra100, rv100)
        if (n_done + 1) % 10 == 0 or n_done + 1 == len(clip_ids):
            log(f"[variants] clip {n_done + 1}/{len(clip_ids)} ({time.time() - t0:.0f} s)")

    results: Dict[str, Dict[float, dict]] = {}
    for name, acc in (("A_repo_pooled_windows", accA), ("B_stitched_10ms", accB),
                      ("B2_stitched_10ms_exact_labels", accB2), ("C_stitched_oracle_10ms", accC), ("D_stitched_100ms_frame", accD),
                      ("S1_seg100ms_dcase", accS1), ("S2_seg1s_repo_components", accS2),
                      ("S3_seg1s_gated_macroF", accS3), ("S4_seg1s_dcase", accS4),
                      ("S4s_seg1s_dcase_strict", accS4s), ("S4m_seg1s_dcase_meandir", accS4m)):
        if name not in use or (name.startswith("C_") and not do_oracle):
            continue
        results[name] = {t: acc[t].compute() for t in thresholds}
    return results


def write_outputs(results: Dict[str, Dict[float, dict]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant, per_thr in results.items():
        for t, m in per_thr.items():
            row = {"variant": variant, "threshold": t}
            row.update({("ER" if k == "ER_20" else "F" if k == "F_20" else k): v
                        for k, v in m.items()})
            rows.append(row)
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with (out_dir / "variants_long.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for variant, per_thr in results.items():
        best_t = min(per_thr, key=lambda t: per_thr[t]["SELD_score"])
        summary[variant] = {"at_0.5": per_thr.get(0.5), "test_optimal_threshold": best_t,
                            "at_test_optimal": per_thr[best_t]}
    (out_dir / "variants_summary.json").write_text(json.dumps(summary, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("Outputs")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--dataset", required=True,
                   choices=["starss23", "tau_nigens2021", "tau_nigens2020", "tau2019"])
    p.add_argument("--split", default="test")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--val-seed", type=int, default=2)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    p.add_argument("--limit-clips", type=int, default=None)
    p.add_argument("--variants", nargs="+", default=None, choices=VARIANTS,
                   help="subset of variants (default: all)")
    args = p.parse_args()

    import yaml
    with open(args.config) as fh:
        raw = yaml.safe_load(fh)
    cfg = raw.get("cfg", raw)
    cache = WindowCache(args.cache)
    ds = build_split_dataset(args.dataset, args.split, args.data_root, cfg, args.val_seed)
    t0 = time.time()
    load0 = machine_load()
    results = evaluate_variants(cache, ds, args.thresholds, limit_clips=args.limit_clips,
                                variants=args.variants)
    write_outputs(results, args.out_dir)
    prov = {
        "script": "analysis/diagnostics/run_metric_variants.py",
        "argv": sys.argv[1:], "cache": str(args.cache), "cache_meta": {
            k: cache.meta.get(k) for k in ("name", "checkpoint", "config", "dataset", "split",
                                           "device", "torch", "platform", "started", "finished",
                                           "elapsed_s", "load_at_start", "load_at_end")},
        "n_windows": cache.n_windows, "thresholds": list(args.thresholds),
        "variants": list(results), "elapsed_s": round(time.time() - t0, 1),
        "load_at_start": load0, "load_at_end": machine_load(),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "S1-S4 are a re-implementation of DCASE-style segment scores "
                "(definitions in analysis/diagnostics/seld_variants.py)",
    }
    (args.out_dir / "provenance.json").write_text(json.dumps(prov, indent=1))
    print(f"[variants] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
