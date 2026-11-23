#!/usr/bin/env python3
"""
analysis/diagnostics/check_against_public_dcase.py
==============================================================================
Equivalence check: segment scorer of ``seld_variants.py`` vs. the public DCASE baseline code
==============================================================================

The DCASE-style rows of ``run_metric_variants.py`` (``S1_seg100ms_dcase``, ``S4_seg1s_dcase``) are scored
with ``seld_variants.SegmentCounts(official_compat=True)``, a numba re-implementation of the arithmetic of
the public DCASE-2022/2023 SELD baseline (``SELD_evaluation_metrics.py``).  This script feeds the same random
reference/prediction pairs to both implementations and reports the largest difference of ER, F, LE/180, LR
and the SELD score.

The public file is **not** part of this repository and is not redistributed (its header states an MIT licence).
Fetch it yourself into a directory of your choice, e.g.

    mkdir -p /tmp/dcase_ref && cd /tmp/dcase_ref
    curl -LO https://raw.githubusercontent.com/sharathadavanne/seld-dcase2023/main/SELD_evaluation_metrics.py

and run

    python analysis/diagnostics/check_against_public_dcase.py --official-dir /tmp/dcase_ref

The file was written for an older NumPy (it uses the removed alias ``np.float``) and imports ``IPython``;
both are shimmed for the duration of the check only (the public file itself is used unmodified).

Verified when this script was written (2026-10-26, 18 random cases: 5 classes, 3 tracks, segments of 1, 4 and
10 frames, three clips each): maximum difference 2e-8.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import types
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.diagnostics.seld_variants import SegmentCounts  # noqa: E402


def _to_blocks(act: np.ndarray, vec: np.ndarray, seg_frames: int) -> Dict[int, dict]:
    """(frames, slots, classes) activity + (frames, slots, classes, 3) vectors -> the nested dictionary the
    public implementation expects: ``{segment: {class: [[frame_ids, [rows per frame]]]}}`` with one
    ``[track_id, x, y, z]`` row per active slot (list order = slot order)."""
    n_frames, n_slots, n_classes = act.shape
    n_blocks = (n_frames + seg_frames - 1) // seg_frames
    out: Dict[int, dict] = {b: {} for b in range(n_blocks)}
    for b in range(n_blocks):
        for c in range(n_classes):
            frames, vals = [], []
            for f in range(b * seg_frames, min(n_frames, (b + 1) * seg_frames)):
                rows = [[n, *vec[f, n, c]] for n in range(n_slots) if act[f, n, c]]
                if rows:
                    frames.append(f - b * seg_frames)
                    vals.append(rows)
            if frames:
                out[b][c] = [[frames, vals]]
    return out


def _load_public_module(official_dir: Path):
    """Import the public ``SELD_evaluation_metrics.py`` from ``official_dir`` with the two shims, then undo them."""
    path = Path(official_dir) / "SELD_evaluation_metrics.py"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found (see the instructions in the module docstring)")
    had_alias = hasattr(np, "float")
    old_ipython = sys.modules.get("IPython")
    if not had_alias:
        np.float = float  # type: ignore[attr-defined]
    sys.modules["IPython"] = types.SimpleNamespace(embed=lambda *a, **k: None)  # type: ignore[assignment]
    try:
        spec = importlib.util.spec_from_file_location("public_dcase_seld_metrics", path)
        module = importlib.util.module_from_spec(spec)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec.loader.exec_module(module)  # type: ignore[union-attr]
    finally:
        if old_ipython is None:
            sys.modules.pop("IPython", None)
        else:
            sys.modules["IPython"] = old_ipython
    return module


def run_check(official_dir: Path, seg_frames=(1, 4, 10), seeds: int = 6, n_classes: int = 5, n_slots: int = 3,
              clips: int = 3, verbose: bool = False) -> Tuple[float, int]:
    """Largest absolute difference over (ER, F, LE/180, LR, SELD) and the number of random cases compared."""
    official = _load_public_module(Path(official_dir))
    worst, cases = 0.0, 0
    for k in seg_frames:
        for seed in range(seeds):
            rng = np.random.default_rng(100 * k + seed)
            ref = official.SELDMetrics(doa_threshold=20, nb_classes=n_classes, average="macro")
            mine = SegmentCounts(n_classes, k)
            for _ in range(clips):
                n_frames = int(rng.integers(30, 70))
                ref_a = rng.random((n_frames, n_slots, n_classes)) < 0.2
                ref_a[0, 0, :] = True                      # every class has at least one reference event
                rv = rng.normal(size=(n_frames, n_slots, n_classes, 3))
                rv /= np.linalg.norm(rv, axis=-1, keepdims=True)
                pred_a = ref_a & (rng.random(ref_a.shape) < 0.75) | (rng.random(ref_a.shape) < 0.07)
                pv = rv + rng.normal(0, 0.3 if seed % 2 else 0.8, rv.shape)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ref.update_seld_scores(_to_blocks(pred_a, pv, k), _to_blocks(ref_a, rv, k))
                mine.update(pred_a, pv.astype(np.float32), ref_a, rv.astype(np.float32))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                er, f, le, lr, seld, _ = ref.compute_seld_scores()
            m = mine.compute()
            d = max(abs(er - m["ER"]), abs(f - m["F"]), abs(le - m["LE_CD"]) / 180.0, abs(lr - m["LR_CD"]),
                    abs(seld - m["SELD_score"]))
            worst = max(worst, float(d))
            cases += 1
            if verbose or d > 1e-4:
                print(f"segments of {k:2d} frames, seed {seed}: |diff| = {d:.2e}"
                      f"  public (ER, F, LE, LR, SELD) = {[round(float(x), 4) for x in (er, f, le, lr, seld)]}"
                      f"  here = {[round(m[key], 4) for key in ('ER', 'F', 'LE_CD', 'LR_CD', 'SELD_score')]}")
    return worst, cases


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("The public file")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--official-dir", type=Path, required=True,
                    help="directory containing the public SELD_evaluation_metrics.py")
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    worst, cases = run_check(args.official_dir, verbose=args.verbose)
    print(f"max |difference| over ER, F, LE/180, LR, SELD ({cases} random cases): {worst:.3e}")
    if worst > args.tol:
        print(f"MISMATCH (tolerance {args.tol:g})")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
