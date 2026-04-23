"""
tests/test_seld_class_balance.py
=================================
Class-balance and class-representativeness analysis for STARSS23 and
TAU-NIGENS2021, for each available split, in both clip-mode and windowed-mode.

All analysis is annotation-only (CSV-based) — no audio is loaded, so the
script runs in seconds even on large datasets.

For each dataset × split the script reports:

  Frame-level table (one row per class)
  ───────────────────────────────────────
  • active frames       — unique (clip, frame_100ms) pairs where class is active
  • % of total frames   — active / total_unique_frames × 100
  • pos_weight          — neg_frames / pos_frames  (used in the BCE loss)

  Coverage table (one row per class)
  ────────────────────────────────────
  • clip coverage       — % of clips that contain ≥1 active frame of the class
  • window coverage     — % of sliding windows (default 10 s / 5 s hop)
                          that contain ≥1 active frame; proxy for how often
                          per epoch the model receives gradient for each class

  Cross-split comparison (train vs test)
  ────────────────────────────────────────
  • Δ% = test% − train%  (distribution shift indicator per class)

  Summary
  ────────
  • Imbalance ratio  max(pos_weight) / min(pos_weight)
  • Most / least represented class per metric

Output
------
  Terminal : formatted tables
  CSV      : results/class_balance_{DATASET}_{SPLIT}.csv

Usage
-----
  # Analyse both datasets, both modes, default 10 s / 5 s windows
  .venv/bin/python tests/test_seld_class_balance.py

  # Single dataset
  .venv/bin/python tests/test_seld_class_balance.py --dataset STARSS23

  # Custom window
  .venv/bin/python tests/test_seld_class_balance.py --window-s 5 --hop-s 2.5

  # Skip CSV export
  .venv/bin/python tests/test_seld_class_balance.py --no-csv

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

RESULTS_DIR = _ROOT / 'results'

# ── Dataset metadata ──────────────────────────────────────────────────────────

ROOTS: Dict[str, Path] = {
    'STARSS23':       _ROOT / 'data' / 'STARSS23',
    'TAU-NIGENS2021': _ROOT / 'data' / 'TAU-NIGENS2021',
}

CLASS_LABELS: Dict[str, List[str]] = {
    'STARSS23': [
        'Female speech', 'Male speech', 'Clapping', 'Telephone', 'Laughter',
        'Domestic sounds', 'Walk/footsteps', 'Door', 'Music',
        'Musical instr.', 'Water tap', 'Bell', 'Knock',
    ],
    'TAU-NIGENS2021': [
        'Alarm', 'Crying baby', 'Crash', 'Barking dog',
        'Female scream', 'Female speech', 'Footsteps', 'Knocking on door',
        'Male scream', 'Male speech', 'Ringing phone', 'Piano',
    ],
}

N_CLASSES: Dict[str, int] = {
    'STARSS23':       13,
    'TAU-NIGENS2021': 12,
}

# Official split → metadata subdirectories
SPLIT_DIRS: Dict[str, Dict[str, List[str]]] = {
    'STARSS23': {
        'train': ['dev-train-sony', 'dev-train-tau'],
        'test':  ['dev-test-sony',  'dev-test-tau'],
    },
    'TAU-NIGENS2021': {
        'train': ['dev-train'],
        'val':   ['dev-val'],
        'test':  ['dev-test'],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Annotation loading
# ══════════════════════════════════════════════════════════════════════════════

def _collect_annotations(
    dataset: str,
    split: str,
) -> Tuple[Dict[str, Dict[int, set]], int]:
    """
    Parse all annotation CSVs for a dataset/split.

    Returns:
        per_clip  : clip_stem → {frame_100ms → set(class_idx)}
        n_classes : number of classes for this dataset
    """
    root      = ROOTS[dataset]
    meta_root = root / 'metadata_dev'
    subdirs   = SPLIT_DIRS[dataset][split]
    n_cls     = N_CLASSES[dataset]

    per_clip: Dict[str, Dict[int, set]] = {}

    for sub in subdirs:
        sub_path = meta_root / sub
        if not sub_path.exists():
            print(f'  [WARN] {sub_path} not found — skipping')
            continue
        for csv_p in sorted(sub_path.glob('*.csv')):
            frames: Dict[int, set] = defaultdict(set)
            with open(csv_p) as fh:
                for line in fh:
                    parts = line.strip().split(',')
                    if len(parts) < 2:
                        continue
                    frame_idx = int(parts[0])
                    cls_idx   = int(parts[1])
                    if cls_idx < n_cls:
                        frames[frame_idx].add(cls_idx)
            per_clip[csv_p.stem] = dict(frames)

    return per_clip, n_cls


# ══════════════════════════════════════════════════════════════════════════════
# Statistics
# ══════════════════════════════════════════════════════════════════════════════

def _frame_stats(
    per_clip: Dict[str, Dict[int, set]],
    n_cls: int,
) -> Tuple[int, List[int]]:
    """Total unique (clip, frame) pairs and per-class active frame counts."""
    total  = 0
    active = [0] * n_cls
    for frames in per_clip.values():
        total += len(frames)
        for classes in frames.values():
            for c in classes:
                active[c] += 1
    return total, active


def _clip_coverage(
    per_clip: Dict[str, Dict[int, set]],
    n_cls: int,
) -> List[int]:
    """For each class: number of clips that contain at least one active frame."""
    clip_hits = [0] * n_cls
    for frames in per_clip.values():
        present: set = set()
        for classes in frames.values():
            present.update(classes)
        for c in present:
            clip_hits[c] += 1
    return clip_hits


def _window_coverage(
    per_clip: Dict[str, Dict[int, set]],
    n_cls: int,
    window_frames: int,
    hop_frames: int,
) -> Tuple[int, List[int]]:
    """
    Simulate sliding-window on annotation frames (100 ms resolution).

    A window is positive for class c if any annotation frame inside it has c.
    The window slides from frame 0 to the last annotated frame of each clip.

    Returns:
        total_windows : total window count across all clips
        win_hits      : per-class count of windows containing each class
    """
    total_windows = 0
    win_hits = [0] * n_cls

    for frames in per_clip.values():
        if not frames:
            continue
        max_frame = max(frames.keys())
        start = 0
        while start <= max_frame:
            end = start + window_frames      # exclusive upper bound
            present: set = set()
            for f in range(start, end):
                if f in frames:
                    present.update(frames[f])
            total_windows += 1
            for c in present:
                if c < n_cls:
                    win_hits[c] += 1
            start += hop_frames

    return total_windows, win_hits


def _pos_weights(total_frames: int, active: List[int]) -> List[float]:
    return [(total_frames - max(a, 1)) / max(a, 1) for a in active]


# ══════════════════════════════════════════════════════════════════════════════
# Printing
# ══════════════════════════════════════════════════════════════════════════════

def _sep(char: str = '─', w: int = 80) -> None:
    print(char * w)


def _bar(value: float, total: float, width: int = 18) -> str:
    filled = int(width * value / max(total, 1))
    return '█' * filled + '░' * (width - filled)


def _print_split_analysis(
    dataset: str,
    split: str,
    total_frames: int,
    active: List[int],
    n_clips: int,
    clip_hits: List[int],
    total_windows: int,
    win_hits: List[int],
    window_s: float,
    hop_s: float,
) -> None:
    labels  = CLASS_LABELS[dataset]
    n_cls   = len(labels)
    pw_list = _pos_weights(total_frames, active)

    _sep()
    print(f'  {dataset}  |  split={split}  |  '
          f'clips={n_clips}  |  frames={total_frames:,}  |  '
          f'windows={total_windows:,} ({window_s:.0f}s/{hop_s:.0f}s)')
    _sep()

    # ── Frame-level + pos_weight ──────────────────────────────────────────────
    print(f'\n  {"─"*76}')
    print(f'  {"Frame-level activity and pos_weight (BCE imbalance correction)"}')
    print(f'  {"─"*76}')
    hdr = f'  {"cls":>3}  {"Class":<20}  {"Frames":>8}  {"Fr%":>6}  {"pos_wt":>8}  bar'
    print(hdr)
    print(f'  {"─"*3}  {"─"*20}  {"─"*8}  {"─"*6}  {"─"*8}  {"─"*18}')

    for c in range(n_cls):
        pct = 100.0 * active[c] / max(total_frames, 1)
        bar = _bar(active[c], total_frames)
        print(f'  {c:3d}  {labels[c]:<20}  {active[c]:8,}  {pct:6.2f}%  '
              f'{pw_list[c]:8.1f}  {bar}')

    # ── Coverage table ────────────────────────────────────────────────────────
    print(f'\n  {"─"*76}')
    print(f'  {"Sample-level coverage (clip-mode vs window-mode)"}')
    print(f'  {"─"*76}')
    print(f'  {"cls":>3}  {"Class":<20}  {"ClipCov":>8}  {"WinCov":>8}  '
          f'  clip bar          win bar')
    print(f'  {"─"*3}  {"─"*20}  {"─"*8}  {"─"*8}')

    for c in range(n_cls):
        c_pct = 100.0 * clip_hits[c] / max(n_clips, 1)
        w_pct = 100.0 * win_hits[c]  / max(total_windows, 1)
        c_bar = _bar(clip_hits[c], n_clips)
        w_bar = _bar(win_hits[c],  total_windows)
        print(f'  {c:3d}  {labels[c]:<20}  {c_pct:7.1f}%  {w_pct:7.1f}%'
              f'  {c_bar}  {w_bar}')

    # ── Summary ───────────────────────────────────────────────────────────────
    pw_max_c = max(range(n_cls), key=lambda c: pw_list[c])
    pw_min_c = min(range(n_cls), key=lambda c: pw_list[c])
    ratio    = pw_list[pw_max_c] / max(pw_list[pw_min_c], 1e-9)

    print(f'\n  Imbalance ratio (max_pos_weight / min_pos_weight) : {ratio:.1f}×')
    print(f'  Most  underrepresented: [{pw_max_c:2d}] {labels[pw_max_c]:<20}  '
          f'pos_weight = {pw_list[pw_max_c]:.1f}')
    print(f'  Least underrepresented: [{pw_min_c:2d}] {labels[pw_min_c]:<20}  '
          f'pos_weight = {pw_list[pw_min_c]:.2f}')
    print()


def _print_shift_table(
    dataset: str,
    total_train: int,
    active_train: List[int],
    total_test: int,
    active_test: List[int],
) -> None:
    labels = CLASS_LABELS[dataset]
    n_cls  = len(labels)

    _sep('═', 80)
    print(f'  {dataset}  |  Distribution shift: train → test')
    _sep('═', 80)
    print(f'  {"cls":>3}  {"Class":<20}  {"Train%":>7}  {"Test%":>7}  '
          f'{"Δ%":>7}  note')
    print(f'  {"─"*3}  {"─"*20}  {"─"*7}  {"─"*7}  {"─"*7}  {"─"*16}')

    shifts = []
    for c in range(n_cls):
        tr_pct = 100.0 * active_train[c] / max(total_train, 1)
        te_pct = 100.0 * active_test[c]  / max(total_test,  1)
        delta  = te_pct - tr_pct
        shifts.append(abs(delta))
        if   delta >  2.0:  note = '▲ more in test'
        elif delta < -2.0:  note = '▼ less in test'
        else:               note = '≈ stable'
        print(f'  {c:3d}  {labels[c]:<20}  {tr_pct:7.2f}%  {te_pct:7.2f}%  '
              f'{delta:+7.2f}%  {note}')

    max_c = max(range(n_cls), key=lambda c: shifts[c])
    print(f'\n  Largest shift: [{max_c}] {labels[max_c]}  '
          f'(|Δ| = {shifts[max_c]:.2f}%)')
    print()


# ══════════════════════════════════════════════════════════════════════════════
# CSV export
# ══════════════════════════════════════════════════════════════════════════════

def _export_csv(
    dataset: str,
    split: str,
    total_frames: int,
    active: List[int],
    n_clips: int,
    clip_hits: List[int],
    total_windows: int,
    win_hits: List[int],
    window_s: float,
    hop_s: float,
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fname   = RESULTS_DIR / f'class_balance_{dataset}_{split}.csv'
    labels  = CLASS_LABELS[dataset]
    n_cls   = len(labels)
    pw_list = _pos_weights(total_frames, active)

    with open(fname, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow([
            'class_idx', 'class_name',
            'active_frames', 'active_frames_pct',
            'pos_weight',
            'clip_coverage_pct',
            f'window_coverage_pct_w{window_s:.0f}s_h{hop_s:.0f}s',
        ])
        for c in range(n_cls):
            fr_pct = 100.0 * active[c]    / max(total_frames,  1)
            cl_pct = 100.0 * clip_hits[c] / max(n_clips,       1)
            wi_pct = 100.0 * win_hits[c]  / max(total_windows, 1)
            writer.writerow([
                c, labels[c],
                active[c], f'{fr_pct:.4f}',
                f'{pw_list[c]:.4f}',
                f'{cl_pct:.4f}',
                f'{wi_pct:.4f}',
            ])

    # JSON sidecar with aggregate counts (read by plot_class_balance.py)
    meta = {
        'dataset':    dataset,
        'split':      split,
        'n_clips':    n_clips,
        'n_windows':  total_windows,
        'window_s':   window_s,
        'hop_s':      hop_s,
    }
    meta_path = RESULTS_DIR / f'class_balance_{dataset}_{split}_meta.json'
    with open(meta_path, 'w') as fh:
        json.dump(meta, fh, indent=2)

    return fname


# ══════════════════════════════════════════════════════════════════════════════
# Per-dataset driver
# ══════════════════════════════════════════════════════════════════════════════

def analyse_dataset(
    dataset: str,
    window_s: float,
    hop_s: float,
    export_csv: bool,
) -> None:
    window_frames = max(1, round(window_s * 10))
    hop_frames    = max(1, round(hop_s    * 10))

    _sep('═', 80)
    print(f'  DATASET : {dataset}')
    print(f'  Window  : {window_s} s  →  {window_frames} annotation frames @ 100 ms')
    print(f'  Hop     : {hop_s} s  →  {hop_frames} annotation frames @ 100 ms')
    _sep('═', 80)
    print()

    splits   = list(SPLIT_DIRS[dataset].keys())
    all_data = {}   # split → (total_frames, active)

    for split in splits:
        print(f'  Loading {split} annotations …', end=' ', flush=True)
        per_clip, n_cls = _collect_annotations(dataset, split)
        n_clips         = len(per_clip)
        print(f'{n_clips} clips loaded', flush=True)

        total_frames, active    = _frame_stats(per_clip, n_cls)
        clip_hits               = _clip_coverage(per_clip, n_cls)
        total_windows, win_hits = _window_coverage(
            per_clip, n_cls, window_frames, hop_frames
        )
        all_data[split] = (total_frames, active)

        print()
        _print_split_analysis(
            dataset, split,
            total_frames, active,
            n_clips, clip_hits,
            total_windows, win_hits,
            window_s, hop_s,
        )

        if export_csv:
            out = _export_csv(
                dataset, split,
                total_frames, active,
                n_clips, clip_hits,
                total_windows, win_hits,
                window_s, hop_s,
            )
            print(f'  CSV → {out.relative_to(_ROOT)}')
            print()

    if 'train' in all_data and 'test' in all_data:
        t_total, t_active = all_data['train']
        e_total, e_active = all_data['test']
        _print_shift_table(dataset, t_total, t_active, e_total, e_active)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description='Class-balance and coverage analysis for SELD datasets.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--dataset',
                   choices=['STARSS23', 'TAU-NIGENS2021', 'both'],
                   default='both',
                   help='Dataset(s) to analyse.')
    p.add_argument('--window-s', type=float, default=10.0,
                   help='Sliding-window duration in seconds.')
    p.add_argument('--hop-s',    type=float, default=5.0,
                   help='Sliding-window hop in seconds.')
    p.add_argument('--no-csv',   action='store_true',
                   help='Skip CSV export to results/.')
    args = p.parse_args()

    datasets = (
        ['STARSS23', 'TAU-NIGENS2021'] if args.dataset == 'both'
        else [args.dataset]
    )

    for ds in datasets:
        if not ROOTS[ds].exists():
            print(f'[SKIP] {ds} — root not found at {ROOTS[ds]}')
            continue
        analyse_dataset(ds, args.window_s, args.hop_s, not args.no_csv)

    print('Done.')


if __name__ == '__main__':
    main()
