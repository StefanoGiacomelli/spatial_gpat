#!/usr/bin/env python3
"""
auxiliary/datasets/analyze_starss23.py
=======================================
Class-occurrence bar plots for STARSS23 — full-clip and windowed mode.

Figure layout (2 rows × 3 cols):
  Row 0 — full-clip:  how many clips in each split contain at least one
           occurrence of each class (binary presence per clip).
  Row 1 — windowed:   how many 10-s windows (stride 2.5 s) contain at
           least one frame of each class, plus a "Silent" bar (no events)
           shown in black.

Splits (3 cols):
  col 0 — train   (80 % of dev-train-sony + dev-train-tau, stratified by room)
  col 1 — val     (remaining 20 %, same stratification)
  col 2 — test    (dev-test-sony + dev-test-tau, no split)

STARSS23 annotation resolution: 100 ms / frame.

Run
---
    python auxiliary/datasets/analyze_starss23.py  [--root PATH]

Output
------
    figures/STARSS23_class_occurrences.svg
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_ROOT  = Path(__file__).resolve().parent.parent.parent
FIGDIR = _ROOT / 'figures'
sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Dataset constants ──────────────────────────────────────────────────────────

FRAME_DUR   = 0.1        # seconds — STARSS23 annotation resolution (100 ms)
WINDOW_S    = 10.0       # windowed-mode clip length
HOP_S       = 2.5        # windowed-mode hop
WINDOW_FRAMES = int(WINDOW_S / FRAME_DUR)   # 100
HOP_FRAMES    = int(HOP_S   / FRAME_DUR)    # 25

VAL_FRACTION = 0.2
VAL_SEED     = 2

CLASS_NAMES = [
    'Female speech',
    'Male speech',
    'Clapping',
    'Telephone',
    'Laughter',
    'Domestic sounds',
    'Walk, footsteps',
    'Door open/close',
    'Music',
    'Musical instr.',
    'Water tap',
    'Bell',
    'Knock',
]
N_CLASSES = 13

SPLIT_LABELS = ['train', 'val', 'test']
SILENT_COLOR = '#2d2d2d'


# ── CSV parsing ────────────────────────────────────────────────────────────────

def parse_csv(csv_path: Path) -> List[Tuple[int, int]]:
    """Return (frame, class_idx) pairs from a STARSS23 annotation CSV."""
    events: List[Tuple[int, int]] = []
    with open(csv_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            try:
                events.append((int(parts[0]), int(parts[1])))
            except (ValueError, IndexError):
                continue   # skip header or malformed rows
    return events


# ── Split builder (mirrors STARSS23Dataset._collect_clips) ────────────────────

def _room_key(csv_path: Path) -> str:
    parent = csv_path.parent.name          # 'dev-train-sony' or 'dev-train-tau'
    site   = 'sony' if 'sony' in parent else 'tau'
    parts  = csv_path.stem.split('_')     # fold3_room21_mix001
    return f'{site}_{parts[1]}'


def collect_splits(meta_root: Path) -> Dict[str, List[Path]]:
    """Return {'train': [...], 'val': [...], 'test': [...]} CSV path lists."""
    # ── test ──────────────────────────────────────────────────────────────────
    test_paths: List[Path] = []
    for sub in ('dev-test-sony', 'dev-test-tau'):
        test_paths.extend(sorted((meta_root / sub).glob('*.csv')))

    # ── train / val — stratified by room ──────────────────────────────────────
    all_train: List[Path] = []
    for sub in ('dev-train-sony', 'dev-train-tau'):
        all_train.extend(sorted((meta_root / sub).glob('*.csv')))

    groups: Dict[str, List[Path]] = {}
    for p in all_train:
        groups.setdefault(_room_key(p), []).append(p)

    rng = np.random.default_rng(VAL_SEED)
    train_paths: List[Path] = []
    val_paths:   List[Path] = []

    for _key, room_clips in sorted(groups.items()):
        n_val = max(1, round(len(room_clips) * VAL_FRACTION))
        shuffle_idx = rng.permutation(len(room_clips))
        val_set = set(shuffle_idx[:n_val].tolist())
        for i, p in enumerate(room_clips):
            (val_paths if i in val_set else train_paths).append(p)

    return {'train': train_paths, 'val': val_paths, 'test': test_paths}


# ── Count functions ────────────────────────────────────────────────────────────

def count_clip_occurrences(csv_paths: List[Path]) -> np.ndarray:
    """Count clips that contain ≥1 frame of each class (shape: N_CLASSES)."""
    counts = np.zeros(N_CLASSES, dtype=int)
    for p in csv_paths:
        events = parse_csv(p)
        present = {cls for _, cls in events if 0 <= cls < N_CLASSES}
        for c in present:
            counts[c] += 1
    return counts


def count_window_occurrences(csv_paths: List[Path]) -> Tuple[np.ndarray, int, int]:
    """
    Slide WINDOW_FRAMES / HOP_FRAMES over each clip's annotation.
    Returns (class_counts, silent_count, total_windows) where:
      class_counts[c] = windows containing ≥1 frame of class c
      silent_count    = windows with no events at all
      total_windows   = silent_count + number of non-silent windows
    """
    class_counts  = np.zeros(N_CLASSES, dtype=int)
    silent_count  = 0
    total_windows = 0

    for p in csv_paths:
        events = parse_csv(p)
        if not events:
            # entirely silent clip — skip (no audio length available)
            silent_count  += 1
            total_windows += 1
            continue

        max_frame = max(f for f, _ in events)
        n_frames  = max_frame + 1

        start = 0
        while start + WINDOW_FRAMES <= n_frames:
            win_end = start + WINDOW_FRAMES
            present = {
                cls for (f, cls) in events
                if start <= f < win_end and 0 <= cls < N_CLASSES
            }
            total_windows += 1
            if present:
                for c in present:
                    class_counts[c] += 1
            else:
                silent_count += 1
            start += HOP_FRAMES

        # partial tail window (if remainder ≥ HOP_FRAMES, include it)
        if start < n_frames and (n_frames - start) >= HOP_FRAMES:
            win_end = n_frames
            present = {
                cls for (f, cls) in events
                if start <= f < win_end and 0 <= cls < N_CLASSES
            }
            total_windows += 1
            if present:
                for c in present:
                    class_counts[c] += 1
            else:
                silent_count += 1

    return class_counts, silent_count, total_windows


# ── Plotting ───────────────────────────────────────────────────────────────────

def _bar(
    ax: plt.Axes,
    counts: np.ndarray,
    labels: List[str],
    color,
    ylabel: str,
    title: str,
) -> None:
    x = np.arange(len(counts))
    bars = ax.bar(x, counts, color=color, width=0.7, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=6.5)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=1)
    ax.set_axisbelow(True)
    for rect, v in zip(bars.patches, counts):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.5,
            str(v),
            ha='center', va='bottom', fontsize=8,
        )


def _bar_with_silent(
    ax: plt.Axes,
    counts: np.ndarray,
    labels: List[str],
    silent: int,
    color,
    ylabel: str,
    title: str,
) -> None:
    """Bar plot with an extra 'No-Event' bar in black at the right."""
    x_cls    = np.arange(len(counts))
    x_silent = len(counts)          # one slot after the last class

    bars  = ax.bar(x_cls,    counts, color=color,       width=0.7, zorder=2)
    s_bar = ax.bar(x_silent, silent, color=SILENT_COLOR, width=0.7, zorder=2)

    all_x      = list(x_cls) + [x_silent]
    all_labels = labels + ['No-Event']
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, rotation=40, ha='right', fontsize=6.5)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=1)
    ax.set_axisbelow(True)

    all_bars = list(bars.patches) + list(s_bar.patches)
    all_vals = list(counts) + [silent]
    for rect, v in zip(all_bars, all_vals):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.5,
            str(v),
            ha='center', va='bottom', fontsize=8,
            color='black',
        )


def plot_figure(
    splits: Dict[str, List[Path]],
    figpath: Path,
) -> None:
    cmap   = plt.get_cmap('tab10')
    colors = [cmap(0), cmap(1), cmap(2)]   # blue, orange, green

    # ── Determine sort order from training split ───────────────────────────────
    train_paths = splits['train']
    train_clip_counts               = count_clip_occurrences(train_paths)
    train_win_counts, _, _          = count_window_occurrences(train_paths)

    clip_order  = np.argsort(train_clip_counts)[::-1]   # descending by train freq
    win_order   = np.argsort(train_win_counts)[::-1]
    clip_labels = [CLASS_NAMES[i] for i in clip_order]
    win_labels  = [CLASS_NAMES[i] for i in win_order]

    fig, axes = plt.subplots(
        2, 3,
        figsize=(18, 9),
        constrained_layout=True,
    )
    fig.suptitle(
        'STARSS23 — class occurrences per split  (sorted by training frequency)\n'
        f'(window: {WINDOW_S} s, hop: {HOP_S} s, frame res: {int(FRAME_DUR*1000)} ms)',
        fontsize=11,
    )

    for col, split_name in enumerate(SPLIT_LABELS):
        csv_paths = splits[split_name]
        color     = colors[col]
        n_clips   = len(csv_paths)

        # ── Row 0: full-clip ─────────────────────────────────────────────────
        clip_counts = count_clip_occurrences(csv_paths)
        _bar(
            axes[0, col],
            clip_counts[clip_order],
            clip_labels,
            color,
            ylabel='clips with class',
            title=f'{split_name.upper()}  ({n_clips} clips) — full-clip',
        )

        # ── Row 1: windowed ──────────────────────────────────────────────────
        win_counts, silent_count, total_windows = count_window_occurrences(csv_paths)
        _bar_with_silent(
            axes[1, col],
            win_counts[win_order],
            win_labels,
            silent_count,
            color,
            ylabel='windows with class',
            title=f'{split_name.upper()}  ({total_windows} windows) — windowed',
        )

    figpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figpath, dpi=600, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f'[fig] {figpath.relative_to(_ROOT)}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='STARSS23 class occurrence analysis')
    parser.add_argument(
        '--root',
        type=Path,
        default=_ROOT / 'data' / 'STARSS23',
        help='Path to STARSS23 root directory',
    )
    args = parser.parse_args()

    data_root = args.root
    meta_root = data_root / 'metadata_dev'

    if not meta_root.exists():
        print(f'[ERROR] metadata_dev not found at {meta_root}')
        sys.exit(1)

    print('Collecting clips …')
    splits = collect_splits(meta_root)
    for name, paths in splits.items():
        print(f'  {name:5s}: {len(paths):3d} clips')

    print('Counting occurrences …')
    plot_figure(splits, FIGDIR / 'STARSS23_class_occurrences.svg')
    print(f'\nDone — figure saved to {(FIGDIR / "STARSS23_class_occurrences.svg").relative_to(_ROOT)}')


if __name__ == '__main__':
    main()
