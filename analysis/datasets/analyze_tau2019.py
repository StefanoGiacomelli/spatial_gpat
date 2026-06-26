#!/usr/bin/env python3
"""
analysis/datasets/analyze_tau2019.py
======================================
Exploratory analysis of the TAU2019 dataset.

Figures saved to figures/ (SVG, 600 DPI):
    TAU2019_doa_polar.svg         — azimuth + elevation polar plots per class
    TAU2019_cooccurrence.svg      — class co-occurrence heatmap
    TAU2019_event_duration.svg    — event duration distributions per class
    TAU2019_clip_timeline.svg     — Gantt-style timeline for 5 random clips

Run
---
    python analysis/datasets/analyze_tau2019.py

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np

_ROOT   = Path(__file__).resolve().parents[2]
FIGDIR  = _ROOT / 'figures' / 'datasets'
DATA_DIR = _ROOT / 'data' / 'TAU2019'
sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CLASS_NAMES = [
    'clearthroat', 'cough', 'doorslam', 'drawer', 'keyboard',
    'keysDrop', 'knock', 'laughter', 'pageturn', 'phone', 'speech',
]
N_CLASSES = 11


# ── CSV parsing ───────────────────────────────────────────────────────────────

def parse_csv(csv_path: Path) -> List[dict]:
    """Return list of events: {cls, start, end, azi, ele, duration}."""
    events   = []
    class_map = {n: i for i, n in enumerate(CLASS_NAMES)}
    with open(csv_path, newline='') as fh:
        for row in csv.DictReader(fh):
            label = row['sound_event_recording']
            if label not in class_map:
                continue
            start = float(row['start_time'])
            end   = float(row['end_time'])
            events.append({
                'cls':      class_map[label],
                'start':    start,
                'end':      end,
                'duration': end - start,
                'azi':      float(row['azi']),
                'ele':      float(row['ele']),
            })
    return events


def collect_all(dev_root: Path, eval_root: Path) -> List[Tuple[Path, List[dict]]]:
    """Collect all clips from dev (splits 1-4) and eval."""
    clips = []
    meta_dev = dev_root / 'metadata_dev'
    for split_n in [1, 2, 3, 4]:
        for csv_p in sorted(meta_dev.glob(f'split{split_n}_*.csv')):
            clips.append((csv_p, parse_csv(csv_p)))
    meta_eval = eval_root / 'metadata_eval'
    if meta_eval.exists():
        for csv_p in sorted(meta_eval.glob('split0_*.csv')):
            clips.append((csv_p, parse_csv(csv_p)))
    return clips


# ── DOA polar plot ────────────────────────────────────────────────────────────

def plot_doa_polar(all_clips: List[Tuple[Path, List[dict]]]) -> None:
    """Polar histograms: azimuth (top row) and elevation (bottom row) per class."""
    fig, axes = plt.subplots(
        2, N_CLASSES,
        figsize=(24, 6),
        subplot_kw={'projection': 'polar'},
    )
    cmap   = plt.get_cmap('tab20', N_CLASSES)
    colors = [cmap(i) for i in range(N_CLASSES)]

    for c in range(N_CLASSES):
        azis_deg = []
        eles_deg = []
        for _, evts in all_clips:
            for e in evts:
                if e['cls'] == c:
                    azis_deg.append(e['azi'])
                    eles_deg.append(e['ele'])

        # ── Top row: azimuth ─────────────────────────────────────────────────
        ax_az = axes[0, c]
        ax_az.set_theta_zero_location('N')
        ax_az.set_theta_direction(-1)
        tick_angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        ax_az.set_xticks(tick_angles)
        ax_az.set_xticklabels(
            [f'{int(np.degrees(a)):d}°' for a in tick_angles],
            fontsize=4,
        )
        ax_az.tick_params(axis='x', pad=1)
        ax_az.set_yticks([])
        ax_az.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)

        if azis_deg:
            az_rad = np.radians(np.array(azis_deg) % 360)
            bins   = np.linspace(0, 2 * np.pi, 25)
            ax_az.hist(az_rad, bins=bins, color=colors[c], alpha=0.8)

        ax_az.set_title(CLASS_NAMES[c], fontsize=6.5, pad=3, fontweight='bold')

        # ── Bottom row: elevation ─────────────────────────────────────────────
        ax_el = axes[1, c]
        ax_el.set_theta_zero_location('N')
        ax_el.set_theta_direction(-1)
        el_ticks = np.linspace(-np.pi / 2, np.pi / 2, 7)
        el_ticks_pos = (el_ticks + np.pi / 2)  # shift to [0, π]
        ax_el.set_xticks(el_ticks_pos)
        ax_el.set_xticklabels(
            [f'{int(np.degrees(a) - 90):d}°' for a in el_ticks_pos],
            fontsize=4,
        )
        ax_el.tick_params(axis='x', pad=1)
        ax_el.set_yticks([])
        ax_el.set_thetamin(0)
        ax_el.set_thetamax(180)
        ax_el.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)

        if eles_deg:
            el_rad = np.radians(np.array(eles_deg) + 90)  # shift to [0, 180°]
            bins   = np.linspace(0, np.pi, 13)
            ax_el.hist(el_rad, bins=bins, color=colors[c], alpha=0.8)

    axes[0, 0].set_ylabel('Azimuth', fontsize=8, labelpad=15)
    axes[1, 0].set_ylabel('Elevation', fontsize=8, labelpad=15)
    fig.suptitle(
        'TAU2019 — DOA distributions per class  (top: azimuth, bottom: elevation)',
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIGDIR / 'TAU2019_doa_polar.svg'
    fig.savefig(out, dpi=600, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f'[fig] {out.relative_to(_ROOT)}')


# ── Co-occurrence heatmap ─────────────────────────────────────────────────────

def plot_cooccurrence(all_clips: List[Tuple[Path, List[dict]]]) -> None:
    """Build and save the cooccurrence plot."""

    mat = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for _, evts in all_clips:
        classes = {e['cls'] for e in evts}
        for c1 in classes:
            for c2 in classes:
                mat[c1, c2] += 1

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(mat, cmap='YlOrRd')
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(CLASS_NAMES, fontsize=7)
    plt.colorbar(im, ax=ax, label='n_clips')
    ax.set_title('TAU2019 — class co-occurrence in same clip')
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, str(mat[i, j]), ha='center', va='center', fontsize=5)
    plt.tight_layout()
    out = FIGDIR / 'TAU2019_cooccurrence.svg'
    fig.savefig(out, dpi=600, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f'[fig] {out.relative_to(_ROOT)}')


# ── Event duration boxplot ────────────────────────────────────────────────────

def plot_event_durations(all_clips: List[Tuple[Path, List[dict]]]) -> None:
    """Build and save the event durations plot."""

    durations = defaultdict(list)
    for _, evts in all_clips:
        for e in evts:
            durations[e['cls']].append(e['duration'])

    data_list = [durations.get(c, [0.0]) for c in range(N_CLASSES)]

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.boxplot(data_list, labels=CLASS_NAMES, showfliers=True)
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Duration (s)')
    ax.set_title('TAU2019 — event duration per class')
    plt.tight_layout()
    out = FIGDIR / 'TAU2019_event_duration.svg'
    fig.savefig(out, dpi=600, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f'[fig] {out.relative_to(_ROOT)}')


# ── Clip timeline (Gantt) ─────────────────────────────────────────────────────

def plot_timelines(all_clips: List[Tuple[Path, List[dict]]]) -> None:
    """5 clips selected randomly, one per split folder."""
    rng = np.random.default_rng(42)

    # Group clips by split number (split1, split2, split3, split4, eval/split0)
    by_split: dict = defaultdict(list)
    for csv_p, evts in all_clips:
        stem = csv_p.stem                        # e.g. 'split1_room1_mix001'
        sn   = stem.split('_')[0].replace('split', '')
        by_split[sn].append((csv_p, evts))

    sample = []
    for sn in sorted(by_split.keys()):
        pool = by_split[sn]
        if pool:
            sample.append(pool[rng.integers(0, len(pool))])
        if len(sample) >= 5:
            break

    cmap = plt.get_cmap('tab20', N_CLASSES)
    n    = len(sample)
    fig, axes = plt.subplots(n, 1, figsize=(14, 0.8 * n + 1), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, (csv_p, evts) in zip(axes, sample):
        max_t = max((e['end'] for e in evts), default=1.0)
        for e in evts:
            ax.barh(0, e['end'] - e['start'], left=e['start'],
                    height=0.6, color=cmap(e['cls']), alpha=0.85)
        ax.set_xlim(0, max_t + 0.1)
        ax.set_yticks([])
        ax.set_ylabel(csv_p.stem, fontsize=6, rotation=0, ha='right', labelpad=4)
        ax.tick_params(axis='x', labelsize=6)

    patches = [
        mpatches.Patch(color=cmap(c), label=CLASS_NAMES[c])
        for c in range(N_CLASSES)
    ]
    fig.legend(handles=patches, ncol=6, loc='lower center',
               fontsize=6, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle('TAU2019 — clip event timelines (5 random clips)', fontsize=9)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    out = FIGDIR / 'TAU2019_clip_timeline.svg'
    fig.savefig(out, dpi=600, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f'[fig] {out.relative_to(_ROOT)}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    """Run the command-line workflow."""

    if not DATA_DIR.exists():
        print(f'[ERROR] TAU2019 data not found at {DATA_DIR}')
        sys.exit(1)

    FIGDIR.mkdir(parents=True, exist_ok=True)

    dev_root  = DATA_DIR / 'dev'
    eval_root = DATA_DIR / 'eval'

    print('Collecting clips …')
    all_clips = collect_all(dev_root, eval_root)
    print(f'  {len(all_clips)} clips, '
          f'{sum(len(e) for _, e in all_clips)} events total')

    print('Generating figures …')
    plot_doa_polar(all_clips)
    plot_cooccurrence(all_clips)
    plot_event_durations(all_clips)
    plot_timelines(all_clips)
    print(f'\nAll figures saved to {FIGDIR.relative_to(_ROOT)}')


if __name__ == '__main__':
    main()
