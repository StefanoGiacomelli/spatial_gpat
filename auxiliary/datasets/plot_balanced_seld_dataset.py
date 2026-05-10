#!/usr/bin/env python3
"""
auxiliary/datasets/plot_balanced_seld_dataset.py
=================================================
Visualise the BalancedSTARSS23 training distribution.

Figure 1 — Class distributions  (figures/balanced_starss23_class_dist.svg)
    2 rows × 3 cols:
      Row 0 — full-clip mode    (clips per class)
      Row 1 — windowed mode     (windows per class, 10 s / 2.5 s hop)
      Col 0 — train  (stacked bars: STARSS23 + NIGENS + TAU2019 + mixup)
      Col 1 — val    (STARSS23 val, single colour)
      Col 2 — test   (STARSS23 test, single colour)
    Classes are sorted by training count (descending).
    Windowed row includes a dark "No-Event" bar.

Figure 2 — DOA marginal distributions  (figures/balanced_starss23_doa.svg)
    5 rows × 3 cols = 15 subplots
    Subplots 0-12: one per class
    Subplot  13  : combined "All Classes"
    Subplot  14  : empty
    Each subplot: azimuth histogram (blue) and elevation histogram (green).

Run
---
    python auxiliary/datasets/plot_balanced_seld_dataset.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

_ROOT  = Path(__file__).resolve().parent.parent.parent
FIGDIR = _ROOT / 'figures'
sys.path.insert(0, str(_ROOT))

from datasets.balanced import BalancedSTARSS23Dataset
from datasets.starss23  import STARSS23Dataset
from datasets.utils     import SELDDataset

# ── Constants ──────────────────────────────────────────────────────────────────

STARSS_ROOT  = _ROOT / 'data' / 'STARSS23'
NIGENS_ROOT  = _ROOT / 'data' / 'TAU-NIGENS2021'
TAU2019_ROOT = _ROOT / 'data' / 'TAU2019'

MODEL_SR     = 24_000
HOP_LENGTH   = 240
N_TRACKS     = 3
N_CLASSES    = 13
VAL_FRACTION = 0.2
VAL_SEED     = 2
WINDOW_S     = 10.0
HOP_S        = 2.5
FRAME_DUR    = 0.1   # seconds per annotation frame

SOURCES = ('starss23', 'nigens', 'tau2019', 'mixup')

CLASS_NAMES = [
    'Female speech',    # 0
    'Male speech',      # 1
    'Clapping',         # 2
    'Telephone',        # 3
    'Laughter',         # 4
    'Domestic sounds',  # 5
    'Walk/footsteps',   # 6
    'Door open/close',  # 7
    'Music',            # 8
    'Musical instr.',   # 9
    'Water tap',        # 10
    'Bell',             # 11
    'Knock',            # 12
]

# Source colours for stacked bars
SRC_COLORS = {
    'starss23': '#4C72B0',   # blue
    'nigens':   '#DD8452',   # orange
    'tau2019':  '#55A868',   # green
    'mixup':    '#9467BD',   # purple
}
SILENT_COLOR  = '#2d2d2d'
STARSS_SINGLE = '#4C72B0'    # single-dataset columns (val, test)


# ── Event helpers ──────────────────────────────────────────────────────────────

def _classes_in_item(item: tuple, ds: BalancedSTARSS23Dataset) -> Tuple[str, set]:
    src = item[0]
    if src == 'starss23':
        evts = ds._starss_ds._events[item[1]]
    elif src == 'nigens':
        evts = ds._nigens_proj[item[1]]
    elif src == 'tau2019':
        evts = ds._tau2019_proj[item[1]]
    elif src == 'mixup':
        ci_a, ci_b, ds_b_type = item[1], item[2], item[3]
        evts_a = list(ds._starss_ds._events[ci_a])
        if ds_b_type == 'starss23':
            evts_b = list(ds._starss_ds._events[ci_b])
        elif ds_b_type == 'nigens':
            evts_b = list(ds._nigens_proj[ci_b])
        else:  # 'tau2019'
            evts_b = list(ds._tau2019_proj[ci_b])
        evts = evts_a + evts_b
    else:
        return src, set()
    return src, {cls for (_, cls, _, _, _) in evts if 0 <= cls < N_CLASSES}


def _classes_in_window(item: tuple, ds: BalancedSTARSS23Dataset) -> Tuple[str, set]:
    src = item[0]
    if src == 'starss23':
        ci, start = item[1], item[2]
        evts     = ds._starss_ds._events[ci]
        orig_sr  = ds._starss_ds.orig_sr
        win_samp = ds._win_samp['starss23']
    elif src == 'nigens':
        ci, start = item[1], item[2]
        evts     = ds._nigens_proj[ci]
        orig_sr  = ds._nigens_ds.orig_sr
        win_samp = ds._win_samp['nigens']
    elif src == 'tau2019':
        ci, start = item[1], item[2]
        evts     = ds._tau2019_proj[ci]
        orig_sr  = ds._tau2019_ds.orig_sr
        win_samp = ds._win_samp['tau2019']
    elif src == 'mixup':
        ci_a, ci_b, ds_b_type, start = item[1], item[2], item[3], item[4]
        evts_a = list(ds._starss_ds._events[ci_a])
        if ds_b_type == 'starss23':
            evts_b = list(ds._starss_ds._events[ci_b])
        elif ds_b_type == 'nigens':
            evts_b = list(ds._nigens_proj[ci_b])
        else:  # 'tau2019'
            evts_b = list(ds._tau2019_proj[ci_b])
        evts     = evts_a + evts_b
        orig_sr  = ds._starss_ds.orig_sr   # start is always in STARSS23 SR
        win_samp = ds._win_samp['starss23']
    else:
        return src, set()
    start_f = int(start / orig_sr / FRAME_DUR)
    end_f   = int((start + win_samp) / orig_sr / FRAME_DUR)
    return src, {cls for (f, cls, _, _, _) in evts
                 if start_f <= f < end_f and 0 <= cls < N_CLASSES}


# ── Balanced train clip-level counts ──────────────────────────────────────────

def _balanced_clip_counts(ds_fc: BalancedSTARSS23Dataset) -> Dict[str, np.ndarray]:
    """
    Per-class item counts by source (full-clip).
    Returns dict[source_name → ndarray shape (N_CLASSES,)].
    """
    counts = {s: np.zeros(N_CLASSES, int) for s in SOURCES}
    for it in ds_fc._items:
        src, classes = _classes_in_item(it, ds_fc)
        for c in classes:
            counts[src][c] += 1
    return counts


# ── Balanced train window-level counts ────────────────────────────────────────

def _balanced_win_counts(ds_win: BalancedSTARSS23Dataset) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Per-class window counts by source (windowed mode).
    Returns (dict[source_name → ndarray], n_silent).
    """
    counts  = {s: np.zeros(N_CLASSES, int) for s in SOURCES}
    n_silent = 0
    for it in ds_win._items:
        src, classes = _classes_in_window(it, ds_win)
        if not classes:
            n_silent += 1
        for c in classes:
            counts[src][c] += 1
    return counts, n_silent


# ── STARSS23 counts (val / test) ───────────────────────────────────────────────

def _starss_clip_counts(ds: SELDDataset) -> np.ndarray:
    counts = np.zeros(N_CLASSES, int)
    for ci, _ss, _ in ds._items:
        cls_set = {cls for (_, cls, *_) in ds._events[ci] if cls < N_CLASSES}
        for c in cls_set:
            counts[c] += 1
    return counts


def _starss_win_counts(ds: SELDDataset) -> Tuple[np.ndarray, int]:
    counts   = np.zeros(N_CLASSES, int)
    n_silent = 0
    spc = round(ds.orig_sr * FRAME_DUR)   # samples per annotation frame
    for ci, ss, _ in ds._items:
        s_f = ss // spc
        e_f = (ss + ds.window_samples) // spc
        cls_set = {
            cls for (f, cls, *_) in ds._events[ci]
            if s_f <= f < e_f and cls < N_CLASSES
        }
        if cls_set:
            for c in cls_set:
                counts[c] += 1
        else:
            n_silent += 1
    return counts, n_silent


# ── DOA data ──────────────────────────────────────────────────────────────────

def _collect_doa(ds_fc: BalancedSTARSS23Dataset) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Return dict[class_idx → (az_array, el_array)] from all full-clip items."""
    by_class: Dict[int, Tuple[list, list]] = {c: ([], []) for c in range(N_CLASSES)}
    for item in ds_fc._items:
        src = item[0]
        if src == 'starss23':
            evts = ds_fc._starss_ds._events[item[1]]
        elif src == 'nigens':
            evts = ds_fc._nigens_proj[item[1]]
        elif src == 'tau2019':
            evts = ds_fc._tau2019_proj[item[1]]
        elif src == 'mixup':
            ci_a, ci_b, ds_b_type = item[1], item[2], item[3]
            evts_a = list(ds_fc._starss_ds._events[ci_a])
            if ds_b_type == 'starss23':
                evts_b = list(ds_fc._starss_ds._events[ci_b])
            elif ds_b_type == 'nigens':
                evts_b = list(ds_fc._nigens_proj[ci_b])
            else:  # 'tau2019'
                evts_b = list(ds_fc._tau2019_proj[ci_b])
            evts = evts_a + evts_b
        else:
            continue
        for (_, cls, _, az, el) in evts:
            if 0 <= cls < N_CLASSES:
                by_class[cls][0].append(float(az))
                by_class[cls][1].append(float(el))
    return {c: (np.array(azs), np.array(els)) for c, (azs, els) in by_class.items()}


# ── Plotting helpers ───────────────────────────────────────────────────────────

def _stacked_bar(
    ax: plt.Axes,
    counts_by_src: Dict[str, np.ndarray],   # source → (N_CLASSES,) sorted
    labels: List[str],
    title: str,
    ylabel: str,
    silent: Optional[int] = None,
) -> None:
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels), float)
    for src in SOURCES:
        vals = counts_by_src[src]
        if vals.sum() == 0:
            continue
        ax.bar(x, vals, bottom=bottom, color=SRC_COLORS[src], width=0.7,
               zorder=2, label=src.upper())
        bottom += vals

    # "No-Event" bar appended after the class bars
    if silent is not None:
        x_s = len(labels)
        ax.bar(x_s, silent, color=SILENT_COLOR, width=0.7, zorder=2, label='No-Event')
        all_x      = list(x) + [x_s]
        all_labels = labels + ['No-Event']
        all_totals = list(bottom.astype(int)) + [silent]
    else:
        all_x      = list(x)
        all_labels = labels
        all_totals = list(bottom.astype(int))

    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, rotation=40, ha='right', fontsize=6)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=1)
    ax.set_axisbelow(True)
    for xi, v in zip(all_x, all_totals):
        if v > 0:
            ax.text(xi, v + 0.3, str(v), ha='center', va='bottom', fontsize=6.5)


def _simple_bar(
    ax: plt.Axes,
    counts: np.ndarray,
    labels: List[str],
    color: str,
    title: str,
    ylabel: str,
    silent: Optional[int] = None,
) -> None:
    x = np.arange(len(counts))
    ax.bar(x, counts, color=color, width=0.7, zorder=2)
    if silent is not None:
        x_s = len(counts)
        ax.bar(x_s, silent, color=SILENT_COLOR, width=0.7, zorder=2)
        all_x      = list(x) + [x_s]
        all_labels = labels + ['No-Event']
        all_vals   = list(counts) + [silent]
    else:
        all_x, all_labels, all_vals = list(x), labels, list(counts)
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, rotation=40, ha='right', fontsize=6)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=1)
    ax.set_axisbelow(True)
    for xi, v in zip(all_x, all_vals):
        if v > 0:
            ax.text(xi, v + 0.3, str(v), ha='center', va='bottom', fontsize=6.5)


# ── Figure 1 ──────────────────────────────────────────────────────────────────

def plot_class_distributions(
    ds_fc:   BalancedSTARSS23Dataset,
    ds_win:  BalancedSTARSS23Dataset,
    starss_val_fc:  SELDDataset,
    starss_val_win: SELDDataset,
    starss_te_fc:   SELDDataset,
    starss_te_win:  SELDDataset,
    figpath: Path,
) -> None:
    print('  Computing clip-level counts …')
    tr_clip = _balanced_clip_counts(ds_fc)
    print('  Computing window-level counts …')
    tr_win, tr_silent = _balanced_win_counts(ds_win)

    val_clip              = _starss_clip_counts(starss_val_fc)
    val_win, val_silent   = _starss_win_counts(starss_val_win)
    te_clip               = _starss_clip_counts(starss_te_fc)
    te_win, te_silent     = _starss_win_counts(starss_te_win)

    # Sort order from training clip totals (descending)
    tr_total_clip = sum(tr_clip[s] for s in SOURCES)
    clip_order    = np.argsort(tr_total_clip)[::-1]
    clip_labels   = [CLASS_NAMES[i] for i in clip_order]

    tr_total_win  = sum(tr_win[s] for s in SOURCES)
    win_order     = np.argsort(tr_total_win)[::-1]
    win_labels    = [CLASS_NAMES[i] for i in win_order]

    # Sort all arrays by order
    tr_clip_s  = {s: tr_clip[s][clip_order]   for s in SOURCES}
    tr_win_s   = {s: tr_win[s][win_order]     for s in SOURCES}
    val_clip_s = val_clip[clip_order]
    val_win_s  = val_win[win_order]
    te_clip_s  = te_clip[clip_order]
    te_win_s   = te_win[win_order]

    n_train_clips = len(ds_fc)
    n_train_wins  = len(ds_win)
    n_val_clips   = sum(1 for _ in starss_val_fc._items)
    n_val_wins    = sum(1 for _ in starss_val_win._items)
    n_te_clips    = sum(1 for _ in starss_te_fc._items)
    n_te_wins     = sum(1 for _ in starss_te_win._items)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10), constrained_layout=True)
    fig.suptitle(
        'BalancedSTARSS23 — class occurrences per split  '
        '(sorted by training frequency)\n'
        f'(window: {WINDOW_S:.0f} s, hop: {HOP_S:.1f} s, '
        f'frame res: {int(FRAME_DUR*1000)} ms)',
        fontsize=11,
    )

    # ── Row 0: full-clip ──────────────────────────────────────────────────────
    _stacked_bar(
        axes[0, 0], tr_clip_s, clip_labels,
        title=f'TRAIN  ({n_train_clips} items) — full-clip',
        ylabel='items with class',
    )
    _simple_bar(
        axes[0, 1], val_clip_s, clip_labels, STARSS_SINGLE,
        title=f'VAL  ({n_val_clips} clips, STARSS23) — full-clip',
        ylabel='clips with class',
    )
    _simple_bar(
        axes[0, 2], te_clip_s, clip_labels, '#2ca02c',
        title=f'TEST  ({n_te_clips} clips, STARSS23) — full-clip',
        ylabel='clips with class',
    )

    # ── Row 1: windowed ───────────────────────────────────────────────────────
    _stacked_bar(
        axes[1, 0], tr_win_s, win_labels,
        title=f'TRAIN  ({n_train_wins} windows) — windowed',
        ylabel='windows with class',
        silent=tr_silent,
    )
    _simple_bar(
        axes[1, 1], val_win_s, win_labels, STARSS_SINGLE,
        title=f'VAL  ({n_val_wins} windows, STARSS23) — windowed',
        ylabel='windows with class',
        silent=val_silent,
    )
    _simple_bar(
        axes[1, 2], te_win_s, win_labels, '#2ca02c',
        title=f'TEST  ({n_te_wins} windows, STARSS23) — windowed',
        ylabel='windows with class',
        silent=te_silent,
    )

    # Legend for train column
    patches = [
        mpatches.Patch(color=SRC_COLORS[s], label=s.upper()) for s in SOURCES
    ] + [mpatches.Patch(color=SILENT_COLOR, label='No-Event')]
    axes[0, 0].legend(handles=patches, fontsize=7, loc='upper right',
                      framealpha=0.8)
    axes[1, 0].legend(handles=patches, fontsize=7, loc='upper right',
                      framealpha=0.8)

    figpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figpath, dpi=600, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f'  [fig1] {figpath.relative_to(_ROOT)}')


# ── Figure 2 ──────────────────────────────────────────────────────────────────

def plot_doa_distributions(
    ds_fc:   BalancedSTARSS23Dataset,
    figpath: Path,
) -> None:
    """
    Per-class DOA figure with the same layout as STARSS23_doa_analysis.svg:
    scatter (az × el) at centre, azimuth histogram on top, elevation histogram
    on the right. One 2×2 panel per class + one Overall panel.
    """
    print('  Collecting DOA events …')
    doa = _collect_doa(ds_fc)   # Dict[int, Tuple[np.ndarray, np.ndarray]]

    N_PER_ROW = 4
    n_slots   = N_CLASSES + 1   # per-class + Overall
    n_rows    = math.ceil(n_slots / N_PER_ROW)

    cmap   = plt.get_cmap('tab20')
    colors = [cmap(i % cmap.N) for i in range(N_CLASSES)]

    fig = plt.figure(figsize=(14.0, 3.8 * n_rows))
    outer_gs = GridSpec(
        n_rows, N_PER_ROW,
        figure  = fig,
        hspace  = 0.55,
        wspace  = 0.30,
        left    = 0.05,
        right   = 0.97,
        top     = 0.92,
        bottom  = 0.06,
    )

    def _draw_panel(slot_idx, az, el, title, color, scatter_data=None):
        row = slot_idx // N_PER_ROW
        col = slot_idx % N_PER_ROW

        inner = GridSpecFromSubplotSpec(
            2, 2,
            subplot_spec  = outer_gs[row, col],
            height_ratios = [1, 3],
            width_ratios  = [3, 1],
            hspace        = 0.05,
            wspace        = 0.05,
        )
        ax_az = fig.add_subplot(inner[0, 0])
        ax_sc = fig.add_subplot(inner[1, 0])
        ax_el = fig.add_subplot(inner[1, 1])
        ax_az.sharex(ax_sc)
        ax_el.sharey(ax_sc)

        # ── Scatter (centre) ──────────────────────────────────────────────────
        if scatter_data is not None:
            for az_c, el_c, col_c in scatter_data:
                if len(az_c):
                    ax_sc.scatter(az_c, el_c, s=1.0, alpha=0.15,
                                  color=col_c, linewidths=0, rasterized=True)
        else:
            if len(az):
                ax_sc.scatter(az, el, s=1.5, alpha=0.25,
                              color=color, linewidths=0, rasterized=True)

        ax_sc.set_xlim(-180, 180)
        ax_sc.set_ylim(-90, 90)
        ax_sc.set_xlabel('Azimuth (°)', fontsize=6, labelpad=1)
        ax_sc.set_ylabel('Elevation (°)', fontsize=6, labelpad=1)
        ax_sc.tick_params(axis='both', labelsize=5)
        ax_sc.set_xticks([-180, -90, 0, 90, 180])
        ax_sc.set_yticks([-90, -45, 0, 45, 90])

        # ── Azimuth histogram (top) ───────────────────────────────────────────
        n_pts = (len(az) if scatter_data is None
                 else sum(len(d[0]) for d in scatter_data))
        if scatter_data is not None:
            az_lists = [np.array(d[0]) for d in scatter_data]
            non_empty = [(a, d[2]) for a, d in zip(az_lists, scatter_data) if len(a)]
            if non_empty:
                ax_az.hist([a for a, _ in non_empty], bins=72, range=(-180, 180),
                           stacked=True, color=[c for _, c in non_empty],
                           alpha=0.85, linewidth=0)
        else:
            if len(az):
                ax_az.hist(az, bins=72, range=(-180, 180),
                           color=color, alpha=0.85, linewidth=0)

        ax_az.set_title(f'{title}\n({n_pts:,} frames)', fontsize=7,
                        pad=2, fontweight='bold')
        ax_az.tick_params(labelbottom=False, labelsize=5)
        ax_az.set_yticks([])
        ax_az.spines[['top', 'right', 'left']].set_visible(False)

        # ── Elevation histogram rotated (right) ───────────────────────────────
        if scatter_data is not None:
            el_lists = [np.array(d[1]) for d in scatter_data]
            non_empty = [(e, d[2]) for e, d in zip(el_lists, scatter_data) if len(e)]
            if non_empty:
                ax_el.hist([e for e, _ in non_empty], bins=36, range=(-90, 90),
                           stacked=True, orientation='horizontal',
                           color=[c for _, c in non_empty], alpha=0.85, linewidth=0)
        else:
            if len(el):
                ax_el.hist(el, bins=36, range=(-90, 90), orientation='horizontal',
                           color=color, alpha=0.85, linewidth=0)

        ax_el.tick_params(labelleft=False, labelsize=5)
        ax_el.set_xticks([])
        ax_el.spines[['top', 'right', 'bottom']].set_visible(False)

    # ── Per-class panels ──────────────────────────────────────────────────────
    for i, label in enumerate(CLASS_NAMES):
        az_arr, el_arr = doa.get(i, (np.array([]), np.array([])))
        _draw_panel(i, az_arr, el_arr, label, colors[i])

    # ── Overall panel ─────────────────────────────────────────────────────────
    scatter_data = [
        (doa.get(c, (np.array([]), np.array([])))[0],
         doa.get(c, (np.array([]), np.array([])))[1],
         colors[c])
        for c in range(N_CLASSES)
    ]
    _draw_panel(N_CLASSES, np.array([]), np.array([]), 'Overall', None,
                scatter_data=scatter_data)

    # ── Hide remaining empty slots ────────────────────────────────────────────
    for j in range(n_slots, n_rows * N_PER_ROW):
        ax_d = fig.add_subplot(outer_gs[j // N_PER_ROW, j % N_PER_ROW])
        ax_d.set_visible(False)

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=colors[c], markersize=5, label=CLASS_NAMES[c])
        for c in range(N_CLASSES)
    ]
    fig.legend(
        handles        = handles,
        loc            = 'lower center',
        ncol           = min(N_CLASSES, 7),
        fontsize       = 6,
        frameon        = False,
        bbox_to_anchor = (0.5, 0.0),
        bbox_transform = fig.transFigure,
    )

    fig.suptitle(
        'BalancedSTARSS23 — DOA distribution per class'
        ' (train split, native 100 ms resolution)',
        fontsize = 10,
        y        = 0.995,
    )

    figpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figpath, dpi=600, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f'  [fig2] {figpath.relative_to(_ROOT)}')


# ── Main ───────────────────────────────────────────────────────────────────────

def _check_roots() -> List[str]:
    missing = []
    for name, root in [('STARSS23', STARSS_ROOT),
                        ('TAU-NIGENS2021', NIGENS_ROOT),
                        ('TAU2019', TAU2019_ROOT)]:
        if not root.exists():
            missing.append(f'{name} ({root})')
    return missing


def main() -> None:
    missing = _check_roots()
    if missing:
        for m in missing:
            print(f'[ERROR] Data root not found: {m}')
        sys.exit(1)

    common_bal = dict(
        starss_root  = STARSS_ROOT,
        nigens_root  = NIGENS_ROOT,
        tau2019_root = TAU2019_ROOT,
        model_sr     = MODEL_SR,
        hop_length   = HOP_LENGTH,
        n_tracks     = N_TRACKS,
        val_fraction = VAL_FRACTION,
        val_seed     = VAL_SEED,
        augment      = None,
        balance_seed = 2,
    )
    common_starss = dict(
        root         = STARSS_ROOT,
        model_sr     = MODEL_SR,
        hop_length   = HOP_LENGTH,
        n_tracks     = N_TRACKS,
        val_fraction = VAL_FRACTION,
        val_seed     = VAL_SEED,
        augment      = None,
    )

    print('Building BalancedSTARSS23Dataset (full-clip) …')
    ds_fc  = BalancedSTARSS23Dataset(window_s=None,     hop_s=None,  **common_bal)
    print('Building BalancedSTARSS23Dataset (windowed) …')
    ds_win = BalancedSTARSS23Dataset(window_s=WINDOW_S, hop_s=HOP_S, **common_bal)

    print('Building STARSS23 val/test datasets …')
    starss_val_fc  = STARSS23Dataset(split='val',  window_s=None,     hop_s=None,  **common_starss)
    starss_val_win = STARSS23Dataset(split='val',  window_s=WINDOW_S, hop_s=HOP_S, **common_starss)
    starss_te_fc   = STARSS23Dataset(split='test', window_s=None,     hop_s=None,  **common_starss)
    starss_te_win  = STARSS23Dataset(split='test', window_s=WINDOW_S, hop_s=HOP_S, **common_starss)

    FIGDIR.mkdir(parents=True, exist_ok=True)

    print('\nPlotting Figure 1 (class distributions) …')
    plot_class_distributions(
        ds_fc, ds_win,
        starss_val_fc, starss_val_win,
        starss_te_fc,  starss_te_win,
        figpath=FIGDIR / 'balanced_starss23_class_dist.svg',
    )

    print('Plotting Figure 2 (DOA marginals) …')
    plot_doa_distributions(
        ds_fc,
        figpath=FIGDIR / 'balanced_starss23_doa.svg',
    )

    print('\nDone.')


if __name__ == '__main__':
    main()
