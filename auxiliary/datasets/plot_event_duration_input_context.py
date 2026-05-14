#!/usr/bin/env python3
"""
auxiliary/datasets/plot_event_duration_input_context.py
========================================================
4 × 2 figure of event-duration statistics for the input-context experiments.

  Rows : STARSS23 | TAU-NIGENS2021 | TAU2019 | BalancedSTARSS23
  Cols :
    Left  — full-clip mode: boxplot of event durations per class as seen by
            the model (annotation segments extracted from the actual dataset
            objects, same code path as training).
    Right — windowed mode: grouped boxplot (w10 / w20 / w40, hop = 2 s) of
            the *effective* event duration inside each window — i.e. the
            portion of each event that the model sees in one training window.
            Long events are counted once per window they span, with the
            truncated duration.  Short events (< window_s) appear fully.

Statistics source
-----------------
Events are loaded from actual ``SELDDataset._events`` objects, NOT from raw
CSVs.  BalancedSTARSS23 reflects the true clip-selection / projection /
mixup-union logic of ``BalancedSTARSS23Dataset._build_clip_index``.
TAU2019 uses ``mixed_mode=True`` (dev splits 2–4, 300 clips), consistent
with its role inside BalancedSTARSS23.

Windowing simulation uses the same 50 % threshold as
``BalancedSTARSS23Dataset._expand_to_windows``:
    include window if  start + window_frames // 2 <= clip_len_frames

Y-axis: log scale in both columns; independent range per subplot.

Usage
-----
    .venv/bin/python auxiliary/datasets/plot_event_duration_input_context.py \\
        [--starss-root  data/STARSS23]        \\
        [--tau-root     data/TAU-NIGENS2021]  \\
        [--tau2019-root data/TAU2019]         \\
        [--out          figures/input_context_event_duration.svg]
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker


# ── Class name tables ─────────────────────────────────────────────────────────

STARSS23_CLASSES = [
    'Female speech', 'Male speech', 'Clapping', 'Telephone', 'Laughter',
    'Domestic sounds', 'Walk, footsteps', 'Door open/close', 'Music',
    'Musical instrument', 'Water tap', 'Bell', 'Knock',
]  # 13 classes, index 0–12

NIGENS_CLASSES = [
    'Alarm', 'Baby cry', 'Crash', 'Dog bark', 'Female scream',
    'Female speech', 'Footstep', 'Knock', 'Male scream', 'Male speech',
    'Phone ring', 'Piano',
]  # 12 classes, index 0–11

TAU2019_CLASSES = [
    'Clearthroat', 'Cough', 'Doorslam', 'Drawer', 'Keyboard',
    'KeysDrop', 'Knock', 'Laughter', 'Pageturn', 'Phone', 'Speech',
]  # 11 classes, index 0–10

BALANCED_CLASSES = STARSS23_CLASSES  # 13-class STARSS23 projection space

# ── Windowing parameters for the right column ─────────────────────────────────

WINDOW_SIZES_S = [10.0, 20.0, 40.0]   # three window sizes to compare
HOP_S          = 2.0                   # fixed hop for all window sizes
FRAME_DUR      = 0.1                   # 100 ms annotation resolution (all datasets)

# Colours for w10 / w20 / w40
WIN_COLORS  = ['#4e9af1', '#f1914e', '#5cb85c']
WIN_LABELS  = ['w10 (10 s)', 'w20 (20 s)', 'w40 (40 s)']


# ══════════════════════════════════════════════════════════════════════════════
# Event segmentation
# ══════════════════════════════════════════════════════════════════════════════

def get_event_segments(
    events: List[Tuple],
) -> List[Tuple[int, int, int, int]]:
    """Convert frame-level events to contiguous segments.

    Parameters
    ----------
    events : list of (frame_100ms, class_idx, source_idx, az, el)

    Returns
    -------
    list of (class_idx, source_idx, start_frame, end_frame)
        ``start_frame`` and ``end_frame`` are inclusive, in 100 ms units.
    """
    class_source_frames: Dict = defaultdict(lambda: defaultdict(list))
    for tup in events:
        frame, cls, src = int(tup[0]), int(tup[1]), int(tup[2])
        class_source_frames[cls][src].append(frame)

    segments: List[Tuple[int, int, int, int]] = []
    for cls, src_dict in class_source_frames.items():
        for src, frames in src_dict.items():
            frames = sorted(set(frames))
            seg_start = seg_end = frames[0]
            for f in frames[1:]:
                if f == seg_end + 1:
                    seg_end = f
                else:
                    segments.append((cls, src, seg_start, seg_end))
                    seg_start = seg_end = f
            segments.append((cls, src, seg_start, seg_end))
    return segments


# ══════════════════════════════════════════════════════════════════════════════
# Duration collection
# ══════════════════════════════════════════════════════════════════════════════

def collect_fullclip_durations(
    all_events: List[List[Tuple]],
) -> Dict[int, List[float]]:
    """Return ``{class_idx: [duration_s, ...]}`` for full-clip mode."""
    durs: Dict[int, List[float]] = defaultdict(list)
    for events in all_events:
        for cls, src, s, e in get_event_segments(events):
            durs[cls].append((e - s + 1) * FRAME_DUR)
    return durs


def _clip_len_frames(src: Union[Path, float]) -> int:
    """Clip length in 100 ms frames.  Accepts a Path or a duration in seconds."""
    if isinstance(src, (int, float)):
        return round(float(src) / FRAME_DUR)
    return round(sf.info(str(src)).duration / FRAME_DUR)


def collect_windowed_durations(
    all_events: List[List[Tuple]],
    clip_refs:  List[Union[Path, float]],   # Path → read duration; float → duration in s
    window_s:   float,
    hop_s:      float,
) -> Dict[int, List[float]]:
    """Return ``{class_idx: [eff_dur_s, ...]}`` simulating windowed training.

    For every window that passes the 50 % threshold
    (``start + win_frames // 2 <= clip_len_frames``),
    the effective duration of each event intersected with that window is recorded.
    """
    win_frames = round(window_s / FRAME_DUR)
    hop_frames = round(hop_s    / FRAME_DUR)

    durs: Dict[int, List[float]] = defaultdict(list)

    for events, ref in zip(all_events, clip_refs):
        segs = get_event_segments(events)
        T    = _clip_len_frames(ref)

        w_start = 0
        while w_start + win_frames // 2 <= T:
            w_end = w_start + win_frames   # exclusive

            for cls, src, e_start, e_end in segs:
                eff_start     = max(e_start, w_start)
                eff_end_excl  = min(e_end + 1, w_end)
                if eff_end_excl > eff_start:
                    durs[cls].append((eff_end_excl - eff_start) * FRAME_DUR)

            w_start += hop_frames

    return durs


# ══════════════════════════════════════════════════════════════════════════════
# Dataset loading
# ══════════════════════════════════════════════════════════════════════════════

def _audio_paths(ds) -> List[Path]:
    return [Path(p) for (p, _) in ds.clips]


def load_starss23(root: Path):
    from datasets.starss23 import STARSS23Dataset
    ds = STARSS23Dataset(
        root       = root,
        split      = 'train',
        model_sr   = 24000,
        hop_length = 240,
        n_tracks   = 3,
        window_s   = None,
    )
    return ds._events, _audio_paths(ds), STARSS23_CLASSES


def load_tau_nigens(root: Path):
    from datasets.tau_nigens import TAUNIGENSDataset
    ds = TAUNIGENSDataset(
        root       = root,
        split      = 'train',
        model_sr   = 24000,
        hop_length = 240,
        n_tracks   = 3,
        window_s   = None,
    )
    return ds._events, _audio_paths(ds), NIGENS_CLASSES


def load_tau2019(root: Path):
    from datasets.tau2019 import TAU2019Dataset
    # mixed_mode=True: dev splits 2-4 (300 clips), consistent with balanced usage
    ds = TAU2019Dataset(
        root       = root,
        split      = 'train',
        mixed_mode = True,
        model_sr   = 24000,
        hop_length = 240,
        n_tracks   = 3,
        window_s   = None,
    )
    return ds._events, _audio_paths(ds), TAU2019_CLASSES


def load_balanced(starss_root: Path, nigens_root: Path, tau2019_root: Path):
    """Load BalancedSTARSS23Dataset and return events + clip refs for each item.

    For windowed-mode simulation, ``clip_refs`` carries the *effective* clip
    duration in seconds:
      - single clips   → audio duration from soundfile
      - mixup pairs    → min(dur_a, dur_b) in seconds (mirrors _expand_to_windows)
    """
    from datasets.balanced import BalancedSTARSS23Dataset

    ds = BalancedSTARSS23Dataset(
        starss_root  = starss_root,
        nigens_root  = nigens_root,
        tau2019_root = tau2019_root,
        model_sr     = 24000,
        hop_length   = 240,
        n_tracks     = 3,
        window_s     = None,   # full-clip mode → _items are clip-level tuples
    )

    all_events: List[List[Tuple]] = []
    clip_refs:  List[Union[Path, float]] = []

    def _dur(path: Path) -> float:
        return sf.info(str(path)).duration

    for item in ds._items:
        t = item[0]

        if t == 'starss23':
            ci = item[1]
            all_events.append(ds._starss_ds._events[ci])
            clip_refs.append(Path(ds._starss_ds.clips[ci][0]))

        elif t == 'nigens':
            ci = item[1]
            all_events.append(ds._nigens_proj[ci])
            clip_refs.append(Path(ds._nigens_ds.clips[ci][0]))

        elif t == 'tau2019':
            ci = item[1]
            all_events.append(ds._tau2019_proj[ci])
            clip_refs.append(Path(ds._tau2019_ds.clips[ci][0]))

        elif t == 'mixup':
            ci_a, ci_b, type_b = item[1], item[2], item[3]

            evts_a = ds._starss_ds._events[ci_a]
            path_a = Path(ds._starss_ds.clips[ci_a][0])
            dur_a  = _dur(path_a)

            if type_b == 'starss23':
                evts_b = ds._starss_ds._events[ci_b]
                dur_b  = _dur(Path(ds._starss_ds.clips[ci_b][0]))
            elif type_b == 'nigens':
                evts_b = ds._nigens_proj[ci_b]
                dur_b  = _dur(Path(ds._nigens_ds.clips[ci_b][0]))
            else:  # 'tau2019'
                evts_b = ds._tau2019_proj[ci_b]
                dur_b  = _dur(Path(ds._tau2019_ds.clips[ci_b][0]))

            # Union events; effective clip length = min of the two
            all_events.append(list(evts_a) + list(evts_b))
            clip_refs.append(min(dur_a, dur_b))   # float → effective duration

    return all_events, clip_refs, BALANCED_CLASSES


# ══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_data_list(durs: Dict[int, List[float]], n_classes: int) -> List[List[float]]:
    """Build an ordered list of duration arrays (one per class, never empty)."""
    return [durs.get(c) or [FRAME_DUR] for c in range(n_classes)]


def plot_fullclip(ax: plt.Axes, durs: Dict[int, List[float]], class_names: List[str]) -> None:
    """Standard boxplot — one box per class, full-clip durations."""
    n = len(class_names)
    data = _make_data_list(durs, n)

    bp = ax.boxplot(
        data,
        positions  = list(range(n)),
        widths     = 0.55,
        showfliers = True,
        flierprops = dict(marker='.', markersize=2, alpha=0.4,
                          markerfacecolor='#888888', markeredgecolor='none'),
        medianprops   = dict(color='#c0392b', linewidth=1.5),
        boxprops      = dict(color='#2c3e50'),
        whiskerprops  = dict(color='#2c3e50', linestyle='--', linewidth=0.8),
        capprops      = dict(color='#2c3e50'),
        patch_artist  = True,
    )
    for patch in bp['boxes']:
        patch.set_facecolor('#dce9f5')

    ax.set_yscale('log')
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=[1, 2, 5]))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f'{v:.0f} s' if v >= 1 else f'{v:.1f} s'
    ))
    ax.set_xticks(range(n))
    ax.set_xticklabels(class_names, fontsize=6.5, rotation=45, ha='right')
    ax.set_xlim(-0.7, n - 0.3)
    ax.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.6)
    ax.set_xlabel('')
    ax.set_ylabel('Event duration (s)', fontsize=8)


def plot_windowed(ax: plt.Axes, durs_list: List[Dict[int, List[float]]],
                  class_names: List[str]) -> None:
    """Grouped boxplot — three boxes per class (w10, w20, w40).

    Parameters
    ----------
    durs_list : [durs_w10, durs_w20, durs_w40]
    """
    n        = len(class_names)
    n_win    = len(WINDOW_SIZES_S)
    gwidth   = 0.8              # total width of one class group
    bwidth   = gwidth / n_win   # width of each individual box

    # Offsets from class centre for each window size
    offsets = [(i - (n_win - 1) / 2) * bwidth for i in range(n_win)]

    for wi, (durs, color, label, offset) in enumerate(
            zip(durs_list, WIN_COLORS, WIN_LABELS, offsets)):
        data      = _make_data_list(durs, n)
        positions = [c + offset for c in range(n)]

        bp = ax.boxplot(
            data,
            positions  = positions,
            widths     = bwidth * 0.85,
            showfliers = True,
            flierprops = dict(marker='.', markersize=1.5, alpha=0.3,
                              markerfacecolor=color, markeredgecolor='none'),
            medianprops   = dict(color='#333333', linewidth=1.2),
            boxprops      = dict(color='#333333', linewidth=0.8),
            whiskerprops  = dict(color='#555555', linestyle='--', linewidth=0.7),
            capprops      = dict(color='#555555', linewidth=0.7),
            patch_artist  = True,
        )
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax.set_yscale('log')
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=[1, 2, 5]))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f'{v:.0f} s' if v >= 1 else f'{v:.1f} s'
    ))
    ax.set_xticks(range(n))
    ax.set_xticklabels(class_names, fontsize=6.5, rotation=45, ha='right')
    ax.set_xlim(-0.7, n - 0.3)
    ax.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.6)
    ax.set_xlabel('')
    ax.set_ylabel('Effective duration (s)', fontsize=8)

    # Legend
    patches = [mpatches.Patch(facecolor=c, alpha=0.7, label=l)
               for c, l in zip(WIN_COLORS, WIN_LABELS)]
    ax.legend(handles=patches, fontsize=7, loc='upper right',
              framealpha=0.85, edgecolor='none')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

ROW_CONFIG = [
    ('STARSS23',          load_starss23,   'starss_root'),
    ('TAU-NIGENS2021',    load_tau_nigens, 'tau_root'),
    ('TAU2019',           load_tau2019,    'tau2019_root'),
    ('BalancedSTARSS23',  None,            None),   # special handling
]

def main():
    p = argparse.ArgumentParser(
        description='Plot event-duration statistics for input-context experiments.'
    )
    p.add_argument('--starss-root',  type=Path,
                   default=_ROOT / 'data' / 'STARSS23')
    p.add_argument('--tau-root',     type=Path,
                   default=_ROOT / 'data' / 'TAU-NIGENS2021')
    p.add_argument('--tau2019-root', type=Path,
                   default=_ROOT / 'data' / 'TAU2019')
    p.add_argument('--out',          type=Path,
                   default=_ROOT / 'figures' / 'input_context_event_duration.svg')
    args = p.parse_args()

    starss_root  = args.starss_root.resolve()
    tau_root     = args.tau_root.resolve()
    tau2019_root = args.tau2019_root.resolve()

    # ── Load all datasets ─────────────────────────────────────────────────────
    print('Loading STARSS23 …')
    s23_events, s23_refs, s23_cls = load_starss23(starss_root)

    print('Loading TAU-NIGENS2021 …')
    tng_events, tng_refs, tng_cls = load_tau_nigens(tau_root)

    print('Loading TAU2019 …')
    t19_events, t19_refs, t19_cls = load_tau2019(tau2019_root)

    print('Loading BalancedSTARSS23 …')
    bal_events, bal_refs, bal_cls = load_balanced(starss_root, tau_root, tau2019_root)

    all_rows = [
        ('STARSS23',         s23_events, s23_refs, s23_cls),
        ('TAU-NIGENS2021',   tng_events, tng_refs, tng_cls),
        ('TAU2019',          t19_events, t19_refs, t19_cls),
        ('BalancedSTARSS23', bal_events, bal_refs, bal_cls),
    ]

    # ── Precompute durations ──────────────────────────────────────────────────
    print('Computing full-clip durations …')
    fullclip_durs = [collect_fullclip_durations(evts) for evts, _, _ in
                     [(r[1], r[2], r[3]) for r in all_rows]]

    windowed_durs = []   # list of [durs_w10, durs_w20, durs_w40] per row
    for row_name, evts, refs, _ in all_rows:
        row_wins = []
        for ws in WINDOW_SIZES_S:
            print(f'  Windowed {row_name} w={ws:.0f}s …')
            row_wins.append(collect_windowed_durations(evts, refs, ws, HOP_S))
        windowed_durs.append(row_wins)

    # ── Build figure ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        nrows   = 4,
        ncols   = 2,
        figsize = (22, 26),
    )
    fig.subplots_adjust(hspace=0.55, wspace=0.28,
                        left=0.06, right=0.98, top=0.96, bottom=0.04)

    for row_idx, (row_name, evts, refs, cls_names) in enumerate(all_rows):
        ax_l = axes[row_idx, 0]
        ax_r = axes[row_idx, 1]
        n_cls = len(cls_names)

        # Left: full-clip boxplot
        plot_fullclip(ax_l, fullclip_durs[row_idx], cls_names)
        ax_l.set_title(f'{row_name} — full-clip  ({n_cls} classes, {len(evts)} clips)',
                       fontsize=9, fontweight='bold', pad=4)

        # Right: grouped windowed boxplot
        plot_windowed(ax_r, windowed_durs[row_idx], cls_names)
        ax_r.set_title(f'{row_name} — windowed (hop = {HOP_S:.0f} s)',
                       fontsize=9, fontweight='bold', pad=4)

    # Column headers
    axes[0, 0].annotate('Full-clip mode', xy=(0.5, 1.12), xycoords='axes fraction',
                         ha='center', fontsize=12, fontweight='bold', color='#2c3e50')
    axes[0, 1].annotate('Windowed mode comparison', xy=(0.5, 1.12),
                         xycoords='axes fraction', ha='center', fontsize=12,
                         fontweight='bold', color='#2c3e50')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.out), dpi=600, format='svg', bbox_inches='tight')
    print(f'\nSaved → {args.out}')


if __name__ == '__main__':
    main()
