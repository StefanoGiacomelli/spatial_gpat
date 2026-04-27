"""
tests/plot_audioset_strong.py — AudioSet-Strong Dataset Analysis
================================================================
Pre-training analysis of the AudioSet-Strong dataset.  All statistics are
computed **post-windowing** (on the training item index), so they directly
reflect what the model sees during training at each window duration.

Outputs
-------
``results/audioset_strong_stats_train.csv``
    Per-class statistics for the training split:
    class_index, mid, display_name, n_events_train,
    pos_windows_10s, pos_rate_10s

``results/audioset_strong_window_stats.csv``
    Label density (pos_rate) and negative-window fraction for each
    window_s in {0.5, 1, 2, 5, 10} seconds — aggregated over all classes.

``figures/audioset_strong_class_dist.svg``
    Bar chart of the 100 most frequent classes (by number of positive
    windows at window_s=10s).

``figures/audioset_strong_events_per_clip.svg``
    Histogram of the number of (unique-class) events per clip.

``figures/audioset_strong_neg_fraction_vs_window.svg``
    Line plot: fraction of negative-only windows as a function of window_s.

Usage
-----
::

    # Uses default paths from configs/datasets/AudioSetStrong.yaml
    .venv/bin/python tests/plot_audioset_strong.py

    # Override window sizes to analyze
    .venv/bin/python tests/plot_audioset_strong.py \\
        --window-sizes 0.5 1 2 5 10

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Project path setup ────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml

# ══════════════════════════════════════════════════════════════════════════════
# Style constants
# ══════════════════════════════════════════════════════════════════════════════

FIGURE_DPI     = 600
FIGURE_FORMAT  = 'svg'
FONT_FAMILY    = 'DejaVu Sans'
GRID_ALPHA     = 0.25
SPINE_COLOR    = '#cccccc'
BG_COLOR       = '#f9f9f9'

BAR_COLOR      = '#3498db'
BAR_ALPHA      = 0.80
LINE_COLOR     = '#e74c3c'
LINE_WIDTH     = 2.0
MARKER_STYLE   = 'o'
MARKER_SIZE    = 7

# ══════════════════════════════════════════════════════════════════════════════
# Defaults
# ══════════════════════════════════════════════════════════════════════════════

_DATASET_CFG    = _ROOT / 'configs' / 'datasets' / 'AudioSetStrong.yaml'
_RESULTS_DIR    = _ROOT / 'results'
_FIGURES_DIR    = _ROOT / 'figures'
_DEFAULT_WINDOW_SIZES = [0.5, 1.0, 2.0, 5.0, 10.0]


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description='Analyze AudioSet-Strong dataset statistics post-windowing.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--window-sizes', type=float, nargs='+',
                   default=_DEFAULT_WINDOW_SIZES,
                   help='Window durations in seconds to analyze.')
    p.add_argument('--val-fraction', type=float, default=0.2,
                   help='Fraction of clips held out for validation.')
    p.add_argument('--val-seed',    type=int,   default=42,
                   help='RNG seed for train/val split.')
    p.add_argument('--top-n-classes', type=int, default=100,
                   help='Number of most-frequent classes to show in bar chart.')
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Data loading helpers (standalone — no Dataset instantiation)
# ══════════════════════════════════════════════════════════════════════════════

def _load_class_map(class_map_csv: Path) -> Tuple[Dict[str, int], Dict[int, str], Dict[int, str]]:
    """
    Returns:
        mid2idx   : MID → int index
        idx2mid   : int index → MID
        idx2name  : int index → display_name
    """
    mid2idx:  Dict[str, int] = {}
    idx2mid:  Dict[int, str] = {}
    idx2name: Dict[int, str] = {}
    with open(class_map_csv, newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            idx  = int(row['index'])
            mid  = row['mid'].strip()
            name = row['display_name'].strip()
            mid2idx[mid] = idx
            idx2mid[idx] = mid
            idx2name[idx] = name
    return mid2idx, idx2mid, idx2name


def _load_events(
    strong_tsv: Path,
    mid2idx: Dict[str, int],
) -> Dict[str, List[Tuple[float, float, int]]]:
    """Parse strong TSV → {clip_id: [(start_s, end_s, class_idx), ...]}."""
    import warnings
    events: Dict[str, List[Tuple[float, float, int]]] = {}
    skipped = 0
    with open(strong_tsv, newline='') as fh:
        reader = csv.reader(fh, delimiter='\t')
        for row in reader:
            if not row or row[0].startswith('#') or row[0] == 'segment_id':
                continue
            if len(row) == 4:
                clip_id, start_s, end_s, mid = row
                presence = 'PRESENT'
            elif len(row) == 5:
                clip_id, start_s, end_s, mid, presence = row
            else:
                continue
            if presence.strip().upper() != 'PRESENT':
                continue
            class_idx = mid2idx.get(mid.strip())
            if class_idx is None:
                skipped += 1
                continue
            events.setdefault(clip_id.strip(), []).append(
                (float(start_s), float(end_s), class_idx)
            )
    if skipped:
        warnings.warn(f'{skipped} events skipped (MID not in class_map_csv)')
    return events


def _collect_clips(audio_dirs: List[Path], events: Dict) -> List[str]:
    """Return sorted clip_ids that have both annotations and an audio file.

    Searches all directories in ``audio_dirs``; handles Y-prefix and
    ``{ytid}_{start_ms}`` clip_id format.
    """
    import re as _re
    audio_lookup: Dict[str, bool] = {}
    for audio_dir in audio_dirs:
        if not audio_dir.exists():
            continue
        for p in audio_dir.glob('*.wav'):
            stem = p.stem
            audio_lookup[stem] = True
            if stem.startswith('Y'):
                audio_lookup[stem[1:]] = True

    found = []
    missing = 0
    for cid in sorted(events.keys()):
        if cid in audio_lookup:
            found.append(cid)
        else:
            ytid = _re.sub(r'_\d+$', '', cid)
            if ytid in audio_lookup:
                found.append(cid)
            else:
                missing += 1
    if missing:
        print(f'  [WARN] {missing} clips in TSV have no matching audio file — skipped.')
    return found


def _split_clips(clips: List[str], val_fraction: float, val_seed: int) -> List[str]:
    """Return training clips after train/val split."""
    rng     = np.random.RandomState(val_seed)
    indices = np.arange(len(clips))
    rng.shuffle(indices)
    n_val   = max(1, int(len(indices) * val_fraction))
    train_i = sorted(indices[n_val:])
    return [clips[i] for i in train_i]


def _build_item_index(clips: List[str], window_s: float) -> List[Tuple[str, float]]:
    """Slide a window over each clip (hop = window_s, no overlap)."""
    clip_dur = 10.0
    items: List[Tuple[str, float]] = []
    for cid in clips:
        t = 0.0
        while t < clip_dur:
            items.append((cid, t))
            t += window_s
            t  = round(t, 10)
    return items


def _count_pos_neg(
    items: List[Tuple[str, float]],
    events: Dict[str, List[Tuple[float, float, int]]],
    window_s: float,
    n_cls: int,
) -> Tuple[np.ndarray, int]:
    """
    Returns:
        pos_counts : (n_cls,) array — number of windows with ≥1 event per class
        n_negative : number of windows with no active label
    """
    pos_counts = np.zeros(n_cls, dtype=np.int64)
    n_negative = 0
    for clip_id, t_start in items:
        t_end  = t_start + window_s
        active = set()
        for ev_s, ev_e, cls in events.get(clip_id, []):
            if min(ev_e, t_end) - max(ev_s, t_start) > 0.0:
                active.add(cls)
        if active:
            for c in active:
                pos_counts[c] += 1
        else:
            n_negative += 1
    return pos_counts, n_negative


# ══════════════════════════════════════════════════════════════════════════════
# Plot helpers
# ══════════════════════════════════════════════════════════════════════════════

def _style_ax(ax, xlabel: str = '', ylabel: str = '', title: str = '') -> None:
    """Apply shared style to an axes object."""
    ax.set_facecolor(BG_COLOR)
    ax.set_xlabel(xlabel, fontsize=10, fontfamily=FONT_FAMILY)
    ax.set_ylabel(ylabel, fontsize=10, fontfamily=FONT_FAMILY)
    if title:
        ax.set_title(title, fontsize=11, fontfamily=FONT_FAMILY)
    ax.grid(True, alpha=GRID_ALPHA, linewidth=0.6)
    ax.tick_params(labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COLOR)
    ax.tick_params(axis='both', labelsize=8.5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(FONT_FAMILY)


def _save_fig(fig, path: Path) -> None:
    fig.savefig(path, dpi=FIGURE_DPI, format=FIGURE_FORMAT,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Saved: {path.relative_to(_ROOT)}')


# ══════════════════════════════════════════════════════════════════════════════
# Analysis functions
# ══════════════════════════════════════════════════════════════════════════════

def plot_class_distribution(
    pos_counts_10s: np.ndarray,
    idx2name: Dict[int, str],
    top_n: int,
    out_path: Path,
) -> None:
    """Bar chart of the top-N most frequent classes at window_s=10s."""
    order   = np.argsort(pos_counts_10s)[::-1][:top_n]
    counts  = pos_counts_10s[order]
    labels  = [idx2name.get(int(i), str(i)) for i in order]

    fig, ax = plt.subplots(figsize=(20, 6))
    fig.patch.set_facecolor(BG_COLOR)

    x = np.arange(len(labels))
    ax.bar(x, counts, color=BAR_COLOR, alpha=BAR_ALPHA, width=0.85, linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=5.5, fontfamily=FONT_FAMILY)
    _style_ax(ax,
              ylabel='Positive windows (window_s=10s)',
              title=f'Top-{top_n} classes by positive-window count — AudioSet-Strong (train)')
    fig.tight_layout()
    _save_fig(fig, out_path)


def plot_events_per_clip(
    events: Dict[str, List],
    clips: List[str],
    out_path: Path,
) -> None:
    """Histogram of unique class count per clip in the training set."""
    counts = []
    for cid in clips:
        n_classes = len({cls for _, _, cls in events.get(cid, [])})
        counts.append(n_classes)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG_COLOR)

    max_val = max(counts) if counts else 1
    bins    = min(max_val + 1, 60)
    ax.hist(counts, bins=bins, color=BAR_COLOR, alpha=BAR_ALPHA,
            edgecolor='white', linewidth=0.4)
    mean_c = np.mean(counts)
    ax.axvline(mean_c, color=LINE_COLOR, linewidth=LINE_WIDTH,
               linestyle='--', label=f'Mean = {mean_c:.1f}')
    ax.legend(fontsize=9, prop={'family': FONT_FAMILY})
    _style_ax(ax,
              xlabel='Number of unique classes per clip',
              ylabel='Number of clips',
              title='Unique class count per clip — AudioSet-Strong training set')
    fig.tight_layout()
    _save_fig(fig, out_path)


def plot_neg_fraction_vs_window(
    window_sizes: List[float],
    neg_fractions: List[float],
    out_path: Path,
) -> None:
    """Line plot of negative-window fraction vs window duration."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG_COLOR)

    ax.plot(window_sizes, neg_fractions,
            color=LINE_COLOR, linewidth=LINE_WIDTH,
            marker=MARKER_STYLE, markersize=MARKER_SIZE,
            markerfacecolor='white', markeredgewidth=1.5)

    for ws, nf in zip(window_sizes, neg_fractions):
        ax.annotate(f'{nf:.1%}', (ws, nf),
                    textcoords='offset points', xytext=(0, 8),
                    ha='center', fontsize=8, fontfamily=FONT_FAMILY)

    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.set_xticks(window_sizes)
    ax.set_xticklabels([f'{w:.1f}s' for w in window_sizes],
                       fontsize=9, fontfamily=FONT_FAMILY)
    _style_ax(ax,
              xlabel='Window duration (s)',
              ylabel='Fraction of negative-only windows',
              title='Negative-window fraction vs window duration — training set')
    fig.tight_layout()
    _save_fig(fig, out_path)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = _parse_args()

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load config ───────────────────────────────────────────────────────────
    if not _DATASET_CFG.exists():
        print(f'[ERROR] Dataset config not found: {_DATASET_CFG}')
        sys.exit(1)
    with open(_DATASET_CFG) as fh:
        ds_cfg = yaml.safe_load(fh)

    data_cfg   = ds_cfg.get('data', {})
    audio_dirs = [Path(data_cfg['audio_root_train'])]
    if data_cfg.get('audio_root_unbalanced'):
        audio_dirs.append(Path(data_cfg['audio_root_unbalanced']))
    tsv_train  = Path(data_cfg['strong_tsv_train'])
    class_map  = Path(data_cfg['class_map'])
    n_cls      = 456   # AudioSet-Strong label map

    print(f'Audio dirs : {[str(d) for d in audio_dirs]}')
    print(f'TSV train  : {tsv_train}')
    print(f'Class map  : {class_map}')
    print()

    # ── Load class map ─────────────────────────────────────────────────────────
    print('Loading class map...')
    mid2idx, idx2mid, idx2name = _load_class_map(class_map)
    print(f'  {len(mid2idx)} classes loaded.')

    # ── Load events ──────────────────────────────────────────────────────────
    print('Loading strong annotations (train TSV)...')
    events = _load_events(tsv_train, mid2idx)
    print(f'  {len(events)} clips with annotations.')

    # ── Collect + split clips ─────────────────────────────────────────────────
    print('Discovering audio files...')
    all_clips    = _collect_clips(audio_dirs, events)
    train_clips  = _split_clips(all_clips, args.val_fraction, args.val_seed)
    n_val        = len(all_clips) - len(train_clips)
    print(f'  Total clips: {len(all_clips)}  '
          f'Train: {len(train_clips)}  Val: {n_val}')
    print()

    # ── Per-class stats at window_s=10s ───────────────────────────────────────
    print('Computing per-class stats at window_s=10s...')
    items_10s   = _build_item_index(train_clips, window_s=10.0)
    pos_10s, n_neg_10s = _count_pos_neg(items_10s, events, 10.0, n_cls)
    n_items_10s = len(items_10s)
    pos_rate_10s = pos_10s / max(n_items_10s, 1)

    # ── Write per-class CSV ───────────────────────────────────────────────────
    stats_csv = _RESULTS_DIR / 'audioset_strong_stats_train.csv'
    with open(stats_csv, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow([
            'class_index', 'mid', 'display_name',
            'pos_windows_10s', 'pos_rate_10s',
        ])
        for idx in range(n_cls):
            writer.writerow([
                idx,
                idx2mid.get(idx, ''),
                idx2name.get(idx, ''),
                int(pos_10s[idx]),
                f'{pos_rate_10s[idx]:.6f}',
            ])
    print(f'  Saved: {stats_csv.relative_to(_ROOT)}')

    # ── Window stats across multiple window sizes ─────────────────────────────
    print('\nComputing window stats across window sizes:', args.window_sizes)
    window_stats_rows = []
    neg_fractions     = []

    for ws in sorted(args.window_sizes):
        items_ws = _build_item_index(train_clips, window_s=ws)
        pos_ws, n_neg_ws = _count_pos_neg(items_ws, events, ws, n_cls)
        n_items_ws = len(items_ws)
        neg_frac   = n_neg_ws / max(n_items_ws, 1)
        neg_fractions.append(neg_frac)

        # Active classes: those with at least 1 positive window
        n_active_cls = int((pos_ws > 0).sum())
        mean_pos_rate = float(pos_ws.sum()) / max(n_items_ws * n_cls, 1)  # density

        row = {
            'window_s':          ws,
            'n_items':           n_items_ws,
            'n_negative':        n_neg_ws,
            'neg_fraction':      f'{neg_frac:.4f}',
            'n_active_classes':  n_active_cls,
            'mean_label_density': f'{mean_pos_rate:.6f}',
        }
        window_stats_rows.append(row)
        print(
            f'  window_s={ws:4.1f}s  items={n_items_ws:7d}  '
            f'neg_frac={neg_frac:.1%}  '
            f'active_cls={n_active_cls}  '
            f'label_density={mean_pos_rate:.4f}'
        )

    window_csv = _RESULTS_DIR / 'audioset_strong_window_stats.csv'
    with open(window_csv, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(window_stats_rows[0].keys()))
        writer.writeheader()
        writer.writerows(window_stats_rows)
    print(f'  Saved: {window_csv.relative_to(_ROOT)}')

    # ── Figures ───────────────────────────────────────────────────────────────
    print('\nGenerating figures...')

    plot_class_distribution(
        pos_counts_10s = pos_10s,
        idx2name       = idx2name,
        top_n          = args.top_n_classes,
        out_path       = _FIGURES_DIR / 'audioset_strong_class_dist.svg',
    )

    plot_events_per_clip(
        events   = events,
        clips    = train_clips,
        out_path = _FIGURES_DIR / 'audioset_strong_events_per_clip.svg',
    )

    sorted_ws  = sorted(args.window_sizes)
    neg_frac_s = [
        float(r['neg_fraction']) for r in
        sorted(window_stats_rows, key=lambda r: r['window_s'])
    ]
    plot_neg_fraction_vs_window(
        window_sizes  = sorted_ws,
        neg_fractions = neg_frac_s,
        out_path      = _FIGURES_DIR / 'audioset_strong_neg_fraction_vs_window.svg',
    )

    print('\nAnalysis complete.')


if __name__ == '__main__':
    main()
