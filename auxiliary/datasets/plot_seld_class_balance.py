"""
tests/plot_class_balance.py
============================
Four-panel figure comparing class imbalance and coverage across datasets and
splits, using the CSV files produced by ``tests/test_class_balance.py``.

Layout (2 × 2)
--------------
  [STARSS23   train]  [STARSS23   test]
  [TAU-NIGENS train]  [TAU-NIGENS test]

Each panel contains two side-by-side sub-axes (inner GridSpec 30 / 70):
  Left  — pos_weight on a log scale (imbalance severity for the BCE loss)
  Right — Clip coverage % and Window coverage % as grouped horizontal bars

Classes are sorted by pos_weight descending (rarest / most problematic at top),
giving a natural severity-ranking read.

Output
------
  figures/seld_class_balance.svg   (600 DPI)

Usage
-----
  .venv/bin/python tests/plot_class_balance.py

  # Custom CSV directory and output
  .venv/bin/python tests/plot_class_balance.py --csv-dir results/ --out figures/

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np

_ROOT       = Path(__file__).resolve().parent.parent
RESULTS_DIR = _ROOT / 'results'
FIGURES_DIR = _ROOT / 'figures'

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ══════════════════════════════════════════════════════════════════════════════
# Style
# ══════════════════════════════════════════════════════════════════════════════

FIGURE_DPI    = 600
FIGURE_FORMAT = 'svg'
FIGSIZE       = (18, 15)     # wide enough for 2 columns, tall for 13-class bars

FONT_FAMILY   = 'DejaVu Sans'
TITLE_FS      = 10
SUBTITLE_FS   = 8.5
TICK_FS       = 7.5
LABEL_FS      = 8
LEGEND_FS     = 8

BG_COLOR      = '#f8f8f8'
SPINE_COLOR   = '#cccccc'
GRID_ALPHA    = 0.22

# Bar colours
C_PW   = '#D95F02'    # orange-red   — pos_weight (imbalance severity)
C_CLIP = '#1F78B4'    # deep blue    — clip coverage
C_WIN  = '#A6CEE3'    # light blue   — window coverage

BAR_H  = 0.38         # half-height per bar in the grouped right panel
BAR_H_PW = 0.55       # bar height in the left (pos_weight) panel


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def _load_csv(path: Path) -> Optional[Dict[str, list]]:
    """Read a class-balance CSV and return a dict of lists (one per column)."""
    if not path.exists():
        return None
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh)
        rows   = list(reader)
    if not rows:
        return None
    # Detect the window-coverage column (name encodes parameters)
    win_col = next(
        (k for k in rows[0] if k.startswith('window_coverage_pct')), None
    )
    return {
        'class_name':         [r['class_name'] for r in rows],
        'active_frames_pct':  [float(r['active_frames_pct']) for r in rows],
        'pos_weight':         [float(r['pos_weight']) for r in rows],
        'clip_coverage_pct':  [float(r['clip_coverage_pct']) for r in rows],
        'window_coverage_pct':[float(r[win_col]) for r in rows] if win_col else
                              [0.0] * len(rows),
    }


def _sort_descending_pw(data: Dict[str, list]) -> Dict[str, list]:
    """Return a copy sorted by pos_weight ascending (so rarest appears at top in barh)."""
    order = sorted(range(len(data['pos_weight'])),
                   key=lambda i: data['pos_weight'][i])
    return {k: [v[i] for i in order] for k, v in data.items()}


# ══════════════════════════════════════════════════════════════════════════════
# Panel drawing
# ══════════════════════════════════════════════════════════════════════════════

def _style_ax(ax, bg=BG_COLOR, spine_col=SPINE_COLOR):
    ax.set_facecolor(bg)
    for sp in ax.spines.values():
        sp.set_color(spine_col)
    ax.tick_params(labelsize=TICK_FS, colors='#444444')


def _draw_panel(
    ax_pw:  plt.Axes,
    ax_cv:  plt.Axes,
    data:   Dict[str, list],
    title:  str,
    n_clips: int,
    n_windows: int,
) -> None:
    """
    Draw one cell: left axis = pos_weight (log), right axis = Clip% + Win%.
    ``data`` must already be sorted (ascending pw → rarest at top in barh).
    """
    n = len(data['class_name'])
    y = np.arange(n)

    # ── Left: pos_weight (log scale) ─────────────────────────────────────────
    _style_ax(ax_pw)
    bars_pw = ax_pw.barh(
        y, data['pos_weight'], height=BAR_H_PW,
        color=C_PW, alpha=0.85, linewidth=0,
    )
    ax_pw.set_xscale('log')
    ax_pw.set_xlim(left=0.5)
    ax_pw.set_yticks(y)
    ax_pw.set_yticklabels(data['class_name'], fontsize=TICK_FS)
    ax_pw.set_ylim(-0.6, n - 0.4)
    ax_pw.xaxis.grid(True, alpha=GRID_ALPHA, linestyle='--', linewidth=0.6)
    ax_pw.set_axisbelow(True)
    ax_pw.set_xlabel('pos_weight  (log)', fontsize=LABEL_FS, color='#333333')

    # Annotate value on bars wider than a threshold
    for bar, pw in zip(bars_pw, data['pos_weight']):
        label = f'{pw:.0f}' if pw >= 10 else f'{pw:.1f}'
        x_end = bar.get_width()
        ax_pw.text(
            x_end * 1.08, bar.get_y() + bar.get_height() / 2,
            label, va='center', ha='left', fontsize=5.5, color='#555555',
        )

    # Title spanning both inner axes — set on left axis with no box clipping
    ax_pw.set_title(title, fontsize=TITLE_FS, fontweight='bold',
                    color='#222222', pad=6, loc='left')

    # ── Right: Clip% and Window% grouped bars ─────────────────────────────────
    _style_ax(ax_cv)
    clip_arr = np.array(data['clip_coverage_pct'])
    win_arr  = np.array(data['window_coverage_pct'])

    ax_cv.barh(
        y + BAR_H / 2, clip_arr, height=BAR_H,
        color=C_CLIP, alpha=0.88, linewidth=0,
        label=f'Clip coverage %  (/ {n_clips} clips)',
    )
    ax_cv.barh(
        y - BAR_H / 2, win_arr,  height=BAR_H,
        color=C_WIN,  alpha=0.88, linewidth=0,
        label=f'Window coverage %  (/ {n_windows} windows, 10 s / 5 s hop)',
    )

    ax_cv.set_xlim(0, 115)
    ax_cv.set_ylim(-0.6, n - 0.4)
    ax_cv.xaxis.grid(True, alpha=GRID_ALPHA, linestyle='--', linewidth=0.6)
    ax_cv.set_axisbelow(True)
    ax_cv.set_xlabel('Coverage  (%)', fontsize=LABEL_FS, color='#333333')
    ax_cv.yaxis.set_visible(False)

    # Value labels on coverage bars (only if bar is wide enough to read)
    for yi, (cv, wv) in enumerate(zip(clip_arr, win_arr)):
        if cv >= 3:
            ax_cv.text(cv + 1.5, yi + BAR_H / 2,
                       f'{cv:.0f}%', va='center', ha='left',
                       fontsize=5.5, color='#333333')
        if wv >= 3:
            ax_cv.text(wv + 1.5, yi - BAR_H / 2,
                       f'{wv:.0f}%', va='center', ha='left',
                       fontsize=5.5, color='#333333')

    # ── Per-panel legend (inside coverage axis, upper-right — rare classes
    #    have short bars there, leaving room for the box) ───────────────────
    ax_cv.legend(
        loc='upper right',
        fontsize=6.0,
        framealpha=0.92,
        edgecolor=SPINE_COLOR,
        handlelength=1.1,
        handleheight=0.9,
        borderpad=0.5,
        labelspacing=0.35,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Clip/window count helpers
# ══════════════════════════════════════════════════════════════════════════════

# Fallback counts used when the JSON sidecar produced by test_seld_class_balance.py
# is not present (e.g., on a first run before the sidecar has been generated).
_CLIP_WINDOW_COUNTS_FALLBACK: Dict[str, Tuple[int, int]] = {
    'STARSS23_train':       (90,  2980),
    'STARSS23_test':        (78,  2316),
    'TAU-NIGENS2021_train': (400, 4800),
    'TAU-NIGENS2021_test':  (100, 1200),
}


def _load_counts(csv_dir: Path, ds: str, split: str) -> Tuple[int, int]:
    """Return (n_clips, n_windows) from the JSON sidecar if available."""
    meta_path = csv_dir / f'class_balance_{ds}_{split}_meta.json'
    if meta_path.exists():
        with open(meta_path) as fh:
            meta = json.load(fh)
        return int(meta['n_clips']), int(meta['n_windows'])
    return _CLIP_WINDOW_COUNTS_FALLBACK.get(f'{ds}_{split}', (0, 0))


# ══════════════════════════════════════════════════════════════════════════════
# Main figure
# ══════════════════════════════════════════════════════════════════════════════

# Four panels: (dataset, split, csv_stem, panel_title)
PANELS = [
    ('STARSS23',       'train', 'class_balance_STARSS23_train',       'STARSS23 — Train'),
    ('STARSS23',       'test',  'class_balance_STARSS23_test',        'STARSS23 — Test'),
    ('TAU-NIGENS2021', 'train', 'class_balance_TAU-NIGENS2021_train', 'TAU-NIGENS2021 — Train'),
    ('TAU-NIGENS2021', 'test',  'class_balance_TAU-NIGENS2021_test',  'TAU-NIGENS2021 — Test'),
]


def build_figure(csv_dir: Path, out_dir: Path) -> Path:
    plt.rcParams['font.family'] = FONT_FAMILY

    fig = plt.figure(figsize=FIGSIZE)
    fig.patch.set_facecolor('white')

    outer_gs = GridSpec(
        2, 2, figure=fig,
        hspace=0.48, wspace=0.55,
        top=0.93, bottom=0.03, left=0.01, right=0.99,
    )

    for idx, (ds, split, csv_stem, title) in enumerate(PANELS):
        row, col = divmod(idx, 2)

        csv_path = csv_dir / f'{csv_stem}.csv'
        data = _load_csv(csv_path)
        if data is None:
            ax = fig.add_subplot(outer_gs[row, col])
            ax.text(0.5, 0.5, f'CSV not found:\n{csv_path.name}',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=9, color='#cc0000')
            ax.set_axis_off()
            continue

        data = _sort_descending_pw(data)
        n_clips, n_windows = _load_counts(csv_dir, ds, split)
        panel_title = f'{title}  ({n_clips} clips)' if n_clips > 0 else title

        inner_gs = GridSpecFromSubplotSpec(
            1, 2,
            subplot_spec=outer_gs[row, col],
            width_ratios=[0.34, 0.66],
            wspace=0.06,
        )
        ax_pw = fig.add_subplot(inner_gs[0])
        ax_cv = fig.add_subplot(inner_gs[1], sharey=ax_pw)

        _draw_panel(ax_pw, ax_cv, data, panel_title, n_clips, n_windows)

    # ── Super-title ────────────────────────────────────────────────────────────
    fig.suptitle(
        'Class Imbalance & Sample Coverage — STARSS23 and TAU-NIGENS2021',
        fontsize=12, fontweight='bold', color='#111111', y=0.975,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'seld_class_balance.{FIGURE_FORMAT}'
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description='Plot class-balance figure from CSV files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--csv-dir', type=Path, default=RESULTS_DIR,
                   help='Directory containing class_balance_*.csv files.')
    p.add_argument('--out', type=Path, default=FIGURES_DIR,
                   help='Output directory for the SVG figure.')
    args = p.parse_args()

    out = build_figure(args.csv_dir, args.out)
    print(f'Figure saved → {out.relative_to(_ROOT)}')


if __name__ == '__main__':
    main()
