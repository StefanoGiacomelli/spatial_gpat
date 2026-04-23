"""
tests/plot_shallow_seld_nas_foa.py
=========================================
Scatter plot of NAS experiment configurations.

Reads a NAS profiling CSV from ``results/`` and produces one SVG figure in
``figures/``.  The figure filename is derived from the CSV filename:

    nas_configs_profile_{group_slug}_{timestamp}.csv
        →  fig_nas_scatter_{group_slug}.svg

Usage
-----
    # latest CSV (all groups)
    python tests/plot_shallow_seld_nas_foa.py

    # specific CSV
    python tests/plot_shallow_seld_nas_foa.py --csv results/nas_configs_profile_foa_shallow_spec_20260405_120000.csv

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

import argparse
import csv
import pathlib
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_ROOT       = pathlib.Path(__file__).parent.parent
RESULTS_DIR = _ROOT / 'results'
FIGURES_DIR = _ROOT / 'figures'

# ══════════════════════════════════════════════════════════════════════════════
#  STYLE CONTROLS
# ══════════════════════════════════════════════════════════════════════════════

FIGURE_DPI    = 600
FIGURE_FORMAT = 'svg'

FIGSIZE       = (13, 8)

FONT_FAMILY   = 'DejaVu Sans'
GRID_ALPHA    = 0.20
SPINE_COLOR   = '#cccccc'
BG_COLOR      = '#f9f9f9'

TITLE_FS      = 13
LABEL_FS      = 11
TICK_FS       = 9
LEGEND_FS     = 7.5

# Marker size: linearly mapped from [min_params, max_params] → [MIN_MS, MAX_MS]
MIN_MS        = 80     # area units (s= in scatter)
MAX_MS        = 600

ALPHA         = 0.82
EDGE_COLOR    = 'white'
EDGE_LW       = 0.6

# Colourmap: sample 24 evenly-spaced colours from a continuous perceptual map.
# nipy_spectral covers the full visible spectrum without repeating hues.
CMAP_NAME     = 'nipy_spectral'

# Short-token → display name for legend
_TOKEN_DISPLAY = {
    'spe': 'spectral',
    'gab': 'gabor',
    'cb':  'ConvBlock',
    'rn':  'ResNet',
    'fe':  'FormantEnh',
    'tt':  'TrackTransf',
    'bg':  'BiGRU',
    'mg':  'MHA-MGU',
}


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _latest_csv() -> pathlib.Path:
    candidates = sorted(RESULTS_DIR.glob('nas_configs_profile_*.csv'))
    if not candidates:
        raise FileNotFoundError(f'No NAS profiling CSV found in {RESULTS_DIR}')
    return candidates[-1]


def _figure_slug(csv_path: pathlib.Path) -> str:
    """Derive a figure-name slug from a CSV path.

    ``nas_configs_profile_foa_shallow_spec_20260405_120000.csv``
    →  ``foa_shallow_spec``

    Falls back to the CSV stem if the expected pattern is not matched.
    """
    stem = csv_path.stem  # e.g. nas_configs_profile_foa_shallow_spec_20260405_120000
    # Strip leading 'nas_configs_profile_' prefix and trailing timestamp
    m = re.match(r'^nas_configs_profile_(.+?)_\d{8}_\d{6}$', stem)
    if m:
        return m.group(1)
    # fallback: strip known prefix only
    prefix = 'nas_configs_profile_'
    if stem.startswith(prefix):
        return stem[len(prefix):]
    return stem


def _load(csv_path: pathlib.Path) -> list[dict]:
    rows = []
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            if row.get('status', '') != 'OK':
                continue
            try:
                row['_gflops']  = float(row['gflops'])
                row['_t_ms']    = float(row['t_ms'])
                row['_params']  = int(row['total_params'])
            except (ValueError, KeyError):
                continue
            # build display label: inp_early_late_smoother
            tokens = [row.get('input_type', ''), row.get('spatial_early', ''),
                      row.get('spatial_late', ''), row.get('smoother', '')]
            row['_label'] = '_'.join(
                _TOKEN_DISPLAY.get(t, t) for t in tokens
            )
            rows.append(row)
    return rows


def _marker_sizes(params: np.ndarray) -> np.ndarray:
    lo, hi = params.min(), params.max()
    if hi == lo:
        return np.full_like(params, (MIN_MS + MAX_MS) / 2, dtype=float)
    return MIN_MS + (MAX_MS - MIN_MS) * (params - lo) / (hi - lo)


def _colours(n: int) -> list:
    cmap = plt.get_cmap(CMAP_NAME)
    # Sample n evenly-spaced values in [0.05, 0.95] to avoid the very dark
    # endpoints of nipy_spectral (near-black at 0 and near-white at 1).
    return [cmap(v) for v in np.linspace(0.05, 0.95, n)]


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def fig_scatter(rows: list[dict], out_path: pathlib.Path, slug: str) -> None:
    gflops  = np.array([r['_gflops']  for r in rows])
    t_ms    = np.array([r['_t_ms']    for r in rows])
    params  = np.array([r['_params']  for r in rows])
    labels  = [r['_label'] for r in rows]

    sizes   = _marker_sizes(params)
    colours = _colours(len(rows))

    plt.rcParams['font.family'] = FONT_FAMILY

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor('white')
    ax.set_facecolor(BG_COLOR)

    # ── Grid ──────────────────────────────────────────────────────────────────
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=GRID_ALPHA, zorder=0)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COLOR)

    # ── Scatter ───────────────────────────────────────────────────────────────
    for i, (row, x, y, s, c, lbl) in enumerate(
            zip(rows, gflops, t_ms, sizes, colours, labels)):
        marker = '^' if row.get('smoother', '') == 'mg' else 'o'
        ax.scatter(x, y, s=s, color=c, alpha=ALPHA, marker=marker,
                   edgecolors=EDGE_COLOR, linewidths=EDGE_LW,
                   zorder=3, label=lbl)

    # ── Axes labels / title ───────────────────────────────────────────────────
    ax.set_xlabel('GFLOPs', fontsize=LABEL_FS)
    ax.set_ylabel('Forward time  (ms)', fontsize=LABEL_FS)
    ax.set_title(
        f'NAS Configurations — Efficiency Overview  [{slug}]\n'
        '(marker size ∝ total parameters)',
        fontsize=TITLE_FS, pad=10,
    )
    ax.tick_params(labelsize=TICK_FS)

    # ── Legend: configs (internal, 2 columns) ────────────────────────────────
    legend = ax.legend(
        loc='upper left',
        bbox_to_anchor=(0.01, 0.99),
        borderaxespad=0,
        fontsize=LEGEND_FS,
        framealpha=0.88,
        edgecolor=SPINE_COLOR,
        title='input_early_late_smooth',
        title_fontsize=LEGEND_FS,
        handlelength=1.0,
        handleheight=1.0,
        markerscale=0.85,
        ncol=2,
        columnspacing=0.8,
        labelspacing=0.8,
    )

    # ── Param-size reference legend (bottom-right inside axes) ────────────────
    ref_params = np.array([params.min(), np.median(params), params.max()])
    ref_sizes  = _marker_sizes(ref_params)
    ref_labels = [f'{int(p/1e6):.0f} M' for p in ref_params]
    for rs, rl in zip(ref_sizes, ref_labels):
        ax.scatter([], [], s=rs, color='#888888', alpha=0.6,
                   edgecolors=EDGE_COLOR, linewidths=EDGE_LW,
                   label=rl)
    # Add a second legend for the size reference
    handles, hlabels = ax.get_legend_handles_labels()
    size_handles = handles[len(rows):]
    size_labels  = hlabels[len(rows):]
    ax.add_artist(legend)   # keep the config legend
    ax.legend(size_handles, size_labels,
              loc='lower right',
              fontsize=LEGEND_FS,
              framealpha=0.88,
              edgecolor=SPINE_COLOR,
              title='Total params',
              title_fontsize=LEGEND_FS,
              handlelength=1.5,
              markerscale=1.0,
              scatterpoints=1,
              labelspacing=1.8,   # vertical gap so large markers don't overlap
              )

    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIGURE_DPI, format=FIGURE_FORMAT,
                bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description='Plot NAS profiling scatter.')
    parser.add_argument('--csv', default=None,
                        help='Path to NAS profiling CSV (default: latest in results/)')
    args = parser.parse_args()

    csv_path = pathlib.Path(args.csv).resolve() if args.csv else _latest_csv()
    print(f'Loading: {csv_path}')

    rows = _load(csv_path)
    if not rows:
        print('[ERROR] No valid (status=OK) rows found.')
        return

    slug     = _figure_slug(csv_path)
    out_path = FIGURES_DIR / f'nas_scatter_{slug}.{FIGURE_FORMAT}'

    print(f'Plotting {len(rows)} configurations  →  {out_path.name}')
    fig_scatter(rows, out_path, slug)


if __name__ == '__main__':
    main()
