#!/usr/bin/env python3
"""
analysis/profiling/plot_spatial_modules.py
========================================
Visualization of spatial_modules profiling results.

Reads the most recent (or a specified) CSV from ``results/`` and produces
3 SVG figures at 600 DPI in ``figures/``:

  fig1_violin.svg            Forward-time violin plots by algorithmic family
  fig2_lines.svg             Forward time vs batch size (one panel per device)
  fig3_heatmap.svg           Heatmap: median time | memory | params | GFLOPs

All style parameters are exposed as constants at the top of this file.

Usage
-----
    python analysis/profiling/plot_spatial_modules.py
    python analysis/profiling/plot_spatial_modules.py --csv results/profiling/spatial_modules_profile_<ts>.csv

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

import argparse
import csv
import pathlib


import numpy as np
import matplotlib
matplotlib.use('Agg')                       # headless-safe; comment out for interactive use
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches


_ROOT       = pathlib.Path(__file__).resolve().parents[2]
RESULTS_DIR = _ROOT / 'results' / 'profiling'
FIGURES_DIR = _ROOT / 'figures' / 'profiling'

# ══════════════════════════════════════════════════════════════════════════════
#  STYLE CONTROLS — edit freely
# ══════════════════════════════════════════════════════════════════════════════

# ── Output ────────────────────────────────────────────────────────────────────
FIGURE_DPI    = 600
FIGURE_FORMAT = 'svg'     # 'svg' | 'pdf' | 'png'

# ── Global typography ─────────────────────────────────────────────────────────
FONT_FAMILY   = 'DejaVu Sans'
GRID_ALPHA    = 0.25       # shared grid transparency
SPINE_COLOR   = '#cccccc'  # shared axes-spine color

# ── Figure 1 — Violin plots by algorithmic family ─────────────────────────────
F1_FIGSIZE          = (22, 7)
F1_SUPTITLE         = 'Forward Time Distribution (by Algorithmic Family)'
F1_SUPTITLE_FS      = 15
F1_SUPTITLE_Y       = 1.02
F1_SUBPLOT_TITLE_FS = 12
F1_YLABEL           = 'Forward time  (ms)'
F1_YLABEL_FS        = 10
F1_TICK_FS          = 8.5
F1_VIOLIN_ALPHA     = 0.72
F1_VIOLIN_EDGE      = '#888888'
F1_MEAN_LW          = 2.0
F1_EXTR_LW          = 1.2
F1_POINT_SIZE       = 18    # scatter point area (s= parameter)
F1_POINT_ALPHA      = 0.60
F1_LOG_SCALE        = True  # log y-axis
F1_HSPACE           = 0.35

# ── Figure 2 — Line plots per device ─────────────────────────────────────────
F2_FIGSIZE_PER_COL  = (9, 6)   # per-device column; total width scales with N_devices
F2_SUPTITLE         = 'Forward time vs batch size — all modules'
F2_SUPTITLE_FS      = 14
F2_SUPTITLE_Y       = 1.03
F2_SUBPLOT_TITLE_FS = 11
F2_XLABEL           = 'Batch size'
F2_YLABEL           = 'Forward time  (ms)'
F2_LABEL_FS         = 10
F2_TICK_FS          = 9
F2_LINE_WIDTH       = 1.6
F2_MARKER_SIZE      = 4
F2_ERROR_ALPHA      = 0.12
F2_LOG_Y            = True
F2_LEGEND_FS        = 6.5
F2_LEGEND_NCOL      = 3
F2_LEGEND_Y         = -0.22   # vertical offset below axes (in axes units)

# ── Figure 3 — Heatmap ────────────────────────────────────────────────────────
F3_FIGSIZE          = (10, 14)
F3_TITLE            = 'Module comparison — CPU / B = 1'
F3_TITLE_FS         = 13
F3_COL_LABEL_FS     = 10
F3_ROW_LABEL_FS     = 8.5
F3_CELL_FS          = 8.0
F3_FAMILY_LABEL_FS  = 8
F3_SEP_COLOR        = '#444444'
F3_SEP_LW           = 1.5
F3_CELL_EDGE        = 'white'
F3_CELL_EDGE_LW     = 0.6
F3_CMAP_GOOD        = '#2ecc71'   # green  → best (smallest)
F3_CMAP_MID         = '#fffde7'   # near-white midpoint
F3_CMAP_BAD         = '#e74c3c'   # red    → worst (largest)
F3_ZERO_COLOR       = '#d5d8dc'   # neutral gray for zero values

# ══════════════════════════════════════════════════════════════════════════════
#  ALGORITHMIC FAMILIES
# ══════════════════════════════════════════════════════════════════════════════

FAMILIES: dict = {
    'Conv-based': [
        'ConvBlock', 'MultiScaleChannelAttention', 'ResNetBlock',
        'DoubleConv', 'CrossStitching', 'FormantEnhancementModule',
    ],
    'Transformer / Attention': [
        'ConformerBlock', 'AttentiveStatisticsPooling',
        'PositionalEncoder', 'TrackTransformer', 'CrossStitchBridge',
    ],
    'Recurrent': [
        'BiGRU', 'MHA_MGU',
    ],
    'DSP': [
        'SpectralProcessor', 'LearnableGaborFilterBank',
        'GaborFrontEnd',
    ],
}

FAMILY_COLORS: dict = {
    'Conv-based':              '#2980b9',
    'Transformer / Attention': '#e67e22',
    'Recurrent':               '#8e44ad',
    'DSP':                     '#27ae60',
}

# Display names for module x-tick labels in fig1 (full or renamed forms)
_MODULE_DISPLAY: dict = {
    'MultiScaleChannelAttention': 'MultiScaleChannelAttention',
    'FormantEnhancementModule':   'FormantEnhancer',
    'AttentiveStatisticsPooling': 'AttentiveStatisticsPooling',
    'LearnableGaborFilterBank':   'GaborFilterbank',
}

# Display names for family legend headers (fig2)
_FAMILY_DISPLAY: dict = {
    'Conv-based':              'Conv-based',
    'Transformer / Attention': 'Transformer / Attention',
    'Recurrent':               'Recurrent',
    'DSP':                     'DSP',
}

# Device colors and markers (used in fig1 overlay and fig2 lines)
_DEVICE_COLORS  = {'cpu': '#2c3e50', 'cuda': '#c0392b', 'mps': '#d35400'}
_DEVICE_MARKERS = {'cpu': 'o',       'cuda': 's',       'mps': '^'}

# Per-module line styles (cycled within family, used in fig2)
_LINESTYLES = ['-', '--', '-.', ':', (0, (5, 2)), (0, (3, 1, 1, 1))]
_MARKERS    = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', 'h', '<', '>', 'p', 'H', '1', '2', '3', '4']

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _latest_csv(results_dir: pathlib.Path) -> pathlib.Path:
    files = sorted(results_dir.glob('spatial_modules_profile_*.csv'))
    if not files:
        raise FileNotFoundError(f'No profiling CSV found in {results_dir}')
    return files[-1]


def _load_csv(path: pathlib.Path) -> list:
    """Load CSV into a list of dicts; convert numeric fields to float/int."""
    rows = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _family_of(module: str) -> str:
    for fam, mods in FAMILIES.items():
        if module in mods:
            return fam
    return 'Unknown'


def _module_order() -> list:
    """Canonical heatmap order: family-grouped, preserving FAMILIES dict order."""
    return [m for mods in FAMILIES.values() for m in mods]


def _linestyle_for(module: str):
    fam  = _family_of(module)
    idx  = FAMILIES.get(fam, []).index(module) if module in FAMILIES.get(fam, []) else 0
    return _LINESTYLES[idx % len(_LINESTYLES)]


def _marker_for(module: str) -> str:
    fam  = _family_of(module)
    idx  = FAMILIES.get(fam, []).index(module) if module in FAMILIES.get(fam, []) else 0
    return _MARKERS[idx % len(_MARKERS)]



def _apply_common_axes(ax, tick_fs: int):
    """Set spine/tick style shared across figures."""
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
    ax.tick_params(axis='both', labelsize=tick_fs, colors='#333333')


def _save(fig, stem: str, out_dir: pathlib.Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f'{stem}.{FIGURE_FORMAT}'
    fig.savefig(p, dpi=FIGURE_DPI, bbox_inches='tight', format=FIGURE_FORMAT)
    print(f'  Saved: {p.relative_to(_ROOT)}')
    plt.close(fig)


def _extract_cpu_b1(rows: list, devices: list) -> dict:
    """Return {module: {col: value}} for cpu (or first device), batch_size=1."""
    ref_dev = 'cpu' if 'cpu' in devices else devices[0]
    data: dict = {}
    for row in rows:
        if row['device'] == ref_dev and int(row['batch_size']) == 1:
            mod = row['module']
            data[mod] = {
                'median_ms':    float(row.get('median_ms',   row.get('mean_ms', 0))),
                'mean_ms':      float(row.get('mean_ms',     0)),
                'memory_mb':    float(row.get('memory_mb',   0)),
                'total_params': int(row.get('total_params',  0)),
                'gflops':       float(row.get('gflops',      0)),
            }
    return data


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Violin plots by algorithmic family
# ══════════════════════════════════════════════════════════════════════════════

def fig1_violin(rows: list, out_dir: pathlib.Path) -> None:
    """One subplot per family; violins over all (batch_size × device) timings."""
    devices  = sorted({r['device'] for r in rows})
    n_fam    = len(FAMILIES)
    rng      = np.random.default_rng(42)

    fig, axes = plt.subplots(1, n_fam, figsize=F1_FIGSIZE)
    fig.suptitle(F1_SUPTITLE, fontsize=F1_SUPTITLE_FS,
                 fontfamily=FONT_FAMILY, y=F1_SUPTITLE_Y)

    # Build a legend for devices (scatter overlay)
    dev_legend_handles = [
        mpatches.Patch(
            facecolor=_DEVICE_COLORS.get(d, '#555555'),
            label=d, alpha=F1_POINT_ALPHA,
        )
        for d in devices
    ]

    for ax, (family, fam_modules) in zip(axes, FAMILIES.items()):
        fam_color = FAMILY_COLORS[family]
        present   = [m for m in fam_modules
                     if any(r['module'] == m for r in rows)]
        positions = list(range(1, len(present) + 1))

        # Gather timing vectors per module (all batch_sizes × all devices)
        datasets = []
        for mod in present:
            vals = [float(r['mean_ms'])
                    for r in rows if r['module'] == mod]
            datasets.append(vals)

        if datasets:
            parts = ax.violinplot(
                datasets, positions=positions,
                showmeans=True, showmedians=False, showextrema=True,
            )
            for body in parts['bodies']:
                body.set_facecolor(fam_color)
                body.set_edgecolor(F1_VIOLIN_EDGE)
                body.set_alpha(F1_VIOLIN_ALPHA)
            if 'cmeans' in parts:
                parts['cmeans'].set_edgecolor('white')
                parts['cmeans'].set_linewidth(F1_MEAN_LW)
            for key in ('cbars', 'cmins', 'cmaxes'):
                if key in parts:
                    parts[key].set_edgecolor(fam_color)
                    parts[key].set_linewidth(F1_EXTR_LW)

        # Scatter overlay: one dot per (batch_size, device) measurement
        for pos, mod in zip(positions, present):
            for dev in devices:
                dev_rows = [r for r in rows
                            if r['module'] == mod and r['device'] == dev]
                if not dev_rows:
                    continue
                vals   = [float(r['mean_ms']) for r in dev_rows]
                jitter = rng.uniform(-0.14, 0.14, size=len(vals))
                ax.scatter(
                    np.array([pos] * len(vals)) + jitter, vals,
                    s=F1_POINT_SIZE, alpha=F1_POINT_ALPHA,
                    color=_DEVICE_COLORS.get(dev, '#555555'),
                    marker=_DEVICE_MARKERS.get(dev, 'o'), zorder=4,
                )

        ax.set_xlim(0.3, len(present) + 0.7)
        ax.set_xticks(positions)
        display_names = [_MODULE_DISPLAY.get(m, m) for m in present]
        ax.set_xticklabels(display_names, rotation=32, ha='right', fontsize=F1_TICK_FS,
                           fontfamily=FONT_FAMILY)
        ax.set_ylabel(F1_YLABEL, fontsize=F1_YLABEL_FS, fontfamily=FONT_FAMILY)
        ax.set_title(family, fontsize=F1_SUBPLOT_TITLE_FS,
                     fontfamily=FONT_FAMILY, color=fam_color, pad=8)
        if F1_LOG_SCALE:
            ax.set_yscale('log')
        ax.grid(True, axis='y', alpha=GRID_ALPHA, which='both', linestyle=':')
        _apply_common_axes(ax, F1_TICK_FS)

    # Device legend in every subplot
    for ax in axes:
        ax.legend(
            handles=dev_legend_handles,
            title='device', title_fontsize=F1_TICK_FS,
            fontsize=F1_TICK_FS, loc='upper right',
            framealpha=0.85, edgecolor=SPINE_COLOR,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=F1_HSPACE)
    _save(fig, 'spat_modules_fig1_violin', out_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Line plots per device
# ══════════════════════════════════════════════════════════════════════════════

def fig2_lines(rows: list, out_dir: pathlib.Path) -> None:
    """One subplot per device; forward time vs batch size for all modules."""
    devices     = sorted({r['device'] for r in rows})
    batch_sizes = sorted({int(r['batch_size']) for r in rows})
    all_modules = _module_order()
    n_dev       = len(devices)

    fig, axes = plt.subplots(
        1, n_dev,
        figsize=(F2_FIGSIZE_PER_COL[0] * n_dev, F2_FIGSIZE_PER_COL[1]),
        sharey=True,
    )
    if n_dev == 1:
        axes = [axes]

    fig.suptitle(F2_SUPTITLE, fontsize=F2_SUPTITLE_FS,
                 fontfamily=FONT_FAMILY, y=F2_SUPTITLE_Y)

    # Precompute per-family legend handles (used after axes loop)
    fam_legend_handles = {}
    for family, fam_mods in FAMILIES.items():
        present_mods = [m for m in fam_mods if any(r['module'] == m for r in rows)]
        handles = []
        for mod in present_mods:
            handles.append(plt.Line2D(
                [0], [0],
                color=FAMILY_COLORS[family],
                linewidth=F2_LINE_WIDTH,
                linestyle=_linestyle_for(mod),
                marker=_marker_for(mod),
                markersize=F2_MARKER_SIZE,
                label=mod,
            ))
        fam_legend_handles[family] = handles

    for ax, dev in zip(axes, devices):
        for mod in all_modules:
            mod_rows = sorted(
                [r for r in rows if r['module'] == mod and r['device'] == dev],
                key=lambda r: int(r['batch_size']),
            )
            if not mod_rows:
                continue
            fam    = _family_of(mod)
            color  = FAMILY_COLORS[fam]
            xs     = [int(r['batch_size']) for r in mod_rows]
            ys     = np.array([float(r['mean_ms']) for r in mod_rows])
            errs   = np.array([float(r['std_ms'])  for r in mod_rows])

            ax.plot(
                xs, ys,
                color=color,
                linewidth=F2_LINE_WIDTH,
                linestyle=_linestyle_for(mod),
                marker=_marker_for(mod),
                markersize=F2_MARKER_SIZE,
                label=mod,
            )
            ax.fill_between(xs, ys - errs, ys + errs,
                            alpha=F2_ERROR_ALPHA, color=color)

        ax.set_title(f'device: {dev}', fontsize=F2_SUBPLOT_TITLE_FS,
                     fontfamily=FONT_FAMILY)
        ax.set_xlabel(F2_XLABEL, fontsize=F2_LABEL_FS, fontfamily=FONT_FAMILY)
        if ax is axes[0]:
            ax.set_ylabel(F2_YLABEL, fontsize=F2_LABEL_FS, fontfamily=FONT_FAMILY)
        ax.set_xscale('log', base=2)
        ax.set_xticks(batch_sizes)
        ax.set_xticklabels([str(b) for b in batch_sizes], fontsize=F2_TICK_FS)
        if F2_LOG_Y:
            ax.set_yscale('log')
        ax.grid(True, alpha=GRID_ALPHA, which='both', linestyle=':')
        _apply_common_axes(ax, F2_TICK_FS)

    fig.tight_layout(rect=[0, 0.0, 1, 0.96])

    # 4 separate family legend boxes placed side-by-side below the axes.
    # Each box contains only its family's modules → guaranteed visual isolation.
    n_fam = len(FAMILIES)
    fam_items = list(FAMILIES.items())
    for fi, (family, _) in enumerate(fam_items):
        handles = fam_legend_handles.get(family, [])
        if not handles:
            continue
        x_center = (fi + 0.5) / n_fam
        leg = fig.legend(
            handles=handles,
            title=_FAMILY_DISPLAY.get(family, family),
            title_fontproperties={'family': FONT_FAMILY,
                                   'size': F2_LEGEND_FS + 0.5,
                                   'weight': 'bold'},
            loc='upper center',
            ncol=1,
            fontsize=F2_LEGEND_FS,
            bbox_to_anchor=(x_center, F2_LEGEND_Y),
            framealpha=0.9,
            edgecolor=FAMILY_COLORS[family],
            prop={'family': FONT_FAMILY, 'size': F2_LEGEND_FS},
            handlelength=2.2,
        )
    _save(fig, 'spat_modules_fig2_lines', out_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Heatmap
# ══════════════════════════════════════════════════════════════════════════════

def fig3_heatmap(rows: list, out_dir: pathlib.Path) -> None:
    """Heatmap: median time | memory | params | GFLOPs — CPU, B=1."""
    devices  = sorted({r['device'] for r in rows})
    cpu_data = _extract_cpu_b1(rows, devices)

    modules  = [m for m in _module_order() if m in cpu_data]
    n_mods   = len(modules)

    # Heatmap columns definition: (csv_key, header_label, format_function)
    HEATMAP_COLS = [
        ('median_ms',    'Median time\n(ms)',   lambda v: f'{v:.3f}'),
        ('memory_mb',    'Memory\n(MB)',         lambda v: f'{v:.3f}'),
        ('total_params', 'Total\nparams',        lambda v: f'{int(v):,}'),
        ('gflops',       'GFLOPs',               lambda v: f'{v:.3f}'),
    ]
    n_cols = len(HEATMAP_COLS)

    # Build colormap: green → near-white → red
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'gwr', [F3_CMAP_GOOD, F3_CMAP_MID, F3_CMAP_BAD], N=256,
    )

    # Per-column normalization (exclude zero values)
    col_norms = []
    for col_key, _, _ in HEATMAP_COLS:
        vals     = [cpu_data[m][col_key] for m in modules]
        nonzero  = [v for v in vals if v > 0]
        vmin     = min(nonzero) if nonzero else 0.0
        vmax     = max(nonzero) if nonzero else 1.0
        col_norms.append((vmin, vmax))

    # Family boundary info (for separators)
    fam_blocks = []   # list of (family, start_idx, end_idx)
    offset = 0
    for fam, fam_mods in FAMILIES.items():
        present = [m for m in fam_mods if m in cpu_data]
        fam_blocks.append((fam, offset, offset + len(present)))
        offset += len(present)

    # ── Draw ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=F3_FIGSIZE)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_mods)
    ax.set_aspect('auto')
    ax.axis('off')

    for row_idx, mod in enumerate(modules):
        y_bot = n_mods - row_idx - 1   # bottom of cell (top-to-bottom order)
        md    = cpu_data[mod]

        for col_idx, ((col_key, _, fmt_fn), (vmin, vmax)) in enumerate(
                zip(HEATMAP_COLS, col_norms)):
            val = md[col_key]

            if val == 0:
                face_color = F3_ZERO_COLOR
                text_color = '#444444'
            else:
                t          = (val - vmin) / max(vmax - vmin, 1e-12)
                t          = min(max(t, 0.0), 1.0)
                rgba       = cmap(t)
                face_color = rgba
                lum        = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = 'white' if lum < 0.50 else '#1a1a1a'

            ax.add_patch(mpatches.FancyBboxPatch(
                (col_idx + 0.02, y_bot + 0.02), 0.96, 0.96,
                boxstyle='square,pad=0',
                facecolor=face_color,
                edgecolor=F3_CELL_EDGE,
                linewidth=F3_CELL_EDGE_LW,
                zorder=2,
            ))
            ax.text(
                col_idx + 0.5, y_bot + 0.5, fmt_fn(val),
                ha='center', va='center',
                fontsize=F3_CELL_FS, color=text_color,
                fontfamily=FONT_FAMILY, zorder=3,
            )

    # Row labels (module names, colored by family)
    for row_idx, mod in enumerate(modules):
        fam   = _family_of(mod)
        y_pos = n_mods - row_idx - 0.5
        ax.text(
            -0.12, y_pos, mod,
            ha='right', va='center',
            fontsize=F3_ROW_LABEL_FS,
            color=FAMILY_COLORS.get(fam, '#333333'),
            fontfamily=FONT_FAMILY, clip_on=False,
        )

    # Column headers (top)
    for col_idx, (_, col_label, _) in enumerate(HEATMAP_COLS):
        ax.text(
            col_idx + 0.5, n_mods + 0.25, col_label,
            ha='center', va='bottom',
            fontsize=F3_COL_LABEL_FS, fontfamily=FONT_FAMILY,
            fontweight='bold', clip_on=False,
        )

    # Family separators (horizontal lines) and right-side labels
    for fam, start, end in fam_blocks:
        # Separator line above this family block (i.e. at y = n_mods - start)
        if start > 0:
            y_sep = n_mods - start
            ax.plot([0, n_cols], [y_sep, y_sep],
                    color=F3_SEP_COLOR, linewidth=F3_SEP_LW, zorder=5)
        # Family label to the right
        mid_y = n_mods - (start + end) / 2
        ax.text(
            n_cols + 0.15, mid_y, fam,
            ha='left', va='center',
            fontsize=F3_FAMILY_LABEL_FS,
            color=FAMILY_COLORS[fam],
            fontfamily=FONT_FAMILY, clip_on=False,
            rotation=90, rotation_mode='anchor',
        )

    # Color legend (vertical colorbar to the right)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.subplots_adjust(left=0.22, right=0.76, top=0.97, bottom=0.02)
    cbar_ax = fig.add_axes([0.88, 0.20, 0.016, 0.60])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels(['Best\n(col. min)', 'Worst\n(col. max)'])
    cbar.ax.tick_params(labelsize=F3_FAMILY_LABEL_FS)
    cbar.outline.set_edgecolor(SPINE_COLOR)

    _save(fig, 'spat_modules_fig3_heatmap', out_dir)


def main() -> None:
    """Run the command-line workflow."""

    parser = argparse.ArgumentParser(
        description='Plot spatial_modules profiling results.')
    parser.add_argument(
        '--csv', default=None, metavar='PATH',
        help='Path to profiling CSV (default: latest in results/).',
    )
    args = parser.parse_args()

    csv_path = (pathlib.Path(args.csv).resolve() if args.csv
                else _latest_csv(RESULTS_DIR))
    print(f'Loading: {csv_path.relative_to(_ROOT)}')
    rows = _load_csv(csv_path)
    print(f'  {len(rows)} rows | devices: {sorted({r["device"] for r in rows})}')

    # Apply global font
    plt.rcParams['font.family'] = FONT_FAMILY

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print('Generating figures...')
    fig1_violin(rows, FIGURES_DIR)
    fig2_lines(rows, FIGURES_DIR)
    fig3_heatmap(rows, FIGURES_DIR)
    print('All figures saved.')


if __name__ == '__main__':
    main()
