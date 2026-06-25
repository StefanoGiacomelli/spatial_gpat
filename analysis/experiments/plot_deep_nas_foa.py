#!/usr/bin/env python3
"""
analysis/experiments/plot_deep_nas_foa.py — NAS Phase 2 FOA Deep Experiments Figures
=======================================================================
Reads the CSV/JSON assets produced by ``analysis/experiments/analyze_deep_nas_foa.py`` and
generates all publication-quality figures (600 DPI SVG) for the deep_spec
phase.

Inputs  (results/)
------------------
  deep_nas_foa_summary.csv      (required)
  deep_nas_foa_tb_curves.csv    (required)
  deep_nas_foa_val_curves.csv   (required)
  deep_nas_foa_per_class.csv    (required)
  deep_nas_foa_best.json        (optional — winner annotation)
  shallow_nas_foa_summary.csv   (optional — cross-phase comparison figure)

Outputs  (figures/)
-------------------
  deep_learning_curves.svg       — val SELD per epoch, one color per run
  deep_train_loss_curves.svg     — train loss components (3 panels, one color per run)
  deep_val_loss_components.svg   — val loss components (4 panels, one color per run)
  deep_training_time.svg         — training duration bar chart
  deep_bar_test_seld.svg         — test SELD ranking bar chart
  deep_layer_comparison.svg      — l1 vs l2 grouped bars per arch family
  deep_seed_stability.svg        — e1_l2 seed pair stability (log Y on metrics panel)
  deep_ablation_heatmap.svg      — n_early × n_late SELD heatmap
  deep_multi_metric_heatmap.svg  — 7 exp × 8 metrics normalised heatmap
  deep_vs_shallow_comparison.svg — cross-phase best-run comparison (log Y, if available)

Usage
-----
  python analysis/experiments/plot_deep_nas_foa.py
  python analysis/experiments/plot_deep_nas_foa.py --results-dir results/ --out figures/

Author
------
Stefano Giacomelli, Ph.D. candidate — DISIM, University of L'Aquila
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / 'results' / 'experiments'
FIGURES_DIR = ROOT / 'figures' / 'experiments'

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Style ──────────────────────────────────────────────────────────────────────
FIGURE_DPI    = 600
FIGURE_FORMAT = 'svg'

plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
})

# ── Per-arch encoding (used only for bar charts / heatmaps that group by arch) ─
_ARCH_COLOR = {
    'e1': '#1F78B4',
    'e2': '#33A02C',
    'e3': '#E31A1C',
}
_ARCH_LABEL = {
    'e1': 'Early 1L (ResNet×1)',
    'e2': 'Early 2L (ResNet×2)',
    'e3': 'Early 3L (ResNet×3)',
}
_DEPTH_LABEL = {
    'l1': 'Late 1L (TT×1)',
    'l2': 'Late 2L (TT×2)',
}

# Two-seed colors for seed_stability figure
_SEED_COLORS = ['#1F78B4', '#E31A1C']   # blue s1, red s2

METRICS = [
    'SELD_score', 'LE_CD', 'ER_20', 'LR_CD', 'F_20',
    'sem_accuracy', 'sem_f1', 'sem_auroc',
]
LOWER_BETTER = {'SELD_score', 'LE_CD', 'ER_20'}


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _n_colors(exps: List[str]) -> Dict[str, tuple]:
    """One distinct tab10 color per experiment (alphabetical order)."""
    cmap = plt.cm.get_cmap('tab10')
    sorted_exps = sorted(set(exps))
    return {exp: cmap(i) for i, exp in enumerate(sorted_exps)}


def _save(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches='tight', dpi=FIGURE_DPI, format=FIGURE_FORMAT)
    plt.close(fig)
    print(f'  Saved {out.name}')


def _parse_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Add arch/depth/smoother/n_early/n_late/n_smoother columns if missing."""
    if 'arch' not in df.columns:
        def _p(exp):
            m = re.match(r'exp_e(\d+)_l(\d+)_s(\d+)', exp)
            ne, nl, ns = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return pd.Series({'arch': f'e{ne}', 'depth': f'l{nl}',
                               'smoother': f's{ns}',
                               'n_early': ne, 'n_late': nl, 'n_smoother': ns})
        df = df.join(df['exp'].apply(_p))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — val SELD learning curves (one color per run)
# ══════════════════════════════════════════════════════════════════════════════

def fig_learning_curves(
    summary_df: pd.DataFrame,
    val_curves_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Build and save the learning curves figure."""

    seld_curves = val_curves_df[val_curves_df['metric'] == 'SELD_score']
    colors = _n_colors(summary_df['exp'].tolist())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.45, linestyle='--', linewidth=0.5)

    # Grey shading between the two e1_l2 seeds
    e1l2_exps = ['exp_e1_l2_s1', 'exp_e1_l2_s2']
    e1l2 = seld_curves[seld_curves['exp'].isin(e1l2_exps)]
    if len(e1l2['exp'].unique()) == 2:
        piv = e1l2.pivot(index='epoch', columns='exp', values='value')
        ax.fill_between(
            piv.index, piv.min(axis=1), piv.max(axis=1),
            alpha=0.12, color='#888888', label='e1_l2 seed range',
        )

    legend_handles = []
    for _, row in summary_df.sort_values('exp').iterrows():
        exp = row['exp']
        color = colors[exp]
        curve = seld_curves[seld_curves['exp'] == exp] \
                    .sort_values('epoch')['value'].values
        if len(curve) == 0:
            continue
        label = exp.replace('exp_', '')
        line, = ax.plot(range(len(curve)), curve, color=color,
                        linewidth=1.4, alpha=0.85, label=label)
        be = int(row['best_epoch'])
        if be < len(curve):
            ax.plot(be, curve[be], '*', color=color, markersize=9, zorder=5)
        legend_handles.append(Line2D([0], [0], color=color, linewidth=1.4, label=label))

    legend_handles.append(
        Line2D([0], [0], color='k', marker='*', linestyle='none',
               markersize=7, label='best val epoch')
    )
    ax.legend(handles=legend_handles, loc='upper right', ncol=2, framealpha=0.85)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('val/SELD_score  (↓ better)')
    ax.set_title('Phase 2 — Validation SELD Score (deep_spec)')
    fig.tight_layout()
    _save(fig, out_dir / f'deep_learning_curves.{FIGURE_FORMAT}')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — train loss curves (3 panels, one color per run, legend top subplot)
# ══════════════════════════════════════════════════════════════════════════════

def fig_train_loss_curves(
    summary_df: pd.DataFrame,
    tb_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Build and save the train loss curves figure."""

    components = [
        ('train_loss_epoch', 'train/loss (total)'),
        ('train_loss_tpit',  'train/loss_tpit'),
        ('train_loss_bce',   'train/loss_bce'),
    ]
    colors = _n_colors(summary_df['exp'].tolist())

    fig, axes = plt.subplots(len(components), 1, figsize=(7, 9), sharex=True)

    legend_handles = []
    for ax_idx, (ax, (tag, ylabel)) in enumerate(zip(axes, components)):
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.45, linestyle='--', linewidth=0.5)
        sub = tb_df[tb_df['metric'] == tag]
        for _, row in summary_df.sort_values('exp').iterrows():
            exp = row['exp']
            vals = sub[sub['exp'] == exp].sort_values('epoch')['value'].values
            if len(vals) == 0:
                continue
            color = colors[exp]
            label = exp.replace('exp_', '')
            ax.plot(range(len(vals)), vals, color=color, linewidth=1.2, alpha=0.85)
            if ax_idx == 0:
                legend_handles.append(
                    Line2D([0], [0], color=color, linewidth=1.2, label=label)
                )
        ax.set_ylabel(ylabel)

    axes[0].legend(handles=legend_handles, loc='upper right',
                   ncol=2, framealpha=0.85)
    axes[-1].set_xlabel('Epoch')
    fig.suptitle('Phase 2 — Training Loss Components (deep_spec)', y=1.01)
    fig.tight_layout()
    _save(fig, out_dir / f'deep_train_loss_curves.{FIGURE_FORMAT}')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — val loss components (4 panels, one color per run, legend top subplot)
# ══════════════════════════════════════════════════════════════════════════════

def fig_val_loss_components(
    summary_df: pd.DataFrame,
    tb_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Build and save the val loss components figure."""

    components = [
        ('val_loss_tpit', 'val/loss_tpit'),
        ('val_loss_sed',  'val/loss_sed'),
        ('val_loss_doa',  'val/loss_doa'),
        ('val_loss_bce',  'val/loss_bce'),
    ]
    colors = _n_colors(summary_df['exp'].tolist())

    fig, axes = plt.subplots(len(components), 1, figsize=(7, 12), sharex=True)

    legend_handles = []
    for ax_idx, (ax, (tag, ylabel)) in enumerate(zip(axes, components)):
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.45, linestyle='--', linewidth=0.5)
        sub = tb_df[tb_df['metric'] == tag]
        for _, row in summary_df.sort_values('exp').iterrows():
            exp = row['exp']
            vals = sub[sub['exp'] == exp].sort_values('epoch')['value'].values
            if len(vals) == 0:
                continue
            color = colors[exp]
            label = exp.replace('exp_', '')
            ax.plot(range(len(vals)), vals, color=color, linewidth=1.2, alpha=0.85)
            if ax_idx == 0:
                legend_handles.append(
                    Line2D([0], [0], color=color, linewidth=1.2, label=label)
                )
        ax.set_ylabel(ylabel)

    axes[0].legend(handles=legend_handles, loc='upper right',
                   ncol=2, framealpha=0.85)
    axes[-1].set_xlabel('Epoch')
    fig.suptitle('Phase 2 — Validation Loss Components (deep_spec)', y=1.005)
    fig.tight_layout()
    _save(fig, out_dir / f'deep_val_loss_components.{FIGURE_FORMAT}')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — training time bar chart
# ══════════════════════════════════════════════════════════════════════════════

def fig_training_time(summary_df: pd.DataFrame, out_dir: Path) -> None:
    """Build and save the training time figure."""

    sub = summary_df.sort_values('duration_h').copy()
    colors = sub['arch'].map(_ARCH_COLOR).values

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, alpha=0.45, linestyle='--', linewidth=0.5)
    bars = ax.barh(range(len(sub)), sub['duration_h'].values,
                   color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub['exp'].str.replace('exp_', '').values, fontsize=7)
    ax.set_xlabel('Training duration  (hours)')
    ax.set_title('Phase 2 — Training Times (deep_spec)')

    for bar, val in zip(bars, sub['duration_h'].values):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}h', va='center', fontsize=7)

    arch_handles = [mpatches.Patch(color=c, label=_ARCH_LABEL[a])
                    for a, c in _ARCH_COLOR.items()]
    ax.legend(handles=arch_handles, loc='lower right', framealpha=0.9)
    ax.set_xlim(0, sub['duration_h'].max() * 1.18)
    fig.tight_layout()
    _save(fig, out_dir / f'deep_training_time.{FIGURE_FORMAT}')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — test SELD ranking bar chart
# ══════════════════════════════════════════════════════════════════════════════

def fig_bar_test_seld(summary_df: pd.DataFrame, out_dir: Path) -> None:
    """Build and save the bar test seld figure."""

    sub = summary_df.sort_values('test/SELD_score').copy()
    colors  = sub['arch'].map(_ARCH_COLOR).values
    hatches = sub['depth'].map({'l1': '', 'l2': '//'}).values

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.45, linestyle='--', linewidth=0.5)
    x = np.arange(len(sub))
    vals = sub['test/SELD_score'].values

    for i, (xi, val) in enumerate(zip(x, vals)):
        ax.bar(xi, val, color=colors[i], hatch=hatches[i],
               edgecolor='white', linewidth=0.5)
        ax.text(xi, val + 0.004, f'{val:.4f}',
                ha='center', va='bottom', fontsize=6.5, rotation=40)

    ax.set_xticks(x)
    ax.set_xticklabels(sub['exp'].str.replace('exp_', '').values,
                       rotation=35, ha='right', fontsize=7)
    ax.set_ylabel('test/SELD_score  (↓ better)')
    ax.set_title('Phase 2 — Test SELD Score Ranking (deep_spec)')

    legend_handles = [
        mpatches.Patch(color=c, label=_ARCH_LABEL[a])
        for a, c in _ARCH_COLOR.items()
    ] + [
        mpatches.Patch(facecolor='white', edgecolor='k', hatch='',  label='Late 1L'),
        mpatches.Patch(facecolor='white', edgecolor='k', hatch='//', label='Late 2L'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', framealpha=0.9, fontsize=7)
    ax.set_ylim(0, vals.max() * 1.14)
    fig.tight_layout()
    _save(fig, out_dir / f'deep_bar_test_seld.{FIGURE_FORMAT}')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 — layer depth comparison (l1 vs l2 per arch)
# ══════════════════════════════════════════════════════════════════════════════

def fig_layer_comparison(summary_df: pd.DataFrame, out_dir: Path) -> None:
    """Build and save the layer comparison figure."""

    archs = ['e1', 'e2', 'e3']
    fig, axes = plt.subplots(1, len(archs), figsize=(12, 5), sharey=True)

    for ax, arch in zip(axes, archs):
        sub = summary_df[summary_df['arch'] == arch].sort_values('depth')
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, alpha=0.45, linestyle='--', linewidth=0.5)

        x_pos, bar_vals, bar_errs = [], [], []
        for depth in ['l1', 'l2']:
            grp = sub[sub['depth'] == depth]['test/SELD_score'].values
            if len(grp) == 0:
                continue
            x_pos.append(depth)
            bar_vals.append(grp.mean())
            bar_errs.append(grp.std() if len(grp) > 1 else 0.0)

        x = np.arange(len(x_pos))
        bars = ax.bar(x, bar_vals,
                      yerr=bar_errs if any(e > 0 for e in bar_errs) else None,
                      color=_ARCH_COLOR[arch], alpha=0.85, edgecolor='white',
                      capsize=5, linewidth=0.5, width=0.55)
        for bar, val, err in zip(bars, bar_vals, bar_errs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + (err or 0) + 0.004,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(x_pos, fontsize=8)
        ax.set_title(_ARCH_LABEL[arch], fontsize=8)
        ax.set_xlabel('Late spatial depth')
        if arch == archs[0]:
            ax.set_ylabel('test/SELD_score  (↓ better)')

    fig.suptitle('Phase 2 — Late Depth Effect by Architecture Family '
                 '(error bars = seed std)', y=1.02)
    fig.tight_layout()
    _save(fig, out_dir / f'deep_layer_comparison.{FIGURE_FORMAT}')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 7 — seed stability (e1_l2 only)
#   Left  : val SELD curves — blue (s1) vs red (s2)
#   Right : test metrics bar chart — log Y axis, Δ embedded in tick labels
# ══════════════════════════════════════════════════════════════════════════════

def fig_seed_stability(
    summary_df: pd.DataFrame,
    val_curves_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Build and save the seed stability figure."""

    seeds = summary_df[
        (summary_df['arch'] == 'e1') & (summary_df['depth'] == 'l2')
    ].sort_values('smoother')

    if len(seeds) < 2:
        print('  Skipping seed_stability (fewer than 2 seeds for e1_l2)')
        return

    seld_curves = val_curves_df[val_curves_df['metric'] == 'SELD_score']

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4))

    # ── Left: val SELD curves (blue s1, red s2) ──────────────────────────────
    ax_left.set_axisbelow(True)
    ax_left.grid(True, alpha=0.45, linestyle='--', linewidth=0.5)
    curves_list = []
    for (_, row), color in zip(seeds.iterrows(), _SEED_COLORS):
        curve = seld_curves[seld_curves['exp'] == row['exp']] \
                    .sort_values('epoch')['value'].values
        curves_list.append(curve)
        be = int(row['best_epoch'])
        ax_left.plot(range(len(curve)), curve, color=color, linewidth=1.3,
                     label=row['exp'].replace('exp_', ''))
        if be < len(curve):
            ax_left.plot(be, curve[be], '*', color=color, markersize=9, zorder=5)
    if len(curves_list) == 2 and len(curves_list[0]) == len(curves_list[1]):
        ax_left.fill_between(range(len(curves_list[0])),
                             curves_list[0], curves_list[1],
                             alpha=0.15, color='#888888')
    ax_left.set_xlabel('Epoch')
    ax_left.set_ylabel('val/SELD_score  (↓ better)')
    ax_left.set_title('val SELD — e1_l2 seed comparison')
    ax_left.legend(fontsize=7, framealpha=0.85)

    # ── Right: test metrics — log Y, Δ in tick labels ────────────────────────
    ax_right.set_axisbelow(True)
    ax_right.yaxis.grid(True, alpha=0.45, linestyle='--', linewidth=0.5)
    x = np.arange(len(METRICS))
    width = 0.35
    row_a = seeds.iloc[0]
    row_b = seeds.iloc[1]

    vals_a = [float(row_a.get(f'test/{m}', float('nan'))) for m in METRICS]
    vals_b = [float(row_b.get(f'test/{m}', float('nan'))) for m in METRICS]

    for k, (vals, color) in enumerate(zip([vals_a, vals_b], _SEED_COLORS)):
        exp_name = (row_a if k == 0 else row_b)['exp'].replace('exp_', '')
        bars = ax_right.bar(x + k * width, vals, width,
                            color=color, label=exp_name,
                            alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals):
            if np.isnan(val):
                continue
            # Multiplicative offset for log scale
            ax_right.text(bar.get_x() + bar.get_width() / 2,
                          val * 1.12,
                          f'{val:.3f}', ha='center', va='bottom',
                          fontsize=5.5, rotation=70)

    # Δ embedded in tick labels: "MetricName\nΔ=x.xxx"
    tick_labels = []
    for m, va, vb in zip(METRICS, vals_a, vals_b):
        delta = abs(va - vb)
        tick_labels.append(f'{m}\nΔ={delta:.3f}')

    ax_right.set_yscale('log')
    # Headroom: set top to max value * 5 to accommodate rotated annotations
    all_vals = [v for v in vals_a + vals_b if not np.isnan(v) and v > 0]
    if all_vals:
        ax_right.set_ylim(top=max(all_vals) * 5)

    ax_right.set_xticks(x + width / 2)
    ax_right.set_xticklabels(tick_labels, rotation=0, ha='center', fontsize=6.5)
    ax_right.set_ylabel('metric value  (log scale)')
    ax_right.set_title('Test metrics — e1_l2 seeds')
    ax_right.legend(fontsize=7, framealpha=0.85)

    fig.suptitle('Phase 2 — Seed Stability Analysis (exp_e1_l2_s1 vs exp_e1_l2_s2)',
                 y=1.02)
    fig.tight_layout()
    _save(fig, out_dir / f'deep_seed_stability.{FIGURE_FORMAT}')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 8 — ablation heatmap (n_early × n_late)
# ══════════════════════════════════════════════════════════════════════════════

def fig_ablation_heatmap(summary_df: pd.DataFrame, out_dir: Path) -> None:
    """Build and save the ablation heatmap figure."""

    pivot = (summary_df
             .groupby(['n_early', 'n_late'])['test/SELD_score']
             .mean()
             .unstack('n_late'))
    pivot.index.name   = 'n_early (ResNet layers)'
    pivot.columns.name = 'n_late (TT layers)'

    fig, ax = plt.subplots(figsize=(7, 5))
    vmin = summary_df['test/SELD_score'].min() - 0.02
    vmax = summary_df['test/SELD_score'].max() + 0.02
    sns.heatmap(
        pivot, ax=ax,
        cmap='RdYlGn_r',
        annot=True, fmt='.4f', annot_kws={'size': 10},
        vmin=vmin, vmax=vmax,
        cbar_kws={'shrink': 0.8, 'label': 'SELD_score (mean over seeds)'},
        linewidths=0.5, linecolor='#cccccc',
    )
    ax.set_title('Phase 2 — Architecture Depth Ablation\n'
                 'Test SELD Score  (mean over seeds, lower = better)')
    ax.set_xlabel('n_late  (Track Transformer layers)')
    ax.set_ylabel('n_early  (ResNet layers)')
    fig.tight_layout()
    _save(fig, out_dir / f'deep_ablation_heatmap.{FIGURE_FORMAT}')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 9 — multi-metric normalised heatmap
# ══════════════════════════════════════════════════════════════════════════════

def fig_multi_metric_heatmap(
    summary_df: pd.DataFrame,
    out_dir: Path,
    best_exp: Optional[str] = None,
) -> None:
    """Build and save the multi metric heatmap figure."""

    sub = summary_df.sort_values('test/SELD_score').copy()
    row_labels = sub['exp'].str.replace('exp_', '').values

    matrix = np.zeros((len(sub), len(METRICS)))
    annot  = np.empty_like(matrix, dtype=object)
    for j, m in enumerate(METRICS):
        col = f'test/{m}'
        vals = sub[col].values.astype(float)
        lo, hi = vals.min(), vals.max()
        norm = (vals - lo) / (hi - lo + 1e-9)
        if m in LOWER_BETTER:
            norm = 1.0 - norm
        matrix[:, j] = norm
        for i, v in enumerate(vals):
            annot[i, j] = f'{v:.4f}'

    fig, ax = plt.subplots(figsize=(11, 5))
    hm = sns.heatmap(
        matrix, ax=ax,
        cmap='RdYlGn',
        vmin=0, vmax=1,
        annot=annot, fmt='',
        annot_kws={'size': 7},
        xticklabels=METRICS,
        yticklabels=row_labels,
        cbar_kws={'label': 'normalised score  (1 = best)', 'shrink': 0.8},
        linewidths=0.3,
    )
    hm.collections[0].colorbar.set_ticks([])

    winner_exp = best_exp or sub.iloc[0]['exp'].replace('exp_', '')
    for i, lbl in enumerate(row_labels):
        if lbl == winner_exp.replace('exp_', ''):
            ax.annotate('★', xy=(len(METRICS) + 0.15, i + 0.5),
                        xycoords='data', ha='left', va='center',
                        fontsize=11, color='#B8860B', clip_on=False)
            break

    ax.set_title('Phase 2 — Multi-Metric Comparison  (1 = best, normalised per column)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    fig.tight_layout()
    _save(fig, out_dir / f'deep_multi_metric_heatmap.{FIGURE_FORMAT}')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 10 — cross-phase comparison (deep best vs shallow best)
#   Log Y axis; Δ(shallow→deep) embedded in x-axis tick labels
# ══════════════════════════════════════════════════════════════════════════════

def fig_cross_phase_comparison(
    summary_df: pd.DataFrame,
    shallow_df: Optional[pd.DataFrame],
    out_dir: Path,
) -> None:
    """Build and save the cross phase comparison figure."""

    deep_best = summary_df.sort_values('test/SELD_score').iloc[0]
    phases: List[Dict] = [
        {'label': 'Deep best\n' + deep_best['exp'].replace('exp_', ''),
         'color': '#E31A1C',
         'vals': {m: float(deep_best.get(f'test/{m}', float('nan'))) for m in METRICS}},
    ]
    if shallow_df is not None:
        shallow_best = shallow_df.sort_values('test/SELD_score').iloc[0]
        phases.insert(0, {
            'label': 'Shallow best\n' + shallow_best['exp'].replace('exp_', ''),
            'color': '#1F78B4',
            'vals': {m: float(shallow_best.get(f'test/{m}', float('nan'))) for m in METRICS},
        })

    n_metrics = len(METRICS)
    n_phases  = len(phases)
    x = np.arange(n_metrics)
    w = 0.7 / n_phases

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.45, linestyle='--', linewidth=0.5, which='both')

    for k, phase in enumerate(phases):
        offsets = x + (k - n_phases / 2 + 0.5) * w
        vals = [phase['vals'].get(m, float('nan')) for m in METRICS]
        bars = ax.bar(offsets, vals, w, color=phase['color'],
                      label=phase['label'], alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals):
            if not np.isnan(val) and val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        val * 1.12,
                        f'{val:.3f}', ha='center', va='bottom',
                        fontsize=6, rotation=40)

    # Δ(shallow→deep) embedded in x-axis tick labels when both phases present
    if len(phases) == 2:
        tick_labels = []
        for m in METRICS:
            v0 = phases[0]['vals'].get(m, float('nan'))
            v1 = phases[1]['vals'].get(m, float('nan'))
            if np.isnan(v0) or np.isnan(v1):
                tick_labels.append(m)
                continue
            delta = v1 - v0
            sign = '+' if delta >= 0 else ''
            if m in LOWER_BETTER:
                arrow = '↓' if delta < 0 else '↑'
                col_marker = '✓' if delta < 0 else '✗'
            else:
                arrow = '↑' if delta > 0 else '↓'
                col_marker = '✓' if delta > 0 else '✗'
            tick_labels.append(f'{m}\n{sign}{delta:.3f} {arrow}')
    else:
        tick_labels = METRICS

    ax.set_yscale('log')
    # Headroom: top = max positive val * 4 to fit rotated annotations
    all_vals = [v for phase in phases
                for v in phase['vals'].values()
                if not np.isnan(v) and v > 0]
    if all_vals:
        ax.set_ylim(top=max(all_vals) * 4)

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, ha='center', fontsize=7.5)
    ax.set_ylabel('metric value  (log scale)')
    ax.set_title('Cross-Phase Comparison: Phase 1 (shallow) → Phase 2 (deep)')
    ax.legend(loc='upper right', framealpha=0.9)
    fig.tight_layout()
    _save(fig, out_dir / f'deep_vs_shallow_comparison.{FIGURE_FORMAT}')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def _load(results_dir: Path, filename: str, required: bool = True) -> Optional[pd.DataFrame]:
    path = results_dir / filename
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f'Required CSV not found: {path}\n'
                f'Run analysis/experiments/analyze_deep_nas_foa.py first.'
            )
        print(f'  Optional {filename} not found — skipping dependent figures.')
        return None
    return pd.read_csv(path)


def main() -> None:
    """Run the command-line workflow."""

    p = argparse.ArgumentParser(
        description='Generate phase-2 deep NAS figures from pre-computed CSVs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--results-dir', type=Path, default=RESULTS_DIR)
    p.add_argument('--out', type=Path, default=FIGURES_DIR)
    args = p.parse_args()

    rd  = args.results_dir
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    print('\nLoading assets…')
    summary_df    = _load(rd, 'deep_nas_foa_summary.csv')
    tb_df         = _load(rd, 'deep_nas_foa_tb_curves.csv')
    val_curves_df = _load(rd, 'deep_nas_foa_val_curves.csv')
    shallow_df    = _load(rd, 'shallow_nas_foa_summary.csv', required=False)

    summary_df = _parse_factors(summary_df)

    best_exp = None
    best_path = rd / 'deep_nas_foa_best.json'
    if best_path.exists():
        best_exp = json.loads(best_path.read_text()).get('winner')

    print('\nGenerating figures…')
    fig_learning_curves(summary_df, val_curves_df, out)
    fig_train_loss_curves(summary_df, tb_df, out)
    fig_val_loss_components(summary_df, tb_df, out)
    fig_training_time(summary_df, out)
    fig_bar_test_seld(summary_df, out)
    fig_layer_comparison(summary_df, out)
    fig_seed_stability(summary_df, val_curves_df, out)
    fig_ablation_heatmap(summary_df, out)
    fig_multi_metric_heatmap(summary_df, out, best_exp=best_exp)
    fig_cross_phase_comparison(summary_df, shallow_df, out)

    print('\nDone.')


if __name__ == '__main__':
    main()
