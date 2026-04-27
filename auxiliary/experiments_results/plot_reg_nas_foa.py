"""
tests/plot_reg_nas_foa.py — NAS Phase 3 FOA Regularised Experiments Figures
=============================================================================
Reads the CSV/JSON assets produced by ``tests/analyze_reg_nas_foa.py`` and
generates all publication-quality figures (600 DPI SVG) for the deep_reg
cross-stitch ablation phase.

Inputs  (results/)
------------------
  reg_nas_foa_summary.csv     (required)
  reg_nas_foa_tb_curves.csv   (required)
  reg_nas_foa_val_curves.csv  (required)
  reg_nas_foa_deltas.csv      (required)
  reg_nas_foa_per_class.csv   (required)
  deep_nas_foa_summary.csv    (optional — cross-phase comparison)
  shallow_nas_foa_summary.csv (optional — cross-phase comparison)

Outputs  (figures/)
-------------------
  reg_learning_curves.svg       — val SELD per epoch, all 4 conditions
  reg_train_loss_curves.svg     — train loss components (3 panels)
  reg_val_loss_components.svg   — val loss components (4 panels)
  reg_bar_test_seld.svg         — test SELD ranking with Δ vs baseline
  reg_delta_bar.svg             — diverging bar: Δ per metric × cs variant
  reg_cs_heatmap.svg            — 2×2 cross-stitch factorial heatmap
  reg_per_class_f1_heatmap.svg  — 13-class × 4-condition F1 heatmap
  reg_multi_metric_heatmap.svg  — 4 exp × 8 metrics normalised heatmap
  reg_vs_deep_vs_shallow.svg    — cross-phase pipeline progression (if avail)

Usage
-----
  python tests/plot_reg_nas_foa.py
  python tests/plot_reg_nas_foa.py --results-dir results/ --out figures/

Author
------
Stefano Giacomelli, Ph.D. candidate — DISIM, University of L'Aquila
"""

from __future__ import annotations

import argparse
import json
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
ROOT        = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / 'results'
FIGURES_DIR = ROOT / 'figures'

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

# ── Visual encoding ────────────────────────────────────────────────────────────
_CS_LABEL  = {'cs00': 'No CS (baseline)', 'cs01': 'Late CS', 'cs10': 'Early CS', 'cs11': 'Both CS'}
_CS_COLOR  = {'cs00': '#444444', 'cs01': '#1F78B4', 'cs10': '#33A02C', 'cs11': '#E31A1C'}
_CS_LS     = {'cs00': '-',  'cs01': '--', 'cs10': ':',  'cs11': '-.'}
_CS_MARKER = {'cs00': 'o',  'cs01': 's',  'cs10': '^',  'cs11': 'D'}
_CS_HATCH  = {'cs00': '',   'cs01': '///', 'cs10': '...', 'cs11': 'xxx'}

METRICS = [
    'SELD_score', 'LE_CD', 'ER_20', 'LR_CD', 'F_20',
    'sem_accuracy', 'sem_f1', 'sem_auroc',
]
LOWER_BETTER = {'SELD_score', 'LE_CD', 'ER_20'}

CS_ORDER = ['cs00', 'cs01', 'cs10', 'cs11']


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches='tight', dpi=FIGURE_DPI, format=FIGURE_FORMAT)
    plt.close(fig)
    print(f'  Saved {out.name}')


def _legend_handles() -> List:
    handles = []
    for cs in CS_ORDER:
        handles.append(
            Line2D([0], [0], color=_CS_COLOR[cs], linestyle=_CS_LS[cs],
                   marker=_CS_MARKER[cs], markersize=4, linewidth=1.3,
                   label=_CS_LABEL[cs])
        )
    return handles


def _ensure_cs_col(df: pd.DataFrame) -> pd.DataFrame:
    """Add cs_code column from exp if missing."""
    if 'cs_code' not in df.columns:
        import re
        def _extract(exp):
            m = re.search(r'cs(\d{2})', exp)
            return f'cs{m.group(1)}' if m else ''
        df = df.copy()
        df['cs_code'] = df['exp'].apply(_extract)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — val SELD learning curves
# ══════════════════════════════════════════════════════════════════════════════

def fig_learning_curves(
    summary_df: pd.DataFrame,
    val_curves_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    seld_curves = val_curves_df[val_curves_df['metric'] == 'SELD_score']
    plot_order = [cs for cs in CS_ORDER if cs in summary_df['cs_code'].values]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.45, linestyle='--', linewidth=0.5)

    for cs in plot_order:
        row = summary_df[summary_df['cs_code'] == cs]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        exp = row['exp']
        curve = seld_curves[seld_curves['exp'] == exp].sort_values('epoch')['value'].values
        if len(curve) == 0:
            continue
        is_baseline = (cs == 'cs00')
        ax.plot(range(len(curve)), curve,
                color=_CS_COLOR[cs], linestyle=_CS_LS[cs],
                marker=_CS_MARKER[cs], markersize=3 if not is_baseline else 4,
                linewidth=1.8 if is_baseline else 1.2,
                alpha=1.0 if is_baseline else 0.82,
                label=_CS_LABEL[cs],
                zorder=5 if is_baseline else 3)
        be = int(row['best_epoch'])
        if be < len(curve):
            ax.plot(be, curve[be], '*', color=_CS_COLOR[cs],
                    markersize=9 if is_baseline else 7, zorder=6)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('val/SELD_score  (↓ better)')
    ax.set_title('Phase 3 — Validation SELD Score (deep_reg, cross-stitch ablation)\n'
                 'cs00 = dropout-regularized baseline (bold)')
    ax.legend(handles=_legend_handles() + [
        Line2D([0], [0], color='k', marker='*', linestyle='none',
               markersize=7, label='best val epoch')
    ], loc='upper right', framealpha=0.85)
    fig.tight_layout()
    _save(fig, out_dir / f'reg_learning_curves.{FIGURE_FORMAT}')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — train loss curves
# ══════════════════════════════════════════════════════════════════════════════

def fig_train_loss_curves(
    summary_df: pd.DataFrame,
    tb_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    components = [
        ('train_loss_epoch', 'train/loss (total)'),
        ('train_loss_tpit',  'train/loss_tpit'),
        ('train_loss_bce',   'train/loss_bce'),
    ]
    fig, axes = plt.subplots(len(components), 1, figsize=(7, 9), sharex=True)

    for ax, (tag, ylabel) in zip(axes, components):
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.45, linestyle='--', linewidth=0.5)
        sub = tb_df[tb_df['metric'] == tag]
        for cs in CS_ORDER:
            row = summary_df[summary_df['cs_code'] == cs]
            if len(row) == 0:
                continue
            exp = row.iloc[0]['exp']
            vals = sub[sub['exp'] == exp].sort_values('epoch')['value'].values
            if len(vals) == 0:
                continue
            is_baseline = (cs == 'cs00')
            ax.plot(range(len(vals)), vals,
                    color=_CS_COLOR[cs], linestyle=_CS_LS[cs],
                    linewidth=1.8 if is_baseline else 1.2,
                    alpha=1.0 if is_baseline else 0.82,
                    label=_CS_LABEL[cs])
        ax.set_ylabel(ylabel)

    axes[-1].set_xlabel('Epoch')
    axes[0].legend(handles=_legend_handles(), loc='upper right', framealpha=0.9)
    fig.suptitle('Phase 3 — Training Loss Components (deep_reg)')
    fig.tight_layout()
    _save(fig, out_dir / f'reg_train_loss_curves.{FIGURE_FORMAT}')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — val loss components
# ══════════════════════════════════════════════════════════════════════════════

def fig_val_loss_components(
    summary_df: pd.DataFrame,
    tb_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    components = [
        ('val_loss_tpit', 'val/loss_tpit'),
        ('val_loss_sed',  'val/loss_sed'),
        ('val_loss_doa',  'val/loss_doa'),
        ('val_loss_bce',  'val/loss_bce'),
    ]
    fig, axes = plt.subplots(len(components), 1, figsize=(7, 12), sharex=True)

    for ax, (tag, ylabel) in zip(axes, components):
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.45, linestyle='--', linewidth=0.5)
        sub = tb_df[tb_df['metric'] == tag]
        for cs in CS_ORDER:
            row = summary_df[summary_df['cs_code'] == cs]
            if len(row) == 0:
                continue
            exp = row.iloc[0]['exp']
            vals = sub[sub['exp'] == exp].sort_values('epoch')['value'].values
            if len(vals) == 0:
                continue
            is_baseline = (cs == 'cs00')
            ax.plot(range(len(vals)), vals,
                    color=_CS_COLOR[cs], linestyle=_CS_LS[cs],
                    linewidth=1.8 if is_baseline else 1.2,
                    alpha=1.0 if is_baseline else 0.82)
        ax.set_ylabel(ylabel)

    axes[-1].set_xlabel('Epoch')
    axes[0].legend(handles=_legend_handles(), loc='upper center', framealpha=0.9)
    fig.suptitle('Phase 3 — Validation Loss Components (deep_reg)')
    fig.tight_layout()
    _save(fig, out_dir / f'reg_val_loss_components.{FIGURE_FORMAT}')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — test SELD ranking with Δ vs baseline
# ══════════════════════════════════════════════════════════════════════════════

def fig_bar_test_seld(summary_df: pd.DataFrame, out_dir: Path) -> None:
    # Fixed order: baseline first, then Early, Late, Both CS
    FIXED_ORDER = ['cs00', 'cs10', 'cs01', 'cs11']
    sub = summary_df.copy()

    baseline_val = float(sub[sub['cs_code'] == 'cs00']['test/SELD_score'].values[0]) \
        if 'cs00' in sub['cs_code'].values else float('nan')

    present = [cs for cs in FIXED_ORDER if cs in sub['cs_code'].values]
    vals_dict = {
        cs: float(sub[sub['cs_code'] == cs]['test/SELD_score'].values[0])
        for cs in present
        if len(sub[sub['cs_code'] == cs]) > 0
    }

    x = np.arange(len(present))
    bar_width = 0.42

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.45, linestyle='--', linewidth=0.5)

    for xi, cs in enumerate(present):
        val = vals_dict.get(cs, float('nan'))
        if np.isnan(val):
            continue
        ax.bar(xi, val, width=bar_width,
               color=_CS_COLOR.get(cs, '#888888'),
               hatch=_CS_HATCH.get(cs, ''),
               edgecolor='white', linewidth=0.5)
        ax.text(xi, val + 0.003, f'{val:.4f}',
                ha='center', va='bottom', fontsize=7)

    # Tick labels: condition names only
    ax.set_xticks(x)
    ax.set_xticklabels([_CS_LABEL.get(cs, cs) for cs in present], fontsize=7.5)

    # Colored delta annotations below x-axis ticks
    trans = ax.get_xaxis_transform()
    for xi, cs in enumerate(present):
        if cs == 'cs00' or np.isnan(baseline_val):
            continue
        val = vals_dict.get(cs, float('nan'))
        if np.isnan(val):
            continue
        delta = val - baseline_val
        sign = '+' if delta >= 0 else ''
        # SELD is lower-better: positive delta = worse, negative = better
        arrow = '↑' if delta > 0 else '↓'
        col = '#cc0000' if delta > 0 else '#007700'
        ax.text(xi, -0.17, f'Δ={sign}{delta:.4f} {arrow}',
                transform=trans, ha='center', va='top',
                fontsize=6.5, color=col, fontweight='bold')

    ax.set_ylabel('test/SELD_score  (↓ better)')
    ax.set_title('Phase 3 — Test SELD Ranking  (Δ vs cs00)')
    all_vals = [v for v in vals_dict.values() if not np.isnan(v)]
    ax.set_ylim(0, max(all_vals) * 1.12 if all_vals else 1.0)
    fig.subplots_adjust(bottom=0.20)
    _save(fig, out_dir / f'reg_bar_test_seld.{FIGURE_FORMAT}')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — diverging bar: Δ per metric × cs variant
# ══════════════════════════════════════════════════════════════════════════════

def fig_delta_bar(deltas_df: pd.DataFrame, out_dir: Path) -> None:
    variants = [cs for cs in ['cs01', 'cs10', 'cs11'] if cs in deltas_df['cs_code'].values]
    n_variants = len(variants)
    x = np.arange(len(METRICS))
    total_width = 0.72
    w = total_width / n_variants

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_axisbelow(True)

    # Collect all sign-adjusted delta values to set scale/ylim
    all_vals_data = []
    bars_data = []
    for k, cs in enumerate(variants):
        row = deltas_df[deltas_df['cs_code'] == cs].iloc[0]
        vals = []
        for m in METRICS:
            d = float(row.get(f'delta_{m}', float('nan')))
            vals.append(-d if m in LOWER_BETTER else d)
        offsets = x + (k - n_variants / 2 + 0.5) * w
        bars_data.append((offsets, vals, cs))
        all_vals_data.extend([v for v in vals if not np.isnan(v)])

    # Determine symlog threshold and ylim
    nonzero_abs = [abs(v) for v in all_vals_data if v != 0]
    linthresh = min(nonzero_abs) * 0.3 if nonzero_abs else 1e-4
    linthresh = max(linthresh, 1e-5)

    pos_vals = [v for v in all_vals_data if v > 0]
    neg_vals = [v for v in all_vals_data if v < 0]
    top    = max(pos_vals) * 8  if pos_vals else  linthresh * 10
    bottom = min(neg_vals) * 10 if neg_vals else -linthresh * 10

    ax.set_yscale('symlog', linthresh=linthresh)
    ax.set_ylim(bottom=bottom, top=top)
    ax.yaxis.grid(True, alpha=0.4, linestyle='--', linewidth=0.5)
    ax.axhline(0, color='k', linewidth=1.3, zorder=3)

    for offsets, vals, cs in bars_data:
        bars = ax.bar(offsets, vals, w, color=_CS_COLOR[cs],
                      hatch=_CS_HATCH[cs], label=_CS_LABEL[cs],
                      alpha=0.85, edgecolor='white', linewidth=0.5)

        for bar, val in zip(bars, vals):
            if np.isnan(val) or val == 0:
                continue
            sign = '+' if val >= 0 else ''
            # Multiplicative offset outside linear region, additive inside
            if abs(val) >= linthresh:
                y_annot = val * 1.6 if val > 0 else val * 1.6
            else:
                y_annot = (val + linthresh * 0.6 if val >= 0
                           else val - linthresh * 0.6)
            ax.text(bar.get_x() + bar.get_width() / 2, y_annot,
                    f'{sign}{val:.4f}',
                    ha='center', va='bottom' if val >= 0 else 'top',
                    fontsize=5.5, rotation=55)

    ax.set_xticks(x)
    ax.set_xticklabels(METRICS, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Δ metric  (+↑ = improvement, −↓ = degradation)')
    ax.set_title('Phase 3 — Cross-Stitch Effect on All Metrics\n'
                 'Δ vs cs00 baseline  (sign-adjusted: positive = always better)')
    ax.legend(loc='upper right', framealpha=0.9)
    fig.tight_layout()
    _save(fig, out_dir / f'reg_delta_bar.{FIGURE_FORMAT}')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 — 2×2 cross-stitch factorial heatmap
# ══════════════════════════════════════════════════════════════════════════════

def fig_cs_heatmap(summary_df: pd.DataFrame, out_dir: Path) -> None:
    sub = summary_df.copy()
    sub['cs_early_int'] = sub['cs_early'].astype(int)
    sub['cs_late_int']  = sub['cs_late'].astype(int)
    pivot = sub.pivot_table(index='cs_early_int', columns='cs_late_int',
                            values='test/SELD_score', aggfunc='mean')
    pivot.index   = ['cs_early=OFF', 'cs_early=ON']
    pivot.columns = ['cs_late=OFF', 'cs_late=ON']

    vmin = sub['test/SELD_score'].min() - 0.02
    vmax = sub['test/SELD_score'].max() + 0.02

    baseline_val = float(pivot.iloc[0, 0]) if not np.isnan(pivot.iloc[0, 0]) else float('nan')
    annot = np.empty_like(pivot.values, dtype=object)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = float(pivot.iloc[i, j])
            delta = v - baseline_val
            sign = '+' if delta >= 0 else ''
            if i == 0 and j == 0:
                annot[i, j] = f'{v:.4f}\n(baseline)'
            else:
                arrow = '↓' if delta < 0 else '↑'
                annot[i, j] = f'{v:.4f}\n{sign}{delta:.4f} {arrow}'

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        pivot, ax=ax,
        cmap='RdYlGn_r',
        vmin=vmin, vmax=vmax,
        annot=annot, fmt='',
        annot_kws={'size': 9},
        cbar_kws={'shrink': 0.8, 'label': 'SELD_score'},
        linewidths=0.8, linecolor='white',
    )
    ax.set_title('Phase 3 — Cross-Stitch Factorial  (2×2 ablation)\n'
                 'test SELD_score  (lower = better)')
    ax.set_xlabel('Late spatial cross-stitch')
    ax.set_ylabel('Early spatial cross-stitch')
    fig.tight_layout()
    _save(fig, out_dir / f'reg_cs_heatmap.{FIGURE_FORMAT}')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 7 — per-class F1 heatmap (13 classes × 4 conditions)
# ══════════════════════════════════════════════════════════════════════════════

def fig_per_class_f1_heatmap(per_class_df: pd.DataFrame, out_dir: Path) -> None:
    pivot = per_class_df.pivot_table(
        index='class_name', columns='cs_code', values='f1', aggfunc='mean'
    )
    if 'cs00' in pivot.columns:
        pivot = pivot.sort_values('cs00', ascending=True)
    cols = [cs for cs in CS_ORDER if cs in pivot.columns]
    pivot = pivot[cols]
    pivot.columns = [_CS_LABEL[c] for c in cols]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot, ax=ax,
        cmap='RdYlGn',
        vmin=0.0, vmax=1.0,
        annot=True, fmt='.3f', annot_kws={'size': 7.5},
        cbar_kws={'shrink': 0.8, 'label': 'F1 score  (higher = better)'},
        linewidths=0.3, linecolor='#dddddd',
    )
    ax.set_title('Phase 3 — Per-Class F1 Score by Cross-Stitch Condition\n'
                 '(sorted by cs00/baseline F1 ascending — hardest classes at top)')
    ax.set_xlabel('Cross-stitch condition')
    ax.set_ylabel('Sound event class')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')
    fig.tight_layout()
    _save(fig, out_dir / f'reg_per_class_f1_heatmap.{FIGURE_FORMAT}')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 8 — multi-metric normalised heatmap
# ══════════════════════════════════════════════════════════════════════════════

def fig_multi_metric_heatmap(
    summary_df: pd.DataFrame,
    out_dir: Path,
    best_exp: Optional[str] = None,
) -> None:
    sub = summary_df.sort_values('test/SELD_score').copy()
    row_labels = [_CS_LABEL.get(cs, cs) for cs in sub['cs_code'].values]

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

    fig, ax = plt.subplots(figsize=(9, 3))
    hm = sns.heatmap(
        matrix, ax=ax,
        cmap='RdYlGn',
        vmin=0, vmax=1,
        annot=annot, fmt='',
        annot_kws={'size': 7.5},
        xticklabels=METRICS,
        yticklabels=row_labels,
        cbar_kws={'label': 'normalised score  (1 = best)', 'shrink': 0.8},
        linewidths=0.3,
    )
    hm.collections[0].colorbar.set_ticks([])

    best_cs = best_exp or sub.iloc[0]['cs_code']
    for i, lbl in enumerate(row_labels):
        row_cs = sub.iloc[i]['cs_code']
        if row_cs == best_cs:
            ax.annotate('★', xy=(len(METRICS) + 0.1, i + 0.5),
                        xycoords='data', ha='left', va='center',
                        fontsize=11, color='#B8860B', clip_on=False)
            break

    ax.set_title('Phase 3 — Multi-Metric Comparison  (1 = best, normalised per column)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    fig.tight_layout()
    _save(fig, out_dir / f'reg_multi_metric_heatmap.{FIGURE_FORMAT}')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 9 — cross-phase pipeline progression
# ══════════════════════════════════════════════════════════════════════════════

def fig_cross_phase_comparison(
    reg_df: pd.DataFrame,
    deep_df: Optional[pd.DataFrame],
    shallow_df: Optional[pd.DataFrame],
    out_dir: Path,
) -> None:
    phases: List[Dict] = []

    if shallow_df is not None:
        best = shallow_df.sort_values('test/SELD_score').iloc[0]
        phases.append({
            'label': 'Phase 1 (shallow)\n' + best['exp'].replace('exp_', ''),
            'color': '#A6CEE3',
            'vals': {m: float(best.get(f'test/{m}', float('nan'))) for m in METRICS},
        })

    if deep_df is not None:
        best = deep_df.sort_values('test/SELD_score').iloc[0]
        phases.append({
            'label': 'Phase 2 (deep)\n' + best['exp'].replace('exp_', ''),
            'color': '#1F78B4',
            'vals': {m: float(best.get(f'test/{m}', float('nan'))) for m in METRICS},
        })

    reg_best = reg_df.sort_values('test/SELD_score').iloc[0]
    phases.append({
        'label': 'Phase 3 (reg)\n' + _CS_LABEL.get(reg_best['cs_code'], reg_best['cs_code']),
        'color': '#E31A1C',
        'vals': {m: float(reg_best.get(f'test/{m}', float('nan'))) for m in METRICS},
    })

    n_phases    = len(phases)
    x           = np.arange(len(METRICS))
    total_width = 0.72
    w           = total_width / n_phases

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.45, linestyle='--', linewidth=0.5)

    all_pos_vals = []
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
                        fontsize=5.5, rotation=40)
                all_pos_vals.append(val)

    ax.set_yscale('log')
    max_val = max(all_pos_vals) if all_pos_vals else 1.0
    min_val = min(all_pos_vals) if all_pos_vals else 0.1
    ax.set_ylim(bottom=min_val * 0.3, top=max_val * 4)

    # Colored delta annotations below x-axis ticks (one row per phase transition)
    trans     = ax.get_xaxis_transform()
    n_steps   = len(phases) - 1
    y_offsets = [-0.13 - 0.09 * i for i in range(n_steps)]

    for step, y_off in zip(range(n_steps), y_offsets):
        for j, m in enumerate(METRICS):
            v0 = phases[step]['vals'].get(m, float('nan'))
            v1 = phases[step + 1]['vals'].get(m, float('nan'))
            if np.isnan(v0) or np.isnan(v1):
                continue
            delta = v1 - v0
            sign  = '+' if delta >= 0 else ''
            if m in LOWER_BETTER:
                col   = '#007700' if delta < 0 else '#cc0000'
                arrow = '↓' if delta < 0 else '↑'
            else:
                col   = '#007700' if delta > 0 else '#cc0000'
                arrow = '↑' if delta > 0 else '↓'
            ax.text(j, y_off, f'{sign}{delta:.3f}{arrow}',
                    transform=trans, ha='center', va='top',
                    fontsize=5.5, color=col, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(METRICS, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('metric value  (log scale)')
    ax.set_title('NAS-SELD Pipeline Progression: Phase 1 → Phase 2 → Phase 3\n'
                 '(Δ annotations: green = improvement, red = degradation)')
    ax.legend(loc='upper right', framealpha=0.9)
    bottom_pad = 0.20 + 0.07 * max(n_steps - 1, 0)
    fig.subplots_adjust(bottom=bottom_pad)
    _save(fig, out_dir / f'reg_vs_deep_vs_shallow.{FIGURE_FORMAT}')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def _load(results_dir: Path, filename: str, required: bool = True) -> Optional[pd.DataFrame]:
    path = results_dir / filename
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f'Required CSV not found: {path}\n'
                f'Run tests/analyze_reg_nas_foa.py first.'
            )
        print(f'  Optional {filename} not found — skipping dependent figures.')
        return None
    return pd.read_csv(path)


def main() -> None:
    p = argparse.ArgumentParser(
        description='Generate phase-3 regularised NAS figures from pre-computed CSVs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--results-dir', type=Path, default=RESULTS_DIR)
    p.add_argument('--out', type=Path, default=FIGURES_DIR)
    args = p.parse_args()

    rd  = args.results_dir
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    print('\nLoading assets…')
    summary_df    = _load(rd, 'reg_nas_foa_summary.csv')
    tb_df         = _load(rd, 'reg_nas_foa_tb_curves.csv')
    val_curves_df = _load(rd, 'reg_nas_foa_val_curves.csv')
    deltas_df     = _load(rd, 'reg_nas_foa_deltas.csv')
    per_class_df  = _load(rd, 'reg_nas_foa_per_class.csv')
    deep_df       = _load(rd, 'deep_nas_foa_summary.csv', required=False)
    shallow_df    = _load(rd, 'shallow_nas_foa_summary.csv', required=False)

    summary_df = _ensure_cs_col(summary_df)

    best_exp = None
    best_path = rd / 'reg_nas_foa_best.json'
    if best_path.exists():
        best_exp = json.loads(best_path.read_text()).get('cs_code')

    print('\nGenerating figures…')
    fig_learning_curves(summary_df, val_curves_df, out)
    fig_train_loss_curves(summary_df, tb_df, out)
    fig_val_loss_components(summary_df, tb_df, out)
    fig_bar_test_seld(summary_df, out)
    fig_delta_bar(deltas_df, out)
    fig_cs_heatmap(summary_df, out)
    fig_per_class_f1_heatmap(per_class_df, out)
    fig_multi_metric_heatmap(summary_df, out, best_exp=best_exp)
    fig_cross_phase_comparison(summary_df, deep_df, shallow_df, out)

    print('\nDone.')


if __name__ == '__main__':
    main()
