"""
auxiliary/experiments_results/plot_finetune_nas_foa.py
=======================================================
NAS-SELD Phase 4 (Finetune) FOA Figures.

Reads the CSV assets produced by ``analyze_finetune_nas_foa.py`` and
generates publication-quality figures (600 DPI SVG) for the finetune phase.

Inputs  (results/)
------------------
  finetune_nas_foa_summary.csv       (required)
  finetune_nas_foa_val_curves.csv    (required)
  finetune_nas_foa_tb_curves.csv     (required)
  finetune_nas_foa_sem_perclass.csv  (required for Fig A6, A7)
  finetune_nas_foa_sed_perclass.csv  (optional — enables Group C heatmaps)

Outputs  (figures/)
-------------------
Group A — finetune-internal analysis (6 figures):
  finetune_learning_curves.svg        — val SELD per epoch, 2-panel
  finetune_train_loss_curves.svg      — training loss components (3 panels)
  finetune_semantic_cm_heatmap.svg    — 3×2 grid: per-class P/R/F1 per experiment
  finetune_architecture_ablation.svg  — 4-exp val-only bar chart (log Y-axis)
  finetune_sem_perclass_heatmap.svg   — semantic-head per-class F1 (3 panels)

Group C — SED per-class figures (2 figures, gated on --compute-inference):
  finetune_sed_perclass_heatmap.svg  — SED per-class F1 (3 panels, val set)
  finetune_sed_cm_heatmap.svg        — SED N×N confusion matrix (3×2 grid, val set)

Usage
-----
  python auxiliary/experiments_results/plot_finetune_nas_foa.py

Author
------
Stefano Giacomelli, Ph.D. candidate — DISIM, University of L'Aquila
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / 'results'
FIGURES_DIR = ROOT / 'figures'
RUNS_DIR    = ROOT / 'runs' / 'at2seld_nas'

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Style ──────────────────────────────────────────────────────────────────────
FIGURE_DPI    = 600
FIGURE_FORMAT = 'svg'

plt.rcParams.update({
    'font.size':        9,
    'axes.titlesize':   9,
    'axes.labelsize':   8,
    'legend.fontsize':  7,
    'xtick.labelsize':  7,
    'ytick.labelsize':  7,
})

# ── Experiment metadata ────────────────────────────────────────────────────────
ALL_EXPS = [
    'exp_ft_mixed_bce',
    'exp_ft_mixed_focal',
    'exp_ft_tau_nigens_focal',
    'exp_ft_tau2019_focal',
    'exp_topt_BiGRU_n2_h512_merge_sum',
    'exp_topt_TT_n1_ffn8_pe_4tracks',
]
DATA_LOSS_EXPS = [
    'exp_ft_mixed_bce',
    'exp_ft_mixed_focal',
    'exp_ft_tau_nigens_focal',
    'exp_ft_tau2019_focal',
]
ARCH_OPT_EXPS = [
    'exp_topt_BiGRU_n2_h512_merge_sum',
    'exp_topt_TT_n1_ffn8_pe_4tracks',
]
BALANCED_EXPS = [
    'exp_ft_mixed_bce',
    'exp_ft_mixed_focal',
    'exp_topt_BiGRU_n2_h512_merge_sum',
    'exp_topt_TT_n1_ffn8_pe_4tracks',
]

_EXP_SHORT = {
    'exp_ft_mixed_bce':                  'mixed-BCE',
    'exp_ft_mixed_focal':                'mixed-Focal',
    'exp_ft_tau_nigens_focal':           'NIGENS-Focal',
    'exp_ft_tau2019_focal':              'TAU2019-Focal',
    'exp_topt_BiGRU_n2_h512_merge_sum':  'topt-BiGRU',
    'exp_topt_TT_n1_ffn8_pe_4tracks':    'topt-TT4',
}

# Fixed tab10 color per experiment (stable across all figures)
_CMAP = matplotlib.colormaps.get_cmap('tab10')
EXP_COLOR: Dict[str, tuple] = {
    exp: _CMAP(i) for i, exp in enumerate(ALL_EXPS)
}

VAL_METRICS = [
    'SELD_score', 'LE_CD', 'ER_20', 'LR_CD', 'F_20',
    'sem_accuracy', 'sem_f1', 'sem_auroc',
]
LOWER_BETTER = {'SELD_score', 'LE_CD', 'ER_20'}

_METRIC_LABEL = {
    'SELD_score':   'SELD ↓',
    'LE_CD':        'LE_CD ↓',
    'ER_20':        'ER@20° ↓',
    'LR_CD':        'LR_CD ↑',
    'F_20':         'F@20° ↑',
    'sem_accuracy': 'Sem-Acc ↑',
    'sem_f1':       'Sem-F1 ↑',
    'sem_auroc':    'Sem-AUROC ↑',
}

CLASS_NAMES = {
    'STARSS23': [
        'Female speech', 'Male speech', 'Clapping', 'Telephone', 'Laughter',
        'Domestic sounds', 'Walk, footsteps', 'Door open/close', 'Music',
        'Musical instrument', 'Water tap', 'Bell', 'Knock',
    ],
    'TAU-NIGENS2021': [
        'Alarm', 'Baby cry', 'Crash', 'Dog bark', 'Female scream',
        'Female speech', 'Footstep', 'Knock', 'Male scream', 'Male speech',
        'Phone ring', 'Piano',
    ],
    'TAU2019': [
        'Clearthroat', 'Cough', 'Doorslam', 'Drawer', 'Keyboard',
        'KeysDrop', 'Knock', 'Laughter', 'Pageturn', 'Phone', 'Speech',
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig: plt.Figure, stem: str) -> None:
    out = FIGURES_DIR / f'{stem}.{FIGURE_FORMAT}'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches='tight', dpi=FIGURE_DPI, format=FIGURE_FORMAT)
    plt.close(fig)
    print(f'  Saved {out.name}')


def _short(exp: str) -> str:
    return _EXP_SHORT.get(exp, exp.replace('exp_', ''))


def _legend_handles(exps: List[str]) -> List[Line2D]:
    return [Line2D([0], [0], color=EXP_COLOR[e], linewidth=1.4, label=_short(e))
            for e in exps if e in EXP_COLOR]


def _best_epoch(summary_df: pd.DataFrame, exp: str) -> int:
    row = summary_df[summary_df['exp_name'] == exp]
    return int(row['best_epoch'].iloc[0]) if len(row) else 0


def _metric_disp(m: str) -> str:
    return _METRIC_LABEL.get(m, m)


# ══════════════════════════════════════════════════════════════════════════════
# Group A — Finetune-internal figures
# ══════════════════════════════════════════════════════════════════════════════

# ── Fig A1: learning curves ───────────────────────────────────────────────────

def fig_learning_curves(
    summary_df: pd.DataFrame,
    val_curves_df: pd.DataFrame,
) -> None:
    seld = val_curves_df[val_curves_df['metric'] == 'SELD_score']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    groups = [
        ('data_loss', DATA_LOSS_EXPS,  'Data / Loss group'),
        ('arch_opt',  ARCH_OPT_EXPS,   'Architecture optimisation'),
    ]

    for ax, (_, exps, title) in zip(axes, groups):
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.45, linestyle='--', linewidth=0.5)
        handles = []
        for exp in exps:
            sub = seld[seld['exp_name'] == exp].sort_values('epoch')
            if sub.empty:
                continue
            curve = sub['value'].values
            color = EXP_COLOR.get(exp, '#333333')
            ax.plot(range(len(curve)), curve, color=color,
                    linewidth=1.4, alpha=0.85)
            be = _best_epoch(summary_df, exp)
            if be < len(curve):
                ax.plot(be, curve[be], '*', color=color, markersize=9, zorder=5)
            handles.append(Line2D([0], [0], color=color, linewidth=1.4,
                                   label=_short(exp)))

        handles.append(Line2D([0], [0], color='k', marker='*', linestyle='none',
                               markersize=7, label='best val epoch'))
        ax.legend(handles=handles, loc='upper right', framealpha=0.85)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('val/SELD_score  (↓ better)')
        ax.set_title(title)

    fig.suptitle('Finetune — Validation SELD Score', fontsize=10, fontweight='bold')
    fig.tight_layout()
    _save(fig, 'finetune_learning_curves')


# ── Fig A2: training loss curves ──────────────────────────────────────────────

def fig_train_loss_curves(
    summary_df: pd.DataFrame,
    tb_df: pd.DataFrame,
) -> None:
    components = [
        ('train/loss_epoch', 'train/loss (total)'),
        ('train/loss_tpit',  'train/loss_tpit (spatial)'),
        ('train/loss_bce',   'train/loss_bce (semantic)'),
    ]

    fig, axes = plt.subplots(len(components), 1, figsize=(8, 9), sharex=True)

    legend_handles: List[Line2D] = []
    for ax_idx, (ax, (tag, ylabel)) in enumerate(zip(axes, components)):
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.45, linestyle='--', linewidth=0.5)
        for exp in ALL_EXPS:
            sub = tb_df[(tb_df['exp_name'] == exp) & (tb_df['tag'] == tag)] \
                      .sort_values('step')
            if sub.empty:
                continue
            color = EXP_COLOR.get(exp, '#333333')
            ax.plot(sub['step'].values, sub['value'].values,
                    color=color, linewidth=1.2, alpha=0.85)
            if ax_idx == 0:
                legend_handles.append(
                    Line2D([0], [0], color=color, linewidth=1.2, label=_short(exp))
                )
        ax.set_ylabel(ylabel)

    axes[0].legend(handles=legend_handles, loc='upper right',
                   ncol=2, framealpha=0.85)
    axes[-1].set_xlabel('Epoch')
    fig.suptitle('Finetune — Training Loss Components', fontsize=10,
                 fontweight='bold')
    fig.tight_layout()
    _save(fig, 'finetune_train_loss_curves')


# ── Fig A3: semantic N×N confusion matrix heatmap (3×2 grid) ─────────────────

def _draw_nxn_cm(
    ax: plt.Axes,
    exp: str,
    label: str,
    dataset: str,
    cls_names: List[str],
    cm_df: pd.DataFrame,
    extra_filters: Optional[Dict[str, str]] = None,
    run_hint: str = '--compute-semantic-cm',
) -> Optional[object]:
    """
    Draw one N×N confusion matrix panel (row-normalised).
    C[i,j] = P(predict j | true i).  Diagonal = recall per class.
    extra_filters: additional column→value filters (e.g. split, dataset_subset).
    Returns AxesImage or None if no data.
    """
    n_cls = len(cls_names)
    sub   = cm_df[cm_df['exp_name'] == exp]
    if extra_filters:
        for col, val in extra_filters.items():
            if col in sub.columns:
                sub = sub[sub[col] == val]

    if sub.empty:
        ax.text(0.5, 0.5, f'{label}\n(run {run_hint})',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=8, style='italic', color='gray')
        ax.set_title(f'{label}  [{dataset}]', fontsize=8, fontweight='bold')
        ax.axis('off')
        return None

    mat = np.zeros((n_cls, n_cls), dtype=float)
    for _, row in sub.iterrows():
        i = int(row['true_class_idx'])
        j = int(row['pred_class_idx'])
        if i < n_cls and j < n_cls:
            mat[i, j] = float(row['count'])

    # Row-normalise: P(predict j | true i)
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    mat_norm = mat / row_sums

    im = ax.imshow(mat_norm, cmap='Blues', vmin=0, vmax=1, aspect='auto')

    for i in range(n_cls):
        for j in range(n_cls):
            v  = mat_norm[i, j]
            fc = 'white' if v > 0.55 else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=5.5, color=fc)

    ax.set_xticks(range(n_cls))
    ax.set_xticklabels(cls_names, rotation=45, ha='right', fontsize=5.5)
    ax.set_yticks(range(n_cls))
    ax.set_yticklabels(cls_names, fontsize=5.5)
    ax.set_xlabel('Predicted class', fontsize=6)
    ax.set_ylabel('True class',      fontsize=6)
    ax.set_title(f'{label}  [{dataset}]', fontsize=8, fontweight='bold')
    return im


def fig_semantic_cm_heatmap(sem_cm_df: pd.DataFrame) -> None:
    """3×2 grid of N×N row-normalised semantic confusion matrices (one per experiment)."""

    exp_config = [
        ('exp_ft_mixed_bce',                 'mixed-BCE',    'STARSS23',       CLASS_NAMES['STARSS23']),
        ('exp_ft_mixed_focal',               'mixed-Focal',  'STARSS23',       CLASS_NAMES['STARSS23']),
        ('exp_ft_tau_nigens_focal',          'NIGENS-Focal', 'TAU-NIGENS2021', CLASS_NAMES['TAU-NIGENS2021']),
        ('exp_ft_tau2019_focal',             'TAU2019-Focal','TAU2019',        CLASS_NAMES['TAU2019']),
        ('exp_topt_BiGRU_n2_h512_merge_sum', 'topt-BiGRU',  'STARSS23',       CLASS_NAMES['STARSS23']),
        ('exp_topt_TT_n1_ffn8_pe_4tracks',   'topt-TT4',    'STARSS23',       CLASS_NAMES['STARSS23']),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(15, 14))

    for ax, (exp, label, dataset, cls_names) in zip(axes.flatten(), exp_config):
        im = _draw_nxn_cm(ax, exp, label, dataset, cls_names, sem_cm_df)
        if im is not None:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label='P(predict col | true row)')

    fig.suptitle(
        'Finetune — Semantic Classification Head: N×N Confusion Matrix (row-normalised)\n'
        '(GP-AT EPANNs semantic branch — val set)',
        fontsize=10, fontweight='bold',
    )
    fig.tight_layout()
    _save(fig, 'finetune_semantic_cm_heatmap_val')


# ── Fig A4: architecture ablation (val only, log Y-axis) ─────────────────────

def fig_architecture_ablation(summary_df: pd.DataFrame) -> None:
    exps = [
        'exp_ft_mixed_bce',
        'exp_ft_mixed_focal',
        'exp_topt_BiGRU_n2_h512_merge_sum',
        'exp_topt_TT_n1_ffn8_pe_4tracks',
    ]
    seld_5   = ['SELD_score', 'LE_CD', 'LR_CD', 'ER_20', 'F_20']
    col_tmpl = 'best_val_{m}'

    n_exp   = len(exps)
    n_met   = len(seld_5)
    x       = np.arange(n_met)
    bwidth  = 0.18
    offsets = [(i - (n_exp - 1) / 2) * bwidth for i in range(n_exp)]

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=0.45, linestyle='--', linewidth=0.5, which='both')

    for exp, off in zip(exps, offsets):
        row = summary_df[summary_df['exp_name'] == exp]
        if row.empty:
            continue
        color = EXP_COLOR.get(exp, '#888888')
        vals = []
        for m in seld_5:
            col = col_tmpl.format(m=m)
            v   = row[col].values[0] if col in row.columns else float('nan')
            vals.append(float(v) if not np.isnan(float(v)) else 1e-4)
        ax.bar(x + off, vals, width=bwidth * 0.9,
               color=color, alpha=0.85, label=_short(exp))

    ax.set_yscale('log')
    ax.set_ylim(bottom=5e-2)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f'{v:g}')
    )
    ax.set_xticks(x)
    ax.set_xticklabels([_metric_disp(m) for m in seld_5])
    ax.set_ylabel('Score — log scale\n(LE_CD in °, others 0–1)')
    ax.legend(loc='upper right', framealpha=0.85)
    ax.set_title('Architecture Optimisation Ablation — Validation Metrics',
                 fontweight='bold')

    fig.suptitle('Finetune — Architecture Ablation (val)',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    _save(fig, 'finetune_architecture_ablation')


# ── Fig A5: semantic head per-class F1 heatmap ───────────────────────────────

def fig_sem_perclass_heatmap(sem_df: pd.DataFrame) -> None:
    if sem_df.empty:
        print('  [SKIP] finetune_sem_perclass_heatmap: sem_perclass CSV is empty.')
        return

    # Panel layout:
    #   Panel 0  — STARSS23 experiments (mixed_bce + mixed_focal), if available
    #   Panel 1  — NIGENS-Focal single-dataset
    #   Panel 2  — TAU2019-Focal single-dataset

    panels = []

    # STARSS23 joint-trained experiments (available after --compute-semantic-cm)
    starss_exps = ['exp_ft_mixed_bce', 'exp_ft_mixed_focal']
    starss_present = [e for e in starss_exps if e in sem_df['exp_name'].values]
    if starss_present:
        panels.append({
            'exps':      starss_present,
            'cls_names': CLASS_NAMES['STARSS23'],
            'title':     'STARSS23 — Semantic Head per-class F1  '
                         '(mixed-BCE and mixed-Focal)',
        })

    # Single-dataset experiments
    for exp, dataset, label in [
        ('exp_ft_tau_nigens_focal', 'TAU-NIGENS2021', 'NIGENS-Focal'),
        ('exp_ft_tau2019_focal',    'TAU2019',        'TAU2019-Focal'),
    ]:
        if exp in sem_df['exp_name'].values:
            panels.append({
                'exps':      [exp],
                'cls_names': CLASS_NAMES[dataset],
                'title':     f'{label} — Semantic Head per-class F1',
            })

    if not panels:
        print('  [SKIP] finetune_sem_perclass_heatmap: no data in sem_perclass CSV.')
        return

    # Filter to val split when the column is present (avoids mixing test rows)
    val_sem_df = (sem_df[sem_df['split'] == 'val']
                  if 'split' in sem_df.columns else sem_df)

    fig, axes = plt.subplots(len(panels), 1,
                             figsize=(11, 2.8 * len(panels)))
    if len(panels) == 1:
        axes = [axes]

    for ax, panel in zip(axes, panels):
        exps      = panel['exps']
        cls_names = panel['cls_names']
        n_exp     = len(exps)
        n_cls     = len(cls_names)

        mat = np.full((n_exp, n_cls), float('nan'))
        for i, exp in enumerate(exps):
            sub = val_sem_df[val_sem_df['exp_name'] == exp].sort_values('class_idx')
            for _, row in sub.iterrows():
                c = int(row['class_idx'])
                if c < n_cls:
                    mat[i, c] = float(row['f1'])

        # Sort classes by mean F1 ascending
        col_mean     = np.nanmean(mat, axis=0)
        order        = np.argsort(col_mean)
        mat_sorted   = mat[:, order]
        names_sorted = [cls_names[i] for i in order]

        im = ax.imshow(mat_sorted, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

        for i in range(n_exp):
            for j in range(n_cls):
                v = mat_sorted[i, j]
                if not np.isnan(v):
                    fc = 'black' if 0.15 < v < 0.85 else 'white'
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                            fontsize=8, color=fc)

        ax.set_xticks(range(n_cls))
        ax.set_xticklabels(names_sorted, rotation=40, ha='right', fontsize=6.5)
        ax.set_yticks(range(n_exp))
        ax.set_yticklabels([_short(e) for e in exps], fontsize=8)
        ax.set_title(panel['title'], fontsize=8)

        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label='F1')

    fig.suptitle(
        'Finetune — Semantic Classification Head per-class F1\n'
        '(GP-AT EPANNs semantic branch)',
        fontsize=9, fontweight='bold',
    )
    fig.tight_layout()
    _save(fig, 'finetune_sem_perclass_heatmap')


# ══════════════════════════════════════════════════════════════════════════════
# Group C — SED per-class figures (mirror of Group A semantic figures)
# ══════════════════════════════════════════════════════════════════════════════

def fig_sed_perclass_heatmap(sed_df: Optional[pd.DataFrame]) -> None:
    """
    SED activity head per-class F1 heatmap — mirrors fig_sem_perclass_heatmap.
    3 panels (val split only):
      Panel 0 — 4 BalancedSTARSS23 experiments × 13 STARSS23 classes
      Panel 1 — NIGENS-Focal × 12 TAU-NIGENS classes
      Panel 2 — TAU2019-Focal × 11 TAU2019 classes
    """
    if sed_df is None or sed_df.empty:
        print('  [SKIP] finetune_sed_perclass_heatmap: sed_perclass CSV not found. '
              'Run with --compute-inference first.')
        return

    val_df = sed_df[sed_df['split'] == 'val']

    panels = []

    # Balanced exps (val on STARSS23 val split)
    bal_present = [e for e in BALANCED_EXPS if e in val_df['exp_name'].values]
    if bal_present:
        panels.append({
            'exps':           bal_present,
            'cls_names':      CLASS_NAMES['STARSS23'],
            'dataset_subset': 'starss23',
            'title':          'STARSS23 — SED Activity Head per-class F1  '
                              '(BalancedSTARSS23 experiments)',
        })

    for exp, dataset, label, subset in [
        ('exp_ft_tau_nigens_focal', 'TAU-NIGENS2021', 'NIGENS-Focal', 'single'),
        ('exp_ft_tau2019_focal',    'TAU2019',        'TAU2019-Focal', 'single'),
    ]:
        if exp in val_df['exp_name'].values:
            panels.append({
                'exps':           [exp],
                'cls_names':      CLASS_NAMES[dataset],
                'dataset_subset': subset,
                'title':          f'{label} — SED Activity Head per-class F1',
            })

    if not panels:
        print('  [SKIP] finetune_sed_perclass_heatmap: no val data in sed_perclass CSV.')
        return

    fig, axes = plt.subplots(len(panels), 1,
                             figsize=(11, 2.8 * len(panels)))
    if len(panels) == 1:
        axes = [axes]

    for ax, panel in zip(axes, panels):
        exps      = panel['exps']
        cls_names = panel['cls_names']
        subset    = panel['dataset_subset']
        n_exp     = len(exps)
        n_cls     = len(cls_names)

        sub = val_df[val_df['dataset_subset'] == subset]
        mat = np.full((n_exp, n_cls), float('nan'))
        for i, exp in enumerate(exps):
            for _, row in sub[sub['exp_name'] == exp].iterrows():
                c = int(row['class_idx'])
                if c < n_cls:
                    mat[i, c] = float(row['f1'])

        col_mean     = np.nanmean(mat, axis=0)
        order        = np.argsort(col_mean)
        mat_sorted   = mat[:, order]
        names_sorted = [cls_names[i] for i in order]

        im = ax.imshow(mat_sorted, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

        for i in range(n_exp):
            for j in range(n_cls):
                v = mat_sorted[i, j]
                if not np.isnan(v):
                    fc = 'black' if 0.15 < v < 0.85 else 'white'
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                            fontsize=8, color=fc)

        ax.set_xticks(range(n_cls))
        ax.set_xticklabels(names_sorted, rotation=40, ha='right', fontsize=6.5)
        ax.set_yticks(range(n_exp))
        ax.set_yticklabels([_short(e) for e in exps], fontsize=8)
        ax.set_title(panel['title'], fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label='F1')

    fig.suptitle(
        'Finetune — SED Activity Head per-class F1\n'
        '(spatial branch — val set)',
        fontsize=9, fontweight='bold',
    )
    fig.tight_layout()
    _save(fig, 'finetune_sed_perclass_heatmap')


def fig_sed_cm_heatmap(sed_cm_df: Optional[pd.DataFrame]) -> None:
    """
    SED activity head N×N confusion matrix — mirrors fig_semantic_cm_heatmap.
    3×2 grid, one row-normalised matrix per experiment (val split).
    """
    if sed_cm_df is None or sed_cm_df.empty:
        print('  [SKIP] finetune_sed_cm_heatmap: sed_cm_matrix CSV not found. '
              'Run with --compute-inference first.')
        return

    # (exp, short_label, dataset_label, class_list, split, dataset_subset)
    exp_config = [
        ('exp_ft_mixed_bce',                 'mixed-BCE',    'STARSS23',
         CLASS_NAMES['STARSS23'],       'val', 'starss23'),
        ('exp_ft_mixed_focal',               'mixed-Focal',  'STARSS23',
         CLASS_NAMES['STARSS23'],       'val', 'starss23'),
        ('exp_ft_tau_nigens_focal',          'NIGENS-Focal', 'TAU-NIGENS2021',
         CLASS_NAMES['TAU-NIGENS2021'], 'val', 'single'),
        ('exp_ft_tau2019_focal',             'TAU2019-Focal','TAU2019',
         CLASS_NAMES['TAU2019'],        'val', 'single'),
        ('exp_topt_BiGRU_n2_h512_merge_sum', 'topt-BiGRU',  'STARSS23',
         CLASS_NAMES['STARSS23'],       'val', 'starss23'),
        ('exp_topt_TT_n1_ffn8_pe_4tracks',   'topt-TT4',    'STARSS23',
         CLASS_NAMES['STARSS23'],       'val', 'starss23'),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(15, 14))

    for ax, (exp, label, dataset, cls_names, split, subset) in \
            zip(axes.flatten(), exp_config):
        im = _draw_nxn_cm(
            ax, exp, label, dataset, cls_names, sed_cm_df,
            extra_filters={'split': split, 'dataset_subset': subset},
            run_hint='--compute-inference',
        )
        if im is not None:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label='P(predict col | true row)')

    fig.suptitle(
        'Finetune — SED Activity Head: N×N Confusion Matrix (row-normalised)\n'
        '(spatial branch — val set)',
        fontsize=10, fontweight='bold',
    )
    fig.tight_layout()
    _save(fig, 'finetune_sed_cm_heatmap_val')


# ── Semantic CM heatmap — test set ───────────────────────────────────────────

def fig_semantic_cm_heatmap_test(sem_cm_df: pd.DataFrame) -> None:
    """
    3×2 grid of N×N semantic CM matrices — test split.
    Each exp is evaluated on its natural test set:
      balanced exps → STARSS23 test (split='test', dataset_subset='starss23')
      single-dataset exps → native test (split='test', dataset_subset='single')
    """
    if sem_cm_df is None or sem_cm_df.empty:
        print('  [SKIP] finetune_semantic_cm_heatmap_test: sem_cm_matrix CSV empty.')
        return
    if 'split' not in sem_cm_df.columns:
        print('  [SKIP] finetune_semantic_cm_heatmap_test: no split column '
              '(re-run --compute-semantic-cm).')
        return

    exp_config = [
        ('exp_ft_mixed_bce',                 'mixed-BCE',    'STARSS23',
         CLASS_NAMES['STARSS23'],       'test', 'starss23'),
        ('exp_ft_mixed_focal',               'mixed-Focal',  'STARSS23',
         CLASS_NAMES['STARSS23'],       'test', 'starss23'),
        ('exp_ft_tau_nigens_focal',          'NIGENS-Focal', 'TAU-NIGENS2021',
         CLASS_NAMES['TAU-NIGENS2021'], 'test', 'single'),
        ('exp_ft_tau2019_focal',             'TAU2019-Focal','TAU2019',
         CLASS_NAMES['TAU2019'],        'test', 'single'),
        ('exp_topt_BiGRU_n2_h512_merge_sum', 'topt-BiGRU',  'STARSS23',
         CLASS_NAMES['STARSS23'],       'test', 'starss23'),
        ('exp_topt_TT_n1_ffn8_pe_4tracks',   'topt-TT4',    'STARSS23',
         CLASS_NAMES['STARSS23'],       'test', 'starss23'),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(15, 14))
    for ax, (exp, label, dataset, cls_names, split, subset) in \
            zip(axes.flatten(), exp_config):
        im = _draw_nxn_cm(
            ax, exp, label, dataset, cls_names, sem_cm_df,
            extra_filters={'split': split, 'dataset_subset': subset},
            run_hint='--compute-semantic-cm',
        )
        if im is not None:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label='P(predict col | true row)')

    fig.suptitle(
        'Finetune — Semantic Classification Head: N×N Confusion Matrix (row-normalised)\n'
        '(GP-AT EPANNs semantic branch — test set)',
        fontsize=10, fontweight='bold',
    )
    fig.tight_layout()
    _save(fig, 'finetune_semantic_cm_heatmap_test')


# ── SED CM heatmaps — test sets ───────────────────────────────────────────────

def _sed_cm_test_grid(
    sed_cm_df: pd.DataFrame,
    exp_config: list,
    suptitle: str,
    stem: str,
) -> None:
    """Generic helper: draw a 3×2 SED CM grid for a given test subset."""
    if sed_cm_df is None or sed_cm_df.empty:
        print(f'  [SKIP] {stem}: sed_cm_matrix CSV empty.')
        return
    if 'split' not in sed_cm_df.columns:
        print(f'  [SKIP] {stem}: no split column (re-run --compute-inference).')
        return

    fig, axes = plt.subplots(3, 2, figsize=(15, 14))
    for ax, (exp, label, dataset, cls_names, split, subset) in \
            zip(axes.flatten(), exp_config):
        im = _draw_nxn_cm(
            ax, exp, label, dataset, cls_names, sed_cm_df,
            extra_filters={'split': split, 'dataset_subset': subset},
            run_hint='--compute-inference',
        )
        if im is not None:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label='P(predict col | true row)')

    fig.suptitle(suptitle, fontsize=10, fontweight='bold')
    fig.tight_layout()
    _save(fig, stem)


def fig_sed_cm_heatmap_test_starss(sed_cm_df: pd.DataFrame) -> None:
    """SED CM — test on STARSS23 test set (4 balanced exps; single-dataset exps N/A)."""
    exp_config = [
        ('exp_ft_mixed_bce',                 'mixed-BCE',   'STARSS23',
         CLASS_NAMES['STARSS23'], 'test', 'starss23'),
        ('exp_ft_mixed_focal',               'mixed-Focal', 'STARSS23',
         CLASS_NAMES['STARSS23'], 'test', 'starss23'),
        ('exp_ft_tau_nigens_focal',          'NIGENS-Focal','(N/A)',
         CLASS_NAMES['STARSS23'], 'test', 'starss23'),   # will show placeholder
        ('exp_ft_tau2019_focal',             'TAU2019-Focal','(N/A)',
         CLASS_NAMES['STARSS23'], 'test', 'starss23'),   # will show placeholder
        ('exp_topt_BiGRU_n2_h512_merge_sum', 'topt-BiGRU', 'STARSS23',
         CLASS_NAMES['STARSS23'], 'test', 'starss23'),
        ('exp_topt_TT_n1_ffn8_pe_4tracks',   'topt-TT4',   'STARSS23',
         CLASS_NAMES['STARSS23'], 'test', 'starss23'),
    ]
    _sed_cm_test_grid(
        sed_cm_df, exp_config,
        suptitle=('Finetune — SED Activity Head: N×N Confusion Matrix (row-normalised)\n'
                  '(spatial branch — test set: STARSS23)'),
        stem='finetune_sed_cm_heatmap_test_starss',
    )


def fig_sed_cm_heatmap_test_tau(sed_cm_df: pd.DataFrame) -> None:
    """
    SED CM — test on TAU-NIGENS test set.
    Balanced exps: STARSS23 projected space (13 cls).
    NIGENS-Focal: native TAU-NIGENS space (12 cls).
    TAU2019-Focal: N/A.
    """
    exp_config = [
        ('exp_ft_mixed_bce',                 'mixed-BCE',   'STARSS23 (proj)',
         CLASS_NAMES['STARSS23'],       'test', 'tau_nigens'),
        ('exp_ft_mixed_focal',               'mixed-Focal', 'STARSS23 (proj)',
         CLASS_NAMES['STARSS23'],       'test', 'tau_nigens'),
        ('exp_ft_tau_nigens_focal',          'NIGENS-Focal','TAU-NIGENS2021',
         CLASS_NAMES['TAU-NIGENS2021'], 'test', 'single'),
        ('exp_ft_tau2019_focal',             'TAU2019-Focal','(N/A)',
         CLASS_NAMES['TAU2019'],        'test', 'tau_nigens'),  # placeholder
        ('exp_topt_BiGRU_n2_h512_merge_sum', 'topt-BiGRU', 'STARSS23 (proj)',
         CLASS_NAMES['STARSS23'],       'test', 'tau_nigens'),
        ('exp_topt_TT_n1_ffn8_pe_4tracks',   'topt-TT4',   'STARSS23 (proj)',
         CLASS_NAMES['STARSS23'],       'test', 'tau_nigens'),
    ]
    _sed_cm_test_grid(
        sed_cm_df, exp_config,
        suptitle=('Finetune — SED Activity Head: N×N Confusion Matrix (row-normalised)\n'
                  '(spatial branch — test set: TAU-NIGENS2021)\n'
                  'Balanced exps: STARSS23 projected space  |  NIGENS-Focal: native space'),
        stem='finetune_sed_cm_heatmap_test_tau',
    )


def fig_sed_cm_heatmap_test_tau2019(sed_cm_df: pd.DataFrame) -> None:
    """
    SED CM — test on TAU2019 test set.
    Balanced exps: STARSS23 projected space (13 cls).
    TAU2019-Focal: native TAU2019 space (11 cls).
    NIGENS-Focal: N/A.
    """
    exp_config = [
        ('exp_ft_mixed_bce',                 'mixed-BCE',    'STARSS23 (proj)',
         CLASS_NAMES['STARSS23'], 'test', 'tau2019'),
        ('exp_ft_mixed_focal',               'mixed-Focal',  'STARSS23 (proj)',
         CLASS_NAMES['STARSS23'], 'test', 'tau2019'),
        ('exp_ft_tau_nigens_focal',          'NIGENS-Focal', '(N/A)',
         CLASS_NAMES['TAU-NIGENS2021'], 'test', 'tau2019'),   # placeholder
        ('exp_ft_tau2019_focal',             'TAU2019-Focal','TAU2019',
         CLASS_NAMES['TAU2019'],    'test', 'single'),
        ('exp_topt_BiGRU_n2_h512_merge_sum', 'topt-BiGRU',  'STARSS23 (proj)',
         CLASS_NAMES['STARSS23'], 'test', 'tau2019'),
        ('exp_topt_TT_n1_ffn8_pe_4tracks',   'topt-TT4',    'STARSS23 (proj)',
         CLASS_NAMES['STARSS23'], 'test', 'tau2019'),
    ]
    _sed_cm_test_grid(
        sed_cm_df, exp_config,
        suptitle=('Finetune — SED Activity Head: N×N Confusion Matrix (row-normalised)\n'
                  '(spatial branch — test set: TAU2019)\n'
                  'Balanced exps: STARSS23 projected space  |  TAU2019-Focal: native space'),
        stem='finetune_sed_cm_heatmap_test_tau2019',
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Load CSVs ─────────────────────────────────────────────────────────────
    def _load(name: str, required: bool = True) -> Optional[pd.DataFrame]:
        p = RESULTS_DIR / name
        if not p.exists():
            if required:
                raise FileNotFoundError(
                    f'Required CSV not found: {p}\n'
                    f'Run analyze_finetune_nas_foa.py first.'
                )
            return None
        return pd.read_csv(p)

    summary_df   = _load('finetune_nas_foa_summary.csv')
    val_curves   = _load('finetune_nas_foa_val_curves.csv')
    tb_curves    = _load('finetune_nas_foa_tb_curves.csv')
    sem_perclass = _load('finetune_nas_foa_sem_perclass.csv', required=False)
    sem_cm_mat   = _load('finetune_nas_foa_sem_cm_matrix.csv', required=False)
    sed_perclass = _load('finetune_nas_foa_sed_perclass.csv', required=False)
    sed_cm_mat   = _load('finetune_nas_foa_sed_cm_matrix.csv', required=False)

    if sem_perclass is None:
        sem_perclass = pd.DataFrame()
    if sem_cm_mat is None:
        sem_cm_mat = pd.DataFrame()

    print('\n── Group A — finetune-internal figures ──')

    print('[A1] Learning curves…')
    fig_learning_curves(summary_df, val_curves)

    print('[A2] Training loss curves…')
    fig_train_loss_curves(summary_df, tb_curves)

    print('[A3] Semantic N×N confusion matrix heatmap — val (3×2 grid)…')
    fig_semantic_cm_heatmap(sem_cm_mat)

    print('[A3b] Semantic N×N confusion matrix heatmap — test (3×2 grid)…')
    fig_semantic_cm_heatmap_test(sem_cm_mat)

    print('[A4] Architecture ablation (val, log Y)…')
    fig_architecture_ablation(summary_df)

    print('[A5] Semantic per-class F1 heatmap…')
    fig_sem_perclass_heatmap(sem_perclass)

    print('\n── Group C — SED per-class figures ──')

    print('[C1] SED per-class F1 heatmap (val)…')
    fig_sed_perclass_heatmap(sed_perclass)

    print('[C2] SED N×N confusion matrix heatmap — val…')
    fig_sed_cm_heatmap(sed_cm_mat)

    print('[C3] SED N×N confusion matrix heatmap — test STARSS23…')
    fig_sed_cm_heatmap_test_starss(sed_cm_mat)

    print('[C4] SED N×N confusion matrix heatmap — test TAU-NIGENS…')
    fig_sed_cm_heatmap_test_tau(sed_cm_mat)

    print('[C5] SED N×N confusion matrix heatmap — test TAU2019…')
    fig_sed_cm_heatmap_test_tau2019(sed_cm_mat)

    print('\nDone.')


if __name__ == '__main__':
    main()
