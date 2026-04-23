"""
tests/analyze_shallow_nas_foa.py — NAS Phase 1 FOA Shallow Experiments Analysis
================================================================================
Reads metrics JSONs and TensorBoard event files for all 16 shallow-input
FOA NAS experiments (8 shallow_spec + 8 shallow_gab) and produces:

  results/
    shallow_nas_foa_summary.csv       — 16 rows × test metrics + val + duration
    shallow_nas_foa_tb_curves.csv     — training/val loss per epoch (long format)

  figures/
    shallow_learning_curves_spe.svg   — val/SELD_score per epoch, shallow_spec
    shallow_learning_curves_gab.svg   — val/SELD_score per epoch, shallow_gab
    shallow_train_loss_curves.svg     — train loss components (3×2 panel grid)
    shallow_val_loss_components.svg   — val loss components  (4×2 panel grid)
    shallow_training_time.svg         — training duration bar chart
    shallow_bar_test_seld.svg         — test SELD_score ranking bar chart
    shallow_ablation_heatmaps.svg     — 6× 2×2 marginal-effect heatmaps
    shallow_multi_metric_heatmap.svg  — 16 exp × 8 metrics normalised heatmap

Usage
-----
    python tests/analyze_shallow_nas_foa.py

Author
------
Stefano Giacomelli, Ph.D. candidate — DISIM, University of L'Aquila
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
RUNS_DIR    = ROOT / 'runs' / 'foa'
RESULTS_DIR = ROOT / 'results'
FIGURES_DIR = ROOT / 'figures'
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

GROUPS = {
    'shallow_spec': 'spe',
    'shallow_gab':  'gab',
}

# ── Visual style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'figure.dpi': 150,
})

# Colour/style encoding: late × early × smoother
_LATE_COLOR  = {'fe': '#4878CF', 'tt': '#D65F5F'}   # blue / red
_EARLY_STYLE = {'cb': '-',       'rn': '--'}
_SMOOTHER_MARKER = {'bg': 'o', 'mg': 's'}


def _label_for(early: str, late: str, smoother: str) -> str:
    return f'{early}_{late}_{smoother}'


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Parse experiment name → axes
# ══════════════════════════════════════════════════════════════════════════════

def parse_exp_name(exp: str) -> Dict[str, str]:
    """exp_spe_rn_tt_bg → {input:'spe', early:'rn', late:'tt', smoother:'bg'}"""
    parts = exp.split('_')          # ['exp', 'spe', 'rn', 'tt', 'bg']
    return dict(input=parts[1], early=parts[2], late=parts[3], smoother=parts[4])


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Load TensorBoard event data
# ══════════════════════════════════════════════════════════════════════════════

def load_tb(group: str, exp: str) -> Dict:
    """
    Returns a dict with keys:
        duration_h        : float
        train_loss_epoch  : list[float]  (len=20)
        train_loss_tpit   : list[float]
        train_loss_bce    : list[float]
        val_loss_tpit     : list[float]
        val_loss_sed      : list[float]
        val_loss_doa      : list[float]
        val_loss_bce      : list[float]
    """
    tb_dir = RUNS_DIR / group / 'checkpoints' / exp / 'version_0'
    ea = EventAccumulator(str(tb_dir))
    ea.Reload()

    def _vals(tag: str) -> List[float]:
        try:
            return [e.value for e in ea.Scalars(tag)]
        except KeyError:
            return []

    # Training duration from wall_time span of train/loss_step
    try:
        steps = ea.Scalars('train/loss_step')
        duration_h = (steps[-1].wall_time - steps[0].wall_time) / 3600.0
    except (KeyError, IndexError):
        duration_h = float('nan')

    return {
        'duration_h':       duration_h,
        'train_loss_epoch': _vals('train/loss_epoch'),
        'train_loss_tpit':  _vals('train/loss_tpit'),
        'train_loss_bce':   _vals('train/loss_bce'),
        'val_loss_tpit':    _vals('val/loss_tpit'),
        'val_loss_sed':     _vals('val/loss_sed'),
        'val_loss_doa':     _vals('val/loss_doa'),
        'val_loss_bce':     _vals('val/loss_bce'),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Load metrics JSON
# ══════════════════════════════════════════════════════════════════════════════

def load_metrics(group: str, exp: str) -> Dict:
    """Returns dict with test metrics, best_val_SELD, best_epoch, val curves."""
    path = RUNS_DIR / group / 'metrics' / f'{exp}.json'
    data = json.loads(path.read_text())

    val_epochs = [d for d in data if isinstance(d.get('epoch'), int)]
    test_entry = next((d for d in data if d.get('epoch') == 'test'), {})

    best = min(val_epochs, key=lambda x: x['val/SELD_score'])

    result = {
        'best_val_SELD': best['val/SELD_score'],
        'best_epoch':    best['epoch'],
        'val_seld_curve': [d['val/SELD_score'] for d in val_epochs],
    }
    for k, v in test_entry.items():
        if k != 'epoch':
            result[k] = v
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Build master DataFrame
# ══════════════════════════════════════════════════════════════════════════════

def build_dataframe() -> pd.DataFrame:
    rows = []
    for group, inp in GROUPS.items():
        pattern = str(RUNS_DIR / group / 'metrics' / f'exp_{inp}_*.json')
        for fpath in sorted(glob.glob(pattern)):
            if 'confusion' in fpath:
                continue
            exp = Path(fpath).stem
            axes  = parse_exp_name(exp)
            met   = load_metrics(group, exp)
            tb    = load_tb(group, exp)
            row = {'exp': exp, 'group': group, **axes,
                   'duration_h': tb['duration_h'],
                   **{k: v for k, v in met.items()
                      if k not in ('val_seld_curve',)}}
            rows.append(row)

    df = pd.DataFrame(rows)
    df.sort_values('test/SELD_score', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Build TensorBoard curves DataFrame (long format)
# ══════════════════════════════════════════════════════════════════════════════

def build_tb_curves_df() -> pd.DataFrame:
    TB_TAGS = [
        'train_loss_epoch', 'train_loss_tpit', 'train_loss_bce',
        'val_loss_tpit', 'val_loss_sed', 'val_loss_doa', 'val_loss_bce',
    ]
    rows = []
    for group, inp in GROUPS.items():
        pattern = str(RUNS_DIR / group / 'metrics' / f'exp_{inp}_*.json')
        for fpath in sorted(glob.glob(pattern)):
            if 'confusion' in fpath:
                continue
            exp = Path(fpath).stem
            tb  = load_tb(group, exp)
            for tag in TB_TAGS:
                vals = tb.get(tag, [])
                for epoch, val in enumerate(vals):
                    rows.append({'exp': exp, 'group': group,
                                 'metric': tag, 'epoch': epoch, 'value': val})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Figures helpers
# ══════════════════════════════════════════════════════════════════════════════

def _exp_line_kwargs(exp: str) -> Dict:
    ax = parse_exp_name(exp)
    return dict(
        color=_LATE_COLOR[ax['late']],
        linestyle=_EARLY_STYLE[ax['early']],
        marker=_SMOOTHER_MARKER[ax['smoother']],
        markersize=3,
        linewidth=1.2,
        alpha=0.85,
    )


def _group_legend_handles(include_star: bool = False) -> List:
    from matplotlib.lines import Line2D
    handles = []
    for late, col in _LATE_COLOR.items():
        handles.append(mpatches.Patch(color=col, label=f'late={late}'))
    for early, ls in _EARLY_STYLE.items():
        handles.append(Line2D([0], [0], color='k', linestyle=ls,
                               linewidth=1.2, label=f'early={early}'))
    for sm, mk in _SMOOTHER_MARKER.items():
        handles.append(Line2D([0], [0], color='k', marker=mk, linestyle='none',
                               markersize=4, label=f'smooth={sm}'))
    if include_star:
        handles.append(Line2D([0], [0], color='k', marker='*', linestyle='none',
                               markersize=7, label='best val epoch'))
    return handles


_STAGE_LABEL = {
    'input':    'input stage',
    'early':    'early stage',
    'late':     'late stage',
    'smoother': 'smoothing stage',
}

_GROUP_TITLE = {
    'shallow_spec': 'Spectrogram Features Extraction',
    'shallow_gab':  'Gabor Analysis Features',
}


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Figure: val SELD learning curves
# ══════════════════════════════════════════════════════════════════════════════

_LC_TITLES = {
    'shallow_spec': 'Validation SELD Score (Spectral Features extractor)',
    'shallow_gab':  'Validation SELD Score (Gabor Features extractor)',
}

def fig_learning_curves(df: pd.DataFrame) -> None:
    for group in GROUPS:
        sub = df[df['group'] == group]
        fig, ax = plt.subplots(figsize=(7, 4))
        n_epochs = None
        for _, row in sub.iterrows():
            met = load_metrics(group, row['exp'])
            curve = met['val_seld_curve']
            n_epochs = len(curve)
            epochs = list(range(n_epochs))
            kw = _exp_line_kwargs(row['exp'])
            ax.plot(epochs, curve, **kw, label=_label_for(
                row['early'], row['late'], row['smoother']))
            be = int(row['best_epoch'])
            ax.plot(be, curve[be], '*', color=kw['color'], markersize=8, zorder=5)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('val/SELD_score  (↓ better)')
        ax.set_title(_LC_TITLES[group])
        if n_epochs is not None:
            ax.set_xticks(range(n_epochs))
        ax.grid(True, alpha=0.5, linestyle='--', linewidth=0.5, zorder=0)
        ax.legend(handles=_group_legend_handles(include_star=True),
                  loc='upper right', ncol=2, framealpha=0.8)
        fig.tight_layout()
        inp = GROUPS[group]
        out = FIGURES_DIR / f'shallow_learning_curves_{inp}.svg'
        fig.savefig(out, bbox_inches='tight', dpi=600)
        plt.close(fig)
        print(f'  Saved {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Figure: train loss curves (3 components × 2 groups)
# ══════════════════════════════════════════════════════════════════════════════

def fig_train_loss_curves(df: pd.DataFrame) -> None:
    components = [
        ('train_loss_epoch', 'train/loss  (total)'),
        ('train_loss_tpit',  'train/loss_tpit'),
        ('train_loss_bce',   'train/loss_bce'),
    ]
    n_rows, n_cols = len(components), len(GROUPS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 9), sharey='row')

    for col_idx, (group, inp) in enumerate(GROUPS.items()):
        sub = df[df['group'] == group]
        for row_idx, (tag, ylabel) in enumerate(components):
            ax = axes[row_idx][col_idx]
            for _, row in sub.iterrows():
                tb = load_tb(group, row['exp'])
                vals = tb.get(tag, [])
                if not vals:
                    continue
                kw = _exp_line_kwargs(row['exp'])
                ax.plot(range(len(vals)), vals, **kw)
            ax.set_ylabel(ylabel if col_idx == 0 else '')
            ax.set_xlabel('Epoch' if row_idx == n_rows - 1 else '')
            ax.grid(True, alpha=0.5, linestyle='--', linewidth=0.5, zorder=0)
            if row_idx == 0:
                ax.set_title(_GROUP_TITLE[group])

    fig.legend(handles=_group_legend_handles(),
               loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02),
               framealpha=0.9)
    fig.suptitle('Training Loss Components', y=1.01)
    fig.tight_layout()
    out = FIGURES_DIR / 'shallow_train_loss_curves.svg'
    fig.savefig(out, bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f'  Saved {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# 9.  Figure: val loss components (4 components × 2 groups)
# ══════════════════════════════════════════════════════════════════════════════

def fig_val_loss_components(df: pd.DataFrame) -> None:
    components = [
        ('val_loss_tpit', 'val/loss_tpit'),
        ('val_loss_sed',  'val/loss_sed'),
        ('val_loss_doa',  'val/loss_doa'),
        ('val_loss_bce',  'val/loss_bce'),
    ]
    n_rows, n_cols = len(components), len(GROUPS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12), sharey='row')

    for col_idx, (group, inp) in enumerate(GROUPS.items()):
        sub = df[df['group'] == group]
        for row_idx, (tag, ylabel) in enumerate(components):
            ax = axes[row_idx][col_idx]
            for _, row in sub.iterrows():
                tb = load_tb(group, row['exp'])
                vals = tb.get(tag, [])
                if not vals:
                    continue
                kw = _exp_line_kwargs(row['exp'])
                ax.plot(range(len(vals)), vals, **kw)
            ax.set_ylabel(ylabel if col_idx == 0 else '')
            ax.set_xlabel('Epoch' if row_idx == n_rows - 1 else '')
            ax.grid(True, alpha=0.5, linestyle='--', linewidth=0.5, zorder=0)
            if row_idx == 0:
                ax.set_title(_GROUP_TITLE[group])

    fig.legend(handles=_group_legend_handles(),
               loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.01),
               framealpha=0.9)
    fig.suptitle('Validation Loss Components', y=1.005)
    fig.tight_layout()
    out = FIGURES_DIR / 'shallow_val_loss_components.svg'
    fig.savefig(out, bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f'  Saved {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# 10. Figure: training time bar chart
# ══════════════════════════════════════════════════════════════════════════════

def fig_training_time(df: pd.DataFrame) -> None:
    sub = df.sort_values('duration_h').copy()
    colors = sub['smoother'].map({'bg': '#4C72B0', 'mg': '#DD8452'})

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(range(len(sub)), sub['duration_h'].values,
                   color=colors.values, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub['exp'].values, fontsize=7)
    ax.set_xlabel('Training duration  (hours)')
    ax.set_title('Training Times (per Experiment)')

    for i, (bar, val) in enumerate(zip(bars, sub['duration_h'].values)):
        ax.text(val + 0.05, i, f'{val:.2f}h', va='center', fontsize=7)

    legend_handles = [
        mpatches.Patch(color='#4C72B0', label='smoothing=bg  (Bi-Directional GRU)'),
        mpatches.Patch(color='#DD8452', label='smoothing=mg  (MHGU)'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', framealpha=0.9)
    ax.set_xlim(0, sub['duration_h'].max() * 1.15)
    fig.tight_layout()
    out = FIGURES_DIR / 'shallow_training_time.svg'
    fig.savefig(out, bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f'  Saved {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# 11. Figure: test SELD_score bar chart (ranking)
# ══════════════════════════════════════════════════════════════════════════════

def fig_bar_test_seld(df: pd.DataFrame) -> None:
    sub = df.sort_values('test/SELD_score').copy()
    input_colors = {'spe': '#4878CF', 'gab': '#D65F5F'}
    late_hatches = {'fe': '', 'tt': '//'}

    colors  = sub['input'].map(input_colors).values
    hatches = sub['late'].map(late_hatches).values

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.55, linestyle='--', linewidth=0.5)
    x = np.arange(len(sub))
    seld_vals = sub['test/SELD_score'].values
    for i, (xi, val) in enumerate(zip(x, seld_vals)):
        ax.bar(xi, val, color=colors[i], hatch=hatches[i],
               edgecolor='white', linewidth=0.6)
        ax.text(xi, val + 0.005, f'{val:.4f}',
                ha='center', va='bottom', fontsize=6, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(sub['exp'].str.replace('exp_', '').values,
                       rotation=40, ha='right', fontsize=7)
    ax.set_ylabel('test/SELD_score  (↓ better)')
    ax.set_title('Test SELD Score Ranking')

    legend_handles = [
        mpatches.Patch(color='#4878CF', label='input=spe'),
        mpatches.Patch(color='#D65F5F', label='input=gab'),
        mpatches.Patch(facecolor='white', edgecolor='k', hatch='', label='late=fe'),
        mpatches.Patch(facecolor='white', edgecolor='k', hatch='//', label='late=tt'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', framealpha=0.9, fontsize=7)

    ax.set_ylim(0, sub['test/SELD_score'].max() * 1.12)
    fig.tight_layout()
    out = FIGURES_DIR / 'shallow_bar_test_seld.svg'
    fig.savefig(out, bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f'  Saved {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# 12. Figure: ablation heatmaps (6 axis pairs)
# ══════════════════════════════════════════════════════════════════════════════

def fig_ablation_heatmaps(df: pd.DataFrame) -> None:
    AXES = ['input', 'early', 'late', 'smoother']
    PAIRS = [(AXES[i], AXES[j]) for i in range(len(AXES)) for j in range(i+1, len(AXES))]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes_flat = axes.flatten()

    for ax, (a, b) in zip(axes_flat, PAIRS):
        pivot = df.groupby([a, b])['test/SELD_score'].mean().unstack(b)
        # Rename index/columns to stage labels
        pivot.index   = [_STAGE_LABEL.get(v, v) for v in pivot.index]
        pivot.columns = [_STAGE_LABEL.get(v, v) for v in pivot.columns]
        sns.heatmap(
            pivot, ax=ax,
            cmap='RdYlGn_r',
            annot=True, fmt='.4f', annot_kws={'size': 8},
            vmin=df['test/SELD_score'].min() - 0.02,
            vmax=df['test/SELD_score'].max() + 0.02,
            cbar_kws={'shrink': 0.8, 'label': 'SELD_score ↓'},
            linewidths=0.5,
        )
        ax.set_title('')
        ax.set_xlabel(_STAGE_LABEL.get(b, b))
        ax.set_ylabel(_STAGE_LABEL.get(a, a))

    fig.suptitle('Ablation Analysis (Test SELD Score) (lower = better)',
                 y=1.01)
    fig.tight_layout()
    out = FIGURES_DIR / 'shallow_ablation_heatmaps.svg'
    fig.savefig(out, bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f'  Saved {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# 13. Figure: multi-metric normalised heatmap
# ══════════════════════════════════════════════════════════════════════════════

def fig_multi_metric_heatmap(df: pd.DataFrame) -> None:
    # Metrics and their direction (True = lower is better)
    METRICS = [
        ('test/SELD_score', True),
        ('test/LE_CD',      True),
        ('test/ER_20',      True),
        ('test/LR_CD',      False),   # higher is better
        ('test/F_20',       False),
        ('test/sem_accuracy', False),
        ('test/sem_f1',     False),
        ('test/sem_auroc',  False),
    ]

    sub = df.sort_values('test/SELD_score').copy()
    row_labels = sub['exp'].str.replace('exp_', '').values

    # Build normalised matrix — 0=worst, 1=best (green) after inversion for lower-better
    matrix = np.zeros((len(sub), len(METRICS)))
    annot  = np.empty_like(matrix, dtype=object)
    for j, (col, lower_better) in enumerate(METRICS):
        vals = sub[col].values.astype(float)
        lo, hi = vals.min(), vals.max()
        norm = (vals - lo) / (hi - lo + 1e-9)
        if lower_better:
            norm = 1.0 - norm      # invert so green = best
        matrix[:, j] = norm
        for i, v in enumerate(vals):
            annot[i, j] = f'{v:.4f}'

    col_labels = [m[0].replace('test/', '') for m in METRICS]

    fig, ax = plt.subplots(figsize=(11, 6))
    hm = sns.heatmap(
        matrix, ax=ax,
        cmap='RdYlGn',
        vmin=0, vmax=1,
        annot=annot, fmt='',
        annot_kws={'size': 7},
        xticklabels=col_labels,
        yticklabels=row_labels,
        cbar_kws={'label': 'normalised score  (1=best)', 'shrink': 0.8},
        linewidths=0.3,
    )
    hm.collections[0].colorbar.set_ticks([])
    ax.set_title('Shallow NAS experiments - Metrics Heatmap')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    fig.tight_layout()
    out = FIGURES_DIR / 'shallow_multi_metric_heatmap.svg'
    fig.savefig(out, bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f'  Saved {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print('\n[1/3] Loading experiment data…')
    df = build_dataframe()

    # ── Save summary CSV ──────────────────────────────────────────────────────
    csv_out = RESULTS_DIR / 'shallow_nas_foa_summary.csv'
    summary_cols = [
        'exp', 'group', 'input', 'early', 'late', 'smoother',
        'duration_h',
        'test/SELD_score', 'test/LE_CD', 'test/LR_CD',
        'test/ER_20', 'test/F_20',
        'test/sem_accuracy', 'test/sem_f1', 'test/sem_auroc',
        'best_val_SELD', 'best_epoch',
    ]
    df[summary_cols].to_csv(csv_out, index=False, float_format='%.6f')
    print(f'  Saved {csv_out.name}')

    # ── Save TB curves CSV ────────────────────────────────────────────────────
    print('[2/3] Extracting TensorBoard curves…')
    tb_df = build_tb_curves_df()
    tb_out = RESULTS_DIR / 'shallow_nas_foa_tb_curves.csv'
    tb_df.to_csv(tb_out, index=False, float_format='%.6f')
    print(f'  Saved {tb_out.name}  ({len(tb_df):,} rows)')

    # ── Generate figures ──────────────────────────────────────────────────────
    print('[3/3] Generating figures…')
    fig_learning_curves(df)
    fig_train_loss_curves(df)
    fig_val_loss_components(df)
    fig_training_time(df)
    fig_bar_test_seld(df)
    fig_ablation_heatmaps(df)
    fig_multi_metric_heatmap(df)

    print('\nDone.')
    print(f'  results/ → {csv_out.name}, {tb_out.name}')
    print(f'  figures/ → 8 × shallow_*.svg')


if __name__ == '__main__':
    main()
