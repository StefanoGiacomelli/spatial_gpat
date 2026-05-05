#!/usr/bin/env python3
"""
auxiliary/datasets/plot_mixed_seld_dataset.py
=============================================
Visualise 15-class unified distributions across 6 windowing regimes × 3 splits
for the 3-dataset MixedSELD (STARSS23 + TAU-NIGENS2021 + TAU2019).

Layout: 6 rows (full-clip, 10s, 5s, 2.5s, 1s, 0.5s) × 3 cols (train/val/test).

Train column: stacked bars per dataset — base (solid) + aug overlay (lighter).
              Aug overlay = expected oversampling additions to balance each class.
Val / test  : base bars only for all three datasets.

Output
------
figures/mixed_seld_dataset.svg  (SVG, 600 DPI)

Run
---
    python auxiliary/datasets/plot_mixed_seld_dataset.py
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

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from datasets.starss23   import STARSS23Dataset
from datasets.tau_nigens import TAUNIGENSDataset
from datasets.tau2019    import TAU2019Dataset

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_SR     = 24_000
HOP_LENGTH   = 240
N_TRACKS     = 3
VAL_FRACTION = 0.2
VAL_SEED     = 42

N_CLASSES = 15
CLASS_NAMES = [
    'speech',    'clapping',   'telephone', 'laughter',     'domestic',
    'footsteps', 'door',       'music',     'water_tap',    'bell',
    'knock',     'alarm',      'cry.baby',  'crash',        'bark.dog',
]

FRAME_DUR = 0.1  # seconds per CSV annotation frame

REGIMES: List[Tuple[str, Optional[float], Optional[float]]] = [
    ('Full clip', None,  None ),
    ('10 s',      10.0,  5.0  ),
    ('5 s',        5.0,  2.5  ),
    ('2.5 s',      2.5,  1.25 ),
    ('1 s',        1.0,  0.5  ),
    ('0.5 s',      0.5,  0.25 ),
]

COLORS = {
    'starss':      '#4C72B0',
    'starss_aug':  '#97B4D6',
    'nigens':      '#DD8452',
    'nigens_aug':  '#EEC4A0',
    'tau2019':     '#55A868',
    'tau2019_aug': '#AED6B4',
}

# Per-dataset class → unified class index
_STARSS_MAP  = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 8,  9, 10], dtype=np.int64)
_NIGENS_MAP  = np.array([11, 12, 13, 14, 0, 0, 5, 10, 0, 0,  2,  7], dtype=np.int64)
_TAU2019_MAP = np.array([0, 0, 6, 4, 4, 4, 10, 3, 4,  2,  0], dtype=np.int64)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _n_samples(audio_path: Path) -> int:
    try:
        return sf.info(str(audio_path)).frames
    except Exception:
        return 0


def _build_frame_cls(events: List, mapping: np.ndarray) -> Dict[int, set]:
    """frame_100ms → set of unified class indices for one clip."""
    fc: Dict[int, set] = {}
    for frame_100ms, class_idx, *_ in events:
        if class_idx < len(mapping):
            uc = int(mapping[class_idx])
            fc.setdefault(frame_100ms, set()).add(uc)
    return fc


def _window_classes(fc: Dict[int, set], start_s: float, window_s: float) -> set:
    f_lo = int(start_s / FRAME_DUR)
    f_hi = int(math.ceil((start_s + window_s) / FRAME_DUR - 1e-9))
    active: set = set()
    for f in range(f_lo, f_hi):
        if f in fc:
            active |= fc[f]
    return active


def _window_counts(
    events_list: List[List],
    n_samples_list: List[int],
    orig_sr: int,
    mapping: np.ndarray,
    window_s: Optional[float],
    hop_s: Optional[float],
) -> np.ndarray:
    """Per unified-class item counts across clips."""
    counts = np.zeros(N_CLASSES, dtype=np.int64)
    for clip_idx, events in enumerate(events_list):
        fc = _build_frame_cls(events, mapping)
        if window_s is None:
            active: set = set()
            for cls_set in fc.values():
                active |= cls_set
            for c in active:
                counts[c] += 1
        else:
            n_samp = n_samples_list[clip_idx]
            if n_samp == 0:
                continue
            hop_samp = round(hop_s * orig_sr)
            start = 0
            while start < n_samp:
                for c in _window_classes(fc, start / orig_sr, window_s):
                    counts[c] += 1
                start += hop_samp
    return counts


def _aug_counts(counts: np.ndarray) -> np.ndarray:
    """Expected items added by per-class oversampling within this dataset."""
    populated = counts[counts > 0]
    if len(populated) == 0:
        return np.zeros(N_CLASSES, dtype=np.float64)
    max_c = populated.max()
    # Only non-zero classes receive aug (matches _build_item_index logic)
    return np.where(counts > 0, np.maximum(max_c - counts, 0), 0).astype(np.float64)


def _stack_train(ax: plt.Axes, groups: List[Tuple[np.ndarray, str, str]], x: np.ndarray, bar_w: float) -> None:
    bottom = np.zeros(N_CLASSES, dtype=np.float64)
    for counts, base_key, aug_key in groups:
        ax.bar(x, counts, bar_w, bottom=bottom, color=COLORS[base_key])
        bottom += counts.astype(np.float64)
        aug = _aug_counts(counts)
        ax.bar(x, aug, bar_w, bottom=bottom, color=COLORS[aug_key])
        bottom += aug


def _stack_bars(ax: plt.Axes, groups: List[Tuple[np.ndarray, str]], x: np.ndarray, bar_w: float) -> None:
    bottom = np.zeros(N_CLASSES, dtype=np.float64)
    for counts, color_key in groups:
        ax.bar(x, counts, bar_w, bottom=bottom, color=COLORS[color_key])
        bottom += counts.astype(np.float64)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    STARSS_ROOT  = _ROOT / 'data' / 'STARSS23'
    NIGENS_ROOT  = _ROOT / 'data' / 'TAU-NIGENS2021'
    TAU2019_ROOT = _ROOT / 'data' / 'TAU2019'

    common_kw = dict(model_sr=MODEL_SR, hop_length=HOP_LENGTH, n_tracks=N_TRACKS)

    # ── Build full-clip datasets (no audio loaded at init) ───────────────────
    print('Building datasets (full-clip mode — no audio loaded)...')
    ds: Dict[str, object] = {}
    for split in ('train', 'val', 'test'):
        ds[f'starss_{split}'] = STARSS23Dataset(
            root=STARSS_ROOT, split=split,
            val_fraction=VAL_FRACTION, val_seed=VAL_SEED, **common_kw,
        )
        ds[f'nigens_{split}'] = TAUNIGENSDataset(
            root=NIGENS_ROOT, split=split, **common_kw,
        )
        ds[f'tau2019_{split}'] = TAU2019Dataset(
            root=TAU2019_ROOT, split=split, mixed_mode=True, **common_kw,
        )

    for key, d in ds.items():
        print(f'  {key}: {len(d.clips)} clips')

    # ── Clip durations (needed for windowed counting) ────────────────────────
    print('Reading clip durations via sf.info()...')
    ns: Dict[str, List[int]] = {
        key: [_n_samples(p) for p, _ in d.clips]
        for key, d in ds.items()
    }

    # ── Per-regime window counts ─────────────────────────────────────────────
    print('Computing per-regime window counts...')
    regime_data = []
    for label, window_s, hop_s in REGIMES:
        print(f'  {label} ...', end='', flush=True)

        def _counts(name: str, split: str, mapping: np.ndarray) -> np.ndarray:
            key = f'{name}_{split}'
            return _window_counts(ds[key]._events, ns[key], ds[key].orig_sr,
                                  mapping, window_s, hop_s)

        s_tr = _counts('starss',  'train', _STARSS_MAP)
        n_tr = _counts('nigens',  'train', _NIGENS_MAP)
        t_tr = _counts('tau2019', 'train', _TAU2019_MAP)
        s_vl = _counts('starss',  'val',   _STARSS_MAP)
        n_vl = _counts('nigens',  'val',   _NIGENS_MAP)
        t_vl = _counts('tau2019', 'val',   _TAU2019_MAP)
        s_ts = _counts('starss',  'test',  _STARSS_MAP)
        n_ts = _counts('nigens',  'test',  _NIGENS_MAP)
        t_ts = _counts('tau2019', 'test',  _TAU2019_MAP)

        n_tr_items = int(s_tr.sum() + n_tr.sum() + t_tr.sum())
        print(f' {n_tr_items} train items')

        regime_data.append(dict(
            label=label,
            s_tr=s_tr, n_tr=n_tr, t_tr=t_tr,
            s_vl=s_vl, n_vl=n_vl, t_vl=t_vl,
            s_ts=s_ts, n_ts=n_ts, t_ts=t_ts,
        ))

    # ── Plot ─────────────────────────────────────────────────────────────────
    n_rows = len(REGIMES)
    fig, axes = plt.subplots(
        n_rows, 3, figsize=(22, 4.2 * n_rows), constrained_layout=True,
    )

    x     = np.arange(N_CLASSES)
    bar_w = 0.72

    col_titles = [
        'Train  (STARSS23 + TAU-NIGENS2021 + TAU2019)',
        'Val  (STARSS23 + TAU-NIGENS2021 + TAU2019)',
        'Test  (STARSS23 + TAU-NIGENS2021 + TAU2019)',
    ]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=10, fontweight='bold', pad=8)

    for row, rd in enumerate(regime_data):
        ax_tr, ax_vl, ax_ts = axes[row, 0], axes[row, 1], axes[row, 2]

        _stack_train(ax_tr, [
            (rd['s_tr'], 'starss',  'starss_aug'),
            (rd['n_tr'], 'nigens',  'nigens_aug'),
            (rd['t_tr'], 'tau2019', 'tau2019_aug'),
        ], x, bar_w)
        _stack_bars(ax_vl, [
            (rd['s_vl'], 'starss'),
            (rd['n_vl'], 'nigens'),
            (rd['t_vl'], 'tau2019'),
        ], x, bar_w)
        _stack_bars(ax_ts, [
            (rd['s_ts'], 'starss'),
            (rd['n_ts'], 'nigens'),
            (rd['t_ts'], 'tau2019'),
        ], x, bar_w)

        ax_tr.set_ylabel(rd['label'], fontsize=10, fontweight='bold', labelpad=6)
        for ax in (ax_tr, ax_vl, ax_ts):
            ax.set_xticks(x)
            ax.set_xticklabels(CLASS_NAMES, fontsize=7, rotation=45, ha='right')
            ax.tick_params(axis='y', labelsize=8)
            ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.6)
            ax.set_xlim(-0.5, N_CLASSES - 0.5)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor=COLORS['starss'],      label='STARSS23'),
        mpatches.Patch(facecolor=COLORS['starss_aug'],  label='STARSS23 aug'),
        mpatches.Patch(facecolor=COLORS['nigens'],      label='TAU-NIGENS2021'),
        mpatches.Patch(facecolor=COLORS['nigens_aug'],  label='TAU-NIGENS2021 aug'),
        mpatches.Patch(facecolor=COLORS['tau2019'],     label='TAU2019'),
        mpatches.Patch(facecolor=COLORS['tau2019_aug'], label='TAU2019 aug'),
    ]
    fig.legend(
        handles=legend_handles, loc='lower center', ncol=6, fontsize=9,
        bbox_to_anchor=(0.5, -0.012), framealpha=0.9,
    )

    fig.suptitle(
        'MixedSELD — Window-level Class Distribution per Windowing Regime\n'
        f'(sr={MODEL_SR}, hop={HOP_LENGTH}, 15 unified classes)',
        fontsize=13, fontweight='bold',
    )

    out = _ROOT / 'figures' / 'mixed_seld_dataset.svg'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=600, bbox_inches='tight', format='svg')
    print(f'\nFigure saved → {out}')


if __name__ == '__main__':
    main()
