#!/usr/bin/env python3
"""
plot_mixed_seld_dataset.py
==========================
Visualise 13-class distributions across 6 windowing regimes × 3 splits
for the finetune MixedSELD dataset (STARSS23 + TAU-NIGENS).

Layout: 6 rows (full-clip, 10s, 5s, 2.5s, 1s, 0.5s) × 3 cols (train/val/test).

Stacked bars (train column only):
  Blue         — STARSS23 windows containing each class
  Orange       — TAU-NIGENS windows containing each class
  Green/hatched— WeightedRandomSampler oversampling boost (expected extra draws)

Val / test columns show STARSS23-only bars.

No audio is loaded: events come from pre-parsed CSVs; clip durations are
obtained via a single sf.info() call per file.

Dataset roots default to <project_root>/data/STARSS23 and
<project_root>/data/TAU-NIGENS2021 (same as main_finetune.py).

Output
------
<project_root>/figures/plot_mixed_seld_dataset.svg  (SVG, 600 DPI metadata)

Usage
-----
python tests/plot_mixed_seld_dataset.py \\
    --config-file configs/experiments_nas-seld/foa/finetune/exp_ft_mixed_bce.yaml
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

_DEFAULT_STARSS_ROOT = _PROJECT_ROOT / 'data' / 'STARSS23'
_DEFAULT_TAU_ROOT    = _PROJECT_ROOT / 'data' / 'TAU-NIGENS2021'
_DEFAULT_CONFIG      = (_PROJECT_ROOT / 'configs' / 'experiments_nas-seld'
                        / 'foa' / 'finetune' / 'exp_ft_mixed_bce.yaml')

from datasets.starss23           import STARSS23Dataset
from datasets.tau_nigens         import TAUNIGENSDataset
from datasets.mixed_seld_dataset import compute_combined_class_counts

# ── Constants ────────────────────────────────────────────────────────────────

N_CLASSES  = 13
FRAME_DUR  = 0.1          # seconds per CSV annotation frame
ORIG_SR    = 24000        # both datasets are 24 kHz

CLASS_NAMES = [
    'Fem.\nspch', 'Male\nspch', 'Clapping', 'Telephone', 'Laughter',
    'Domestic', 'Footsteps', 'Door', 'Music', 'Instrument',
    'Water\ntap', 'Bell', 'Knock',
]

# (label, window_s, hop_s)  — 50 % overlap for all windowed regimes
REGIMES: List[Tuple[str, Optional[float], Optional[float]]] = [
    ('Full clip', None,  None ),
    ('10 s',      10.0,  5.0  ),
    ('5 s',        5.0,  2.5  ),
    ('2.5 s',      2.5,  1.25 ),
    ('1 s',        1.0,  0.5  ),
    ('0.5 s',      0.5,  0.25 ),
]

COLORS = {
    'starss': '#4C72B0',
    'tau':    '#DD8452',
    'boost':  '#55A868',
}


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Plot class distributions for the finetune MixedSELD dataset',
    )
    p.add_argument('--config-file', default=str(_DEFAULT_CONFIG),
                   help=f'FinetuneDataModule-compatible YAML (default: {_DEFAULT_CONFIG})')
    p.add_argument('--starss-root', default=str(_DEFAULT_STARSS_ROOT),
                   help=f'STARSS23 dataset root (default: {_DEFAULT_STARSS_ROOT})')
    p.add_argument('--tau-root', default=str(_DEFAULT_TAU_ROOT),
                   help=f'TAU-NIGENS2021 dataset root (default: {_DEFAULT_TAU_ROOT})')
    return p.parse_args()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _n_samples(audio_path: Path) -> int:
    """Return frame count of audio file via sf.info; 0 on error."""
    try:
        return sf.info(str(audio_path)).frames
    except Exception:
        return 0


def _build_frame_cls(events: List) -> Dict[int, set]:
    """frame_100ms → set of class indices, for one clip's events."""
    fc: Dict[int, set] = {}
    for frame_100ms, class_idx, *_ in events:
        if class_idx < N_CLASSES:
            fc.setdefault(frame_100ms, set()).add(class_idx)
    return fc


def _clip_active(events: List) -> set:
    """All class indices present anywhere in the clip."""
    return {cls for (_, cls, *_) in events if cls < N_CLASSES}


def _window_classes(fc: Dict[int, set], start_s: float, window_s: float) -> set:
    """Classes active in window [start_s, start_s + window_s)."""
    end_s = start_s + window_s
    f_lo  = int(start_s / FRAME_DUR)
    f_hi  = int(math.ceil(end_s / FRAME_DUR - 1e-9))
    active: set = set()
    for f in range(f_lo, f_hi):
        if f in fc:
            active |= fc[f]
    return active


# ── Per-class window counts ──────────────────────────────────────────────────

def _window_counts(
    events_list: List[List],
    n_samples_list: List[int],
    window_s: Optional[float],
    hop_s: Optional[float],
) -> np.ndarray:
    """
    For each class, count items (windows or clips) containing that class.

    Full-clip mode  : one item per clip; n_samples not used.
    Windowed mode   : one item per (clip, start_sample); clips with
                      n_samples = 0 are skipped.

    Returns ndarray shape (N_CLASSES,) of int64.
    """
    counts = np.zeros(N_CLASSES, dtype=np.int64)

    for clip_idx, events in enumerate(events_list):
        fc = _build_frame_cls(events)

        if window_s is None:
            active = set()
            for cls_set in fc.values():
                active |= cls_set
            for c in active:
                counts[c] += 1
        else:
            n_samp = n_samples_list[clip_idx]
            if n_samp == 0:
                continue
            hop_samp = round(hop_s * ORIG_SR)
            start    = 0
            while start < n_samp:
                start_s = start / ORIG_SR
                for c in _window_classes(fc, start_s, window_s):
                    counts[c] += 1
                start += hop_samp

    return counts


# ── Oversampling boost ────────────────────────────────────────────────────────

def _oversampling_boost(
    starss_ds: STARSS23Dataset,
    tau_ds:    TAUNIGENSDataset,
    starss_ns: List[int],
    tau_ns:    List[int],
    frame_counts: Dict[int, int],
    window_s: Optional[float],
    hop_s:    Optional[float],
) -> np.ndarray:
    """
    Expected extra draws per class due to WeightedRandomSampler.

    Mirrors MixedSELDDataset.clip_weights() exactly:
      weight[item] = max(target / count[c] for c in clip_active_classes)
    where target = median combined frame count.

    The 'boost' for class c = Σ_i max(expected_draws[i] - 1, 0)
    for all items i whose parent clip contains class c.

    Returns ndarray shape (N_CLASSES,) of float64.
    """
    target = max(1, int(np.median(list(frame_counts.values()))))

    def _items_clip_active(ds, ns_list):
        """List of (active_class_set, weight) per item, using clip-level classes."""
        out: List[set] = []
        if window_s is None:
            for events in ds._events:
                out.append(_clip_active(events))
        else:
            hop_samp = round(hop_s * ORIG_SR)
            for clip_idx, events in enumerate(ds._events):
                n_samp = ns_list[clip_idx]
                if n_samp == 0:
                    continue
                active = _clip_active(events)
                start  = 0
                while start < n_samp:
                    out.append(active)
                    start += hop_samp
        return out

    starss_items = _items_clip_active(starss_ds, starss_ns)
    tau_items    = _items_clip_active(tau_ds,    tau_ns)
    all_items    = starss_items + tau_items

    if not all_items:
        return np.zeros(N_CLASSES, dtype=np.float64)

    w = np.array([
        max((target / frame_counts[c] for c in active), default=1.0)
        for active in all_items
    ], dtype=np.float64)

    N      = len(w)
    total  = w.sum()
    if total == 0.0:
        return np.zeros(N_CLASSES, dtype=np.float64)

    exp_draws = N * w / total   # E[draws_i] under sampler

    boost = np.zeros(N_CLASSES, dtype=np.float64)
    for i, active in enumerate(all_items):
        extra = max(exp_draws[i] - 1.0, 0.0)
        if extra > 0.0:
            for c in active:
                boost[c] += extra

    return boost


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # ── Config ───────────────────────────────────────────────────────────────
    with open(args.config_file) as f:
        cfg = yaml.safe_load(f)

    model_sr     = cfg['input']['sr']
    hop_length   = cfg['input'].get('hop_length', 320)
    n_tracks     = cfg.get('heads', {}).get('n_tracks', 2)
    audio_format = cfg.get('data', {}).get('format', 'foa')
    val_fraction = cfg.get('data', {}).get('val_fraction', 0.2)
    val_seed     = cfg.get('data', {}).get('val_seed', 42)

    common_kw = dict(
        model_sr=model_sr, hop_length=hop_length,
        n_tracks=n_tracks, audio_format=audio_format,
    )

    # ── Build full-clip datasets (no audio loading at init) ──────────────────
    print('Building datasets (full-clip mode — no audio loaded)...')
    starss_tr = STARSS23Dataset(
        root=args.starss_root, split='train',
        val_fraction=val_fraction, val_seed=val_seed,
        **common_kw,
    )
    starss_vl = STARSS23Dataset(
        root=args.starss_root, split='val',
        val_fraction=val_fraction, val_seed=val_seed,
        **common_kw,
    )
    starss_ts = STARSS23Dataset(
        root=args.starss_root, split='test',
        val_fraction=val_fraction, val_seed=val_seed,
        **common_kw,
    )
    tau_tr = TAUNIGENSDataset(
        root=args.tau_root, split='train',
        **common_kw,
    )

    print(f'  STARSS23  : {len(starss_tr.clips)} train | '
          f'{len(starss_vl.clips)} val | {len(starss_ts.clips)} test')
    print(f'  TAU-NIGENS: {len(tau_tr.clips)} train')

    # ── Read clip durations (one sf.info call per clip) ─────────────────────
    print('Reading clip durations via sf.info()...')
    sn_tr = [_n_samples(p) for p, _ in starss_tr.clips]
    sn_vl = [_n_samples(p) for p, _ in starss_vl.clips]
    sn_ts = [_n_samples(p) for p, _ in starss_ts.clips]
    tn_tr = [_n_samples(p) for p, _ in tau_tr.clips]

    # ── Combined frame counts (regime-independent, for oversampling weights) ─
    print('Computing combined frame-level class counts...')
    frame_counts = compute_combined_class_counts(starss_tr, tau_tr)

    # ── Per-regime computation ────────────────────────────────────────────────
    print('Computing per-regime window counts...')
    regime_data = []
    for label, window_s, hop_s in REGIMES:
        print(f'  {label} ...', end='', flush=True)

        cts_str = _window_counts(starss_tr._events, sn_tr, window_s, hop_s)
        cts_tau = _window_counts(tau_tr._events,    tn_tr, window_s, hop_s)
        boost   = _oversampling_boost(
            starss_tr, tau_tr, sn_tr, tn_tr,
            frame_counts, window_s, hop_s,
        )
        cts_vl = _window_counts(starss_vl._events, sn_vl, window_s, hop_s)
        cts_ts = _window_counts(starss_ts._events, sn_ts, window_s, hop_s)

        n_tr_items = int(cts_str.sum() + cts_tau.sum())
        print(f' {n_tr_items} train items')

        regime_data.append({
            'label':   label,
            'str_tr':  cts_str,
            'tau_tr':  cts_tau,
            'boost':   boost,
            'str_vl':  cts_vl,
            'str_ts':  cts_ts,
        })

    # ── Plot ─────────────────────────────────────────────────────────────────
    n_rows = len(REGIMES)
    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(22, 4.2 * n_rows),
        constrained_layout=True,
    )

    x     = np.arange(N_CLASSES)
    bar_w = 0.72

    col_titles = [
        'Train  (STARSS23 + TAU-NIGENS)',
        'Val  (STARSS23)',
        'Test  (STARSS23)',
    ]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=11, fontweight='bold', pad=8)

    for row, rd in enumerate(regime_data):
        ax_tr = axes[row, 0]
        ax_vl = axes[row, 1]
        ax_ts = axes[row, 2]

        # ── Train ─────────────────────────────────────────────────────────
        ax_tr.bar(x, rd['str_tr'], bar_w,
                  color=COLORS['starss'], label='STARSS23')
        ax_tr.bar(x, rd['tau_tr'], bar_w,
                  bottom=rd['str_tr'],
                  color=COLORS['tau'], label='TAU-NIGENS')
        boost_bot = (rd['str_tr'] + rd['tau_tr']).astype(np.float64)
        ax_tr.bar(x, rd['boost'], bar_w,
                  bottom=boost_bot,
                  color=COLORS['boost'], hatch='//',
                  edgecolor='white', linewidth=0.5,
                  alpha=0.85, label='Oversampling boost')

        # ── Val / Test ────────────────────────────────────────────────────
        ax_vl.bar(x, rd['str_vl'], bar_w, color=COLORS['starss'])
        ax_ts.bar(x, rd['str_ts'], bar_w, color=COLORS['starss'])

        # ── Per-row formatting ────────────────────────────────────────────
        ax_tr.set_ylabel(rd['label'], fontsize=10, fontweight='bold', labelpad=6)

        for ax in (ax_tr, ax_vl, ax_ts):
            ax.set_xticks(x)
            ax.set_xticklabels(CLASS_NAMES, fontsize=7.5, rotation=45, ha='right')
            ax.tick_params(axis='y', labelsize=8)
            ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.6)
            ax.set_xlim(-0.5, N_CLASSES - 0.5)
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f'{int(v):,}')
            )

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor=COLORS['starss'], label='STARSS23'),
        mpatches.Patch(facecolor=COLORS['tau'],    label='TAU-NIGENS'),
        mpatches.Patch(
            facecolor=COLORS['boost'], hatch='//', edgecolor='white',
            linewidth=0.5, alpha=0.85,
            label='Oversampling boost  (WeightedRandomSampler expected extra draws)',
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc='lower center', ncol=3, fontsize=9,
        bbox_to_anchor=(0.5, -0.012),
        framealpha=0.9,
    )

    fig.suptitle(
        'MixedSELD Dataset — Window-level Class Distribution per Windowing Regime\n'
        f'(format={audio_format}, sr={model_sr}, hop={hop_length})',
        fontsize=13, fontweight='bold',
    )

    out = _PROJECT_ROOT / 'figures' / 'plot_mixed_seld_dataset.svg'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=600, bbox_inches='tight', format='svg')
    print(f'\nFigure saved → {out}')


if __name__ == '__main__':
    main()
