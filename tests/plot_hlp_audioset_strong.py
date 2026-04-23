#!/usr/bin/env python3
"""
plot_hlp_audioset_strong.py
===========================
Class-distribution analysis of the AudioSet-Strong pipeline, showing
— per windowing regime and split — the contributions of:

  Blue          : base window counts  (raw strong annotations, no HLP)
  Orange        : HLP extra counts    (classes activated only by propagation)
  Green/hatched : PSLA oversampling boost (WeightedRandomSampler expected extra draws)

Note on split behaviour (mirrors AudioSetStrongDataModule exactly):
  * TRAIN / VAL : HLP applied (from data.hlp_enabled) + PSLA boost on TRAIN
  * TEST        : HLP NOT applied (ground-truth evaluation labels, raw TSV)

Layout: 5 rows (10 s, 5 s, 2.5 s, 1 s, 0.5 s) × 3 cols (train / val / test).
Each subplot: 456 classes sorted by train-base frequency (descending), log Y scale.

Strategy
--------
1. Instantiate AudioSetStrongDataset once per split (window_s=10 s, no augment)
   to collect clip lists and events — no audio is loaded; all clips are exactly 10 s.
2. Extract the HLP table as a scipy sparse matrix for fast batch propagation.
3. Compute window-class counts analytically for every regime.
4. Compute PSLA boost via scipy sparse matrix products (memory-efficient).

Output
------
<project_root>/figures/plot_hlp_audioset_strong.svg  (SVG, 600 DPI metadata)

Usage
-----
    .venv/bin/python tests/plot_hlp_audioset_strong.py
    .venv/bin/python tests/plot_hlp_audioset_strong.py \\
        --config-file configs/experiments_at2sed/at2sed_bce_10s.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from scipy.sparse import csr_matrix as sp_csr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from datasets.audioset_strong import AudioSetStrongDataset

_DEFAULT_DATASET_CFG = _ROOT / 'configs' / 'datasets' / 'AudioSetStrong.yaml'
_DEFAULT_EXP_CFG     = _ROOT / 'configs' / 'experiments_at2sed' / 'at2sed_bce_10s.yaml'

CLIP_DURATION_S = AudioSetStrongDataset.CLIP_DURATION_S  # 10.0

# Non-overlapping (hop = window) — matches AT2SED training configs
REGIMES: List[Tuple[str, float]] = [
    ('10 s',  10.0),
    ('5 s',    5.0),
    ('2.5 s',  2.5),
    ('1 s',    1.0),
    ('0.5 s',  0.5),
]

COLORS = {
    'base': '#4C72B0',
    'hlp':  '#DD8452',
    'psla': '#55A868',
}

_CHUNK = 32768   # rows per batch in the sparse computation


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Plot HLP + PSLA class distributions for the AT2SED pipeline',
    )
    p.add_argument(
        '--config-file', default=str(_DEFAULT_EXP_CFG),
        help=f'AT2SED experiment YAML (default: {_DEFAULT_EXP_CFG.name})',
    )
    return p.parse_args()


# ── Config ────────────────────────────────────────────────────────────────────

def _load_cfg(exp_yaml: str) -> dict:
    """Merge dataset-level defaults with experiment-level overrides."""
    with open(_DEFAULT_DATASET_CFG) as f:
        cfg = yaml.safe_load(f)
    with open(exp_yaml) as f:
        exp = yaml.safe_load(f)
    for section, vals in exp.items():
        if isinstance(vals, dict) and section in cfg:
            cfg[section].update(vals)
        else:
            cfg[section] = vals
    # Resolve relative class_map path
    cfg['data']['class_map'] = str(_ROOT / cfg['data']['class_map'])
    if 'hlp_table_path' in cfg['data']:
        cfg['data']['hlp_table_path'] = str(_ROOT / cfg['data']['hlp_table_path'])
    return cfg


# ── Display names ─────────────────────────────────────────────────────────────

def _load_display_names(class_map_csv: str) -> Dict[int, str]:
    """Return {class_index: display_name} from the label-map CSV."""
    names: Dict[int, str] = {}
    with open(class_map_csv, newline='') as f:
        for row in csv.DictReader(f):
            names[int(row['index'])] = row.get('display_name', row['mid']).strip('"')
    return names


# ── HLP sparse matrix ─────────────────────────────────────────────────────────

def _hlp_sparse(hlp_table_path: str) -> Optional[sp_csr]:
    """Load HLP table → (C, C) float32 scipy csr_matrix, or None."""
    import torch
    p = Path(hlp_table_path)
    if not p.exists():
        warnings.warn(f'HLP table not found: {p}  — HLP disabled', stacklevel=2)
        return None
    t = torch.load(str(p), weights_only=True).float().numpy()
    return sp_csr(t)


# ── Window items ──────────────────────────────────────────────────────────────

def _window_items(clips: List[str], hop_s: float) -> List[Tuple[str, float]]:
    """All (clip_id, t_start) for a clip list and given hop (= window for non-overlap)."""
    items: List[Tuple[str, float]] = []
    for cid in clips:
        t = 0.0
        while t < CLIP_DURATION_S:
            items.append((cid, t))
            t = round(t + hop_s, 10)
    return items


# ── Core computation ──────────────────────────────────────────────────────────

def _compute_counts(
    clips: List[str],
    events: Dict[str, List[Tuple[float, float, int]]],
    n_cls: int,
    window_s: float,
    hlp_sp: Optional[sp_csr],
    overlap_threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-class window item counts for one split and one window regime.

    Returns:
        base_counts      : (C,) int64 — windows where class c appears in raw labels
        hlp_extra_counts : (C,) int64 — windows where c appears ONLY via HLP
    """
    items = _window_items(clips, window_s)   # non-overlapping: hop = window
    N = len(items)
    if N == 0:
        return np.zeros(n_cls, np.int64), np.zeros(n_cls, np.int64)

    base_counts      = np.zeros(n_cls, dtype=np.int64)
    hlp_extra_counts = np.zeros(n_cls, dtype=np.int64)

    for start in range(0, N, _CHUNK):
        end = min(start + _CHUNK, N)
        sz  = end - start
        mat = np.zeros((sz, n_cls), dtype=np.float32)

        for i, (cid, t_s) in enumerate(items[start:end]):
            t_e = t_s + window_s
            for ev_s, ev_e, cls in events.get(cid, []):
                ov = min(ev_e, t_e) - max(ev_s, t_s)
                if ov <= 0.0:
                    continue
                if overlap_threshold == 0.0 or ov / window_s >= overlap_threshold:
                    mat[i, cls] = 1.0

        base_counts += (mat > 0).sum(axis=0).astype(np.int64)

        if hlp_sp is not None:
            mat_hlp = np.clip((sp_csr(mat) @ hlp_sp).toarray(), 0.0, 1.0)
            hlp_extra_counts += ((mat_hlp > 0) & (mat == 0)).sum(axis=0).astype(np.int64)

    return base_counts, hlp_extra_counts


def _compute_psla_boost(
    clips: List[str],
    events: Dict[str, List[Tuple[float, float, int]]],
    n_cls: int,
    window_s: float,
    hlp_sp: Optional[sp_csr],
    overlap_threshold: float = 0.0,
) -> np.ndarray:
    """
    Expected extra draws per class from WeightedRandomSampler (PSLA formula).

    Formula (mirrors AudioSetStrongDataset._item_weights_psla):
        n_c      = number of training items where class c is active (post-HLP)
        weight_i = Σ_c 1000 / (n_c + 0.01)   for c active in item i
        E[draws_i] = N × weight_i / Σweight
        boost_c  = Σ_{i: c active} max(E[draws_i] − 1, 0)

    Returns (C,) float64.
    """
    items = _window_items(clips, window_s)
    N = len(items)
    if N == 0:
        return np.zeros(n_cls, dtype=np.float64)

    # ── Build sparse active matrix (N, C) ────────────────────────────────────
    all_rows, all_cols = [], []

    for start in range(0, N, _CHUNK):
        end = min(start + _CHUNK, N)
        sz  = end - start
        mat = np.zeros((sz, n_cls), dtype=np.float32)

        for i, (cid, t_s) in enumerate(items[start:end]):
            t_e = t_s + window_s
            for ev_s, ev_e, cls in events.get(cid, []):
                ov = min(ev_e, t_e) - max(ev_s, t_s)
                if ov <= 0.0:
                    continue
                if overlap_threshold == 0.0 or ov / window_s >= overlap_threshold:
                    mat[i, cls] = 1.0

        if hlp_sp is not None:
            mat = np.clip((sp_csr(mat) @ hlp_sp).toarray(), 0.0, 1.0)

        rr, cc = np.where(mat > 0)
        all_rows.extend((rr + start).tolist())
        all_cols.extend(cc.tolist())

    if not all_rows:
        return np.zeros(n_cls, dtype=np.float64)

    active_sp = sp_csr(
        (np.ones(len(all_rows), dtype=np.float32), (all_rows, all_cols)),
        shape=(N, n_cls),
    )

    # ── PSLA weights ─────────────────────────────────────────────────────────
    class_counts = np.array(active_sp.sum(axis=0)).ravel()          # (C,)
    psla_unit    = (1000.0 / (class_counts + 0.01)).astype(np.float32)  # (C,)

    # weight_i = Σ_{c active} psla_unit[c]  = active_sp @ psla_unit
    weights = np.array(active_sp @ psla_unit).ravel().astype(np.float64)
    weights[weights == 0.0] = 1.0   # negative items → weight 1

    total_w        = weights.sum()
    expected_draws = N * weights / total_w
    extra          = np.maximum(expected_draws - 1.0, 0.0).astype(np.float32)

    # boost_c = Σ_{i: c active} extra_i  = active_sp.T @ extra
    boost = np.array(active_sp.T @ extra).ravel().astype(np.float64)
    return boost


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    cfg  = _load_cfg(args.config_file)

    d_cfg = cfg['data']
    m_cfg = cfg.get('model', {})
    t_cfg = cfg.get('training', {})

    n_cls             = int(m_cfg.get('n_cls', 527))
    overlap_threshold = float(d_cfg.get('overlap_threshold', 0.0))
    val_fraction      = float(d_cfg.get('val_fraction', 0.2))
    val_seed          = int(d_cfg.get('val_seed', 42))
    hlp_enabled       = bool(d_cfg.get('hlp_enabled', False))
    hlp_table_path    = d_cfg.get('hlp_table_path', None)
    sampler_formula   = str(t_cfg.get('balanced_sampler_formula', 'psla'))

    class_map_csv  = d_cfg['class_map']
    tsv_train      = d_cfg['strong_tsv_train']
    tsv_test       = d_cfg['strong_tsv_test']
    audio_train    = d_cfg['audio_root_train']
    audio_test     = d_cfg['audio_root_test']
    audio_unbal    = d_cfg.get('audio_root_unbalanced', None)

    display_names = _load_display_names(class_map_csv)

    # ── HLP sparse matrix ────────────────────────────────────────────────────
    hlp_sp = _hlp_sparse(hlp_table_path) if (hlp_enabled and hlp_table_path) else None
    if hlp_enabled and hlp_sp is None:
        print('[WARNING] HLP enabled but table not loaded — proceeding without HLP.')

    # ── Instantiate stub datasets (window_s=10s) to get clip lists + events ──
    # No audio is loaded; item index is built analytically from CLIP_DURATION_S.
    print('Building stub datasets (window_s=10 s, no audio loading)...')

    common_kw = dict(
        strong_tsv=tsv_train, class_map_csv=class_map_csv,
        window_s=10.0, hop_s=10.0, n_cls=n_cls,
        overlap_threshold=overlap_threshold,
        val_fraction=val_fraction, val_seed=val_seed,
    )
    extra_dirs = [audio_unbal] if audio_unbal and Path(audio_unbal).exists() else []

    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        train_ds = AudioSetStrongDataset(
            split='train', audio_dir=audio_train,
            extra_audio_dirs=extra_dirs, **common_kw,
        )
        val_ds = AudioSetStrongDataset(
            split='val', audio_dir=audio_train,
            extra_audio_dirs=extra_dirs, **common_kw,
        )
        test_ds = AudioSetStrongDataset(
            split='test', audio_dir=audio_test,
            strong_tsv=tsv_test, class_map_csv=class_map_csv,
            window_s=10.0, hop_s=10.0, n_cls=n_cls,
            overlap_threshold=overlap_threshold,
        )

    train_clips = [cid for cid, _ in train_ds._clips]
    val_clips   = [cid for cid, _ in val_ds._clips]
    test_clips  = [cid for cid, _ in test_ds._clips]
    train_ev    = train_ds._events
    val_ev      = val_ds._events
    test_ev     = test_ds._events

    print(f'  Clips — train: {len(train_clips)} | val: {len(val_clips)} | test: {len(test_clips)}')
    if not (train_clips or val_clips or test_clips):
        print('[ERROR] No clips found — check that audio dirs are accessible.')
        return

    # ── Per-regime computation ────────────────────────────────────────────────
    # Compute base reference order: sort classes by train base count at 10 s
    print('Computing reference sort order (train, 10 s)...')
    ref_base, _ = _compute_counts(
        train_clips, train_ev, n_cls, 10.0, hlp_sp, overlap_threshold)
    sort_idx = np.argsort(ref_base)[::-1]   # descending frequency
    rank     = np.argsort(sort_idx)          # class_idx → rank (x-position)

    print('Computing per-regime distributions...')
    regime_data = []
    for label, window_s in REGIMES:
        print(f'  {label} ...', end='', flush=True)

        tr_base, tr_hlp = _compute_counts(
            train_clips, train_ev, n_cls, window_s, hlp_sp, overlap_threshold)
        vl_base, vl_hlp = _compute_counts(
            val_clips, val_ev, n_cls, window_s, hlp_sp, overlap_threshold)
        # Test: NO HLP (mirrors DataModule behaviour)
        ts_base, _      = _compute_counts(
            test_clips, test_ev, n_cls, window_s, None, overlap_threshold)

        if sampler_formula == 'psla':
            psla = _compute_psla_boost(
                train_clips, train_ev, n_cls, window_s, hlp_sp, overlap_threshold)
        else:
            psla = np.zeros(n_cls, dtype=np.float64)

        n_tr = int(tr_base.sum() + tr_hlp.sum())
        print(f' {n_tr:,} train labels')

        regime_data.append({
            'label':   label,
            'tr_base': tr_base[sort_idx],
            'tr_hlp':  tr_hlp[sort_idx],
            'tr_psla': psla[sort_idx],
            'vl_base': vl_base[sort_idx],
            'vl_hlp':  vl_hlp[sort_idx],
            'ts_base': ts_base[sort_idx],
        })

    # ── Plot — one figure per window regime ──────────────────────────────────
    x     = np.arange(n_cls)
    bar_w = 1.0   # solid fill, no gap between bars

    # x-axis labels: all 456 class names ordered by train frequency
    x_labels = [
        display_names.get(int(sort_idx[i]), str(sort_idx[i]))
        for i in range(n_cls)
    ]

    legend_handles = [
        mpatches.Patch(facecolor=COLORS['base'],
                       label='Base (raw strong labels)'),
        mpatches.Patch(facecolor=COLORS['hlp'],
                       label='HLP extra  (ancestor classes via propagation)'),
        mpatches.Patch(facecolor=COLORS['psla'], hatch='//',
                       edgecolor='white', linewidth=0.3, alpha=0.85,
                       label='PSLA oversampling boost  (expected extra draws, train only)'),
    ]

    # Width: ~0.14 in per class so all 456 labels are legible at rotation=90°
    fig_w = max(70, n_cls * 0.14)
    fig_h = 22   # 3 rows: each ~6 in plot + space for rotated labels

    out_dir = _ROOT / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)

    for rd in regime_data:
        label     = rd['label']
        file_stem = 'hlp_audioset_strong_' + label.replace(' ', '').replace('.', 'p')

        fig, axes = plt.subplots(
            3, 1,
            figsize=(fig_w, fig_h),
            sharex=True,
            constrained_layout=True,
        )
        ax_tr, ax_vl, ax_ts = axes

        # ── Train ─────────────────────────────────────────────────────────
        ax_tr.bar(x, rd['tr_base'], bar_w, color=COLORS['base'], linewidth=0)
        ax_tr.bar(x, rd['tr_hlp'],  bar_w, bottom=rd['tr_base'],
                  color=COLORS['hlp'], linewidth=0)
        bot_psla = (rd['tr_base'] + rd['tr_hlp']).astype(np.float64)
        ax_tr.bar(x, rd['tr_psla'], bar_w, bottom=bot_psla,
                  color=COLORS['psla'], hatch='//', edgecolor='white',
                  linewidth=0.3, alpha=0.85)
        ax_tr.set_title(
            f'Train  ({len(train_clips)} clips | {sampler_formula.upper()} sampler)',
            fontsize=11, fontweight='bold', loc='left', pad=4,
        )

        # ── Val ────────────────────────────────────────────────────────────
        ax_vl.bar(x, rd['vl_base'], bar_w, color=COLORS['base'], linewidth=0)
        ax_vl.bar(x, rd['vl_hlp'],  bar_w, bottom=rd['vl_base'],
                  color=COLORS['hlp'], linewidth=0)
        ax_vl.set_title(
            f'Val    ({len(val_clips)} clips | HLP applied)',
            fontsize=11, fontweight='bold', loc='left', pad=4,
        )

        # ── Test (no HLP) ──────────────────────────────────────────────────
        ax_ts.bar(x, rd['ts_base'], bar_w, color=COLORS['base'], linewidth=0)
        ax_ts.set_title(
            f'Test   ({len(test_clips)} clips | raw labels, no HLP)',
            fontsize=11, fontweight='bold', loc='left', pad=4,
        )

        # ── Shared x-axis labels (shown only on bottom subplot via sharex) ─
        ax_ts.set_xticks(x)
        ax_ts.set_xticklabels(x_labels, rotation=90, fontsize=4.5, ha='right')

        # ── Common formatting ─────────────────────────────────────────────
        for ax in axes:
            ax.set_yscale('log')
            ax.set_xlim(-0.5, n_cls - 0.5)
            ax.tick_params(axis='y', labelsize=8)
            ax.grid(axis='y', linestyle='--', alpha=0.35, linewidth=0.5)
            ax.set_ylabel('Window count (log)', fontsize=9)

        # ── Per-figure title and legend ───────────────────────────────────
        fig.suptitle(
            f'AudioSet-Strong — {label} window  '
            f'(HLP: {"ON" if hlp_sp is not None else "OFF"} | '
            f'{n_cls} classes sorted by train frequency ↓)',
            fontsize=13, fontweight='bold',
        )
        fig.legend(
            handles=legend_handles, loc='lower center', ncol=3, fontsize=9,
            bbox_to_anchor=(0.5, -0.005), framealpha=0.9,
        )

        out = out_dir / f'{file_stem}.svg'
        fig.savefig(str(out), dpi=600, bbox_inches='tight', format='svg')
        plt.close(fig)
        print(f'  Saved → {out.name}')


if __name__ == '__main__':
    main()
