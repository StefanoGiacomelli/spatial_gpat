#!/usr/bin/env python3
"""
analysis/experiments/analyze_deep_nas_foa.py — NAS Phase 2 FOA Deep Experiments Analysis
==========================================================================
Reads metrics JSONs, TensorBoard event files, YAML hparams and confusion
matrices for all 7 deep_spec FOA NAS experiments and exports structured
assets for ``analysis/experiments/plot_deep_nas_foa.py``.

No figures are generated here.

Inputs
------
  runs/foa/deep_spec/metrics/exp_e*.json
  runs/foa/deep_spec/metrics/exp_e*_confusion_matrix.json
  runs/foa/deep_spec/checkpoints/exp_e*/version_0/hparams.yaml
  runs/foa/deep_spec/checkpoints/exp_e*/version_0/events.out.tfevents.*

Outputs  (results/)
-------------------
  deep_nas_foa_summary.csv     — 7 rows × test metrics + factors + duration
  deep_nas_foa_tb_curves.csv   — training/val loss per epoch (long format)
  deep_nas_foa_val_curves.csv  — val metrics per epoch, all 8 metrics (long)
  deep_nas_foa_per_class.csv   — per-class P/R/F1 from confusion matrices
  deep_nas_foa_best.json       — winner identification for plot scripts

Experiment naming: exp_e{n_early}_l{n_late}_s{n_smoother}
  e = spatial_early.n_layers  (1–3 ResNet blocks)
  l = spatial_late.n_layers   (1–2 Track Transformer layers)
  s = track_smoothing_spatial.n_layers  (1–2 BiGRU layers)

Usage
-----
  python analysis/experiments/analyze_deep_nas_foa.py

Author
------
Stefano Giacomelli, Ph.D. candidate — DISIM, University of L'Aquila
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
RUNS_DIR    = ROOT / 'runs' / 'foa'
RESULTS_DIR = ROOT / 'results' / 'experiments'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Constants ──────────────────────────────────────────────────────────────────
GROUP = 'deep_spec'

METRICS = [
    'SELD_score', 'LE_CD', 'ER_20', 'LR_CD', 'F_20',
    'sem_accuracy', 'sem_f1', 'sem_auroc',
]

STARSS23_CLASS_NAMES = [
    'Female speech', 'Male speech', 'Clapping', 'Telephone', 'Laughter',
    'Domestic sounds', 'Walk/footsteps', 'Door', 'Music',
    'Musical instr.', 'Water tap', 'Bell', 'Knock',
]

TB_TAGS = [
    'train_loss_epoch', 'train_loss_tpit', 'train_loss_bce',
    'val_loss_tpit', 'val_loss_sed', 'val_loss_doa', 'val_loss_bce',
]


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Parsing
# ══════════════════════════════════════════════════════════════════════════════

def parse_exp_name(exp: str) -> Dict:
    """
    exp_e3_l1_s1  →  {arch:'e3', depth:'l1', smoother:'s1',
                       n_early:3, n_late:1, n_smoother:1}
    """
    m = re.match(r'exp_e(\d+)_l(\d+)_s(\d+)', exp)
    if not m:
        raise ValueError(f'Cannot parse deep_spec experiment name: {exp!r}')
    ne, nl, ns = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return {
        'arch':      f'e{ne}',
        'depth':     f'l{nl}',
        'smoother':  f's{ns}',
        'n_early':   ne,
        'n_late':    nl,
        'n_smoother': ns,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Loaders
# ══════════════════════════════════════════════════════════════════════════════

def load_hparams(group: str, exp: str) -> Dict:
    """Read hparams.yaml and return a flat {section.key: value} dict."""
    path = RUNS_DIR / group / 'checkpoints' / exp / 'version_0' / 'hparams.yaml'
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    cfg = raw.get('cfg', raw)
    flat: Dict = {}
    for section, vals in cfg.items():
        if isinstance(vals, dict):
            for k, v in vals.items():
                flat[f'{section}.{k}'] = v
        else:
            flat[section] = vals
    return flat


def load_metrics(group: str, exp: str) -> Dict:
    """
    Return:
        best_val_SELD, best_epoch,
        val_{metric}_curve  (list, one entry per epoch) for all 8 metrics,
        test/{metric}       (flat scalars from the test entry).
    """
    path = RUNS_DIR / group / 'metrics' / f'{exp}.json'
    data = json.loads(path.read_text())

    val_epochs = [d for d in data if isinstance(d.get('epoch'), int)]
    test_entry = next((d for d in data if d.get('epoch') == 'test'), {})

    best = min(val_epochs, key=lambda x: x['val/SELD_score'])

    result: Dict = {
        'best_val_SELD': best['val/SELD_score'],
        'best_epoch':    int(best['epoch']),
    }
    for m in METRICS:
        tag = f'val/{m}'
        result[f'val_{m}_curve'] = [d.get(tag, float('nan')) for d in val_epochs]
    for k, v in test_entry.items():
        if k != 'epoch':
            result[k] = v
    return result


def load_tb(group: str, exp: str) -> Dict:
    """
    Return duration_h + 7 TB loss-curve lists (same tags as shallow script).
    """
    tb_dir = RUNS_DIR / group / 'checkpoints' / exp / 'version_0'
    ea = EventAccumulator(str(tb_dir))
    ea.Reload()

    def _vals(tag: str) -> List[float]:
        try:
            return [e.value for e in ea.Scalars(tag)]
        except KeyError:
            return []

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


def load_confusion_matrix(group: str, exp: str) -> Optional[List]:
    """Return 13×2×2 list or None if file absent."""
    path = RUNS_DIR / group / 'metrics' / f'{exp}_confusion_matrix.json'
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data.get('sem_confusion_matrix')


# ══════════════════════════════════════════════════════════════════════════════
# 3.  DataFrame builders
# ══════════════════════════════════════════════════════════════════════════════

def _iter_exps(group: str) -> List[str]:
    pattern = str(RUNS_DIR / group / 'metrics' / 'exp_e*.json')
    return sorted(
        Path(p).stem for p in glob.glob(pattern)
        if 'confusion' not in p
    )


def build_summary_df() -> pd.DataFrame:
    """7-row DataFrame: factors + test metrics + duration + best_epoch."""
    rows = []
    for exp in _iter_exps(GROUP):
        axes = parse_exp_name(exp)
        met  = load_metrics(GROUP, exp)
        tb   = load_tb(GROUP, exp)
        row: Dict = {'exp': exp, **axes,
                     'duration_h': tb['duration_h'],
                     'best_epoch': met['best_epoch'],
                     'best_val_SELD': met['best_val_SELD']}
        for m in METRICS:
            row[f'test/{m}'] = met.get(f'test/{m}', float('nan'))
        rows.append(row)

    df = pd.DataFrame(rows)
    df.sort_values('test/SELD_score', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def build_tb_curves_df() -> pd.DataFrame:
    """Long-format training/val loss curves from TensorBoard."""
    rows = []
    for exp in _iter_exps(GROUP):
        tb = load_tb(GROUP, exp)
        for tag in TB_TAGS:
            for epoch, val in enumerate(tb.get(tag, [])):
                rows.append({'exp': exp, 'metric': tag,
                             'epoch': epoch, 'value': val})
    return pd.DataFrame(rows)


def build_val_curves_df() -> pd.DataFrame:
    """Long-format validation metric curves for all 8 metrics."""
    rows = []
    for exp in _iter_exps(GROUP):
        met = load_metrics(GROUP, exp)
        for m in METRICS:
            for epoch, val in enumerate(met.get(f'val_{m}_curve', [])):
                rows.append({'exp': exp, 'metric': m,
                             'epoch': epoch, 'value': val})
    return pd.DataFrame(rows)


def build_per_class_df() -> pd.DataFrame:
    """Per-class precision / recall / F1 derived from confusion matrices."""
    rows = []
    for exp in _iter_exps(GROUP):
        axes = parse_exp_name(exp)
        cm   = load_confusion_matrix(GROUP, exp)
        if cm is None:
            continue
        for class_idx, mat in enumerate(cm):
            # mat = [[TN, FP], [FN, TP]]
            tn, fp = int(mat[0][0]), int(mat[0][1])
            fn, tp = int(mat[1][0]), int(mat[1][1])
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            name = (STARSS23_CLASS_NAMES[class_idx]
                    if class_idx < len(STARSS23_CLASS_NAMES) else f'class_{class_idx}')
            rows.append({
                'exp':        exp,
                **axes,
                'class_idx':  class_idx,
                'class_name': name,
                'precision':  prec,
                'recall':     rec,
                'f1':         f1,
                'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run the command-line workflow."""

    print('\n[1/5] Building summary DataFrame…')
    df = build_summary_df()

    summary_cols = [
        'exp', 'arch', 'depth', 'smoother', 'n_early', 'n_late', 'n_smoother',
        'duration_h', 'best_epoch', 'best_val_SELD',
        'test/SELD_score', 'test/LE_CD', 'test/ER_20', 'test/LR_CD', 'test/F_20',
        'test/sem_accuracy', 'test/sem_f1', 'test/sem_auroc',
    ]
    csv_out = RESULTS_DIR / 'deep_nas_foa_summary.csv'
    df[summary_cols].to_csv(csv_out, index=False, float_format='%.6f')
    print(f'  Saved {csv_out.name}  ({len(df)} rows)')

    print('[2/5] Building TB curves DataFrame…')
    tb_df = build_tb_curves_df()
    tb_out = RESULTS_DIR / 'deep_nas_foa_tb_curves.csv'
    tb_df.to_csv(tb_out, index=False, float_format='%.6f')
    print(f'  Saved {tb_out.name}  ({len(tb_df):,} rows)')

    print('[3/5] Building validation curves DataFrame…')
    val_df = build_val_curves_df()
    val_out = RESULTS_DIR / 'deep_nas_foa_val_curves.csv'
    val_df.to_csv(val_out, index=False, float_format='%.6f')
    print(f'  Saved {val_out.name}  ({len(val_df):,} rows)')

    print('[4/5] Building per-class metrics DataFrame…')
    cls_df = build_per_class_df()
    cls_out = RESULTS_DIR / 'deep_nas_foa_per_class.csv'
    cls_df.to_csv(cls_out, index=False, float_format='%.6f')
    print(f'  Saved {cls_out.name}  ({len(cls_df)} rows)')

    print('[5/5] Saving best-run JSON…')
    winner = df.iloc[0]
    best: Dict = {'winner': winner['exp'], 'group': GROUP}
    for m in METRICS:
        best[f'test_{m}'] = float(winner[f'test/{m}'])
    best['best_epoch'] = int(winner['best_epoch'])
    best['duration_h'] = float(winner['duration_h'])

    best_out = RESULTS_DIR / 'deep_nas_foa_best.json'
    best_out.write_text(json.dumps(best, indent=2))
    print(f'  Saved {best_out.name}  (winner: {winner["exp"]})')

    print('\n── Summary (sorted by test SELD ↑ better) ──')
    disp = df[['exp', 'n_early', 'n_late', 'n_smoother',
               'test/SELD_score', 'test/LE_CD', 'test/F_20',
               'best_epoch', 'duration_h']].copy()
    disp.columns = ['exp', 'e', 'l', 's', 'SELD', 'LE_CD', 'F_20', 'best_ep', 'h']
    print(disp.to_string(index=False, float_format='%.4f'))
    print('\nDone.')


if __name__ == '__main__':
    main()
