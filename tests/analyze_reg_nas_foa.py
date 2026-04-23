"""
tests/analyze_reg_nas_foa.py — NAS Phase 3 FOA Regularised Experiments Analysis
=================================================================================
Reads metrics JSONs, TensorBoard event files, YAML hparams and confusion
matrices for the 4 deep_reg FOA NAS experiments (cross-stitch ablation on
the phase-2 winner) and exports structured assets for
``tests/plot_reg_nas_foa.py``.

No figures are generated here.

Inputs
------
  runs/foa/deep_reg/metrics/exp_r*.json
  runs/foa/deep_reg/metrics/exp_r*_confusion_matrix.json
  runs/foa/deep_reg/checkpoints/exp_r*/version_0/hparams.yaml
  runs/foa/deep_reg/checkpoints/exp_r*/version_0/events.out.tfevents.*
  results/deep_nas_foa_best.json  (optional — phase-2 winner for comparison)

Outputs  (results/)
-------------------
  reg_nas_foa_summary.csv     — 4 rows × test metrics + cs factors + duration
  reg_nas_foa_tb_curves.csv   — training/val loss per epoch (long format)
  reg_nas_foa_val_curves.csv  — val metrics per epoch, all 8 metrics (long)
  reg_nas_foa_deltas.csv      — Δ metrics vs cs00 baseline (cs01/cs10/cs11)
  reg_nas_foa_per_class.csv   — per-class P/R/F1 from confusion matrices
  reg_nas_foa_best.json       — winner + comparison to phase-2 (dropout effect)

Experiment naming: exp_r3_t1_cs{XY}
  cs_early_enabled = (X == '1')
  cs_late_enabled  = (Y == '1')
  cs00 = no cross-stitch (regularized baseline = phase-2 winner + dropout)
  cs01 = late cross-stitch only
  cs10 = early cross-stitch only
  cs11 = both early and late cross-stitch

Usage
-----
  python tests/analyze_reg_nas_foa.py

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
ROOT        = Path(__file__).resolve().parent.parent
RUNS_DIR    = ROOT / 'runs' / 'foa'
RESULTS_DIR = ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Constants ──────────────────────────────────────────────────────────────────
GROUP = 'deep_reg'

METRICS = [
    'SELD_score', 'LE_CD', 'ER_20', 'LR_CD', 'F_20',
    'sem_accuracy', 'sem_f1', 'sem_auroc',
]
LOWER_BETTER = {'SELD_score', 'LE_CD', 'ER_20'}

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
    exp_r3_t1_cs01  →  {cs_code:'cs01', cs_early:False, cs_late:True,
                         repeat:'r3', arch_type:'t1'}
    """
    m = re.match(r'exp_(r\d+)_(t\d+)_cs(\d{2})', exp)
    if not m:
        raise ValueError(f'Cannot parse deep_reg experiment name: {exp!r}')
    repeat, arch_type, cs = m.group(1), m.group(2), m.group(3)
    return {
        'repeat':    repeat,
        'arch_type': arch_type,
        'cs_code':   f'cs{cs}',
        'cs_early':  cs[0] == '1',
        'cs_late':   cs[1] == '1',
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Loaders  (same pattern as analyze_deep_nas_foa.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_hparams(group: str, exp: str) -> Dict:
    """Read hparams.yaml and return flat {section.key: value} dict."""
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
    """Return best_val_SELD, best_epoch, val curves, and test metrics."""
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
    """Return duration_h + 7 TB loss-curve lists."""
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
    pattern = str(RUNS_DIR / group / 'metrics' / 'exp_r*.json')
    return sorted(
        Path(p).stem for p in glob.glob(pattern)
        if 'confusion' not in p
    )


def build_summary_df() -> pd.DataFrame:
    """4-row DataFrame: cs factors + test metrics + hparam info + duration."""
    rows = []
    for exp in _iter_exps(GROUP):
        axes    = parse_exp_name(exp)
        met     = load_metrics(GROUP, exp)
        tb      = load_tb(GROUP, exp)
        hparams = load_hparams(GROUP, exp)

        row: Dict = {
            'exp': exp,
            **axes,
            'batch_size':  hparams.get('training.batch_size', float('nan')),
            'duration_h':  tb['duration_h'],
            'best_epoch':  met['best_epoch'],
            'best_val_SELD': met['best_val_SELD'],
        }
        for m in METRICS:
            row[f'test/{m}'] = met.get(f'test/{m}', float('nan'))
        rows.append(row)

    df = pd.DataFrame(rows)
    df.sort_values('cs_code', inplace=True)   # cs00 first for stable ordering
    df.reset_index(drop=True, inplace=True)
    return df


def build_tb_curves_df() -> pd.DataFrame:
    """Long-format training/val loss curves from TensorBoard."""
    rows = []
    for exp in _iter_exps(GROUP):
        axes = parse_exp_name(exp)
        tb   = load_tb(GROUP, exp)
        for tag in TB_TAGS:
            for epoch, val in enumerate(tb.get(tag, [])):
                rows.append({'exp': exp, 'cs_code': axes['cs_code'],
                             'metric': tag, 'epoch': epoch, 'value': val})
    return pd.DataFrame(rows)


def build_val_curves_df() -> pd.DataFrame:
    """Long-format val metric curves for all 8 metrics."""
    rows = []
    for exp in _iter_exps(GROUP):
        axes = parse_exp_name(exp)
        met  = load_metrics(GROUP, exp)
        for m in METRICS:
            for epoch, val in enumerate(met.get(f'val_{m}_curve', [])):
                rows.append({'exp': exp, 'cs_code': axes['cs_code'],
                             'metric': m, 'epoch': epoch, 'value': val})
    return pd.DataFrame(rows)


def compute_deltas(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Δ metric = metric(csXY) − metric(cs00)  for cs01, cs10, cs11.

    For LOWER_BETTER metrics (SELD_score, LE_CD, ER_20):
        negative delta = improvement.
    For others:
        positive delta = improvement.
    The ``improved_{m}`` bool column encodes this consistently.
    """
    baseline = summary_df[summary_df['cs_code'] == 'cs00'].iloc[0]
    variants = summary_df[summary_df['cs_code'] != 'cs00'].copy()

    rows = []
    for _, row in variants.iterrows():
        delta_row: Dict = {
            'cs_code':  row['cs_code'],
            'cs_early': row['cs_early'],
            'cs_late':  row['cs_late'],
        }
        for m in METRICS:
            base_val = float(baseline.get(f'test/{m}', float('nan')))
            curr_val = float(row.get(f'test/{m}', float('nan')))
            delta = curr_val - base_val
            delta_row[f'delta_{m}']    = delta
            # improvement = delta < 0 for lower-better, delta > 0 for higher-better
            if m in LOWER_BETTER:
                delta_row[f'improved_{m}'] = delta < 0
            else:
                delta_row[f'improved_{m}'] = delta > 0
        rows.append(delta_row)

    return pd.DataFrame(rows)


def build_per_class_df() -> pd.DataFrame:
    """Per-class precision / recall / F1 from confusion matrices, all 4 runs."""
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
                'cs_code':    axes['cs_code'],
                'cs_early':   axes['cs_early'],
                'cs_late':    axes['cs_late'],
                'class_idx':  class_idx,
                'class_name': name,
                'precision':  prec,
                'recall':     rec,
                'f1':         f1,
                'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            })
    return pd.DataFrame(rows)


def _load_deep_best() -> Optional[Dict]:
    path = RESULTS_DIR / 'deep_nas_foa_best.json'
    if not path.exists():
        return None
    return json.loads(path.read_text())


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print('\n[1/6] Building summary DataFrame…')
    df = build_summary_df()

    summary_cols = [
        'exp', 'cs_code', 'cs_early', 'cs_late', 'batch_size',
        'duration_h', 'best_epoch', 'best_val_SELD',
        'test/SELD_score', 'test/LE_CD', 'test/ER_20', 'test/LR_CD', 'test/F_20',
        'test/sem_accuracy', 'test/sem_f1', 'test/sem_auroc',
    ]
    csv_out = RESULTS_DIR / 'reg_nas_foa_summary.csv'
    df[summary_cols].to_csv(csv_out, index=False, float_format='%.6f')
    print(f'  Saved {csv_out.name}  ({len(df)} rows)')

    print('[2/6] Building TB curves DataFrame…')
    tb_df = build_tb_curves_df()
    tb_out = RESULTS_DIR / 'reg_nas_foa_tb_curves.csv'
    tb_df.to_csv(tb_out, index=False, float_format='%.6f')
    print(f'  Saved {tb_out.name}  ({len(tb_df):,} rows)')

    print('[3/6] Building validation curves DataFrame…')
    val_df = build_val_curves_df()
    val_out = RESULTS_DIR / 'reg_nas_foa_val_curves.csv'
    val_df.to_csv(val_out, index=False, float_format='%.6f')
    print(f'  Saved {val_out.name}  ({len(val_df):,} rows)')

    print('[4/6] Computing delta metrics vs cs00 baseline…')
    deltas_df = compute_deltas(df)
    delta_out = RESULTS_DIR / 'reg_nas_foa_deltas.csv'
    deltas_df.to_csv(delta_out, index=False, float_format='%.6f')
    print(f'  Saved {delta_out.name}  ({len(deltas_df)} rows)')

    print('[5/6] Building per-class metrics DataFrame…')
    cls_df = build_per_class_df()
    cls_out = RESULTS_DIR / 'reg_nas_foa_per_class.csv'
    cls_df.to_csv(cls_out, index=False, float_format='%.6f')
    print(f'  Saved {cls_out.name}  ({len(cls_df)} rows)')

    print('[6/6] Saving best-run JSON…')
    winner = df.sort_values('test/SELD_score').iloc[0]
    deep_best = _load_deep_best()

    best: Dict = {
        'winner': winner['exp'],
        'group':  GROUP,
        'cs_code': winner['cs_code'],
    }
    for m in METRICS:
        best[f'test_{m}'] = float(winner[f'test/{m}'])
    best['best_epoch'] = int(winner['best_epoch'])
    best['duration_h'] = float(winner['duration_h'])

    # Compare to phase-2 best (quantifies pure dropout effect via cs00 baseline)
    if deep_best is not None:
        baseline = df[df['cs_code'] == 'cs00']
        if len(baseline) > 0:
            bl = baseline.iloc[0]
            best['deep_spec_winner'] = deep_best['winner']
            for m in METRICS:
                bl_val   = float(bl.get(f'test/{m}', float('nan')))
                deep_val = float(deep_best.get(f'test_{m}', float('nan')))
                best[f'dropout_delta_{m}'] = bl_val - deep_val

    best_out = RESULTS_DIR / 'reg_nas_foa_best.json'
    best_out.write_text(json.dumps(best, indent=2))
    print(f'  Saved {best_out.name}  (winner: {winner["exp"]})')

    print('\n── Delta summary (cs01/cs10/cs11 vs cs00 baseline) ──')
    delta_disp = deltas_df[['cs_code'] + [f'delta_{m}' for m in METRICS]].copy()
    print(delta_disp.to_string(index=False, float_format='%+.4f'))
    print('\nDone.')


if __name__ == '__main__':
    main()
