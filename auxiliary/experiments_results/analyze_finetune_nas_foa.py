"""
auxiliary/experiments_results/analyze_finetune_nas_foa.py
==========================================================
NAS-SELD Phase 4 (Finetune) FOA Experiments Analysis.

Reads metrics JSONs, TensorBoard event files, YAML hparams and (where
available) semantic confusion matrices for the 6 finetune experiments, and
exports structured assets for ``plot_finetune_nas_foa.py``.

No figures are generated here.

Inputs
------
  runs/at2seld_nas/finetune/*/metrics/{exp}.json
  runs/at2seld_nas/finetune/*/metrics/{exp}_confusion_matrix.json  (where present)
  runs/at2seld_nas/finetune/*/checkpoints/{exp}/version_0/hparams.yaml
  runs/at2seld_nas/finetune/*/checkpoints/{exp}/version_0/events.out.tfevents.*

Outputs  (results/)
-------------------
  finetune_nas_foa_summary.csv      — 6 rows × factors + val/test metrics + duration
  finetune_nas_foa_val_curves.csv   — val metrics per epoch, all 8 metrics (long)
  finetune_nas_foa_tb_curves.csv    — training loss per epoch (long format)
  finetune_nas_foa_sem_perclass.csv — semantic head per-class P/R/F1
                                      (all 6 exps after --compute-semantic-cm;
                                       tau_nigens and tau2019 from JSON without flag)

With --compute-semantic-cm (gated, ~20 min on M4):
  finetune_nas_foa_sem_perclass.csv — extended with balanced-exp inference rows

With --compute-inference (gated, ~1 h on M4):
  finetune_nas_foa_sed_perclass.csv — SED per-class P/R/F1 from checkpoint inference

Experiment groups
-----------------
  data_loss : exp_ft_mixed_bce, exp_ft_mixed_focal,
              exp_ft_tau_nigens_focal, exp_ft_tau2019_focal
  arch_opt  : exp_topt_BiGRU_n2_h512_merge_sum, exp_topt_TT_n1_ffn8_pe_4tracks

Test metric layout (asymmetric)
--------------------------------
  data_loss / arch_opt on BalancedSTARSS23:
      test_starss/{metric}, test_tau/{metric}, test_tau2019/{metric}
  data_loss on single dataset:
      test/{metric}  (+ test/sem_*)

Usage
-----
  # Static CSVs only (fast):
  python auxiliary/experiments_results/analyze_finetune_nas_foa.py

  # Add semantic CM for balanced experiments (~20 min):
  python auxiliary/experiments_results/analyze_finetune_nas_foa.py \\
      --compute-semantic-cm \\
      --starss-root  data/STARSS23 \\
      --tau-root     data/TAU-NIGENS2021 \\
      --tau2019-root data/TAU2019

  # Full inference — SED per-class (~1 h):
  python auxiliary/experiments_results/analyze_finetune_nas_foa.py \\
      --compute-inference \\
      --starss-root  data/STARSS23 \\
      --tau-root     data/TAU-NIGENS2021 \\
      --tau2019-root data/TAU2019

Author
------
Stefano Giacomelli, Ph.D. candidate — DISIM, University of L'Aquila
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

try:
    from tqdm import tqdm as _tqdm
    def _progress(iterable, **kw):
        return _tqdm(iterable, **kw)
except ImportError:
    def _progress(iterable, desc='', **_kw):
        print(f'  {desc}…')
        return iterable

import numpy as np
import pandas as pd
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent.parent
RUNS_DIR    = ROOT / 'runs' / 'at2seld_nas' / 'finetune'
RESULTS_DIR = ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Constants ──────────────────────────────────────────────────────────────────
VAL_METRICS = [
    'SELD_score', 'LE_CD', 'ER_20', 'LR_CD', 'F_20',
    'sem_accuracy', 'sem_f1', 'sem_auroc',
]
SELD_5 = ['SELD_score', 'LE_CD', 'LR_CD', 'ER_20', 'F_20']

TB_TAGS = [
    'train/loss_epoch', 'train/loss_tpit', 'train/loss_bce',
    'val/loss_tpit', 'val/loss_sed', 'val/loss_doa', 'val/loss_bce',
]

# Groups
DATA_LOSS_EXPS = {
    'exp_ft_mixed_bce', 'exp_ft_mixed_focal',
    'exp_ft_tau_nigens_focal', 'exp_ft_tau2019_focal',
}
ARCH_OPT_EXPS = {
    'exp_topt_BiGRU_n2_h512_merge_sum',
    'exp_topt_TT_n1_ffn8_pe_4tracks',
}
BALANCED_EXPS = {
    'exp_ft_mixed_bce', 'exp_ft_mixed_focal',
    'exp_topt_BiGRU_n2_h512_merge_sum', 'exp_topt_TT_n1_ffn8_pe_4tracks',
}

CLASS_NAMES: Dict[str, List[str]] = {
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
    'BalancedSTARSS23': [
        'Female speech', 'Male speech', 'Clapping', 'Telephone', 'Laughter',
        'Domestic sounds', 'Walk, footsteps', 'Door open/close', 'Music',
        'Musical instrument', 'Water tap', 'Bell', 'Knock',
    ],
}

# Subset labels for the SED per-class CSV (BalancedSTARSS23 test)
BALANCED_TEST_SUBSETS = ['starss23', 'tau_nigens', 'tau2019']


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _iter_exps() -> List[str]:
    return sorted(
        d.name for d in RUNS_DIR.iterdir()
        if d.is_dir() and (d / 'metrics' / f'{d.name}.json').exists()
    )


def _find_best_ckpt(exp: str) -> Optional[Path]:
    ckpt_dir = RUNS_DIR / exp / 'checkpoints' / exp
    if not ckpt_dir.exists():
        return None
    candidates = sorted(p for p in ckpt_dir.glob('*.ckpt') if p.name != 'last.ckpt')
    return candidates[-1] if candidates else None


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Loaders
# ══════════════════════════════════════════════════════════════════════════════

def load_hparams(exp: str) -> Dict:
    """Read hparams.yaml and return a flat {section.key: value} dict."""
    path = RUNS_DIR / exp / 'checkpoints' / exp / 'version_0' / 'hparams.yaml'
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


def load_metrics(exp: str) -> Dict:
    """Return val curves, best-epoch stats, and test metrics (any layout)."""
    path = RUNS_DIR / exp / 'metrics' / f'{exp}.json'
    data = json.loads(path.read_text())

    val_epochs = [d for d in data if isinstance(d.get('epoch'), int)]
    test_entry = next((d for d in data if d.get('epoch') == 'test'), {})

    best = min(val_epochs, key=lambda x: x.get('val/SELD_score', float('inf')))

    result: Dict = {
        'best_val_SELD': best.get('val/SELD_score', float('nan')),
        'best_epoch':    int(best['epoch']),
        'n_epochs':      len(val_epochs),
    }
    for m in VAL_METRICS:
        tag = f'val/{m}'
        result[f'val_{m}_curve'] = [d.get(tag, float('nan')) for d in val_epochs]

    for k, v in test_entry.items():
        if k != 'epoch':
            result[k] = v

    return result


def load_tb(exp: str) -> Dict:
    """Return duration_h + TB loss-curve lists."""
    tb_dir = RUNS_DIR / exp / 'checkpoints' / exp / 'version_0'
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
        'duration_h':   duration_h,
        **{tag.replace('/', '_'): _vals(tag) for tag in TB_TAGS},
    }


def load_sem_confusion_matrix(exp: str) -> Optional[List]:
    """Return n_cls × 2 × 2 list or None if file absent or no sem_CM key."""
    path = RUNS_DIR / exp / 'metrics' / f'{exp}_confusion_matrix.json'
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data.get('sem_confusion_matrix')


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Factor extraction
# ══════════════════════════════════════════════════════════════════════════════

def parse_factors(exp: str, hp: Dict) -> Dict:
    group        = 'data_loss' if exp in DATA_LOSS_EXPS else 'arch_opt'
    dataset_mode = hp.get('semantic.dataset', 'BalancedSTARSS23')

    smoother_type = hp.get('track_smoothing_spatial.type', 'bigru')
    smoother_h    = hp.get('track_smoothing_spatial.hidden_size', 256)
    smoother_nl   = hp.get('track_smoothing_spatial.n_layers', 1)
    merge_mode    = hp.get('track_smoothing_spatial.merge_mode', None)

    tt_nl        = hp.get('spatial_late.n_layers', 1)
    tt_ffn       = hp.get('spatial_late.ffn_expansion', None)
    pe_enable    = hp.get('spatial_late.pe_enable', False)

    return {
        'group':            group,
        'dataset_mode':     dataset_mode,
        'focal_gamma':      hp.get('training.focal_gamma', 0.0),
        'n_tracks':         hp.get('heads.n_tracks', 3),
        'smoother_type':    smoother_type,
        'smoother_h':       smoother_h,
        'smoother_n_layers': smoother_nl,
        'merge_mode':       merge_mode,
        'tt_n_layers':      tt_nl,
        'tt_ffn_expansion': tt_ffn,
        'pe_enable':        pe_enable,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4.  DataFrame builders
# ══════════════════════════════════════════════════════════════════════════════

def build_summary_df() -> pd.DataFrame:
    rows = []
    for exp in _iter_exps():
        hp  = load_hparams(exp)
        met = load_metrics(exp)
        tb  = load_tb(exp)
        fac = parse_factors(exp, hp)

        row: Dict = {
            'exp_name':   exp,
            'duration_h': tb['duration_h'],
            'best_epoch': met['best_epoch'],
            'n_epochs':   met['n_epochs'],
            **fac,
        }

        # Best val (value at best_epoch from curve)
        for m in VAL_METRICS:
            curve = met.get(f'val_{m}_curve', [])
            if curve and met['best_epoch'] < len(curve):
                row[f'best_val_{m}'] = curve[met['best_epoch']]
            else:
                row[f'best_val_{m}'] = float('nan')

        # Test — 3-subset layout (BalancedSTARSS23 / arch_opt experiments)
        for subset in ['starss', 'tau', 'tau2019']:
            for m in SELD_5:
                row[f'test_{subset}_{m}'] = met.get(f'test_{subset}/{m}', float('nan'))

        # Test — single-dataset layout (semantic metrics included)
        for m in SELD_5 + ['sem_accuracy', 'sem_f1', 'sem_auroc']:
            row[f'test_single_{m}'] = met.get(f'test/{m}', float('nan'))

        rows.append(row)

    df = pd.DataFrame(rows)
    df.sort_values('exp_name', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def build_val_curves_df() -> pd.DataFrame:
    rows = []
    for exp in _iter_exps():
        met = load_metrics(exp)
        for m in VAL_METRICS:
            for epoch, val in enumerate(met.get(f'val_{m}_curve', [])):
                rows.append({'exp_name': exp, 'metric': m,
                             'epoch': epoch, 'value': val})
    return pd.DataFrame(rows)


def build_tb_curves_df() -> pd.DataFrame:
    rows = []
    for exp in _iter_exps():
        tb = load_tb(exp)
        for tag in TB_TAGS:
            key = tag.replace('/', '_')
            for step, val in enumerate(tb.get(key, [])):
                rows.append({'exp_name': exp, 'tag': tag,
                             'step': step, 'value': val})
    return pd.DataFrame(rows)


def build_sem_perclass_df() -> pd.DataFrame:
    """Per-class P/R/F1 from semantic confusion matrices (where available)."""
    rows = []
    for exp in _iter_exps():
        cm = load_sem_confusion_matrix(exp)
        if cm is None:
            continue
        hp           = load_hparams(exp)
        dataset_mode = hp.get('semantic.dataset', 'BalancedSTARSS23')
        cls_names    = CLASS_NAMES.get(dataset_mode, CLASS_NAMES['STARSS23'])

        for cls_idx, mat in enumerate(cm):
            # mat = [[TN, FP], [FN, TP]]
            tn, fp = int(mat[0][0]), int(mat[0][1])
            fn, tp = int(mat[1][0]), int(mat[1][1])
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            name = (cls_names[cls_idx] if cls_idx < len(cls_names)
                    else f'class_{cls_idx}')
            rows.append({
                'exp_name':       exp,
                'split':          'val',
                'dataset_subset': 'single',
                'class_idx':      cls_idx,
                'class_name':     name,
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                'precision':  prec,
                'recall':     rec,
                'f1':         f1,
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Inference (gated by --compute-inference)
# ══════════════════════════════════════════════════════════════════════════════

_TARGET_THRESH = 0.5   # matches TARGET_THRESH in experiments_nas_seld/loss.py


def _run_sed_split(
    model,
    loader,
    orig_sr: int,
    device,
    exp: str,
    split: str,
    dataset_subset: str,
    cls_names: List[str],
    rows: List[Dict],
    cm_rows: Optional[List[Dict]] = None,
) -> None:
    """
    Run inference over one DataLoader.
    Appends per-class P/R/F1 to *rows*.
    If *cm_rows* is given, also appends N×N confusion matrix entries.
    """
    import torch
    from experiments_nas_seld.loss import SELDMetrics

    n_cls   = len(cls_names)
    metrics = SELDMetrics(angle_thresh=20.0)
    model.eval()

    # N×N SED co-activation matrix  [true_class, pred_class]
    cm_matrix = np.zeros((n_cls, n_cls), dtype=np.int64) if cm_rows is not None else None

    pbar = _progress(loader,
                     desc=f'{exp} {split}/{dataset_subset}',
                     total=len(loader), leave=False)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for batch in pbar:
            x, target = batch[0].to(device), batch[1].to(device)
            act_logits, doa_tanh, _ = model(x, orig_sr=orig_sr)

            T = min(act_logits.shape[1], target.shape[1])
            metrics.update(act_logits[:, :T], doa_tanh[:, :T], target[:, :T])

            if cm_matrix is not None:
                # act_logits: (B, T, N_tracks, N_classes)
                # target:     (B, T, N_tracks, N_classes, 3)
                true_active = (
                    target[:, :T].norm(dim=-1) > _TARGET_THRESH
                ).any(dim=2)                                        # (B, T, N_cls) bool
                pred_active = (
                    act_logits[:, :T].sigmoid() > 0.5
                ).any(dim=2)                                        # (B, T, N_cls) bool
                B, Tc, _ = true_active.shape
                ta = true_active.reshape(B * Tc, n_cls).cpu().numpy().astype(np.int32)
                pa = pred_active.reshape(B * Tc, n_cls).cpu().numpy().astype(np.int32)
                cm_matrix += ta.T @ pa                              # (n_cls, n_cls)

    for entry in metrics.compute_perclass():
        c    = entry['class_idx']
        name = cls_names[c] if c < len(cls_names) else f'class_{c}'
        rows.append({
            'exp_name':       exp,
            'split':          split,
            'dataset_subset': dataset_subset,
            'class_idx':      c,
            'class_name':     name,
            'tp':             entry['tp'],
            'fp':             entry['fp'],
            'fn':             entry['fn'],
            'precision':      entry['precision'],
            'recall':         entry['recall'],
            'f1':             entry['f1'],
        })

    if cm_rows is not None and cm_matrix is not None:
        for i in range(n_cls):
            true_name = cls_names[i] if i < len(cls_names) else f'class_{i}'
            for j in range(n_cls):
                pred_name = cls_names[j] if j < len(cls_names) else f'class_{j}'
                cm_rows.append({
                    'exp_name':        exp,
                    'split':           split,
                    'dataset_subset':  dataset_subset,
                    'true_class_idx':  i,
                    'true_class_name': true_name,
                    'pred_class_idx':  j,
                    'pred_class_name': pred_name,
                    'count':           int(cm_matrix[i, j]),
                })


def build_sed_perclass_df(args: argparse.Namespace) -> tuple:
    """
    Load each checkpoint and run SED inference on val + all test splits.

    Returns (perclass_df, cm_matrix_df) where:
      perclass_df   — per-class F1 (long format, same schema as sem_perclass)
      cm_matrix_df  — N×N SED confusion matrix in long format
    """
    import torch
    from experiments_nas_seld.train import NASLightningModule
    from datasets.starss23 import STARSS23Dataset
    from datasets.tau_nigens import TAUNIGENSDataset
    from datasets.tau2019 import TAU2019Dataset
    from datasets.balanced import ProjectedSELDDataset, NIGENS_TO_STARSS13, TAU2019_TO_STARSS13
    from datasets.utils import SELDDataset
    from torch.utils.data import DataLoader

    device = torch.device(
        'mps'  if torch.backends.mps.is_available()
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    print(f'  SED inference device: {device}')

    all_rows:   List[Dict] = []
    all_cm_rows: List[Dict] = []

    for exp in _iter_exps():
        ckpt_path = _find_best_ckpt(exp)
        if ckpt_path is None:
            print(f'  [SKIP] {exp}: checkpoint not found')
            continue

        hparam_path = (RUNS_DIR / exp / 'checkpoints' / exp
                       / 'version_0' / 'hparams.yaml')
        with open(hparam_path) as fh:
            raw = yaml.safe_load(fh)
        cfg = raw.get('cfg', raw)

        dataset_mode = cfg.get('semantic', {}).get('dataset', 'STARSS23')
        is_balanced  = (exp in BALANCED_EXPS)

        data = cfg.setdefault('data', {})
        if is_balanced:
            data['starss_root']  = str(args.starss_root.resolve())
            data['tau_root']     = str(args.tau_root.resolve())
            data['tau2019_root'] = str(args.tau2019_root.resolve())
        elif dataset_mode == 'TAU-NIGENS2021':
            data['root_path'] = str(args.tau_root.resolve())
        elif dataset_mode == 'TAU2019':
            data['root_path'] = str(args.tau2019_root.resolve())
        else:
            data['root_path'] = str(args.starss_root.resolve())
        data['num_workers'] = 0

        in_cfg   = cfg['input']
        sr       = in_cfg['sr']
        hop      = in_cfg.get('hop_length', 320)
        n_tracks = cfg['heads']['n_tracks']
        window_s = data.get('window_s')
        hop_s    = data.get('hop_s')
        windowed = window_s is not None

        val_frac = data.get('val_fraction', 0.2)
        val_seed = 2   # forced: ensures full class coverage (incl. Bell) in val split
        cls_starss = CLASS_NAMES['STARSS23']

        print(f'\n  [{exp}] loading checkpoint: {ckpt_path.name}')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            module = NASLightningModule.load_from_checkpoint(
                str(ckpt_path), cfg=cfg, map_location='cpu',
            )
        model   = module.model.eval().to(device)
        orig_sr = module.orig_sr

        def _mk_loader(ds) -> DataLoader:
            is_win = getattr(ds, 'windowed', windowed)
            collate = None if is_win else SELDDataset.collate_fn
            return DataLoader(ds, batch_size=1, shuffle=False,
                              num_workers=0, collate_fn=collate)

        common_kw = dict(model_sr=sr, hop_length=hop, n_tracks=n_tracks,
                         window_s=window_s, hop_s=hop_s)

        # ── VAL split ──────────────────────────────────────────────────────────
        if is_balanced:
            val_ds = STARSS23Dataset(
                root=args.starss_root.resolve(), split='val',
                val_fraction=val_frac, val_seed=val_seed, **common_kw,
            )
            _run_sed_split(model, _mk_loader(val_ds), orig_sr, device,
                           exp, 'val', 'starss23', cls_starss,
                           all_rows, all_cm_rows)

        elif dataset_mode == 'TAU-NIGENS2021':
            val_ds = TAUNIGENSDataset(
                root=args.tau_root.resolve(), split='val', **common_kw,
            )
            _run_sed_split(model, _mk_loader(val_ds), orig_sr, device,
                           exp, 'val', 'single', CLASS_NAMES['TAU-NIGENS2021'],
                           all_rows, all_cm_rows)

        elif dataset_mode == 'TAU2019':
            val_ds = TAU2019Dataset(
                root=args.tau2019_root.resolve(), split='val', fold=1, **common_kw,
            )
            _run_sed_split(model, _mk_loader(val_ds), orig_sr, device,
                           exp, 'val', 'single', CLASS_NAMES['TAU2019'],
                           all_rows, all_cm_rows)

        # ── TEST splits ────────────────────────────────────────────────────────
        if is_balanced:
            starss_test = STARSS23Dataset(
                root=args.starss_root.resolve(), split='test', **common_kw,
            )
            _run_sed_split(model, _mk_loader(starss_test), orig_sr, device,
                           exp, 'test', 'starss23', cls_starss,
                           all_rows, all_cm_rows)

            nigens_test = TAUNIGENSDataset(
                root=args.tau_root.resolve(), split='test', **common_kw,
            )
            nigens_proj = ProjectedSELDDataset(nigens_test, NIGENS_TO_STARSS13)
            _run_sed_split(model, _mk_loader(nigens_proj), orig_sr, device,
                           exp, 'test', 'tau_nigens', cls_starss,
                           all_rows, all_cm_rows)

            tau2019_test = TAU2019Dataset(
                root=args.tau2019_root.resolve(), split='test',
                mixed_mode=True, **common_kw,
            )
            tau2019_proj = ProjectedSELDDataset(tau2019_test, TAU2019_TO_STARSS13)
            _run_sed_split(model, _mk_loader(tau2019_proj), orig_sr, device,
                           exp, 'test', 'tau2019', cls_starss,
                           all_rows, all_cm_rows)

        elif dataset_mode == 'TAU-NIGENS2021':
            test_ds = TAUNIGENSDataset(
                root=args.tau_root.resolve(), split='test', **common_kw,
            )
            _run_sed_split(model, _mk_loader(test_ds), orig_sr, device,
                           exp, 'test', 'single', CLASS_NAMES['TAU-NIGENS2021'],
                           all_rows, all_cm_rows)

        elif dataset_mode == 'TAU2019':
            test_ds = TAU2019Dataset(
                root=args.tau2019_root.resolve(), split='test',
                mixed_mode=True, fold=1, **common_kw,
            )
            _run_sed_split(model, _mk_loader(test_ds), orig_sr, device,
                           exp, 'test', 'single', CLASS_NAMES['TAU2019'],
                           all_rows, all_cm_rows)

    return pd.DataFrame(all_rows), pd.DataFrame(all_cm_rows)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Semantic CM inference (gated by --compute-semantic-cm)
# ══════════════════════════════════════════════════════════════════════════════

def _run_sem_split(
    model,
    loader,
    orig_sr: int,
    device,
    exp: str,
    split: str,
    dataset_subset: str,
    n_classes: int,
    current_cls: List[str],
    rows: List[Dict],
    cm_rows: List[Dict],
) -> None:
    """
    Run semantic-head inference over one DataLoader.
    Appends per-class P/R/F1 to *rows* and N×N co-activation matrix to *cm_rows*.
    Both include split and dataset_subset columns for later filtering.
    """
    import torch

    tp_acc    = np.zeros(n_classes, dtype=np.int64)
    fp_acc    = np.zeros(n_classes, dtype=np.int64)
    fn_acc    = np.zeros(n_classes, dtype=np.int64)
    tn_acc    = np.zeros(n_classes, dtype=np.int64)
    cm_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)

    pbar = _progress(loader, desc=f'{exp} {split}/{dataset_subset}', leave=False)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for batch in pbar:
            x, target = batch[0].to(device), batch[1].to(device)
            _, _, sem_logits = model(x, orig_sr=orig_sr)

            # Replicate presence projection from loss.py SemanticClassificationMetrics
            T_sem = sem_logits.shape[1]
            T_mel = target.shape[1]
            if T_mel != T_sem:
                idx     = torch.linspace(0, T_mel - 1, T_sem,
                                         dtype=torch.long, device=target.device)
                tgt_sem = target[:, idx]
            else:
                tgt_sem = target
            norms    = tgt_sem.norm(dim=-1)
            presence = (norms > _TARGET_THRESH).any(dim=2)    # (B, T, C) bool
            pred_bin = torch.sigmoid(sem_logits) > 0.5        # (B, T, C) bool
            B, T, C  = presence.shape
            ta = presence.reshape(B * T, C).cpu().numpy().astype(np.int32)
            pa = pred_bin.reshape(B * T, C).cpu().numpy().astype(np.int32)
            tp_acc    += (ta * pa).sum(axis=0)
            fp_acc    += ((1 - ta) * pa).sum(axis=0)
            fn_acc    += (ta * (1 - pa)).sum(axis=0)
            tn_acc    += ((1 - ta) * (1 - pa)).sum(axis=0)
            cm_matrix += ta.T @ pa                             # (n_cls, n_cls)

    for cls_idx in range(n_classes):
        tp   = int(tp_acc[cls_idx])
        fp   = int(fp_acc[cls_idx])
        fn   = int(fn_acc[cls_idx])
        tn   = int(tn_acc[cls_idx])
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        name = (current_cls[cls_idx] if cls_idx < len(current_cls)
                else f'class_{cls_idx}')
        rows.append({
            'exp_name':       exp,
            'split':          split,
            'dataset_subset': dataset_subset,
            'class_idx':      cls_idx,
            'class_name':     name,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision':  prec,
            'recall':     rec,
            'f1':         f1,
        })

    for i in range(n_classes):
        ti = current_cls[i] if i < len(current_cls) else f'class_{i}'
        for j in range(n_classes):
            pj = current_cls[j] if j < len(current_cls) else f'class_{j}'
            cm_rows.append({
                'exp_name':        exp,
                'split':           split,
                'dataset_subset':  dataset_subset,
                'true_class_idx':  i,
                'true_class_name': ti,
                'pred_class_idx':  j,
                'pred_class_name': pj,
                'count':           int(cm_matrix[i, j]),
            })


def build_sem_cm_from_inference(args: argparse.Namespace) -> tuple:
    """
    Load each checkpoint and run semantic CM inference on val + all test splits.

    val_seed is forced to 2 to ensure full class coverage (incl. rare classes
    like Bell) in the val split. Note: this differs from the training val_seed=42,
    so val metrics here are for CM visualization only, not for model selection.

    Returns (perclass_df, cm_matrix_df):
      perclass_df  — per-class P/R/F1 with split and dataset_subset columns
      cm_matrix_df — N×N co-activation matrix (long format) with split and
                     dataset_subset for filtering by val/test and dataset
    """
    import torch
    from experiments_nas_seld.train import NASLightningModule
    from datasets.starss23 import STARSS23Dataset
    from datasets.tau_nigens import TAUNIGENSDataset
    from datasets.tau2019 import TAU2019Dataset
    from datasets.balanced import ProjectedSELDDataset, NIGENS_TO_STARSS13, TAU2019_TO_STARSS13
    from datasets.utils import SELDDataset
    from torch.utils.data import DataLoader

    device = torch.device(
        'mps'  if torch.backends.mps.is_available()
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    print(f'  Semantic CM inference device: {device}')

    rows:    List[Dict] = []
    cm_rows: List[Dict] = []

    for exp in _iter_exps():
        ckpt_path = _find_best_ckpt(exp)
        if ckpt_path is None:
            print(f'  [SKIP] {exp}: checkpoint not found')
            continue

        hparam_path = (RUNS_DIR / exp / 'checkpoints' / exp
                       / 'version_0' / 'hparams.yaml')
        with open(hparam_path) as fh:
            raw = yaml.safe_load(fh)
        cfg = raw.get('cfg', raw)

        dataset_mode = cfg.get('semantic', {}).get('dataset', 'STARSS23')
        is_balanced  = (exp in BALANCED_EXPS)

        data = cfg.setdefault('data', {})
        data['num_workers'] = 0
        if is_balanced:
            data['starss_root'] = str(args.starss_root.resolve())
        elif dataset_mode == 'TAU-NIGENS2021':
            data['root_path'] = str(args.tau_root.resolve())
        elif dataset_mode == 'TAU2019':
            data['root_path'] = str(args.tau2019_root.resolve())
        else:
            data['root_path'] = str(args.starss_root.resolve())

        in_cfg   = cfg['input']
        sr       = in_cfg['sr']
        hop      = in_cfg.get('hop_length', 320)
        n_tracks = cfg['heads']['n_tracks']
        window_s = data.get('window_s')
        hop_s    = data.get('hop_s')
        windowed = window_s is not None
        val_frac = data.get('val_fraction', 0.2)
        val_seed = 2   # forced: ensures full class coverage (incl. Bell) in val split

        common_kw = dict(model_sr=sr, hop_length=hop, n_tracks=n_tracks,
                         window_s=window_s, hop_s=hop_s)

        def _mk_loader(ds) -> DataLoader:
            is_w = getattr(ds, 'windowed', windowed)
            col  = None if is_w else SELDDataset.collate_fn
            return DataLoader(ds, batch_size=1, shuffle=False,
                              num_workers=0, collate_fn=col)

        if is_balanced:
            n_classes   = 13
            current_cls = CLASS_NAMES['STARSS23']
            val_ds      = STARSS23Dataset(
                root=args.starss_root.resolve(), split='val',
                val_fraction=val_frac, val_seed=val_seed, **common_kw,
            )
            val_subset  = 'starss23'
        elif dataset_mode == 'TAU-NIGENS2021':
            n_classes   = 12
            current_cls = CLASS_NAMES['TAU-NIGENS2021']
            val_ds      = TAUNIGENSDataset(
                root=args.tau_root.resolve(), split='val', **common_kw,
            )
            val_subset  = 'single'
        elif dataset_mode == 'TAU2019':
            n_classes   = 11
            current_cls = CLASS_NAMES['TAU2019']
            val_ds      = TAU2019Dataset(
                root=args.tau2019_root.resolve(), split='val', fold=1, **common_kw,
            )
            val_subset  = 'single'
        else:
            n_classes   = 13
            current_cls = CLASS_NAMES['STARSS23']
            val_ds      = STARSS23Dataset(
                root=args.starss_root.resolve(), split='val',
                val_fraction=val_frac, val_seed=val_seed, **common_kw,
            )
            val_subset  = 'starss23'

        print(f'\n  [{exp}] loading checkpoint for semantic CM '
              f'({len(val_ds)} windows, window_s={window_s})…')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            module = NASLightningModule.load_from_checkpoint(
                str(ckpt_path), cfg=cfg, map_location='cpu',
            )
        model   = module.model.eval().to(device)
        orig_sr = module.orig_sr

        # ── VAL split ─────────────────────────────────────────────────────────
        _run_sem_split(model, _mk_loader(val_ds), orig_sr, device,
                       exp, 'val', val_subset,
                       n_classes, current_cls, rows, cm_rows)

        # ── TEST splits ───────────────────────────────────────────────────────
        if is_balanced:
            starss_test  = STARSS23Dataset(
                root=args.starss_root.resolve(), split='test', **common_kw,
            )
            _run_sem_split(model, _mk_loader(starss_test), orig_sr, device,
                           exp, 'test', 'starss23',
                           n_classes, current_cls, rows, cm_rows)

            nigens_test  = TAUNIGENSDataset(
                root=args.tau_root.resolve(), split='test', **common_kw,
            )
            nigens_proj  = ProjectedSELDDataset(nigens_test, NIGENS_TO_STARSS13)
            _run_sem_split(model, _mk_loader(nigens_proj), orig_sr, device,
                           exp, 'test', 'tau_nigens',
                           n_classes, current_cls, rows, cm_rows)

            tau2019_test = TAU2019Dataset(
                root=args.tau2019_root.resolve(), split='test',
                mixed_mode=True, **common_kw,
            )
            tau2019_proj = ProjectedSELDDataset(tau2019_test, TAU2019_TO_STARSS13)
            _run_sem_split(model, _mk_loader(tau2019_proj), orig_sr, device,
                           exp, 'test', 'tau2019',
                           n_classes, current_cls, rows, cm_rows)

        elif dataset_mode == 'TAU-NIGENS2021':
            test_ds = TAUNIGENSDataset(
                root=args.tau_root.resolve(), split='test', **common_kw,
            )
            _run_sem_split(model, _mk_loader(test_ds), orig_sr, device,
                           exp, 'test', 'single',
                           n_classes, current_cls, rows, cm_rows)

        elif dataset_mode == 'TAU2019':
            test_ds = TAU2019Dataset(
                root=args.tau2019_root.resolve(), split='test',
                mixed_mode=True, fold=1, **common_kw,
            )
            _run_sem_split(model, _mk_loader(test_ds), orig_sr, device,
                           exp, 'test', 'single',
                           n_classes, current_cls, rows, cm_rows)

    return pd.DataFrame(rows), pd.DataFrame(cm_rows)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Analyze finetune NAS-SELD experiments.'
    )
    parser.add_argument('--compute-inference', action='store_true',
                        help='Load checkpoints and compute SED per-class CSV (~1 h).')
    parser.add_argument('--compute-semantic-cm', action='store_true',
                        help='Run semantic CM inference for all experiments (~20 min). '
                             'Extends finetune_nas_foa_sem_perclass.csv with '
                             'balanced-experiment rows (requires --starss-root).')
    parser.add_argument('--starss-root',  type=Path,
                        default=ROOT / 'data' / 'STARSS23')
    parser.add_argument('--tau-root',     type=Path,
                        default=ROOT / 'data' / 'TAU-NIGENS2021')
    parser.add_argument('--tau2019-root', type=Path,
                        default=ROOT / 'data' / 'TAU2019')
    args = parser.parse_args()

    print('\n[1/4] Building summary DataFrame…')
    df = build_summary_df()
    out = RESULTS_DIR / 'finetune_nas_foa_summary.csv'
    df.to_csv(out, index=False, float_format='%.6f')
    print(f'  Saved {out.name}  ({len(df)} rows)')

    print('[2/4] Building val curves DataFrame…')
    val_df = build_val_curves_df()
    out = RESULTS_DIR / 'finetune_nas_foa_val_curves.csv'
    val_df.to_csv(out, index=False, float_format='%.6f')
    print(f'  Saved {out.name}  ({len(val_df):,} rows)')

    print('[3/4] Building TensorBoard curves DataFrame…')
    tb_df = build_tb_curves_df()
    out = RESULTS_DIR / 'finetune_nas_foa_tb_curves.csv'
    tb_df.to_csv(out, index=False, float_format='%.6f')
    print(f'  Saved {out.name}  ({len(tb_df):,} rows)')

    print('[4/4] Building semantic per-class DataFrame (JSON sources)…')
    sem_df = build_sem_perclass_df()
    print(f'  JSON sources: {len(sem_df)} rows '
          f'({sem_df["exp_name"].nunique() if len(sem_df) else 0} experiments)')

    sem_cm_df = pd.DataFrame()
    if args.compute_semantic_cm:
        print('\n[4b] Running semantic CM checkpoint inference for all experiments…')
        cm_inf_df, sem_cm_df = build_sem_cm_from_inference(args)
        # Inference rows supersede JSON rows for the same experiment
        if not sem_df.empty:
            keep = sem_df[~sem_df['exp_name'].isin(cm_inf_df['exp_name'].unique())]
            sem_df = pd.concat([keep, cm_inf_df], ignore_index=True)
        else:
            sem_df = cm_inf_df
        print(f'  After merge: {len(sem_df)} rows '
              f'({sem_df["exp_name"].nunique()} experiments)')

        out = RESULTS_DIR / 'finetune_nas_foa_sem_cm_matrix.csv'
        sem_cm_df.to_csv(out, index=False, float_format='%.6f')
        print(f'  Saved {out.name}  ({len(sem_cm_df)} rows)')
    else:
        print('  Pass --compute-semantic-cm to add balanced-experiment rows.')

    out = RESULTS_DIR / 'finetune_nas_foa_sem_perclass.csv'
    sem_df.to_csv(out, index=False, float_format='%.6f')
    print(f'  Saved {out.name}  ({len(sem_df)} rows)')

    if args.compute_inference:
        print('\n[5/5] Running checkpoint inference for SED per-class metrics…')
        sed_df, sed_cm_df = build_sed_perclass_df(args)
        out = RESULTS_DIR / 'finetune_nas_foa_sed_perclass.csv'
        sed_df.to_csv(out, index=False, float_format='%.6f')
        print(f'  Saved {out.name}  ({len(sed_df)} rows)')
        out = RESULTS_DIR / 'finetune_nas_foa_sed_cm_matrix.csv'
        sed_cm_df.to_csv(out, index=False, float_format='%.6f')
        print(f'  Saved {out.name}  ({len(sed_cm_df)} rows)')
    else:
        print('\n  Skipping SED inference (pass --compute-inference to enable).')

    # ── Console summary ───────────────────────────────────────────────────────
    print('\n── Summary (sorted by exp_name) ──')
    cols = ['exp_name', 'group', 'dataset_mode', 'focal_gamma',
            'best_val_SELD_score', 'best_epoch', 'duration_h']
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False, float_format='%.4f'))
    print('\nDone.')


if __name__ == '__main__':
    main()
