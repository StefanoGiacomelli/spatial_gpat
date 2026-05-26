#!/usr/bin/env python3
"""
auxiliary/activation_test.py
==============================================================================
SED Activation Threshold Diagnostic Script
==============================================================================

Tests NAS-SELD, Standalone-Conformer and Standalone-ResNet models across a
configurable range of act_thresh values to diagnose under/over-detection,
calibration issues, and localization vs. detection failure modes.

Inference is executed *once* per model; threshold evaluation re-uses cached
logits without additional forward passes.

Usage
-----
    python auxiliary/activation_test.py \\
        --dataset starss23 --split test \\
        --root /path/to/STARSS23 \\
        --nas-config     runs/at2seld_nas/finetune/exp_ft_mixed_focal/checkpoints/exp_ft_mixed_focal/version_0/hparams.yaml \\
        --nas-checkpoint runs/at2seld_nas/finetune/exp_ft_mixed_focal/checkpoints/exp_ft_mixed_focal/30_0.4702.ckpt \\
        --conformer-config     runs/standalone_seld/conformer_starss23/checkpoints/conformer_starss23/version_0/hparams.yaml \\
        --conformer-checkpoint runs/standalone_seld/conformer_starss23/checkpoints/conformer_starss23/42_0.4671.ckpt \\
        --resnet-config     runs/standalone_seld/resnet_starss23/checkpoints/resnet_starss23/version_0/hparams.yaml \\
        --resnet-checkpoint runs/standalone_seld/resnet_starss23/checkpoints/resnet_starss23/29_0.4366.ckpt \\
        --thresholds 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5 \\
        --pool-size 128 --sample-policy random --seed 2 \\
        --device mps --output-dir ./activation_test_outputs
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

# ── sys.path: make project root importable ────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Optional dependencies ─────────────────────────────────────────────────────
try:
    import sed_eval  # type: ignore
    _SED_EVAL = True
except ImportError:
    _SED_EVAL = False

try:
    import pandas as pd  # type: ignore
    _PANDAS = True
except ImportError:
    _PANDAS = False

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

CLASS_NAMES: Dict[str, List[str]] = {
    'starss23': [
        'Female speech', 'Male speech', 'Clapping', 'Telephone', 'Laughter',
        'Domestic sounds', 'Walk, footsteps', 'Door open/close', 'Music',
        'Musical instrument', 'Water tap', 'Bell', 'Knock',
    ],
    'balanced_starss23': [
        'Female speech', 'Male speech', 'Clapping', 'Telephone', 'Laughter',
        'Domestic sounds', 'Walk, footsteps', 'Door open/close', 'Music',
        'Musical instrument', 'Water tap', 'Bell', 'Knock',
    ],
    'tau_nigens': [
        'Alarm', 'Baby cry', 'Crash', 'Dog bark', 'Female scream',
        'Female speech', 'Footstep', 'Knock', 'Male scream', 'Male speech',
        'Phone ring', 'Piano',
    ],
    'tau2019': [
        'Clearthroat', 'Cough', 'Doorslam', 'Drawer', 'Keyboard',
        'KeysDrop', 'Knock', 'Laughter', 'Pageturn', 'Phone', 'Speech',
    ],
}

_TGT_THRESH   = 0.5   # norm threshold for ground-truth activity
_ANGLE_THRESH = 20.0  # degrees, DCASE standard

# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class InferenceCache:
    """CPU-cached forward-pass outputs for one model."""
    act_logits_list: List[torch.Tensor]   # per batch (B, T, N, C)
    doa_tanh_list:   List[torch.Tensor]   # per batch (B, T, N, C, 3)
    targets_list:    List[torch.Tensor]   # per batch (B, T, N, C, 3)
    n_classes:  int
    n_tracks:   int
    n_frames:   int                       # total frame*batch count across pool
    orig_sr:    int
    hop_length: int
    model_name: str
    # Populated lazily
    probs_flat: Optional[np.ndarray] = None   # sigmoid of all logits, flattened
    tgt_flat:   Optional[np.ndarray] = None   # all GT activity flags, flattened

    @property
    def frame_duration(self) -> float:
        return self.hop_length / self.orig_sr

    def build_aggregates(self) -> None:
        if self.probs_flat is not None and self.tgt_flat is not None:
            return  # already populated on device during inference
        probs, tgts = [], []
        for logits, tgt in zip(self.act_logits_list, self.targets_list):
            T = min(logits.shape[1], tgt.shape[1])
            probs.append(torch.sigmoid(logits[:, :T]).numpy().ravel())
            act = (tgt[:, :T].norm(dim=-1) > _TGT_THRESH).numpy()
            tgts.append(act.ravel())
        self.probs_flat = np.concatenate(probs).astype(np.float32)
        self.tgt_flat   = np.concatenate(tgts).astype(bool)


@dataclass
class ThresholdResult:
    threshold:       float
    seld_score:      float = float('nan')
    er_20:           float = float('nan')
    f_20:            float = float('nan')
    le_cd:           float = float('nan')
    lr_cd:           float = float('nan')
    pred_active:     int   = 0
    tgt_active:      int   = 0
    pred_tgt_ratio:  float = float('nan')
    pred_active_pct: float = float('nan')
    tgt_active_pct:  float = float('nan')
    mean_prob:       float = float('nan')
    median_prob:     float = float('nan')
    max_prob:        float = float('nan')
    mean_prob_on_active:   float = float('nan')
    mean_prob_on_inactive: float = float('nan')
    sed_eval_er:    float = float('nan')
    sed_eval_f1:    float = float('nan')
    sed_eval_prec:  float = float('nan')
    sed_eval_rec:   float = float('nan')
    perclass: List[Dict[str, Any]] = field(default_factory=list)
    # per-class sed_eval metrics: {class_name: {sed_f1, sed_prec, sed_rec}}
    perclass_sed: Dict[str, Dict[str, float]] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Config helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_yaml_cfg(path: Path) -> dict:
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    return raw.get('cfg', raw)


def _extract_model_params(cfg: dict) -> Tuple[int, int, int, Optional[float], Optional[float]]:
    """Return (sr, hop_length, n_tracks, window_s, hop_s)."""
    inp      = cfg.get('input', {})
    sr       = inp.get('sr', 24000)
    hop      = inp.get('hop_length', 320)
    n_tracks = cfg.get('heads', {}).get('n_tracks',
               cfg.get('spatial_late', {}).get('n_tracks', 3))
    data     = cfg.get('data', {})
    window_s = data.get('window_s', None)
    hop_s    = data.get('hop_s', None)
    return sr, hop, n_tracks, window_s, hop_s


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Dataset building + pool selection
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset(dataset_name: str, split: str, root: Path, cfg: dict,
                  val_seed: int = 2):
    """Instantiate the appropriate dataset class from cfg."""
    from datasets.utils import SELDDataset  # noqa (needed for collate_fn ref)

    sr, hop, n_tracks, window_s, hop_s = _extract_model_params(cfg)
    val_frac = cfg.get('data', {}).get('val_fraction', 0.2)

    kw = dict(model_sr=sr, hop_length=hop, n_tracks=n_tracks,
              window_s=window_s, hop_s=hop_s)

    name = dataset_name.lower()
    if name in ('starss23', 'balanced_starss23'):
        from datasets.starss23 import STARSS23Dataset
        return STARSS23Dataset(root=root, split=split,
                               val_fraction=val_frac, val_seed=val_seed, **kw)
    elif name == 'tau_nigens':
        from datasets.tau_nigens import TAUNIGENSDataset
        return TAUNIGENSDataset(root=root, split=split, **kw)
    elif name == 'tau2019':
        from datasets.tau2019 import TAU2019Dataset
        return TAU2019Dataset(root=root, split=split,
                              fold=1, mixed_mode=(split == 'test'), **kw)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name!r}. "
                         "Choose from: starss23, tau_nigens, tau2019.")


def select_pool_indices(dataset, pool_size: int, policy: str,
                        seed: int) -> List[int]:
    """
    Select pool_size indices from dataset.

    Policies:
      first            — first N indices
      random           — N random indices (reproducible with seed)
      balanced-active  — prefer windows that are forced to contain events
                         (uses dataset._items[i][2] = is_forced when available)
    """
    n = len(dataset)
    if pool_size <= 0 or pool_size >= n:
        return list(range(n))

    if policy == 'first':
        return list(range(pool_size))

    rng = np.random.default_rng(seed)

    if policy == 'balanced-active':
        items = getattr(dataset, '_items', None)
        if items and isinstance(items[0], tuple) and len(items[0]) == 3:
            forced = [i for i, it in enumerate(items) if it[2]]
            if len(forced) >= pool_size:
                return rng.choice(forced, pool_size, replace=False).tolist()
            remaining = pool_size - len(forced)
            non_forced = list(set(range(n)) - set(forced))
            extra = rng.choice(non_forced,
                               min(remaining, len(non_forced)),
                               replace=False).tolist()
            print(f"  [balanced-active] {len(forced)} forced + {len(extra)} random")
            return forced + extra
        else:
            warnings.warn("balanced-active: cannot check activity without loading data "
                          "— falling back to random.", stacklevel=2)

    # random (default)
    return rng.choice(n, pool_size, replace=False).tolist()


def _count_pool_activity(dataset, indices: List[int]) -> Tuple[int, float]:
    """Return (n_samples_with_any_active, mean_active_frames_ratio)."""
    n_active = 0
    ratios   = []
    for idx in indices:
        try:
            _, tgt = dataset[idx]          # (T, N, C, 3)
            active = (tgt.norm(dim=-1) > _TGT_THRESH).any(dim=(1, 2))  # (T,) bool
            has_active = active.any().item()
            n_active  += int(has_active)
            ratios.append(active.float().mean().item())
        except Exception:
            pass
    mean_ratio = float(np.mean(ratios)) if ratios else float('nan')
    return n_active, mean_ratio


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Model loading
# ──────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_nas_model(cfg_path: Path, ckpt_path: Path,
                   device: torch.device) -> Tuple[nn.Module, dict, int]:
    """Load NASLightningModule checkpoint. Returns (model, cfg, orig_sr)."""
    from experiments_nas_seld.train import NASLightningModule

    cfg = load_yaml_cfg(cfg_path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        module = NASLightningModule.load_from_checkpoint(
            str(ckpt_path), cfg=cfg, map_location='cpu'
        )
    model   = module.model.eval().to(device)
    orig_sr = getattr(module, 'orig_sr', cfg.get('input', {}).get('sr', 24000))
    return model, cfg, orig_sr


def load_standalone_model(cfg_path: Path, ckpt_path: Path,
                          device: torch.device) -> Tuple[nn.Module, dict, int]:
    """Load StandaloneSELDModule checkpoint. Returns (model, cfg, orig_sr)."""
    from experiments_standalone_seld.train import StandaloneSELDModule

    cfg = load_yaml_cfg(cfg_path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        module = StandaloneSELDModule.load_from_checkpoint(
            str(ckpt_path), cfg=cfg, map_location='cpu'
        )
    model   = module.model.eval().to(device)
    orig_sr = getattr(module, 'orig_sr', cfg.get('input', {}).get('sr', 24000))
    return model, cfg, orig_sr


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Inference
# ──────────────────────────────────────────────────────────────────────────────

def run_inference(model: nn.Module, loader, device: torch.device,
                  orig_sr: int, hop_length: int,
                  is_nas: bool, model_name: str) -> InferenceCache:
    """
    Run forward passes over *loader* and cache logits + targets on CPU.
    Sigmoid and activity norm are computed on device per batch; the resulting
    float32/bool arrays are immediately moved to CPU to free GPU memory.
    Returns InferenceCache with aggregates pre-built.
    """
    act_list, doa_list, tgt_list = [], [], []
    probs_parts: List[np.ndarray] = []
    tgt_activity_parts: List[np.ndarray] = []
    n_frames_total = 0

    model.eval()
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for batch in tqdm(loader, desc=f'{model_name}', unit='batch', dynamic_ncols=True):
            x, target = batch[0].to(device), batch[1].to(device)
            if is_nas:
                act_logits, doa_tanh, _ = model(x, orig_sr=orig_sr)
            else:
                act_logits, doa_tanh = model(x, orig_sr=orig_sr)

            T = min(act_logits.shape[1], target.shape[1])
            act_t = act_logits[:, :T]
            tgt_t = target[:, :T]

            # Compute probability and activity on device before moving to CPU
            probs_parts.append(torch.sigmoid(act_t).cpu().numpy().ravel().astype(np.float32))
            tgt_activity_parts.append(
                (tgt_t.norm(dim=-1) > _TGT_THRESH).cpu().numpy().ravel())

            act_list.append(act_t.cpu())
            doa_list.append(doa_tanh[:, :T].cpu())
            tgt_list.append(tgt_t.cpu())
            n_frames_total += act_t.shape[0] * T

    if not act_list:
        raise RuntimeError(f"No batches processed for model {model_name!r}.")

    sample = act_list[0]
    n_tracks  = sample.shape[2]
    n_classes = sample.shape[3]

    cache = InferenceCache(
        act_logits_list=act_list,
        doa_tanh_list=doa_list,
        targets_list=tgt_list,
        n_classes=n_classes,
        n_tracks=n_tracks,
        n_frames=n_frames_total,
        orig_sr=orig_sr,
        hop_length=hop_length,
        model_name=model_name,
    )
    cache.probs_flat = np.concatenate(probs_parts)
    cache.tgt_flat   = np.concatenate(tgt_activity_parts).astype(bool)
    return cache


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Threshold evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_threshold(cache: InferenceCache, threshold: float,
                       class_names: List[str]) -> ThresholdResult:
    """Evaluate one threshold using cached logits. Returns ThresholdResult."""
    from experiments_nas_seld.loss import SELDMetrics

    metrics = SELDMetrics(angle_thresh=_ANGLE_THRESH, act_thresh=threshold)
    for logits, doa, tgt in zip(cache.act_logits_list,
                                 cache.doa_tanh_list,
                                 cache.targets_list):
        metrics.update(logits, doa, tgt)

    m = metrics.compute()

    # Diagnostic activity stats from flat arrays
    probs = cache.probs_flat
    tgts  = cache.tgt_flat
    preds = (probs > threshold)

    n_total     = len(probs)
    pred_active = int(preds.sum())
    tgt_active  = int(tgts.sum())
    ratio       = pred_active / max(tgt_active, 1)

    mean_on_act  = float(probs[tgts].mean())   if tgts.any()  else float('nan')
    mean_on_inac = float(probs[~tgts].mean())  if (~tgts).any() else float('nan')

    # Per-class breakdown from SELDMetrics
    perclass_raw = metrics.compute_perclass()
    perclass = []
    for row in perclass_raw:
        c   = row['class_idx']
        name = class_names[c] if c < len(class_names) else f'class_{c}'
        # Per-class activity counts (any-track, ignoring localization)
        pc_tgt, pc_pred = 0, 0
        for logits_b, tgt_b in zip(cache.act_logits_list, cache.targets_list):
            T = min(logits_b.shape[1], tgt_b.shape[1])
            t_act = (tgt_b[:, :T, :, c, :].norm(dim=-1) > _TGT_THRESH).any(dim=-1)
            p_act = (torch.sigmoid(logits_b[:, :T, :, c]) > threshold).any(dim=-1)
            pc_tgt  += t_act.sum().item()
            pc_pred += p_act.sum().item()
        perclass.append({
            'class_idx':  c,
            'class_name': name,
            'tgt_active':  pc_tgt,
            'pred_active': pc_pred,
            'pred_tgt_ratio': pc_pred / max(pc_tgt, 1),
            'f1':        row['f1'],
            'precision': row['precision'],
            'recall':    row['recall'],
            'tp': row['tp'], 'fp': row['fp'], 'fn': row['fn'],
        })

    return ThresholdResult(
        threshold=threshold,
        seld_score=m.get('SELD_score', float('nan')),
        er_20=m.get('ER_20', float('nan')),
        f_20=m.get('F_20', float('nan')),
        le_cd=m.get('LE_CD', float('nan')),
        lr_cd=m.get('LR_CD', float('nan')),
        pred_active=pred_active,
        tgt_active=tgt_active,
        pred_tgt_ratio=ratio,
        pred_active_pct=100.0 * pred_active / max(n_total, 1),
        tgt_active_pct=100.0 * tgt_active  / max(n_total, 1),
        mean_prob=float(probs.mean()),
        median_prob=float(np.median(probs)),
        max_prob=float(probs.max()),
        mean_prob_on_active=mean_on_act,
        mean_prob_on_inactive=mean_on_inac,
        perclass=perclass,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 6.  sed_eval integration
# ──────────────────────────────────────────────────────────────────────────────

def _cache_to_binary(cache: InferenceCache,
                     threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (pred_T×C, tgt_T×C) binary arrays, any-track collapsed."""
    pred_list, tgt_list = [], []
    for logits, tgt in zip(cache.act_logits_list, cache.targets_list):
        T = min(logits.shape[1], tgt.shape[1])
        B = logits.shape[0]
        for b in range(B):
            p = (torch.sigmoid(logits[b, :T])          > threshold).any(dim=1).numpy()  # (T, C)
            t = (tgt[b, :T].norm(dim=-1) > _TGT_THRESH).any(dim=1).numpy()              # (T, C)
            pred_list.append(p)
            tgt_list.append(t)
    return np.concatenate(pred_list, axis=0), np.concatenate(tgt_list, axis=0)


def _binary_to_events(binary: np.ndarray, frame_dur: float,
                      class_names: List[str]) -> list:
    T, C = binary.shape
    events = []
    for c in range(C):
        label   = class_names[c] if c < len(class_names) else str(c)
        active  = binary[:, c].astype(bool)
        changes = np.diff(np.concatenate([[False], active, [False]]).astype(int))
        onsets  = np.where(changes == 1)[0]
        offsets = np.where(changes == -1)[0]
        for on, off in zip(onsets, offsets):
            events.append({'event_label':  label,
                           'event_onset':  on  * frame_dur,
                           'event_offset': off * frame_dur})
    return events


def compute_sed_eval_metrics(cache: InferenceCache, threshold: float,
                             class_names: List[str]) -> Dict[str, float]:
    if not _SED_EVAL:
        return {}
    pred_bin, tgt_bin = _cache_to_binary(cache, threshold)
    frame_dur = cache.frame_duration

    ref_events  = _binary_to_events(tgt_bin,  frame_dur, class_names)
    est_events  = _binary_to_events(pred_bin, frame_dur, class_names)

    seg_metrics = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=class_names,
        time_resolution=frame_dur,
    )
    seg_metrics.evaluate(reference_event_list=ref_events,
                         estimated_event_list=est_events)
    ov = seg_metrics.results_overall_metrics()
    cw = seg_metrics.results_class_wise_metrics()
    overall = {
        'sed_eval_er':   ov.get('error_rate',  {}).get('error_rate',  float('nan')),
        'sed_eval_f1':   ov.get('f_measure',   {}).get('f_measure',   float('nan')),
        'sed_eval_prec': ov.get('f_measure',   {}).get('precision',   float('nan')),
        'sed_eval_rec':  ov.get('f_measure',   {}).get('recall',      float('nan')),
    }
    perclass_sed = {}
    for label, stats in cw.items():
        fm = stats.get('f_measure', {})
        perclass_sed[label] = {
            'sed_f1':   fm.get('f_measure', float('nan')),
            'sed_prec': fm.get('precision',  float('nan')),
            'sed_rec':  fm.get('recall',     float('nan')),
        }
    overall['_perclass_sed'] = perclass_sed
    return overall


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Reporting helpers
# ──────────────────────────────────────────────────────────────────────────────

_SEP  = '=' * 80
_DASH = '-' * 80


def _fmt(v: float, decimals: int = 4, pct: bool = False) -> str:
    if np.isnan(v):
        return '   N/A'
    return f'{v * 100:.1f}%' if pct else f'{v:.{decimals}f}'


def print_header(args, model_specs: List[Tuple[str, Optional[Path], Optional[Path]]],
                 n_split: int) -> None:
    print(f'\n{_SEP}')
    print('ACTIVATION THRESHOLD DIAGNOSTIC TEST')
    print(_SEP)
    print(f'\nRepository diagnostic script: auxiliary/activation_test.py\n')
    print('Dataset:')
    print(f'  name:                  {args.dataset.upper()}')
    print(f'  root:                  {args.root}')
    print(f'  split:                 {args.split}')
    print(f'  total samples:         {n_split}')
    eff = min(args.pool_size, n_split) if args.pool_size > 0 else n_split
    print(f'  requested pool size:   {args.pool_size if args.pool_size > 0 else "ALL"}')
    print(f'  effective pool size:   {eff}')
    print(f'  sample policy:         {args.sample_policy}')
    print(f'  seed:                  {args.seed}')
    print(f'\nThresholds:')
    print(f'  {args.thresholds}')
    print(f'\nModels:')
    for i, (name, cfg_p, ckpt_p) in enumerate(model_specs, 1):
        print(f'  {i}. {name}')
        print(f'       config:     {cfg_p}')
        print(f'       checkpoint: {ckpt_p}')
    print(f'\nDevice:  {args.device}')
    print(f'Output:  {args.output_dir}')
    print(f'\n{_SEP}')


def print_model_info(name: str, model: nn.Module, cfg: dict,
                     orig_sr: int) -> None:
    total, trainable = count_parameters(model)
    n_tracks  = cfg.get('heads', {}).get('n_tracks', '?')
    n_classes = cfg.get('heads', {}).get('n_classes', '?')
    sr        = cfg.get('input', {}).get('sr', orig_sr)
    hop       = cfg.get('input', {}).get('hop_length', '?')
    print(f'\n{_DASH}')
    print(f'Model: {name}')
    print(_DASH)
    print(f'  Total parameters:     {total:,}')
    print(f'  Trainable parameters: {trainable:,}')
    print(f'  Device:               {next(model.parameters()).device}')
    print(f'  Input sr:             {sr} Hz   hop_length: {hop}')
    print(f'  Output: act_logits (B, T, {n_tracks}, {n_classes})  '
          f'doa_tanh (B, T, {n_tracks}, {n_classes}, 3)')
    print(f'  Status: loaded ✓')


def print_inference_summary(cache: InferenceCache) -> None:
    probs = cache.probs_flat
    tgts  = cache.tgt_flat
    print(f'\n{_DASH}')
    print(f'Inference summary: {cache.model_name}')
    print(_DASH)
    print(f'  Batches processed:     {len(cache.act_logits_list)}')
    print(f'  Total frame×batch:     {cache.n_frames:,}')
    print(f'  Classes / tracks:      {cache.n_classes} / {cache.n_tracks}')
    print(f'\n  Activity logits (raw):')
    sample_logits = torch.cat(cache.act_logits_list).numpy()
    print(f'    shape: {list(cache.act_logits_list[0].shape)}  (first batch)')
    print(f'    min / max / mean / std: '
          f'{sample_logits.min():.3f} / {sample_logits.max():.3f} / '
          f'{sample_logits.mean():.3f} / {sample_logits.std():.3f}')
    print(f'\n  Activity probabilities (sigmoid):')
    print(f'    min:    {probs.min():.4f}')
    print(f'    max:    {probs.max():.4f}')
    print(f'    mean:   {probs.mean():.4f}')
    print(f'    median: {np.median(probs):.4f}')
    print(f'    std:    {probs.std():.4f}')
    n_tot = len(tgts)
    n_act = int(tgts.sum())
    print(f'\n  Target activity:')
    print(f'    active count:   {n_act:,} / {n_tot:,}  '
          f'({100.*n_act/max(n_tot,1):.2f}%)')
    if tgts.any():
        print(f'    mean prob on target-active:   {probs[tgts].mean():.4f}')
        print(f'    mean prob on target-inactive: {probs[~tgts].mean():.4f}')
    # DOA stats
    doa_sample = torch.cat(cache.doa_tanh_list).numpy()
    doa_norms  = np.linalg.norm(doa_sample.reshape(-1, 3), axis=-1)
    print(f'\n  Predicted DOA (tanh) norms — mean/median/max: '
          f'{doa_norms.mean():.3f} / {np.median(doa_norms):.3f} / {doa_norms.max():.3f}')
    tgt_sample = torch.cat(cache.targets_list).numpy()
    tgt_norms  = np.linalg.norm(tgt_sample.reshape(-1, 3), axis=-1)
    active_norms = tgt_norms[tgt_norms > _TGT_THRESH]
    if active_norms.size:
        print(f'  Target DOA norms (active) — mean/median/max: '
              f'{active_norms.mean():.3f} / {np.median(active_norms):.3f} / {active_norms.max():.3f}')


def print_threshold_table(name: str, results: List[ThresholdResult]) -> None:
    print(f'\n{_DASH}')
    print(f'Threshold sweep: {name}')
    print(_DASH)
    header = (f"{'act_thr':>7} | {'SELD↓':>6} | {'ER_20↓':>6} | {'F_20↑':>6} | "
              f"{'LE_CD↓':>7} | {'LR_CD↑':>6} | {'pred_act':>8} | "
              f"{'tgt_act':>7} | {'pred/tgt':>8} | {'pred%':>5} | {'tgt%':>5}")
    sep = '-' * len(header)
    print(f'\n{header}')
    print(sep)
    for r in results:
        pred_tgt = f'{r.pred_tgt_ratio:.3f}' if not np.isnan(r.pred_tgt_ratio) else '  N/A'
        print(
            f'{r.threshold:>7.3f} | '
            f'{r.seld_score:>6.4f} | '
            f'{r.er_20:>6.4f} | '
            f'{r.f_20:>6.4f} | '
            f'{r.le_cd:>7.2f} | '
            f'{r.lr_cd:>6.4f} | '
            f'{r.pred_active:>8d} | '
            f'{r.tgt_active:>7d} | '
            f'{pred_tgt:>8} | '
            f'{r.pred_active_pct:>5.1f} | '
            f'{r.tgt_active_pct:>5.1f}'
        )
    print()
    print('Legend:')
    print('  SELD↓ ER_20↓ LE_CD↓  = lower is better')
    print('  F_20↑ LR_CD↑         = higher is better')
    print('  pred/tgt < 1          → under-detection')
    print('  pred/tgt > 1          → over-detection')


def _best_threshold(results: List[ThresholdResult]) -> Dict[str, ThresholdResult]:
    best = {
        'SELD':   min(results, key=lambda r: r.seld_score  if not np.isnan(r.seld_score)  else 1e9),
        'F_20':   max(results, key=lambda r: r.f_20        if not np.isnan(r.f_20)        else -1),
        'ER_20':  min(results, key=lambda r: r.er_20       if not np.isnan(r.er_20)       else 1e9),
        'ratio1': min(results, key=lambda r: abs(r.pred_tgt_ratio - 1.0) if not np.isnan(r.pred_tgt_ratio) else 1e9),
    }
    return best


def print_best_threshold_analysis(name: str, results: List[ThresholdResult]) -> None:
    best = _best_threshold(results)
    r05  = next((r for r in results if abs(r.threshold - 0.5) < 1e-6), results[-1])

    f20_vals = [r.f_20 for r in results if not np.isnan(r.f_20)]
    seld_vals= [r.seld_score for r in results if not np.isnan(r.seld_score)]
    sensitivity = 'LOW'
    if f20_vals and (max(f20_vals) - min(f20_vals)) > 0.1:
        sensitivity = 'HIGH'
    elif f20_vals and (max(f20_vals) - min(f20_vals)) > 0.03:
        sensitivity = 'MEDIUM'

    # Failure mode
    ratio_05 = r05.pred_tgt_ratio if r05 else float('nan')
    ratio_best = best['F_20'].pred_tgt_ratio if not np.isnan(best['F_20'].pred_tgt_ratio) else float('nan')
    le_best    = best['SELD'].le_cd

    if not np.isnan(ratio_05) and ratio_05 < 0.5:
        failure = 'UNDER-DETECTION'
    elif not np.isnan(ratio_05) and ratio_05 > 2.0:
        failure = 'OVER-DETECTION'
    elif not np.isnan(ratio_best) and 0.7 < ratio_best < 1.4 and not np.isnan(le_best) and le_best > 40:
        failure = 'LOCALIZATION'
    elif sensitivity == 'LOW':
        failure = 'UNCLEAR'
    else:
        failure = 'MIXED'

    print(f'\nBest thresholds for {name}:')
    print(f'  best by SELD_score:       act_thr={best["SELD"].threshold:.2f}  '
          f'(SELD={best["SELD"].seld_score:.4f})')
    print(f'  best by F_20:             act_thr={best["F_20"].threshold:.2f}  '
          f'(F_20={best["F_20"].f_20:.4f})')
    print(f'  best by ER_20:            act_thr={best["ER_20"].threshold:.2f}  '
          f'(ER_20={best["ER_20"].er_20:.4f})')
    print(f'  closest pred/tgt=1:       act_thr={best["ratio1"].threshold:.2f}  '
          f'(ratio={best["ratio1"].pred_tgt_ratio:.3f})')
    print(f'\nDiagnostic interpretation:')
    print(f'  At act_thr=0.50: pred/tgt = {ratio_05:.3f}' if not np.isnan(ratio_05)
          else '  At act_thr=0.50: pred/tgt = N/A')
    print(f'  At best thr:     pred/tgt = {ratio_best:.3f}' if not np.isnan(ratio_best)
          else '  At best thr:     pred/tgt = N/A')
    print(f'  Activity prob max    = {results[0].max_prob:.4f}')
    print(f'  Activity prob median = {results[0].median_prob:.4f}')
    print(f'  Under-detection:     {"YES" if not np.isnan(ratio_05) and ratio_05 < 0.7 else "NO"}')
    print(f'  Over-detection:      {"YES" if not np.isnan(ratio_05) and ratio_05 > 1.4 else "NO"}')
    print(f'  Threshold sensitivity: {sensitivity}')
    print(f'  Main failure mode hypothesis: {failure}')

    f20_at_05   = r05.f_20 if r05 and not np.isnan(r05.f_20) else float('nan')
    f20_at_best = best['F_20'].f_20
    seld_at_05  = r05.seld_score if r05 and not np.isnan(r05.seld_score) else float('nan')
    seld_at_best= best['SELD'].seld_score
    if not np.isnan(f20_at_05) and not np.isnan(f20_at_best):
        print(f'  F_20 improvement (0.50 → best): '
              f'{f20_at_05:.4f} → {f20_at_best:.4f}  '
              f'(Δ={f20_at_best - f20_at_05:+.4f})')
    if not np.isnan(seld_at_05) and not np.isnan(seld_at_best):
        print(f'  SELD improvement  (0.50 → best): '
              f'{seld_at_05:.4f} → {seld_at_best:.4f}  '
              f'(Δ={seld_at_best - seld_at_05:+.4f})')


def print_perclass_report(name: str, best_result: ThresholdResult) -> None:
    thr = best_result.threshold
    rows = best_result.perclass
    if not rows:
        print(f'  [no per-class data for {name}]')
        return
    print(f'\n{_DASH}')
    print(f'Per-class diagnostic at best SELD threshold: {name},  act_thr={thr:.3f}')
    print(_DASH)
    hdr = (f"  {'cls':>3}  {'class_name':<22}  {'tgt_act':>7}  {'pred_act':>8}  "
           f"{'pred/tgt':>8}  {'F1':>6}  {'prec':>6}  {'rec':>6}")
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for r in rows:
        ratio_s = f'{r["pred_tgt_ratio"]:.3f}'
        print(f'  {r["class_idx"]:>3}  {r["class_name"]:<22}  '
              f'{r["tgt_active"]:>7}  {r["pred_active"]:>8}  '
              f'{ratio_s:>8}  {r["f1"]:>6.3f}  '
              f'{r["precision"]:>6.3f}  {r["recall"]:>6.3f}')
    # Summary
    sorted_under = sorted(rows, key=lambda r: r['pred_tgt_ratio'])
    sorted_over  = sorted(rows, key=lambda r: r['pred_tgt_ratio'], reverse=True)
    sorted_worst_f1 = sorted(rows, key=lambda r: r['f1'])
    print(f'\n  Classes with strongest under-detection (pred/tgt lowest):')
    for r in sorted_under[:3]:
        print(f'    {r["class_name"]}: pred/tgt={r["pred_tgt_ratio"]:.3f}')
    print(f'  Classes with strongest over-detection (pred/tgt highest):')
    for r in sorted_over[:3]:
        print(f'    {r["class_name"]}: pred/tgt={r["pred_tgt_ratio"]:.3f}')
    print(f'  Classes with worst F1:')
    for r in sorted_worst_f1[:3]:
        print(f'    {r["class_name"]}: F1={r["f1"]:.3f}')


def print_perclass_threshold_sweep(name: str, results: List[ThresholdResult],
                                   use_sed_eval: bool = False) -> None:
    """
    For each class, show F1 at every threshold and mark the best one.
    use_sed_eval=True uses sed_eval per-class F1; False uses SELDMetrics per-class F1.
    """
    if not results:
        return
    thresholds = [r.threshold for r in results]
    # Collect class names from first result with data
    all_classes = []
    for r in results:
        if r.perclass:
            all_classes = [pc['class_name'] for pc in sorted(r.perclass, key=lambda x: x['class_idx'])]
            break
    if not all_classes:
        return

    source = 'sed_eval' if (use_sed_eval and _SED_EVAL) else 'SELDMetrics'
    print(f'\n{_DASH}')
    print(f'Per-class F1 across thresholds: {name}  [{source}]')
    print(_DASH)

    col_w = 7
    thr_header = '  '.join(f'{t:>{col_w}.3f}' for t in thresholds)
    print(f'\n{"class":<22}  {thr_header}   best_thr  best_F1')
    print('-' * (22 + 2 + len(thr_header) + 22))

    for cls_name in all_classes:
        f1s = []
        for r in results:
            if use_sed_eval and _SED_EVAL and r.perclass_sed:
                v = r.perclass_sed.get(cls_name, {}).get('sed_f1', float('nan'))
            else:
                pc = next((x for x in r.perclass if x['class_name'] == cls_name), None)
                v = pc['f1'] if pc is not None else float('nan')
            f1s.append(v)

        valid = [(t, v) for t, v in zip(thresholds, f1s) if not np.isnan(v)]
        if valid:
            best_thr, best_f1 = max(valid, key=lambda x: x[1])
        else:
            best_thr, best_f1 = float('nan'), float('nan')

        row_vals = '  '.join(
            f'{"★" if abs(t - best_thr) < 1e-6 else " "}{v:>{col_w-1}.3f}' if not np.isnan(v) else f'{"":>{col_w}}'
            for t, v in zip(thresholds, f1s)
        )
        best_str = f'{best_thr:.3f}' if not np.isnan(best_thr) else '  N/A'
        f1_str   = f'{best_f1:.3f}'  if not np.isnan(best_f1)  else '  N/A'
        print(f'{cls_name:<22}  {row_vals}   {best_str:>8}  {f1_str:>7}')

    # Summary: which threshold is globally best (most classes)
    best_counts: Dict[float, int] = {t: 0 for t in thresholds}
    for cls_name in all_classes:
        f1s = []
        for r in results:
            if use_sed_eval and _SED_EVAL and r.perclass_sed:
                v = r.perclass_sed.get(cls_name, {}).get('sed_f1', float('nan'))
            else:
                pc = next((x for x in r.perclass if x['class_name'] == cls_name), None)
                v = pc['f1'] if pc is not None else float('nan')
            f1s.append(v)
        valid = [(t, v) for t, v in zip(thresholds, f1s) if not np.isnan(v)]
        if valid:
            best_t, _ = max(valid, key=lambda x: x[1])
            best_counts[best_t] = best_counts.get(best_t, 0) + 1

    print(f'\n  Best threshold per class count:')
    for t in sorted(best_counts):
        bar = '█' * best_counts[t]
        print(f'    thr={t:.3f}  {best_counts[t]:>2} classes  {bar}')


def print_sed_eval_table(name: str, results: List[ThresholdResult]) -> None:
    if not _SED_EVAL:
        return
    has_data = any(not np.isnan(r.sed_eval_er) for r in results)
    if not has_data:
        return
    print(f'\n{_DASH}')
    print(f'sed_eval / DCASE-compatible metrics: {name}')
    print('(Note: segment-based at model frame rate — not official DCASE 100ms resolution)')
    print(_DASH)
    hdr = (f"{'act_thr':>7} | {'sed_ER↓':>8} | {'sed_F1↑':>8} | "
           f"{'Precision↑':>10} | {'Recall↑':>8}")
    print(f'\n{hdr}')
    print('-' * len(hdr))
    for r in results:
        def _f(v): return f'{v:.4f}' if not np.isnan(v) else '   N/A'
        print(f'{r.threshold:>7.3f} | {_f(r.sed_eval_er):>8} | {_f(r.sed_eval_f1):>8} | '
              f'{_f(r.sed_eval_prec):>10} | {_f(r.sed_eval_rec):>8}')


def print_final_comparison(model_names: List[str],
                           results_by_model: Dict[str, List[ThresholdResult]]) -> None:
    print(f'\n{_SEP}')
    print('FINAL COMPARISON ACROSS MODELS')
    print(_SEP)
    print('\nBest internal metrics by model:\n')
    hdr = (f"{'model':<24} | {'best_thr':>8} | {'SELD↓':>6} | {'ER↓':>6} | "
           f"{'F_20↑':>6} | {'LE↓':>6} | {'LR↑':>6} | {'pred/tgt':>8}")
    print(hdr)
    print('-' * len(hdr))
    for name in model_names:
        results = results_by_model.get(name, [])
        if not results:
            print(f'{name:<24} | (no data)')
            continue
        bests = _best_threshold(results)
        b = bests['SELD']
        print(f'{name:<24} | {b.threshold:>8.3f} | '
              f'{b.seld_score:>6.4f} | {b.er_20:>6.4f} | '
              f'{b.f_20:>6.4f} | {b.le_cd:>6.2f} | '
              f'{b.lr_cd:>6.4f} | {b.pred_tgt_ratio:>8.3f}')

    if _SED_EVAL:
        has_sed = any(
            any(not np.isnan(r.sed_eval_er) for r in results_by_model.get(n, []))
            for n in model_names
        )
        if has_sed:
            print('\nBest sed_eval metrics by model:\n')
            hdr2 = (f"{'model':<24} | {'best_thr':>8} | {'sed_ER↓':>7} | "
                    f"{'sed_F1↑':>7} | {'Prec↑':>6} | {'Rec↑':>6}")
            print(hdr2)
            print('-' * len(hdr2))
            for name in model_names:
                results = results_by_model.get(name, [])
                if not results:
                    print(f'{name:<24} | (no data)')
                    continue
                best_f1 = max(results,
                              key=lambda r: r.sed_eval_f1 if not np.isnan(r.sed_eval_f1) else -1)
                def _f(v): return f'{v:.4f}' if not np.isnan(v) else '  N/A'
                print(f'{name:<24} | {best_f1.threshold:>8.3f} | '
                      f'{_f(best_f1.sed_eval_er):>7} | {_f(best_f1.sed_eval_f1):>7} | '
                      f'{_f(best_f1.sed_eval_prec):>6} | {_f(best_f1.sed_eval_rec):>6}')


def print_diagnostic_summary(model_names: List[str],
                             results_by_model: Dict[str, List[ThresholdResult]]) -> None:
    print(f'\n{_SEP}')
    print('DIAGNOSTIC SUMMARY')
    print(_SEP)
    for name in model_names:
        results = results_by_model.get(name, [])
        if not results:
            continue
        bests  = _best_threshold(results)
        r05    = next((r for r in results if abs(r.threshold - 0.5) < 1e-6), results[-1])
        bseld  = bests['SELD']
        bf20   = bests['F_20']
        ratio_05   = r05.pred_tgt_ratio
        ratio_best = bf20.pred_tgt_ratio
        f20_at_05  = r05.f_20
        f20_best   = bf20.f_20
        seld_05    = r05.seld_score
        seld_best  = bseld.seld_score
        # Failure mode
        if not np.isnan(ratio_05) and ratio_05 < 0.5:
            failure = 'UNDER-DETECTION'
        elif not np.isnan(ratio_05) and ratio_05 > 2.0:
            failure = 'OVER-DETECTION'
        elif (not np.isnan(ratio_best) and 0.7 < ratio_best < 1.4
              and not np.isnan(bseld.le_cd) and bseld.le_cd > 40):
            failure = 'LOCALIZATION'
        else:
            failure = 'MIXED / UNCLEAR'

        print(f'\n  Model: {name}')
        print(f'    Best threshold by SELD_score:       {bseld.threshold:.2f}')
        print(f'    Best threshold by F_20:             {bf20.threshold:.2f}')
        print(f'    pred/tgt at act_thr=0.50:           {ratio_05:.3f}' if not np.isnan(ratio_05) else
              f'    pred/tgt at act_thr=0.50:           N/A')
        print(f'    pred/tgt at best threshold:         {ratio_best:.3f}' if not np.isnan(ratio_best) else
              f'    pred/tgt at best threshold:         N/A')
        if not np.isnan(f20_at_05) and not np.isnan(f20_best):
            print(f'    F_20 improvement (0.50→best):       {f20_at_05:.4f}→{f20_best:.4f}  (Δ={f20_best-f20_at_05:+.4f})')
        if not np.isnan(seld_05) and not np.isnan(seld_best):
            print(f'    SELD improvement (0.50→best):       {seld_05:.4f}→{seld_best:.4f}  (Δ={seld_best-seld_05:+.4f})')
        print(f'    Main failure mode hypothesis:       {failure}')

    print(f'\nGlobal interpretation:')
    print('  - If all models improve strongly when lowering act_thresh, '
          'the main issue is SED calibration.')
    print('  - If pred/tgt remains low even at very low thresholds, '
          'activity probabilities are systematically under-confident.')
    print('  - If pred/tgt is reasonable but LE_CD remains high, '
          'localization / DOA tracking is the dominant problem.')
    print('  - If sed_eval SED metrics improve but SELD metrics remain poor, '
          'localization is likely the bottleneck.')
    print('  - If internal metrics and sed_eval disagree strongly, '
          'the internal metric implementation may not be fully aligned with DCASE protocol.')


# ──────────────────────────────────────────────────────────────────────────────
# 8.  Output saving
# ──────────────────────────────────────────────────────────────────────────────

def save_outputs(args, model_names: List[str],
                 results_by_model: Dict[str, List[ThresholdResult]],
                 class_names: List[str],
                 out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── threshold_summary.csv ────────────────────────────────────────────────
    rows = []
    for name in model_names:
        for r in results_by_model.get(name, []):
            rows.append({
                'dataset':    args.dataset,
                'split':      args.split,
                'pool_size':  args.pool_size,
                'sample_policy': args.sample_policy,
                'seed':       args.seed,
                'model_name': name,
                'threshold':  r.threshold,
                'SELD_score': r.seld_score,
                'ER_20':      r.er_20,
                'F_20':       r.f_20,
                'LE_CD':      r.le_cd,
                'LR_CD':      r.lr_cd,
                'pred_active_count': r.pred_active,
                'tgt_active_count':  r.tgt_active,
                'pred_target_ratio': r.pred_tgt_ratio,
                'pred_active_pct':   r.pred_active_pct,
                'tgt_active_pct':    r.tgt_active_pct,
                'mean_activity_prob':            r.mean_prob,
                'median_activity_prob':          r.median_prob,
                'max_activity_prob':             r.max_prob,
                'mean_activity_prob_on_active':  r.mean_prob_on_active,
                'mean_activity_prob_on_inactive':r.mean_prob_on_inactive,
                'sed_eval_ER':   r.sed_eval_er,
                'sed_eval_F1':   r.sed_eval_f1,
                'sed_eval_prec': r.sed_eval_prec,
                'sed_eval_rec':  r.sed_eval_rec,
            })
    if _PANDAS and rows:
        import pandas as pd
        pd.DataFrame(rows).to_csv(out_dir / 'threshold_summary.csv', index=False, float_format='%.6f')
        print(f'  Saved threshold_summary.csv  ({len(rows)} rows)')

    # ── per_class_summary.csv ────────────────────────────────────────────────
    pc_rows = []
    for name in model_names:
        bests = _best_threshold(results_by_model.get(name, []))
        best_r = bests.get('SELD')
        if best_r is None:
            continue
        for pc in best_r.perclass:
            pc_rows.append({
                'dataset':    args.dataset,
                'split':      args.split,
                'pool_size':  args.pool_size,
                'model_name': name,
                'threshold':  best_r.threshold,
                'class_id':   pc['class_idx'],
                'class_name': pc['class_name'],
                'tgt_active_count':  pc['tgt_active'],
                'pred_active_count': pc['pred_active'],
                'pred_target_ratio': pc['pred_tgt_ratio'],
                'F1':        pc['f1'],
                'precision': pc['precision'],
                'recall':    pc['recall'],
                'TP': pc['tp'], 'FP': pc['fp'], 'FN': pc['fn'],
            })
    if _PANDAS and pc_rows:
        import pandas as pd
        pd.DataFrame(pc_rows).to_csv(out_dir / 'per_class_summary.csv', index=False, float_format='%.6f')
        print(f'  Saved per_class_summary.csv  ({len(pc_rows)} rows)')

    # ── diagnostic_summary.json ──────────────────────────────────────────────
    diag = {
        'run_config': {
            'dataset':      args.dataset,
            'split':        args.split,
            'root':         str(args.root),
            'pool_size':    args.pool_size,
            'sample_policy': args.sample_policy,
            'seed':         args.seed,
            'thresholds':   args.thresholds,
            'device':       str(args.device),
        },
        'models': {},
    }
    for name in model_names:
        results = results_by_model.get(name, [])
        if not results:
            continue
        bests = _best_threshold(results)
        r05   = next((r for r in results if abs(r.threshold - 0.5) < 1e-6), results[-1])
        diag['models'][name] = {
            'best_threshold_SELD': bests['SELD'].threshold,
            'best_threshold_F20':  bests['F_20'].threshold,
            'SELD_at_05':          r05.seld_score,
            'F20_at_05':           r05.f_20,
            'ratio_at_05':         r05.pred_tgt_ratio,
            'SELD_at_best':        bests['SELD'].seld_score,
            'F20_at_best':         bests['F_20'].f_20,
            'ratio_at_best':       bests['F_20'].pred_tgt_ratio,
            'threshold_results':   [
                {'threshold': r.threshold, 'SELD': r.seld_score,
                 'F20': r.f_20, 'ER20': r.er_20,
                 'LE_CD': r.le_cd, 'LR_CD': r.lr_cd,
                 'pred_tgt_ratio': r.pred_tgt_ratio}
                for r in results
            ],
        }
    diag['output_dir'] = str(out_dir)
    with open(out_dir / 'diagnostic_summary.json', 'w') as fh:
        json.dump(diag, fh, indent=2, default=str)
    print(f'  Saved diagnostic_summary.json')


# ──────────────────────────────────────────────────────────────────────────────
# 9.  CLI + Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='SED activation threshold diagnostic test',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset
    p.add_argument('--dataset', default='starss23',
                   choices=['starss23', 'balanced_starss23', 'tau_nigens', 'tau2019'],
                   help='Dataset name')
    p.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    p.add_argument('--root', type=Path, required=True,
                   help='Dataset root directory')
    p.add_argument('--val-seed', type=int, default=2,
                   help='Seed for train/val split (only relevant for STARSS23 val split)')
    # Pool
    p.add_argument('--pool-size', type=int, default=128,
                   help='Number of samples to evaluate (<=0 = all)')
    p.add_argument('--sample-policy', default='random',
                   choices=['first', 'random', 'balanced-active'],
                   help='Pool selection policy')
    p.add_argument('--seed', type=int, default=42)
    # Thresholds
    p.add_argument('--thresholds', type=float, nargs='+',
                   default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
                   help='Activity threshold values to sweep')
    # NAS-SELD model
    p.add_argument('--nas-config',     type=Path, default=None)
    p.add_argument('--nas-checkpoint', type=Path, default=None)
    # Standalone Conformer
    p.add_argument('--conformer-config',     type=Path, default=None)
    p.add_argument('--conformer-checkpoint', type=Path, default=None)
    # Standalone ResNet
    p.add_argument('--resnet-config',     type=Path, default=None)
    p.add_argument('--resnet-checkpoint', type=Path, default=None)
    # Which models to run
    p.add_argument('--models', nargs='+',
                   default=['nas_seld_x1', 'standalone_conformer', 'standalone_resnet'],
                   choices=['nas_seld_x1', 'standalone_conformer', 'standalone_resnet'],
                   help='Models to test')
    # Runtime
    p.add_argument('--batch-size',   type=int, default=4)
    p.add_argument('--num-workers',  type=int, default=0)
    p.add_argument('--device',       default='auto',
                   choices=['auto', 'cpu', 'cuda', 'mps'],
                   help='Compute device: auto | cpu | cuda | mps')
    p.add_argument('--output-dir', type=Path, default=Path('./activation_test_outputs'))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.thresholds = sorted(args.thresholds)

    # ── Device ───────────────────────────────────────────────────────────────
    if args.device == 'auto':
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        warnings.warn('CUDA requested but not available. Falling back to CPU.')
        args.device = 'cpu'
    if args.device == 'mps' and not torch.backends.mps.is_available():
        warnings.warn('MPS requested but not available. Falling back to CPU.')
        args.device = 'cpu'
    device = torch.device(args.device)

    # ── Resolve model specs ───────────────────────────────────────────────────
    model_spec_map = {
        'nas_seld_x1':        ('NAS',        args.nas_config,       args.nas_checkpoint,       True),
        'standalone_conformer':('Conformer',  args.conformer_config, args.conformer_checkpoint, False),
        'standalone_resnet':  ('ResNet',      args.resnet_config,    args.resnet_checkpoint,    False),
    }

    active_specs = []
    for logical_name in args.models:
        label, cfg_p, ckpt_p, is_nas = model_spec_map[logical_name]
        if cfg_p is None or ckpt_p is None:
            print(f'[SKIP] {logical_name}: --{logical_name.replace("_","-")}-config/checkpoint '
                  f'not provided.')
            continue
        if not cfg_p.exists():
            print(f'[SKIP] {logical_name}: config not found at {cfg_p}')
            continue
        if not ckpt_p.exists():
            print(f'[SKIP] {logical_name}: checkpoint not found at {ckpt_p}')
            continue
        active_specs.append((logical_name, cfg_p, ckpt_p, is_nas))

    if not active_specs:
        print('No valid models to test. Provide at least one --*-config and --*-checkpoint.')
        sys.exit(1)

    class_names = CLASS_NAMES.get(args.dataset.lower(), [])

    # ── Build reference dataset (use first model's config for pool sizing) ────
    first_cfg = load_yaml_cfg(active_specs[0][1])
    try:
        ref_dataset = build_dataset(args.dataset, args.split, args.root,
                                    first_cfg, val_seed=args.val_seed)
    except Exception as exc:
        print(f'ERROR building reference dataset: {exc}')
        sys.exit(1)

    n_split = len(ref_dataset)
    pool_indices = select_pool_indices(ref_dataset, args.pool_size,
                                       args.sample_policy, args.seed)

    # ── Header ───────────────────────────────────────────────────────────────
    header_specs = [(n, c, k) for n, c, k, _ in active_specs]
    print_header(args, header_specs, n_split)

    # Quick pool stats (may take a moment for large pools)
    if len(pool_indices) <= 256:
        print('\nComputing pool activity statistics…')
        n_active_samples, mean_ratio = _count_pool_activity(ref_dataset, pool_indices)
        print(f'  Samples with ≥1 active event: {n_active_samples}/{len(pool_indices)}'
              f'  ({100.*n_active_samples/max(len(pool_indices),1):.1f}%)')
        print(f'  Mean active frame ratio:       {mean_ratio:.4f}')
    else:
        print(f'\n  Pool size > 256: skipping per-sample activity scan.')

    if not class_names:
        print(f'\n  WARNING: no class names found for dataset "{args.dataset}". '
              f'Using class indices.')

    # sed_eval note
    if not _SED_EVAL:
        print('\n  sed_eval not available: skipping sed_eval metrics.')
        print('  Install with: pip install sed_eval')

    # ── Main evaluation loop ──────────────────────────────────────────────────
    results_by_model: Dict[str, List[ThresholdResult]] = {}
    caches_by_model:  Dict[str, InferenceCache] = {}

    for (logical_name, cfg_p, ckpt_p, is_nas) in active_specs:
        # Load model
        print(f'\n{_DASH}')
        print(f'Loading model: {logical_name}')
        print(_DASH)
        print(f'  Config:      {cfg_p}')
        print(f'  Checkpoint:  {ckpt_p}')
        try:
            cfg = load_yaml_cfg(cfg_p)
            if is_nas:
                model, cfg, orig_sr = load_nas_model(cfg_p, ckpt_p, device)
            else:
                model, cfg, orig_sr = load_standalone_model(cfg_p, ckpt_p, device)
        except Exception as exc:
            print(f'  ERROR loading {logical_name}: {exc}')
            import traceback; traceback.print_exc()
            continue

        print_model_info(logical_name, model, cfg, orig_sr)

        # Build per-model dataset (may differ in sr/n_tracks from reference)
        try:
            dataset = build_dataset(args.dataset, args.split, args.root,
                                    cfg, val_seed=args.val_seed)
        except Exception as exc:
            print(f'  ERROR building dataset for {logical_name}: {exc}')
            continue

        # Align pool indices to this dataset's length
        n_this = len(dataset)
        valid_idx = [i for i in pool_indices if i < n_this]
        if len(valid_idx) < len(pool_indices):
            print(f'  WARNING: {len(pool_indices) - len(valid_idx)} pool indices '
                  f'exceed dataset size ({n_this}). Clipping pool.')

        from torch.utils.data import Subset
        from datasets.utils import SELDDataset
        subset    = Subset(dataset, valid_idx)
        windowed  = getattr(dataset, 'windowed', False)
        collate   = None if windowed else SELDDataset.collate_fn
        bs        = args.batch_size if windowed else 1
        loader    = torch.utils.data.DataLoader(
            subset, batch_size=bs, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate,
        )

        sr, hop, *_ = _extract_model_params(cfg)
        print(f'\n  Running inference on {len(valid_idx)} samples '
              f'(windowed={windowed}, batch_size={bs})…')
        try:
            cache = run_inference(model, loader, device, orig_sr, hop,
                                  is_nas, logical_name)
        except Exception as exc:
            print(f'  ERROR during inference for {logical_name}: {exc}')
            import traceback; traceback.print_exc()
            continue

        caches_by_model[logical_name] = cache
        print_inference_summary(cache)

        # Shape sanity check
        sample_logits = cache.act_logits_list[0]
        print(f'\n  Shape check:')
        print(f'    act_logits[0]: {list(sample_logits.shape)}  '
              f'(B, T, N_tracks={cache.n_tracks}, N_classes={cache.n_classes})')
        expected_cls = len(class_names) if class_names else cache.n_classes
        if cache.n_classes != expected_cls:
            print(f'  WARNING: model outputs {cache.n_classes} classes but '
                  f'class_names has {expected_cls} entries.')

        # Threshold sweep
        print(f'\n  Sweeping {len(args.thresholds)} thresholds…')
        thr_results = []
        for thr in args.thresholds:
            print(f'    act_thresh={thr:.3f}…', end='', flush=True)
            try:
                r = evaluate_threshold(cache, thr, class_names)
                if _SED_EVAL:
                    sed = compute_sed_eval_metrics(cache, thr, class_names)
                    r.sed_eval_er   = sed.get('sed_eval_er',   float('nan'))
                    r.sed_eval_f1   = sed.get('sed_eval_f1',   float('nan'))
                    r.sed_eval_prec = sed.get('sed_eval_prec', float('nan'))
                    r.sed_eval_rec  = sed.get('sed_eval_rec',  float('nan'))
                    r.perclass_sed  = sed.get('_perclass_sed', {})
                thr_results.append(r)
                print(f'  SELD={r.seld_score:.4f}  F20={r.f_20:.4f}  '
                      f'ratio={r.pred_tgt_ratio:.3f}')
            except Exception as exc:
                print(f'  ERROR at thr={thr}: {exc}')

        results_by_model[logical_name] = thr_results

        # Per-model report
        if thr_results:
            print_threshold_table(logical_name, thr_results)
            print_best_threshold_analysis(logical_name, thr_results)
            print_sed_eval_table(logical_name, thr_results)

            # Per-class at best SELD threshold
            bests   = _best_threshold(thr_results)
            best_r  = bests['SELD']
            print_perclass_report(logical_name, best_r)

            # Per-class F1 across all thresholds
            print_perclass_threshold_sweep(logical_name, thr_results, use_sed_eval=False)
            if _SED_EVAL:
                print_perclass_threshold_sweep(logical_name, thr_results, use_sed_eval=True)

        # Free GPU memory
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()

    # ── Final comparison ──────────────────────────────────────────────────────
    model_names = [n for n, *_ in active_specs if n in results_by_model]
    print_final_comparison(model_names, results_by_model)
    print_diagnostic_summary(model_names, results_by_model)

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_dir = args.output_dir
    print(f'\n{_DASH}')
    print(f'Saving outputs to: {out_dir}')
    save_outputs(args, model_names, results_by_model, class_names, out_dir)
    print('\nDone.')


if __name__ == '__main__':
    main()
