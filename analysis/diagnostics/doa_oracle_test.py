#!/usr/bin/env python3
"""
analysis/diagnostics/doa_oracle_test.py
==============================================================================
DOA Oracle Activity Diagnostic Script
==============================================================================

Evaluates DOA estimation quality for NAS-SELD, Standalone-Conformer and
Standalone-ResNet under ORACLE activity: ground-truth activity is used as the
evaluation mask so that SED logits and act_thresh are completely bypassed.

This is complementary to activation_test.py:
  - activation_test.py  → diagnoses SED/activity calibration
  - doa_oracle_test.py  → diagnoses DOA/localization under known activity

Usage
-----
    python analysis/diagnostics/doa_oracle_test.py \\
        --dataset starss23 --split test \\
        --root data/STARSS23 \\
        --nas-config     runs/at2seld_nas/finetune/exp_ft_mixed_focal/checkpoints/exp_ft_mixed_focal/version_0/hparams.yaml \\
        --nas-checkpoint runs/at2seld_nas/finetune/exp_ft_mixed_focal/checkpoints/exp_ft_mixed_focal/30_0.4702.ckpt \\
        --conformer-config     runs/standalone_seld/conformer_starss23/checkpoints/conformer_starss23/version_0/hparams.yaml \\
        --conformer-checkpoint runs/standalone_seld/conformer_starss23/checkpoints/conformer_starss23/29_0.4366.ckpt \\
        --resnet-config     runs/standalone_seld/resnet_starss23/checkpoints/resnet_starss23/version_0/hparams.yaml \\
        --resnet-checkpoint runs/standalone_seld/resnet_starss23/checkpoints/resnet_starss23/42_0.4671.ckpt \\
        --pool-size 256 --sample-policy random --seed 2 \\
        --device mps --output-dir results/diagnostics/doa_oracle_outputs \\
        --coordinate-sanity-check \\
        --angle-thresholds 10 20 30 45 90

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
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
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import pandas as pd
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

_TGT_THRESH = 0.5   # norm threshold for ground-truth activity
_SEP  = '=' * 80
_DASH = '-' * 80

# coordinate transforms for sanity check: name → (sign_x, sign_y, sign_z, perm)
# perm is an index triplet that selects which source axes fill (x, y, z)
_COORD_TRANSFORMS: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = {
    'identity':        ((1,  1,  1), (0, 1, 2)),
    'flip_x':          ((-1, 1,  1), (0, 1, 2)),
    'flip_y':          ((1, -1,  1), (0, 1, 2)),
    'flip_z':          ((1,  1, -1), (0, 1, 2)),
    'flip_xy':         ((-1,-1,  1), (0, 1, 2)),
    'flip_xz':         ((-1, 1, -1), (0, 1, 2)),
    'flip_yz':         ((1, -1, -1), (0, 1, 2)),
    'flip_xyz':        ((-1,-1, -1), (0, 1, 2)),
    'swap_xy':         ((1,  1,  1), (1, 0, 2)),
    'swap_xz':         ((1,  1,  1), (2, 1, 0)),
    'swap_yz':         ((1,  1,  1), (0, 2, 1)),
    'swap_xy_flip_x':  ((-1, 1,  1), (1, 0, 2)),
    'swap_xy_flip_y':  ((1, -1,  1), (1, 0, 2)),
}


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DOACache:
    """CPU-cached forward-pass outputs for one model (numpy arrays)."""
    # Per-batch lists; each entry has shape (B, T, N, C, 3) for doa/targets
    # and (B, T, N, C) for act_logits
    doa_list:        List[np.ndarray]
    targets_list:    List[np.ndarray]
    act_logits_list: List[np.ndarray]  # diagnostic only — NOT used for gating
    # One recording stem per pool sample (parallel to flattened sample order)
    recording_ids:   List[str]
    n_classes:   int
    n_tracks:    int
    n_frames:    int   # total B*T across all batches
    orig_sr:     int
    hop_length:  int
    model_name:  str
    # Lazily built flat arrays (only active slots)
    _pred_active_flat:    Optional[np.ndarray] = field(default=None, repr=False)
    _target_active_flat:  Optional[np.ndarray] = field(default=None, repr=False)
    _rec_active_flat:     Optional[np.ndarray] = field(default=None, repr=False)
    _class_active_flat:   Optional[np.ndarray] = field(default=None, repr=False)
    _track_active_flat:   Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def frame_duration(self) -> float:
        return self.hop_length / self.orig_sr

    def build_flat_active(self) -> None:
        """Flatten all target-active (N_active, 3) arrays for global analysis."""
        pred_parts, tgt_parts, rec_parts, cls_parts, trk_parts = [], [], [], [], []
        # Build per-batch-item recording ID list (len = total pool samples)
        sample_idx = 0
        for b_idx, (doa_b, tgt_b) in enumerate(zip(self.doa_list, self.targets_list)):
            B, T, N, C, _ = doa_b.shape
            # target activity: (B, T, N, C)
            tgt_norm = np.linalg.norm(tgt_b, axis=-1)  # (B, T, N, C)
            active   = tgt_norm > _TGT_THRESH           # (B, T, N, C) bool
            for b in range(B):
                rec_id = (self.recording_ids[sample_idx]
                          if sample_idx < len(self.recording_ids) else 'unknown')
                for n in range(N):
                    for c in range(C):
                        mask = active[b, :, n, c]   # (T,) bool
                        if mask.any():
                            pred_parts.append(doa_b[b, mask, n, c, :])    # (k, 3)
                            tgt_parts.append(tgt_b[b, mask, n, c, :])     # (k, 3)
                            k = int(mask.sum())
                            rec_parts.append(np.full(k, rec_id))
                            cls_parts.append(np.full(k, c, dtype=np.int32))
                            trk_parts.append(np.full(k, n, dtype=np.int32))
                sample_idx += 1

        if pred_parts:
            self._pred_active_flat   = np.concatenate(pred_parts, axis=0)
            self._target_active_flat = np.concatenate(tgt_parts,  axis=0)
            self._rec_active_flat    = np.concatenate(rec_parts,   axis=0)
            self._class_active_flat  = np.concatenate(cls_parts,   axis=0)
            self._track_active_flat  = np.concatenate(trk_parts,   axis=0)
        else:
            empty = np.zeros((0, 3), dtype=np.float32)
            self._pred_active_flat   = empty
            self._target_active_flat = empty
            self._rec_active_flat    = np.array([], dtype=object)
            self._class_active_flat  = np.array([], dtype=np.int32)
            self._track_active_flat  = np.array([], dtype=np.int32)

    @property
    def n_active(self) -> int:
        if self._pred_active_flat is None:
            self.build_flat_active()
        return len(self._pred_active_flat)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Angular error helpers
# ──────────────────────────────────────────────────────────────────────────────

def angular_error_deg(pred: np.ndarray, target: np.ndarray,
                      eps: float = 1e-8) -> np.ndarray:
    """
    Compute element-wise angular error in degrees.
    Both inputs: (..., 3). Normalises internally; handles near-zero pred norms.
    """
    pred_norm = np.linalg.norm(pred,   axis=-1, keepdims=True)
    tgt_norm  = np.linalg.norm(target, axis=-1, keepdims=True)
    pred_n = pred   / (pred_norm + eps)
    tgt_n  = target / (tgt_norm  + eps)
    cos_sim = np.clip((pred_n * tgt_n).sum(axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(cos_sim))


def apply_coord_transform(pred: np.ndarray,
                          signs: Tuple[int, int, int],
                          perm:  Tuple[int, int, int]) -> np.ndarray:
    """Apply one coordinate transform to (..., 3) predicted DOA."""
    permuted = pred[..., perm]          # reorder axes
    return permuted * np.array(signs)   # flip signs


def _pct(arr: np.ndarray, q: float) -> float:
    return float(np.percentile(arr, q)) if len(arr) > 0 else float('nan')


def _stats(arr: np.ndarray) -> Dict[str, float]:
    """Return common descriptive stats dict."""
    if len(arr) == 0:
        nan = float('nan')
        return dict(mean=nan, median=nan, std=nan, min=nan,
                    p25=nan, p75=nan, p90=nan, p95=nan, max=nan)
    return dict(
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        std=float(np.std(arr)),
        min=float(np.min(arr)),
        p25=_pct(arr, 25),
        p75=_pct(arr, 75),
        p90=_pct(arr, 90),
        p95=_pct(arr, 95),
        max=float(np.max(arr)),
    )


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Config helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_yaml_cfg(path: Path) -> dict:
    """Load yaml cfg inputs for this analysis."""

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
# 3.  Dataset building + pool selection
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset(dataset_name: str, split: str, root: Path, cfg: dict,
                  val_seed: int = 2):
    """Instantiate the appropriate dataset class from cfg."""
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
    """Select pool_size indices. pool_size<=0 → full dataset."""
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
            extra = rng.choice(non_forced, min(remaining, len(non_forced)),
                               replace=False).tolist()
            print(f'  [balanced-active] {len(forced)} forced + {len(extra)} random')
            return forced + extra
        else:
            warnings.warn('balanced-active: activity flag unavailable — falling back to random.')
    return rng.choice(n, pool_size, replace=False).tolist()


def get_recording_ids(dataset, pool_indices: List[int]) -> List[str]:
    """Return recording stem string for each pool index."""
    items = getattr(dataset, '_items', None)
    clips = getattr(dataset, 'clips', None)
    if items is None or clips is None:
        return [f'sample_{i:05d}' for i in range(len(pool_indices))]
    ids = []
    for idx in pool_indices:
        try:
            clip_idx = items[idx][0]
            stem = clips[clip_idx][1].stem
        except (IndexError, AttributeError):
            stem = f'sample_{idx:05d}'
        ids.append(stem)
    return ids


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Model loading
# ──────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count trainable and total model parameters."""

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_nas_model(cfg_path: Path, ckpt_path: Path,
                   device: torch.device) -> Tuple[nn.Module, dict, int, nn.Module]:
    """Load NASLightningModule checkpoint. Returns (model, cfg, orig_sr, tpit_loss)."""
    from experiments_nas_seld.train import NASLightningModule
    cfg = load_yaml_cfg(cfg_path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        module = NASLightningModule.load_from_checkpoint(
            str(ckpt_path), cfg=cfg, map_location='cpu')
    model     = module.model.eval().to(device)
    tpit_loss = module.tpit_loss.eval().to(device)
    orig_sr   = getattr(module, 'orig_sr', cfg.get('input', {}).get('sr', 24000))
    return model, cfg, orig_sr, tpit_loss


def load_standalone_model(cfg_path: Path, ckpt_path: Path, device: torch.device,
                          return_tpit: bool = False):
    """Load StandaloneSELDModule checkpoint. Returns (model, cfg, orig_sr), or
    (model, cfg, orig_sr, tpit_loss) when ``return_tpit`` is True (``--align-standalone``)."""
    from experiments_standalone_seld.train import StandaloneSELDModule
    cfg = load_yaml_cfg(cfg_path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        module = StandaloneSELDModule.load_from_checkpoint(
            str(ckpt_path), cfg=cfg, map_location='cpu')
    model   = module.model.eval().to(device)
    orig_sr = getattr(module, 'orig_sr', cfg.get('input', {}).get('sr', 24000))
    if return_tpit:
        return model, cfg, orig_sr, module.tpit_loss.eval().to(device)
    return model, cfg, orig_sr


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Inference
# ──────────────────────────────────────────────────────────────────────────────

def run_inference(model: nn.Module, loader, device: torch.device,
                  orig_sr: int, hop_length: int,
                  is_nas: bool, model_name: str,
                  tpit_loss: Optional[nn.Module] = None) -> Tuple[
                      List[np.ndarray], List[np.ndarray], List[np.ndarray], int]:
    """
    Run forward pass, return (doa_list, target_list, act_logits_list, n_frames).
    All tensors moved to CPU numpy. act_logits stored for secondary diagnostics only.
    For NAS-SELD models with tpit_loss provided, doa predictions are PIT-aligned
    (via tPITLoss.last_best_perm) before being stored.  Standalone models are aligned
    only when a ``tpit_loss`` is passed (CLI ``--align-standalone``); the default
    (historical) behaviour leaves their slots in the raw network order.
    """
    aligned = tpit_loss is not None
    alignment = 'PIT-aligned' if aligned else 'raw/unmatched'
    print(f'  DOA oracle alignment: {alignment}')
    if not aligned and is_nas:
        print('  WARNING: DOA oracle metrics are computed without PIT alignment '
              'and may be unreliable for tPIT models.')
    elif not aligned:
        print('  WARNING: standalone models are trained with tPITLoss (the slot order is arbitrary w.r.t. the '
              'target slots); without --align-standalone these statistics are NOT comparable with the '
              'PIT-aligned NAS-SELD statistics.')

    doa_list, tgt_list, act_list = [], [], []
    n_frames = 0

    model.eval()
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for batch in tqdm(loader, desc=model_name, unit='batch', dynamic_ncols=True):
            x, target = batch[0].to(device), batch[1].to(device)
            if is_nas:
                act_logits, doa_tanh, _ = model(x, orig_sr=orig_sr)
            else:
                act_logits, doa_tanh = model(x, orig_sr=orig_sr)

            T = min(doa_tanh.shape[1], target.shape[1])
            act_t = act_logits[:, :T]
            doa_t = doa_tanh[:, :T]
            tgt_t = target[:, :T]

            # PIT-align predicted DOA tracks to target tracks using tPITLoss permutation
            if tpit_loss is not None:
                tpit_loss(act_t, doa_t, tgt_t)        # sets tpit_loss.last_best_perm
                perm = tpit_loss.last_best_perm        # (B*T*C, N)
                B_, T_, N_, C_, _ = doa_t.shape
                # Flatten to tPITLoss internal layout: (B*T*C, N, 3)
                doa_flat = doa_t.permute(0, 1, 3, 2, 4).reshape(B_ * T_ * C_, N_, 3)
                doa_aligned = torch.gather(
                    doa_flat,
                    1,
                    perm.unsqueeze(-1).expand(-1, -1, 3),
                )  # (B*T*C, N, 3)
                # Restore to (B, T, N, C, 3)
                doa_t = doa_aligned.reshape(B_, T_, C_, N_, 3).permute(0, 1, 3, 2, 4)

            doa_list.append(doa_t.cpu().numpy())         # (B, T, N, C, 3) — aligned
            tgt_list.append(tgt_t.cpu().numpy())         # (B, T, N, C, 3)
            act_list.append(act_t.cpu().numpy())         # (B, T, N, C)
            n_frames += act_t.shape[0] * T

    return doa_list, tgt_list, act_list, n_frames


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Global DOA oracle metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_global_metrics(cache: DOACache,
                           angle_thresholds: List[float],
                           zero_norm_thresholds: List[float]
                           ) -> Dict[str, Any]:
    """Compute all global DOA oracle metrics from cached flat arrays."""
    if cache._pred_active_flat is None:
        cache.build_flat_active()

    pred = cache._pred_active_flat    # (N_active, 3)
    tgt  = cache._target_active_flat  # (N_active, 3)
    n_active = len(pred)

    if n_active == 0:
        return {'n_active': 0, 'warning': 'No target-active slots found.'}

    ae = angular_error_deg(pred, tgt)   # (N_active,)

    # Angular accuracy
    acc = {f'acc_{int(th)}deg': float((ae <= th).mean()) for th in angle_thresholds}

    # Predicted norm diagnostics
    pred_norms = np.linalg.norm(pred, axis=-1)   # (N_active,)
    tgt_norms  = np.linalg.norm(tgt,  axis=-1)

    # All-slot pred norm (need to compute separately)
    all_pred_norms = np.concatenate([
        np.linalg.norm(d, axis=-1).ravel() for d in cache.doa_list
    ])
    all_tgt_norms = np.concatenate([
        np.linalg.norm(t, axis=-1).ravel() for t in cache.targets_list
    ])
    all_tgt_active_mask = all_tgt_norms > _TGT_THRESH
    inactive_pred_norms = all_pred_norms[~all_tgt_active_mask]

    # Zero-vector collapse
    zero_active = {
        f'zero_norm_active_lt_{th:.2f}': float((pred_norms < th).mean())
        for th in zero_norm_thresholds
    }
    zero_inactive = {}
    if len(inactive_pred_norms) > 0:
        zero_inactive = {
            f'zero_norm_inactive_lt_{th:.2f}': float((inactive_pred_norms < th).mean())
            for th in zero_norm_thresholds
        }

    return {
        'n_active':         n_active,
        'n_total_slots':    len(all_tgt_active_mask),
        'active_pct':       100.0 * n_active / max(len(all_tgt_active_mask), 1),
        'ae_stats':         _stats(ae),
        **acc,
        'pred_norm_all':    dict(
            mean=float(np.mean(all_pred_norms)),
            median=float(np.median(all_pred_norms)),
            p90=_pct(all_pred_norms, 90),
            max=float(np.max(all_pred_norms)),
        ),
        'pred_norm_active': dict(
            mean=float(np.mean(pred_norms)),
            median=float(np.median(pred_norms)),
            p90=_pct(pred_norms, 90),
            max=float(np.max(pred_norms)),
        ),
        'pred_norm_inactive': dict(
            mean=float(np.mean(inactive_pred_norms)) if len(inactive_pred_norms) > 0 else float('nan'),
            median=float(np.median(inactive_pred_norms)) if len(inactive_pred_norms) > 0 else float('nan'),
            p90=_pct(inactive_pred_norms, 90),
            max=float(np.max(inactive_pred_norms)) if len(inactive_pred_norms) > 0 else float('nan'),
        ),
        'target_norm_active': dict(
            mean=float(np.mean(tgt_norms)),
            median=float(np.median(tgt_norms)),
            min=float(np.min(tgt_norms)),
            max=float(np.max(tgt_norms)),
        ),
        **zero_active,
        **zero_inactive,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Coordinate sanity check
# ──────────────────────────────────────────────────────────────────────────────

def compute_coord_sanity(cache: DOACache) -> Dict[str, Any]:
    """Test all 13 coordinate transforms and return per-transform AE stats."""
    if cache._pred_active_flat is None:
        cache.build_flat_active()
    pred = cache._pred_active_flat
    tgt  = cache._target_active_flat
    if len(pred) == 0:
        return {}

    identity_ae = angular_error_deg(pred, tgt)
    id_mean   = float(np.mean(identity_ae))
    id_median = float(np.median(identity_ae))

    results = {}
    for name, (signs, perm) in _COORD_TRANSFORMS.items():
        t_pred = apply_coord_transform(pred, signs, perm)
        ae     = angular_error_deg(t_pred, tgt)
        m      = float(np.mean(ae))
        med    = float(np.median(ae))
        results[name] = {
            'mean_AE':            m,
            'median_AE':          med,
            'p90_AE':             _pct(ae, 90),
            'mean_improvement':   id_mean   - m,
            'median_improvement': id_median - med,
        }
    best = min(results, key=lambda k: results[k]['median_AE'])
    mismatch = (results[best]['median_improvement'] > 10.0 and best != 'identity')
    return {'transforms': results, 'best': best, 'mismatch_suspected': mismatch}


# ──────────────────────────────────────────────────────────────────────────────
# 8.  Static vs moving source diagnostic
# ──────────────────────────────────────────────────────────────────────────────

def _find_segments(active_mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return list of (start, end_excl) for contiguous True runs in active_mask."""
    changes = np.diff(np.concatenate([[False], active_mask, [False]]).astype(int))
    starts  = np.where(changes ==  1)[0]
    ends    = np.where(changes == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def classify_segments(cache: DOACache,
                      static_thresh: float,
                      moving_thresh: float) -> Dict[str, Any]:
    """
    For each contiguous active segment per (batch_item, track, class),
    compute trajectory span (max angular distance over the segment),
    classify as static/quasi-static/moving, and compute per-group AE stats.
    """
    rows: List[Dict[str, Any]] = []

    sample_idx = 0
    for doa_b, tgt_b in zip(cache.doa_list, cache.targets_list):
        B, T, N, C, _ = doa_b.shape
        tgt_norm  = np.linalg.norm(tgt_b, axis=-1)
        active    = tgt_norm > _TGT_THRESH   # (B, T, N, C)

        for b in range(B):
            rec_id = (cache.recording_ids[sample_idx]
                      if sample_idx < len(cache.recording_ids) else 'unknown')
            for n in range(N):
                for c in range(C):
                    segs = _find_segments(active[b, :, n, c])
                    for s, e in segs:
                        if e - s < 2:
                            continue
                        seg_tgt  = tgt_b[b, s:e, n, c, :]   # (L, 3)
                        seg_pred = doa_b[b, s:e, n, c, :]   # (L, 3)
                        # Trajectory span: angular distance between first and last GT frame
                        span = float(angular_error_deg(seg_tgt[:1], seg_tgt[-1:])[0])
                        # Per-slot AE
                        ae_seg = angular_error_deg(seg_pred, seg_tgt)
                        pred_norms = np.linalg.norm(seg_pred, axis=-1)
                        rows.append({
                            'rec_id':      rec_id,
                            'track':       n,
                            'class_idx':   c,
                            'span_deg':    span,
                            'n_slots':     e - s,
                            'ae_mean':     float(np.mean(ae_seg)),
                            'ae_median':   float(np.median(ae_seg)),
                            'ae_p90':      _pct(ae_seg, 90),
                            'acc_20':      float((ae_seg <= 20.0).mean()),
                            'pred_norm_median': float(np.median(pred_norms)),
                            'zero_norm_lt01':   float((pred_norms < 0.1).mean()),
                        })
            sample_idx += 1

    if not rows:
        return {'error': 'No segments found (pool too small or no active events).'}

    def _group(cond):
        sub = [r for r in rows if cond(r['span_deg'])]
        if not sub:
            return {'n_events': 0, 'n_slots': 0,
                    'mean_AE': float('nan'), 'median_AE': float('nan'),
                    'p90_AE': float('nan'), 'acc_20deg': float('nan'),
                    'pred_norm_median': float('nan'), 'zero_norm_lt01': float('nan')}
        ae_m   = np.array([r['ae_mean']   for r in sub])
        ae_med = np.array([r['ae_median'] for r in sub])
        ae_p90 = np.array([r['ae_p90']   for r in sub])
        acc20  = np.array([r['acc_20']    for r in sub])
        pnm    = np.array([r['pred_norm_median'] for r in sub])
        zr     = np.array([r['zero_norm_lt01']   for r in sub])
        total_slots = sum(r['n_slots'] for r in sub)
        return {
            'n_events':        len(sub),
            'n_slots':         total_slots,
            'mean_AE':         float(np.mean(ae_m)),
            'median_AE':       float(np.mean(ae_med)),
            'p90_AE':          float(np.mean(ae_p90)),
            'acc_20deg':       float(np.mean(acc20)),
            'pred_norm_median':float(np.median(pnm)),
            'zero_norm_lt01':  float(np.mean(zr)),
        }

    result = {
        'static':      _group(lambda s: s <= static_thresh),
        'quasi_static':_group(lambda s: static_thresh < s <= moving_thresh),
        'moving':      _group(lambda s: s > moving_thresh),
    }

    # Interpretation
    st  = result['static']['median_AE']
    mv  = result['moving']['median_AE']
    if not np.isnan(st) and not np.isnan(mv):
        if st < 20 and mv > 40:
            result['interpretation'] = 'Likely moving-source tracking problem.'
        elif st > 40:
            result['interpretation'] = 'Localization issue is not limited to moving trajectories.'
        else:
            result['interpretation'] = 'No clear static vs. moving split observed.'
    else:
        result['interpretation'] = 'Insufficient data for interpretation.'
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 9.  Temporal smoothness / jitter
# ──────────────────────────────────────────────────────────────────────────────

def compute_temporal_smoothness(cache: DOACache) -> Dict[str, Any]:
    """Compute frame-to-frame angular velocity for target and predicted DOA."""
    tgt_vel_list, pred_vel_list = [], []

    for doa_b, tgt_b in zip(cache.doa_list, cache.targets_list):
        B, T, N, C, _ = doa_b.shape
        tgt_norm = np.linalg.norm(tgt_b, axis=-1)
        active   = tgt_norm > _TGT_THRESH

        for b in range(B):
            for n in range(N):
                for c in range(C):
                    segs = _find_segments(active[b, :, n, c])
                    for s, e in segs:
                        if e - s < 2:
                            continue
                        seg_tgt  = tgt_b[b, s:e, n, c, :]
                        seg_pred = doa_b[b, s:e, n, c, :]
                        # Frame-to-frame angle
                        tgt_vel  = angular_error_deg(seg_tgt[:-1], seg_tgt[1:])
                        pred_vel = angular_error_deg(seg_pred[:-1], seg_pred[1:])
                        tgt_vel_list.append(tgt_vel)
                        pred_vel_list.append(pred_vel)

    if not tgt_vel_list:
        return {'error': 'No multi-frame segments found.'}

    tgt_all  = np.concatenate(tgt_vel_list)
    pred_all = np.concatenate(pred_vel_list)
    eps = 1e-6
    jitter = pred_all / (tgt_all + eps)

    def _s(a):
        return dict(mean=float(np.mean(a)), median=float(np.median(a)), p90=_pct(a, 90))

    # Interpretation
    jitter_med = float(np.median(jitter))
    if jitter_med > 3.0:
        interp = 'HIGH temporal jitter: predicted DOA changes much faster than target.'
    elif jitter_med < 0.3:
        interp = 'Frozen DOA: predicted changes much slower than target (poor tracking).'
    else:
        interp = 'Temporal smoothness is within reasonable range.'

    return {
        'target_velocity':   _s(tgt_all),
        'predicted_velocity':_s(pred_all),
        'jitter_ratio':      _s(jitter),
        'interpretation':    interp,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 10.  Per-class diagnostics
# ──────────────────────────────────────────────────────────────────────────────

def compute_perclass(cache: DOACache,
                     class_names: List[str],
                     min_slots: int = 5) -> List[Dict[str, Any]]:
    """Per-class AE stats and norm diagnostics."""
    if cache._pred_active_flat is None:
        cache.build_flat_active()
    pred   = cache._pred_active_flat
    tgt    = cache._target_active_flat
    cls_id = cache._class_active_flat

    rows = []
    for c in range(cache.n_classes):
        mask = cls_id == c
        if mask.sum() < min_slots:
            rows.append({
                'class_idx': c,
                'class_name': class_names[c] if c < len(class_names) else f'class_{c}',
                'active_slots': int(mask.sum()),
                'mean_AE': float('nan'), 'median_AE': float('nan'),
                'p90_AE': float('nan'), 'acc_20deg': float('nan'),
                'pred_norm_median': float('nan'), 'zero_norm_lt010': float('nan'),
            })
            continue
        ae   = angular_error_deg(pred[mask], tgt[mask])
        pn   = np.linalg.norm(pred[mask], axis=-1)
        rows.append({
            'class_idx':       c,
            'class_name':      class_names[c] if c < len(class_names) else f'class_{c}',
            'active_slots':    int(mask.sum()),
            'mean_AE':         float(np.mean(ae)),
            'median_AE':       float(np.median(ae)),
            'p90_AE':          _pct(ae, 90),
            'acc_20deg':       float((ae <= 20.0).mean()),
            'pred_norm_median':float(np.median(pn)),
            'zero_norm_lt010': float((pn < 0.10).mean()),
        })
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# 11.  Per-recording diagnostics
# ──────────────────────────────────────────────────────────────────────────────

def compute_perrecording(cache: DOACache) -> List[Dict[str, Any]]:
    """Per-recording AE stats."""
    if cache._pred_active_flat is None:
        cache.build_flat_active()
    pred   = cache._pred_active_flat
    tgt    = cache._target_active_flat
    rec_id = cache._rec_active_flat

    unique_recs = np.unique(rec_id)
    rows = []
    for rid in unique_recs:
        mask = rec_id == rid
        if mask.sum() == 0:
            continue
        ae = angular_error_deg(pred[mask], tgt[mask])
        pn = np.linalg.norm(pred[mask], axis=-1)
        rows.append({
            'recording_id':    str(rid),
            'active_slots':    int(mask.sum()),
            'mean_AE':         float(np.mean(ae)),
            'median_AE':       float(np.median(ae)),
            'p90_AE':          _pct(ae, 90),
            'acc_20deg':       float((ae <= 20.0).mean()),
            'pred_norm_median':float(np.median(pn)),
            'zero_norm_lt010': float((pn < 0.10).mean()),
        })
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# 12.  Reporting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _f(v: float, d: int = 2) -> str:
    return f'{v:.{d}f}' if not np.isnan(v) else '  N/A'


def _fsign(v: float, d: int = 2) -> str:
    """Like _f but with explicit +/- sign prefix."""
    if np.isnan(v):
        return '  N/A'
    return f'{v:+.{d}f}'


def print_global_report(name: str, gm: Dict[str, Any],
                        angle_thresholds: List[float],
                        zero_norm_thresholds: List[float]) -> None:
    """Print the global report report section."""

    print(f'\n{_DASH}')
    print(f'Global DOA oracle metrics: {name}')
    print(_DASH)

    if 'warning' in gm:
        print(f'\n  WARNING: {gm["warning"]}')
        return

    ae = gm['ae_stats']
    print(f'\nAngular error statistics:')
    print(f'  evaluated slots: {gm["n_active"]:,}  '
          f'({gm["active_pct"]:.2f}% of all {gm["n_total_slots"]:,} slots)')
    print(f'  mean:   {_f(ae["mean"])}°')
    print(f'  median: {_f(ae["median"])}°')
    print(f'  std:    {_f(ae["std"])}°')
    print(f'  min:    {_f(ae["min"])}°')
    print(f'  p25:    {_f(ae["p25"])}°')
    print(f'  p75:    {_f(ae["p75"])}°')
    print(f'  p90:    {_f(ae["p90"])}°')
    print(f'  p95:    {_f(ae["p95"])}°')
    print(f'  max:    {_f(ae["max"])}°')

    print(f'\nAngular accuracy:')
    for th in angle_thresholds:
        k = f'acc_{int(th)}deg'
        v = gm.get(k, float('nan'))
        print(f'  <={int(th):>3}°: {_f(v*100, 1)}%')

    print(f'\nDOA norm diagnostics:')
    for label, key in [('all slots', 'pred_norm_all'),
                       ('target-active slots', 'pred_norm_active'),
                       ('target-inactive slots', 'pred_norm_inactive')]:
        d = gm.get(key, {})
        print(f'  Predicted norm, {label}:')
        print(f'    mean:   {_f(d.get("mean", float("nan")))}'
              f'  median: {_f(d.get("median", float("nan")))}'
              f'  p90: {_f(d.get("p90", float("nan")))}'
              f'  max: {_f(d.get("max", float("nan")))}')
    d = gm.get('target_norm_active', {})
    print(f'  Target norm, active slots:')
    print(f'    mean:   {_f(d.get("mean", float("nan")))}'
          f'  median: {_f(d.get("median", float("nan")))}'
          f'  min: {_f(d.get("min", float("nan")))}'
          f'  max: {_f(d.get("max", float("nan")))}')

    print(f'\nZero-vector collapse diagnostic:')
    print(f'  On target-active slots:')
    for th in zero_norm_thresholds:
        k = f'zero_norm_active_lt_{th:.2f}'
        v = gm.get(k, float('nan'))
        print(f'    norm < {th:.2f}: {_f(v*100, 1)}%')
    print(f'  On target-inactive slots:')
    for th in zero_norm_thresholds:
        k = f'zero_norm_inactive_lt_{th:.2f}'
        v = gm.get(k, float('nan'))
        print(f'    norm < {th:.2f}: {_f(v*100, 1)}%')


def print_coord_report(name: str, coord: Dict[str, Any]) -> None:
    """Print the coord report report section."""

    print(f'\n{_DASH}')
    print(f'Coordinate transform sanity check: {name}')
    print(_DASH)
    if not coord:
        print('  (no data)')
        return
    tr  = coord.get('transforms', {})
    hdr = (f'{"transform":<20} | {"mean_AE":>7} | {"median_AE":>9} | '
           f'{"p90_AE":>6} | {"mean_impr":>9} | {"median_impr":>11}')
    print(f'\n{hdr}')
    print('-' * len(hdr))
    for tname, stats in tr.items():
        best_mark = ' ★' if tname == coord.get('best') else '  '
        print(f'{tname:<20} |{best_mark}{_f(stats["mean_AE"]):>6}° | '
              f'{_f(stats["median_AE"]):>8}° | '
              f'{_f(stats["p90_AE"]):>5}° | '
              f'{_fsign(stats["mean_improvement"]):>9}° | '
              f'{_fsign(stats["median_improvement"]):>11}°')
    best = coord.get('best', 'identity')
    mismatch = coord.get('mismatch_suspected', False)
    print(f'\nBest transform by median angular error: {best}')
    if mismatch:
        print('  ⚠  Possible coordinate-system mismatch detected.'
              f' ({best} reduces median AE by '
              f'{_f(tr[best]["median_improvement"])}°)')
    else:
        print('  No significant coordinate mismatch detected.')


def print_motion_report(name: str, motion: Dict[str, Any]) -> None:
    """Print the motion report report section."""

    print(f'\n{_DASH}')
    print(f'Static vs moving diagnostic: {name}')
    print(_DASH)
    if 'error' in motion:
        print(f'  {motion["error"]}')
        return
    hdr = (f'{"motion_type":<14} | {"n_events":>8} | {"n_slots":>7} | '
           f'{"mean_AE":>7} | {"median_AE":>9} | {"p90_AE":>6} | '
           f'{"<=20°":>6} | {"norm_med":>8} | {"zero<0.1":>8}')
    print(f'\n{hdr}')
    print('-' * len(hdr))
    for mtype in ('static', 'quasi_static', 'moving'):
        d = motion.get(mtype, {})
        print(f'{mtype:<14} | {d.get("n_events",0):>8} | '
              f'{d.get("n_slots",0):>7} | '
              f'{_f(d.get("mean_AE",float("nan"))):>7} | '
              f'{_f(d.get("median_AE",float("nan"))):>9} | '
              f'{_f(d.get("p90_AE",float("nan"))):>6} | '
              f'{_f(d.get("acc_20deg",float("nan"))*100,1):>5}% | '
              f'{_f(d.get("pred_norm_median",float("nan"))):>8} | '
              f'{_f(d.get("zero_norm_lt01",float("nan"))*100,1):>7}%')
    print(f'\nInterpretation: {motion.get("interpretation", "")}')


def print_smoothness_report(name: str, smooth: Dict[str, Any]) -> None:
    """Print the smoothness report report section."""

    print(f'\n{_DASH}')
    print(f'Temporal smoothness diagnostic: {name}')
    print(_DASH)
    if 'error' in smooth:
        print(f'  {smooth["error"]}')
        return
    for key, label in [('target_velocity', 'Target angular velocity'),
                       ('predicted_velocity', 'Predicted angular velocity'),
                       ('jitter_ratio', 'Jitter ratio (pred/tgt)')]:
        d = smooth.get(key, {})
        print(f'\n{label}:')
        print(f'  mean:   {_f(d.get("mean", float("nan")))}')
        print(f'  median: {_f(d.get("median", float("nan")))}')
        print(f'  p90:    {_f(d.get("p90", float("nan")))}')
    print(f'\nInterpretation: {smooth.get("interpretation", "")}')


def print_perclass_report(name: str, rows: List[Dict[str, Any]]) -> None:
    """Print the perclass report report section."""

    print(f'\n{_DASH}')
    print(f'Per-class DOA diagnostic: {name}')
    print(_DASH)
    hdr = (f'{"cls":>3} | {"class_name":<22} | {"slots":>7} | '
           f'{"mean_AE":>7} | {"med_AE":>6} | {"p90_AE":>6} | '
           f'{"<=20°":>6} | {"norm_med":>8} | {"zero<0.1":>8}')
    print(f'\n{hdr}')
    print('-' * len(hdr))
    for r in rows:
        slots_s = f'{r["active_slots"]:>7}'
        if r['active_slots'] < 5:
            slots_s += ' (skip)'
        print(f'{r["class_idx"]:>3} | {r["class_name"]:<22} | {r["active_slots"]:>7} | '
              f'{_f(r["mean_AE"]):>7} | {_f(r["median_AE"]):>6} | '
              f'{_f(r["p90_AE"]):>6} | {_f(r.get("acc_20deg",float("nan"))*100,1):>5}% | '
              f'{_f(r["pred_norm_median"]):>8} | {_f(r["zero_norm_lt010"]*100,1):>7}%')

    valid = [r for r in rows if not np.isnan(r.get('median_AE', float('nan')))
             and r['active_slots'] >= 5]
    if valid:
        worst5  = sorted(valid, key=lambda r: r['median_AE'], reverse=True)[:5]
        best5   = sorted(valid, key=lambda r: r['median_AE'])[:5]
        zero5   = sorted(valid, key=lambda r: r.get('zero_norm_lt010', 0), reverse=True)[:5]
        print(f'\nClasses with worst median angular error:')
        for i, r in enumerate(worst5, 1):
            print(f'  {i}. {r["class_name"]} — median {_f(r["median_AE"])}°  '
                  f'({r["active_slots"]} slots)')
        print(f'\nClasses with strongest zero-vector collapse (norm<0.1):')
        for i, r in enumerate(zero5, 1):
            print(f'  {i}. {r["class_name"]} — {_f(r["zero_norm_lt010"]*100, 1)}%  '
                  f'({r["active_slots"]} slots)')
        print(f'\nClasses with best localisation:')
        for i, r in enumerate(best5, 1):
            print(f'  {i}. {r["class_name"]} — median {_f(r["median_AE"])}°  '
                  f'({r["active_slots"]} slots)')


def print_perrecording_report(name: str, rows: List[Dict[str, Any]]) -> None:
    """Print the perrecording report report section."""

    if not rows:
        return
    print(f'\n{_DASH}')
    print(f'Per-recording DOA diagnostic: {name}')
    print(_DASH)
    worst10 = sorted(rows, key=lambda r: r.get('median_AE', 0), reverse=True)[:10]
    best5   = sorted(rows, key=lambda r: r.get('median_AE', 999))[:5]
    zero10  = sorted(rows, key=lambda r: r.get('zero_norm_lt010', 0), reverse=True)[:10]

    def _row(r):
        return (f'  {r["recording_id"]:<35}  slots={r["active_slots"]:>6}  '
                f'med_AE={_f(r["median_AE"])}°  '
                f'<=20°={_f(r["acc_20deg"]*100,1)}%  '
                f'zero<0.1={_f(r["zero_norm_lt010"]*100,1)}%')

    print(f'\nWorst recordings by median angular error:')
    for i, r in enumerate(worst10, 1):
        print(f'  {i:>2}. ' + _row(r).lstrip())
    print(f'\nBest recordings by median angular error:')
    for i, r in enumerate(best5, 1):
        print(f'  {i:>2}. ' + _row(r).lstrip())
    print(f'\nWorst recordings by zero-vector collapse:')
    for i, r in enumerate(zero10, 1):
        print(f'  {i:>2}. ' + _row(r).lstrip())


def print_final_comparison(model_names: List[str],
                           global_metrics: Dict[str, Dict],
                           coord_checks: Dict[str, Dict]) -> None:
    """Print the final comparison report section."""

    print(f'\n{_SEP}')
    print('FINAL DOA ORACLE COMPARISON ACROSS MODELS')
    print(_SEP)
    hdr = (f'{"model":<24} | {"slots":>7} | {"mean_AE":>7} | '
           f'{"med_AE":>6} | {"p90_AE":>6} | {"<=20°":>6} | '
           f'{"norm_act_med":>12} | {"zero<0.1":>8} | {"best_coord":>12}')
    print(f'\n{hdr}')
    print('-' * len(hdr))
    for name in model_names:
        gm  = global_metrics.get(name, {})
        cc  = coord_checks.get(name, {})
        ae  = gm.get('ae_stats', {})
        pna = gm.get('pred_norm_active', {})
        zra = gm.get('zero_norm_active_lt_0.10', float('nan'))
        acc = gm.get('acc_20deg', float('nan'))
        best = cc.get('best', 'n/a')
        print(f'{name:<24} | {gm.get("n_active",0):>7,} | '
              f'{_f(ae.get("mean",float("nan"))):>7} | '
              f'{_f(ae.get("median",float("nan"))):>6} | '
              f'{_f(ae.get("p90",float("nan"))):>6} | '
              f'{_f(acc*100 if not np.isnan(acc) else float("nan"),1):>5}% | '
              f'{_f(pna.get("median",float("nan"))):>12} | '
              f'{_f(zra*100 if not np.isnan(zra) else float("nan"),1):>7}% | '
              f'{best:>12}')


def print_diagnostic_summary(model_names: List[str],
                              global_metrics: Dict[str, Dict],
                              coord_checks: Dict[str, Dict],
                              motion_data: Dict[str, Dict],
                              smooth_data: Dict[str, Dict]) -> None:
    """Print the diagnostic summary report section."""

    print(f'\n{_SEP}')
    print('DIAGNOSTIC SUMMARY')
    print(_SEP)

    for name in model_names:
        gm   = global_metrics.get(name, {})
        cc   = coord_checks.get(name, {})
        md   = motion_data.get(name, {})
        sd   = smooth_data.get(name, {})
        ae   = gm.get('ae_stats', {})
        med  = ae.get('median', float('nan'))
        acc20 = gm.get('acc_20deg', float('nan'))
        zra  = gm.get('zero_norm_active_lt_0.10', float('nan'))
        jit  = sd.get('jitter_ratio', {}).get('median', float('nan'))

        # Localisation quality label
        if np.isnan(med):
            loc_label = 'N/A'
        elif med < 10:
            loc_label = 'GOOD'
        elif med < 25:
            loc_label = 'MODERATE'
        elif med < 50:
            loc_label = 'POOR'
        else:
            loc_label = 'COLLAPSED'

        zero_label    = 'YES' if not np.isnan(zra) and zra > 0.20 else 'NO'
        mismatch      = cc.get('mismatch_suspected', False)
        mismatch_label= 'YES' if mismatch else 'NO'
        interp_motion = md.get('interpretation', '')
        if 'tracking' in interp_motion.lower():
            moving_label = 'YES'
        elif 'not limited' in interp_motion.lower():
            moving_label = 'NO (global issue)'
        else:
            moving_label = 'UNCLEAR'
        jit_label = ('YES (jitter)' if not np.isnan(jit) and jit > 3.0 else
                     'YES (frozen)'  if not np.isnan(jit) and jit < 0.3 else
                     'NO' if not np.isnan(jit) else 'N/A')

        st_med = md.get('static',  {}).get('median_AE', float('nan'))
        mv_med = md.get('moving',  {}).get('median_AE', float('nan'))
        def _loc(v):
            if np.isnan(v): return 'N/A'
            return 'GOOD' if v < 10 else 'MODERATE' if v < 25 else 'POOR'

        print(f'\n  Model: {name}')
        print(f'    Localisation under oracle activity:    {loc_label}'
              f'  (median AE = {_f(med)}°, <=20°= {_f(acc20*100 if not np.isnan(acc20) else float("nan"),1)}%)')
        print(f'    Zero-vector collapse (>20% at <0.1):   {zero_label}'
              f'  ({_f(zra*100 if not np.isnan(zra) else float("nan"),1)}% of active slots)')
        print(f'    Possible coordinate mismatch:          {mismatch_label}')
        print(f'    Moving-source tracking problem:        {moving_label}')
        print(f'    Temporal jitter:                       {jit_label}')
        print(f'    Static-source localisation:            {_loc(st_med)}'
              f'  (median AE = {_f(st_med)}°)')
        print(f'    Moving-source localisation:            {_loc(mv_med)}'
              f'  (median AE = {_f(mv_med)}°)')

    print(f'\nGlobal interpretation rules:')
    print('  - If oracle DOA is poor even when target activity is known, '
          'the problem is not only SED.')
    print('  - If predicted DOA norms are close to zero on target-active slots, '
          'suspect DOA head collapse or loss imbalance.')
    print('  - If a coordinate transform strongly improves angular error, '
          'suspect coordinate convention or channel/axis mismatch.')
    print('  - If static sources are localised well but moving sources are not, '
          'suspect trajectory tracking limitations.')
    print('  - If predicted angular velocity >> target velocity, suspect jittery localisation.')
    print('  - If predicted angular velocity << target velocity on moving sources, '
          'suspect frozen DOA estimates.')


# ──────────────────────────────────────────────────────────────────────────────
# 13.  CSV / JSON outputs
# ──────────────────────────────────────────────────────────────────────────────

def save_outputs(args, model_names: List[str],
                 global_metrics: Dict[str, Dict],
                 perclass_data:  Dict[str, List],
                 motion_data:    Dict[str, Dict],
                 perrecording:   Dict[str, List],
                 coord_checks:   Dict[str, Dict],
                 smooth_data:    Dict[str, Dict],
                 out_dir: Path) -> None:
    """Serialize outputs outputs."""

    out_dir.mkdir(parents=True, exist_ok=True)
    if not _PANDAS:
        print('  pandas not available — skipping CSV export.')
        return

    # 1. doa_oracle_summary.csv
    rows = []
    for name in model_names:
        gm  = global_metrics.get(name, {})
        cc  = coord_checks.get(name, {})
        ae  = gm.get('ae_stats', {})
        pna = gm.get('pred_norm_active', {})
        pni = gm.get('pred_norm_inactive', {})
        pna_all = gm.get('pred_norm_all', {})
        tna = gm.get('target_norm_active', {})
        best = cc.get('best', '')
        btr  = cc.get('transforms', {}).get(best, {})
        rows.append({
            'dataset':        args.dataset,
            'split':          args.split,
            'pool_size':      args.pool_size,
            'sample_policy':  args.sample_policy,
            'seed':           args.seed,
            'model_name':     name,
            'evaluated_slots':gm.get('n_active', 0),
            'mean_AE':        ae.get('mean', float('nan')),
            'median_AE':      ae.get('median', float('nan')),
            'std_AE':         ae.get('std', float('nan')),
            'min_AE':         ae.get('min', float('nan')),
            'p25_AE':         ae.get('p25', float('nan')),
            'p75_AE':         ae.get('p75', float('nan')),
            'p90_AE':         ae.get('p90', float('nan')),
            'p95_AE':         ae.get('p95', float('nan')),
            'max_AE':         ae.get('max', float('nan')),
            **{f'acc_{int(th)}deg': gm.get(f'acc_{int(th)}deg', float('nan'))
               for th in args.angle_thresholds},
            'pred_norm_all_mean':        pna_all.get('mean', float('nan')),
            'pred_norm_all_median':      pna_all.get('median', float('nan')),
            'pred_norm_active_mean':     pna.get('mean', float('nan')),
            'pred_norm_active_median':   pna.get('median', float('nan')),
            'pred_norm_inactive_mean':   pni.get('mean', float('nan')),
            'pred_norm_inactive_median': pni.get('median', float('nan')),
            'target_norm_active_mean':   tna.get('mean', float('nan')),
            'target_norm_active_median': tna.get('median', float('nan')),
            **{f'zero_norm_active_lt_{th:.2f}': gm.get(f'zero_norm_active_lt_{th:.2f}', float('nan'))
               for th in args.zero_norm_thresholds},
            'best_coord_transform':          best,
            'best_coord_transform_median_AE':btr.get('median_AE', float('nan')),
            'coordinate_mismatch_suspected': cc.get('mismatch_suspected', False),
        })
    pd.DataFrame(rows).to_csv(out_dir / 'doa_oracle_summary.csv',
                               index=False, float_format='%.6f')
    print(f'  Saved doa_oracle_summary.csv  ({len(rows)} rows)')

    # 2. doa_per_class_summary.csv
    pc_rows = []
    for name in model_names:
        for r in perclass_data.get(name, []):
            pc_rows.append({
                'dataset': args.dataset, 'split': args.split, 'model_name': name,
                **{k: r.get(k, float('nan')) for k in
                   ['class_idx', 'class_name', 'active_slots',
                    'mean_AE', 'median_AE', 'p90_AE', 'acc_20deg',
                    'pred_norm_median', 'zero_norm_lt010']},
            })
    if pc_rows:
        pd.DataFrame(pc_rows).to_csv(out_dir / 'doa_per_class_summary.csv',
                                      index=False, float_format='%.6f')
        print(f'  Saved doa_per_class_summary.csv  ({len(pc_rows)} rows)')

    # 3. doa_motion_summary.csv
    mo_rows = []
    for name in model_names:
        md = motion_data.get(name, {})
        for mtype in ('static', 'quasi_static', 'moving'):
            d = md.get(mtype, {})
            mo_rows.append({
                'dataset': args.dataset, 'split': args.split, 'model_name': name,
                'motion_type': mtype,
                **{k: d.get(k, float('nan')) for k in
                   ['n_events', 'n_slots', 'mean_AE', 'median_AE', 'p90_AE',
                    'acc_20deg', 'pred_norm_median', 'zero_norm_lt01']},
            })
    if mo_rows:
        pd.DataFrame(mo_rows).to_csv(out_dir / 'doa_motion_summary.csv',
                                      index=False, float_format='%.6f')
        print(f'  Saved doa_motion_summary.csv  ({len(mo_rows)} rows)')

    # 4. doa_recording_summary.csv
    rec_rows = []
    for name in model_names:
        for r in perrecording.get(name, []):
            rec_rows.append({
                'dataset': args.dataset, 'split': args.split, 'model_name': name,
                **{k: r.get(k, float('nan')) for k in
                   ['recording_id', 'active_slots', 'mean_AE', 'median_AE',
                    'p90_AE', 'acc_20deg', 'pred_norm_median', 'zero_norm_lt010']},
            })
    if rec_rows:
        pd.DataFrame(rec_rows).to_csv(out_dir / 'doa_recording_summary.csv',
                                       index=False, float_format='%.6f')
        print(f'  Saved doa_recording_summary.csv  ({len(rec_rows)} rows)')

    # 5. doa_diagnostic_summary.json
    summary = {
        'run_config': {
            'dataset': args.dataset, 'split': args.split,
            'pool_size': args.pool_size, 'sample_policy': args.sample_policy,
            'seed': args.seed, 'device': str(args.device),
            'angle_thresholds': args.angle_thresholds,
            'zero_norm_thresholds': args.zero_norm_thresholds,
            'static_thresh_deg': args.static_thresh_deg,
            'moving_thresh_deg': args.moving_thresh_deg,
            'align_standalone': bool(args.align_standalone),
        },
        'models': {
            name: {
                'global_metrics': global_metrics.get(name, {}),
                'coord_check': {k: v for k, v in coord_checks.get(name, {}).items()
                                if k != 'transforms'},
                'motion_summary': motion_data.get(name, {}),
                'temporal_smoothness': smooth_data.get(name, {}),
            }
            for name in model_names
        },
    }

    def _serial(obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, np.bool_):    return bool(obj)
        return obj

    with open(out_dir / 'doa_diagnostic_summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2, default=_serial)
    print('  Saved doa_diagnostic_summary.json')


# ──────────────────────────────────────────────────────────────────────────────
# 14.  CLI + main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    p = argparse.ArgumentParser(
        description='DOA Oracle Activity Diagnostic — evaluates DOA quality '
                    'using ground-truth activity as the evaluation mask.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset
    p.add_argument('--dataset', default='starss23',
                   choices=['starss23', 'balanced_starss23', 'tau_nigens', 'tau2019'])
    p.add_argument('--split',   default='test',
                   choices=['train', 'val', 'test'])
    p.add_argument('--root',    type=Path, required=True,
                   help='Root directory of the dataset.')
    p.add_argument('--val-seed', type=int, default=2)
    # Pool
    p.add_argument('--pool-size',      type=int,   default=256,
                   help='Max samples to evaluate. <=0 → full split.')
    p.add_argument('--sample-policy',  default='random',
                   choices=['first', 'random', 'balanced-active'])
    p.add_argument('--seed',           type=int,   default=2)
    p.add_argument('--batch-size',     type=int,   default=8)
    p.add_argument('--num-workers',    type=int,   default=4)
    p.add_argument('--device',         default='auto',
                   choices=['auto', 'cpu', 'cuda', 'mps'],
                   help='Compute device: auto | cpu | cuda | mps')
    # Models
    p.add_argument('--nas-config',      type=Path, default=None)
    p.add_argument('--nas-checkpoint',  type=Path, default=None)
    p.add_argument('--conformer-config',      type=Path, default=None)
    p.add_argument('--conformer-checkpoint',  type=Path, default=None)
    p.add_argument('--resnet-config',      type=Path, default=None)
    p.add_argument('--resnet-checkpoint',  type=Path, default=None)
    p.add_argument('--models', nargs='+',
                   choices=['nas_seld_x1', 'standalone_conformer', 'standalone_resnet'],
                   default=['nas_seld_x1', 'standalone_conformer', 'standalone_resnet'])
    p.add_argument('--align-standalone', action='store_true',
                   help="PIT-align the DOA slots of the standalone models with the checkpoint's tPITLoss, as "
                        "for NAS-SELD and as the training-time test of StandaloneSELDModule does. Default off = "
                        "historical behaviour of the stored results (raw slot order, not comparable with NAS-SELD).")
    # Diagnostics
    p.add_argument('--coordinate-sanity-check', action='store_true')
    p.add_argument('--angle-thresholds',     type=float, nargs='+',
                   default=[10, 20, 30, 45, 90])
    p.add_argument('--zero-norm-thresholds', type=float, nargs='+',
                   default=[0.05, 0.10, 0.20, 0.50])
    p.add_argument('--static-thresh-deg',  type=float, default=5.0)
    p.add_argument('--moving-thresh-deg',  type=float, default=20.0)
    p.add_argument(
        '--output-dir',
        type=Path,
        default=ROOT / 'results' / 'diagnostics' / 'doa_oracle_outputs',
    )
    return p.parse_args()


def main() -> None:
    """Run the command-line workflow."""

    args = parse_args()
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

    print(_SEP)
    print('DOA ORACLE DIAGNOSTIC TEST')
    print(_SEP)
    print('\nRepository diagnostic script: analysis/diagnostics/doa_oracle_test.py')
    print('\nEvaluation mode:')
    print('  ORACLE_ACTIVITY')
    print('  Ground-truth activity is used as the evaluation mask.')
    print('  SED logits and act_thresh are NOT used for DOA gating.')

    # ── Model spec map ────────────────────────────────────────────────────────
    model_spec_map = {
        'nas_seld_x1':         (args.nas_config,       args.nas_checkpoint,       True),
        'standalone_conformer':(args.conformer_config, args.conformer_checkpoint, False),
        'standalone_resnet':   (args.resnet_config,    args.resnet_checkpoint,    False),
    }
    active_models = [m for m in args.models
                     if model_spec_map[m][0] is not None
                     and model_spec_map[m][1] is not None]
    if not active_models:
        print('\nERROR: No model configs/checkpoints provided. '
              'Pass at least one of --nas-config/--nas-checkpoint, etc.')
        sys.exit(1)

    print(f'\nDataset:')
    print(f'  name:   {args.dataset}')
    print(f'  root:   {args.root}')
    print(f'  split:  {args.split}')
    print(f'\nModels:')
    for i, name in enumerate(active_models, 1):
        cfg_p, ckpt_p, _ = model_spec_map[name]
        print(f'  {i}. {name}')
        print(f'       config:     {cfg_p}')
        print(f'       checkpoint: {ckpt_p}')
    print(f'\nDiagnostics:')
    print(f'  coordinate sanity check: {"enabled" if args.coordinate_sanity_check else "disabled"}')
    print(f'  angle thresholds:    {args.angle_thresholds}°')
    print(f'  zero norm thresholds:{args.zero_norm_thresholds}')
    print(f'  static threshold:    {args.static_thresh_deg}°')
    print(f'  moving threshold:    {args.moving_thresh_deg}°')
    print(f'\nDevice:           {device}')
    print(f'\nOutput directory: {args.output_dir}')
    print(f'\n{_SEP}')

    # ── Results containers ────────────────────────────────────────────────────
    global_metrics:  Dict[str, Dict]  = {}
    perclass_data:   Dict[str, List]  = {}
    motion_data:     Dict[str, Dict]  = {}
    perrecording:    Dict[str, List]  = {}
    coord_checks:    Dict[str, Dict]  = {}
    smooth_data:     Dict[str, Dict]  = {}
    class_names_used: List[str]        = CLASS_NAMES.get(args.dataset, [])

    # ── Per-model loop ────────────────────────────────────────────────────────
    for logical_name in active_models:
        cfg_path, ckpt_path, is_nas = model_spec_map[logical_name]

        print(f'\n{_DASH}')
        print(f'Loading model: {logical_name}')
        print(_DASH)
        print(f'  Config:     {cfg_path}')
        print(f'  Checkpoint: {ckpt_path}')

        try:
            if is_nas:
                model, cfg, orig_sr, tpit_loss = load_nas_model(cfg_path, ckpt_path, device)
            else:
                if args.align_standalone:
                    model, cfg, orig_sr, tpit_loss = load_standalone_model(
                        cfg_path, ckpt_path, device, return_tpit=True)
                else:
                    model, cfg, orig_sr = load_standalone_model(cfg_path, ckpt_path, device)
                    tpit_loss = None
        except Exception as exc:
            print(f'  ERROR loading {logical_name}: {exc}')
            import traceback; traceback.print_exc()
            continue

        total_p, train_p = count_parameters(model)
        sr, hop, n_tracks, window_s, hop_s = _extract_model_params(cfg)
        print(f'  Parameters: {total_p:,} total  /  {train_p:,} trainable')
        print(f'  Device:     {device}')
        print(f'  Input sr:   {orig_sr} Hz   hop_length: {hop}')
        print(f'  Status:     loaded ✓')

        # ── Build dataset & pool ─────────────────────────────────────────────
        try:
            dataset = build_dataset(args.dataset, args.split, args.root,
                                    cfg, val_seed=args.val_seed)
        except Exception as exc:
            print(f'  ERROR building dataset: {exc}')
            continue

        n_split = len(dataset)
        if logical_name == active_models[0]:
            print(f'\nDataset:')
            print(f'  total samples in split:  {n_split:,}')

        pool_indices = select_pool_indices(dataset, args.pool_size,
                                          args.sample_policy, args.seed)
        eff = len(pool_indices)
        if logical_name == active_models[0]:
            print(f'  requested pool size:     '
                  f'{"ALL" if args.pool_size <= 0 else args.pool_size}')
            print(f'  effective pool size:     {eff:,}')
            print(f'  sample policy:           {args.sample_policy}')
            print(f'  seed:                    {args.seed}')

        rec_ids = get_recording_ids(dataset, pool_indices)

        from torch.utils.data import Subset
        from datasets.utils import SELDDataset
        subset   = Subset(dataset, pool_indices)
        windowed = getattr(dataset, 'windowed', False)
        collate  = None if windowed else SELDDataset.collate_fn
        bs       = args.batch_size if windowed else 1
        loader   = torch.utils.data.DataLoader(
            subset, batch_size=bs, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate,
        )

        # ── Inference ────────────────────────────────────────────────────────
        print(f'\n  Running inference on {eff:,} samples '
              f'(windowed={windowed}, batch_size={bs})…')
        try:
            doa_list, tgt_list, act_list, n_frames = run_inference(
                model, loader, device, orig_sr, hop, is_nas, logical_name, tpit_loss)
        except Exception as exc:
            print(f'  ERROR during inference: {exc}')
            import traceback; traceback.print_exc()
            continue

        if not doa_list:
            print('  ERROR: no batches processed.')
            continue

        sample0 = doa_list[0]  # (B, T, N, C, 3)
        B0, T0, N0, C0, _ = sample0.shape
        n_classes_found = C0
        if not class_names_used:
            class_names_used = [f'class_{i}' for i in range(n_classes_found)]

        # ── Build cache ──────────────────────────────────────────────────────
        cache = DOACache(
            doa_list=doa_list,
            targets_list=tgt_list,
            act_logits_list=act_list,
            recording_ids=rec_ids,
            n_classes=n_classes_found,
            n_tracks=N0,
            n_frames=n_frames,
            orig_sr=orig_sr,
            hop_length=hop,
            model_name=logical_name,
        )
        cache.build_flat_active()

        print(f'\n{_DASH}')
        print(f'Inference summary: {logical_name}')
        print(_DASH)
        print(f'  Samples processed:    {eff:,}')
        print(f'  Batches processed:    {len(doa_list):,}')
        print(f'  Total frames (B×T):   {n_frames:,}')
        print(f'  Classes / tracks:     {n_classes_found} / {N0}')
        print(f'\n  Raw model output shape (first batch):')
        print(f'    doa_tanh: {list(sample0.shape)}  (B, T, N_tracks={N0}, C={C0}, xyz=3)')
        tgt0 = tgt_list[0]
        print(f'    target:   {list(tgt0.shape)}')
        if cache.n_active == 0:
            print(f'\n  WARNING: No target-active slots found in selected pool.')
            print('  Use a larger pool or balanced-active sample policy.')
            continue
        tgt_total = n_frames * N0 * C0
        print(f'\n  Target-active slots:')
        print(f'    count:      {cache.n_active:,}')
        print(f'    percentage: {100*cache.n_active/max(tgt_total,1):.2f}% of all {tgt_total:,} slots')

        # ── Compute metrics ──────────────────────────────────────────────────
        print(f'\n  Computing global DOA oracle metrics…')
        gm = compute_global_metrics(cache, args.angle_thresholds,
                                    args.zero_norm_thresholds)
        global_metrics[logical_name] = gm
        print_global_report(logical_name, gm, args.angle_thresholds,
                            args.zero_norm_thresholds)

        if args.coordinate_sanity_check:
            print(f'\n  Running coordinate sanity check…')
            cc = compute_coord_sanity(cache)
            coord_checks[logical_name] = cc
            print_coord_report(logical_name, cc)
        else:
            coord_checks[logical_name] = {}

        print(f'\n  Classifying static/moving trajectories…')
        md = classify_segments(cache, args.static_thresh_deg, args.moving_thresh_deg)
        motion_data[logical_name] = md
        print_motion_report(logical_name, md)

        print(f'\n  Computing temporal smoothness…')
        sd = compute_temporal_smoothness(cache)
        smooth_data[logical_name] = sd
        print_smoothness_report(logical_name, sd)

        print(f'\n  Computing per-class metrics…')
        pc = compute_perclass(cache, class_names_used)
        perclass_data[logical_name] = pc
        print_perclass_report(logical_name, pc)

        print(f'\n  Computing per-recording metrics…')
        pr = compute_perrecording(cache)
        perrecording[logical_name] = pr
        print_perrecording_report(logical_name, pr)

        # ── Free GPU memory ──────────────────────────────────────────────────
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()

    # ── Final cross-model report ──────────────────────────────────────────────
    if active_models:
        print_final_comparison(active_models, global_metrics, coord_checks)
        print_diagnostic_summary(active_models, global_metrics, coord_checks,
                                 motion_data, smooth_data)

        print(f'\n{_DASH}')
        print('Saving outputs…')
        save_outputs(args, active_models, global_metrics, perclass_data,
                     motion_data, perrecording, coord_checks, smooth_data,
                     args.output_dir)
        print('Done.')


if __name__ == '__main__':
    main()
