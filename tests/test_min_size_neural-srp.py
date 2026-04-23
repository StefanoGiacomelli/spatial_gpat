#!/usr/bin/env python3
"""
test_min_size_neural-srp.py — Binary search for minimum input window size
=======================================================================
Empirically determines the smallest T_samples accepted by:
  1. NeuralSRPEncoder (encoder only)
  2. SpatialSRPModel  (encoder + DOA/activity/distance heads)

for both eval-mode (forward only) and train-mode (forward + backward pass).

No dataset required — all inputs are synthetic random tensors.

Usage
-----
    python tests/test_min_size_neural-srp.py
    python tests/test_min_size_neural-srp.py --config-file configs/experiments_spatial/mic/srp_baseline.yaml

Architecture notes
------------------
The theoretical lower bound is T_samples = win_size (one output frame):
    T_frames = (T_samples - win_size) // hop_size + 1
All subsequent layers (Conv2d k=3 p=1, GRU, Linear) accept T_frames=1.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

import argparse
import sys
import traceback
from pathlib import Path

import torch
import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from spatial_modules.neural_srp_modules import NeuralSRPEncoder, SpatialSRPModel


# ── Default config (mirrors srp_baseline.yaml) ────────────────────────────────

_DEFAULTS = dict(
    # encoder
    n_mics            = 4,
    win_size          = 480,
    hop_size          = 240,
    sr                = 24000,
    tau_max           = 32,
    n_conv_channels   = 128,
    n_conv_layers     = 3,
    f_pool_size       = [2, 2, 2],
    t_pool_size       = [1, 1, 1],
    dropout_rate      = 0.1,
    rnn_hidden        = 128,
    n_rnn_layers      = 2,
    fnn_pairwise_size = 128,
    d_out             = 256,
    pair_agg_mode     = 'sum',
    metadata_fusion   = 'late_concat',
    bidirectional     = True,
    # heads
    n_tracks          = 2,
    # test
    batch_size        = 2,
)


def _load_cfg(config_file: Path) -> dict:
    with open(config_file) as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = dict(_DEFAULTS)
    enc = raw.get('encoder', {})
    for k in ('n_mics', 'win_size', 'hop_size', 'tau_max', 'n_conv_channels',
              'n_conv_layers', 'f_pool_size', 't_pool_size', 'dropout_rate',
              'rnn_hidden', 'n_rnn_layers', 'fnn_pairwise_size', 'd_out',
              'pair_agg_mode', 'metadata_fusion', 'bidirectional'):
        if k in enc:
            cfg[k] = enc[k]
    cfg['sr'] = int(raw.get('data', {}).get('sr', cfg['sr']))
    cfg['n_tracks'] = int(raw.get('heads', {}).get('n_tracks', cfg['n_tracks']))
    return cfg


# ── Model builders ────────────────────────────────────────────────────────────

def _build_encoder(cfg: dict) -> NeuralSRPEncoder:
    return NeuralSRPEncoder(
        n_mics            = cfg['n_mics'],
        win_size          = cfg['win_size'],
        hop_size          = cfg['hop_size'],
        tau_max           = cfg['tau_max'],
        n_conv_channels   = cfg['n_conv_channels'],
        n_conv_layers     = cfg['n_conv_layers'],
        f_pool_size       = cfg['f_pool_size'],
        t_pool_size       = cfg['t_pool_size'],
        dropout_rate      = cfg['dropout_rate'],
        rnn_hidden        = cfg['rnn_hidden'],
        n_rnn_layers      = cfg['n_rnn_layers'],
        fnn_pairwise_size = cfg['fnn_pairwise_size'],
        d_out             = cfg['d_out'],
        pair_agg_mode     = cfg['pair_agg_mode'],
        metadata_fusion   = cfg['metadata_fusion'],
        bidirectional     = cfg['bidirectional'],
    )


def _build_full_model(cfg: dict) -> SpatialSRPModel:
    encoder_cfg = {k: cfg[k] for k in (
        'n_mics', 'win_size', 'hop_size', 'tau_max', 'n_conv_channels',
        'n_conv_layers', 'f_pool_size', 't_pool_size', 'dropout_rate',
        'rnn_hidden', 'n_rnn_layers', 'fnn_pairwise_size', 'd_out',
        'pair_agg_mode', 'metadata_fusion', 'bidirectional',
    )}
    return SpatialSRPModel(encoder_cfg, n_tracks=cfg['n_tracks'])


# ── Check functions ───────────────────────────────────────────────────────────

def _synthetic_inputs(T_samples: int, cfg: dict):
    B  = cfg['batch_size']
    M  = cfg['n_mics']
    wav     = torch.randn(B, M, T_samples)
    mic_pos = torch.randn(B, M, 3) * 0.05   # ~5cm spread
    return wav, mic_pos


def _is_finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all())


def _check_encoder_eval(T: int, cfg: dict) -> bool:
    try:
        model = _build_encoder(cfg).eval()
        wav, mic_pos = _synthetic_inputs(T, cfg)
        with torch.no_grad():
            out = model(wav, mic_pos)
        expected_T = (T - cfg['win_size']) // cfg['hop_size'] + 1
        assert out.shape == (cfg['batch_size'], expected_T, cfg['d_out']), \
            f"shape mismatch: {out.shape}"
        return _is_finite(out)
    except Exception:
        return False


def _check_encoder_train(T: int, cfg: dict) -> bool:
    try:
        model = _build_encoder(cfg).train()
        wav, mic_pos = _synthetic_inputs(T, cfg)
        out  = model(wav, mic_pos)
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            if p.grad is not None and not _is_finite(p.grad):
                return False
        return _is_finite(out)
    except Exception:
        return False


def _check_full_eval(T: int, cfg: dict) -> bool:
    try:
        model = _build_full_model(cfg).eval()
        wav, mic_pos = _synthetic_inputs(T, cfg)
        with torch.no_grad():
            pred = model(wav, mic_pos)
        expected_T = (T - cfg['win_size']) // cfg['hop_size'] + 1
        B, N = cfg['batch_size'], cfg['n_tracks']
        assert pred['doa'].shape      == (B, expected_T, N, 3)
        assert pred['activity'].shape == (B, expected_T, N)
        assert pred['distance'].shape == (B, expected_T, N)
        return (
            _is_finite(pred['doa'])
            and _is_finite(pred['activity'])
            and _is_finite(pred['distance'])
        )
    except Exception:
        return False


def _check_full_train(T: int, cfg: dict) -> bool:
    try:
        model = _build_full_model(cfg).train()
        wav, mic_pos = _synthetic_inputs(T, cfg)
        pred = model(wav, mic_pos)
        loss = pred['doa'].sum() + pred['activity'].sum() + pred['distance'].sum()
        loss.backward()
        for p in model.parameters():
            if p.grad is not None and not _is_finite(p.grad):
                return False
        return True
    except Exception:
        return False


# ── Binary search ─────────────────────────────────────────────────────────────

def binary_search_min(check_fn, low: int, high: int) -> int:
    """Return smallest T in [low, high] where check_fn(T) is True, or high+1 if none."""
    if not check_fn(high):
        return high + 1   # even max size fails — architecture issue
    while low < high:
        mid = (low + high) // 2
        if check_fn(mid):
            high = mid
        else:
            low = mid + 1
    return low


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Binary search for minimum input size accepted by the spatial model.'
    )
    p.add_argument('--config-file', type=str, default=None,
                   help='Path to experiment YAML. Defaults match srp_baseline.yaml.')
    args = p.parse_args()

    cfg = _DEFAULTS.copy()
    if args.config_file:
        cfg = _load_cfg(Path(args.config_file))

    win_size = cfg['win_size']
    hop_size = cfg['hop_size']
    sr       = cfg['sr']
    max_T    = 10 * sr   # 10 seconds

    print(f'\n{"=" * 60}')
    print(f'NeuralSRP minimum input size binary search')
    print(f'{"=" * 60}')
    print(f'  win_size={win_size}  hop_size={hop_size}  sr={sr}')
    print(f'  search range: [{win_size}, {max_T}] samples')
    print(f'  batch_size={cfg["batch_size"]}  n_tracks={cfg["n_tracks"]}')
    print()

    cases = [
        ('Encoder  eval ', _check_encoder_eval),
        ('Encoder  train', _check_encoder_train),
        ('FullModel eval ', _check_full_eval),
        ('FullModel train', _check_full_train),
    ]

    results = []
    for label, fn in cases:
        print(f'  Searching: {label} ...', end='', flush=True)
        check = lambda T, _fn=fn: _fn(T, cfg)
        min_T  = binary_search_min(check, win_size, max_T)
        if min_T > max_T:
            print(f'  FAILED (even {max_T} samples fails!)')
            results.append((label, None, None, None))
        else:
            T_frames = (min_T - win_size) // hop_size + 1
            dur_ms   = 1000.0 * min_T / sr
            print(f'  {min_T} samples ({dur_ms:.1f} ms, {T_frames} frame(s))')
            results.append((label, min_T, dur_ms, T_frames))

    print()
    print(f'{"─" * 60}')
    print(f'{"Case":<20} {"min_T_samples":>14} {"duration_ms":>12} {"T_frames":>10}')
    print(f'{"─" * 60}')
    for label, min_T, dur_ms, T_frames in results:
        if min_T is None:
            print(f'{label:<20} {"FAILED":>14}')
        else:
            print(f'{label:<20} {min_T:>14d} {dur_ms:>11.1f} {T_frames:>10d}')
    print(f'{"─" * 60}')

    # Practical recommendation
    actual_min = max((r[1] for r in results if r[1] is not None), default=None)
    if actual_min is not None:
        dur_s  = actual_min / sr
        frames = (actual_min - win_size) // hop_size + 1
        annot_res_ms = hop_size / sr * 1000   # annotation frame duration in ms
        n_annot = max(1, round(actual_min / sr / (hop_size / sr)))
        print()
        print('Practical recommendation')
        print(f'  Absolute minimum     : {actual_min} samples = {dur_s*1000:.1f} ms '
              f'({frames} encoder frame(s))')
        print(f'  Annotation resolution: {annot_res_ms:.1f} ms / frame '
              f'(hop_size={hop_size} @ {sr} Hz)')
        print(f'  For ≥1 annotation frame : {hop_size} samples = {hop_size/sr*1000:.1f} ms')
        print(f'  For ≥10 annotation frames: {10*hop_size} samples = {10*hop_size/sr*1000:.1f} ms '
              f'(≈ {10*hop_size/sr:.2f} s)')
        print(f'  Recommended minimum  : ≥ 1 s ({sr} samples) for meaningful BiGRU context')
    print()


if __name__ == '__main__':
    main()
