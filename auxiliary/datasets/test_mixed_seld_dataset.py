#!/usr/bin/env python3
"""
auxiliary/datasets/test_mixed_seld_dataset.py
==============================================
End-to-end functional test for MixedSELDDataset (STARSS23 + TAU-NIGENS2021 +
TAU2019) in both windowed (10 s / 5 s hop) and full-clip modes.

What is reported
----------------
  ① Split statistics (annotation-based, no audio loaded):

       Split layout matching MixedSELDDataModule:
         Train  : STARSS23 train + TAU-NIGENS2021 train + TAU2019 train
         Val    : STARSS23 val  + TAU2019 val
         Test   : STARSS23 test + TAU2019 test

       For each dataset × split:
         • n_clips, n_items (windowed and full-clip side by side)
         • Per-class frame counts and % of total (15-class unified schema)
         • Imbalance ratio

  ② Batch-shape verification (loads actual audio):
       • DataLoader in windowed and/or full-clip mode
       • Checks wav dtype/shape (4 ch FOA), target shape (…, 15, 3)

Output
------
  results/mixed_seld_stats.json

Run
---
    python auxiliary/datasets/test_mixed_seld_dataset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datasets.starss23   import STARSS23Dataset
from datasets.tau_nigens import TAUNIGENSDataset
from datasets.tau2019    import TAU2019Dataset
from datasets.utils      import (MixedSELDDataset, SELDDataset,
                                  STARSS_TO_UNIFIED, NIGENS_TO_UNIFIED,
                                  TAU2019_TO_UNIFIED)

# ── Constants ──────────────────────────────────────────────────────────────────

STARSS_ROOT  = _ROOT / 'data' / 'STARSS23'
NIGENS_ROOT  = _ROOT / 'data' / 'TAU-NIGENS2021'
TAU2019_ROOT = _ROOT / 'data' / 'TAU2019'
RESULTS_DIR  = _ROOT / 'results'

MODEL_SR     = 24_000
HOP_LENGTH   = 240
N_TRACKS     = 3
N_CLASSES    = 15        # unified schema
VAL_FRACTION = 0.2
VAL_SEED     = 42
WINDOW_S     = 10.0
HOP_S        = 5.0

CLASS_NAMES = [
    'speech', 'clapping', 'telephone', 'laughter', 'domestic',
    'footsteps', 'door', 'music', 'water_tap', 'bell', 'knock',
    'alarm', 'crying_baby', 'crash', 'barking_dog',
]


# ── Dataset factory ────────────────────────────────────────────────────────────

def _starss(split: str, window_s: Optional[float],
            hop_s: Optional[float]) -> STARSS23Dataset:
    return STARSS23Dataset(
        root         = STARSS_ROOT,
        split        = split,
        model_sr     = MODEL_SR,
        hop_length   = HOP_LENGTH,
        n_tracks     = N_TRACKS,
        window_s     = window_s,
        hop_s        = hop_s,
        augment      = None,
        val_fraction = VAL_FRACTION,
        val_seed     = VAL_SEED,
    )


def _nigens(split: str, window_s: Optional[float],
            hop_s: Optional[float]) -> TAUNIGENSDataset:
    return TAUNIGENSDataset(
        root       = NIGENS_ROOT,
        split      = split,
        model_sr   = MODEL_SR,
        hop_length = HOP_LENGTH,
        n_tracks   = N_TRACKS,
        window_s   = window_s,
        hop_s      = hop_s,
        augment    = None,
    )


def _tau2019(split: str, window_s: Optional[float],
             hop_s: Optional[float]) -> TAU2019Dataset:
    return TAU2019Dataset(
        root       = TAU2019_ROOT,
        split      = split,
        model_sr   = MODEL_SR,
        hop_length = HOP_LENGTH,
        n_tracks   = N_TRACKS,
        window_s   = window_s,
        hop_s      = hop_s,
        augment    = None,
        mixed_mode = True,
    )


def _mixed(starss: STARSS23Dataset,
           nigens: TAUNIGENSDataset,
           tau:    TAU2019Dataset) -> MixedSELDDataset:
    return MixedSELDDataset(
        datasets = [starss, nigens, tau],
        mappings = [STARSS_TO_UNIFIED, NIGENS_TO_UNIFIED, TAU2019_TO_UNIFIED],
    )


# ── Per-class frame counting ───────────────────────────────────────────────────

def _count_frames_unified(ds, mapping, n_src_cls: int) -> Dict[int, int]:
    """
    Count unique (clip_idx, frame_100ms) pairs per *unified* class.
    Uses the dataset's own class space and remaps to unified.
    """
    counts: Dict[int, int] = {c: 0 for c in range(N_CLASSES)}
    for clip_idx, events in enumerate(ds._events):
        seen: set = set()
        for frame_100ms, class_idx, *_ in events:
            if class_idx >= n_src_cls:
                continue
            key = (clip_idx, frame_100ms, class_idx)
            if key not in seen:
                seen.add(key)
                u_cls = int(mapping[class_idx])
                counts[u_cls] += 1
    for c in range(N_CLASSES):
        if counts[c] == 0:
            counts[c] = 1
    return counts


# ── Display helpers ────────────────────────────────────────────────────────────

W = 76

def _sep(char='─') -> str:
    return char * W


def _print_class_table(counts: Dict[int, int], header: str) -> None:
    total = sum(counts.values())
    print(f'\n  {header}')
    print(_sep())
    print(f'  {"idx":>3}  {"unified class":22}  {"frames":>8}  {"% total":>8}')
    print(_sep())
    for idx in range(N_CLASSES):
        c   = counts[idx]
        pct = 100.0 * c / total
        print(f'  {idx:>3}  {CLASS_NAMES[idx]:22}  {c:>8,}  {pct:>7.2f}%')
    print(_sep())
    max_c = max(counts.values())
    min_c = min(counts.values())
    print(f'  Imbalance ratio  max/min: {max_c / min_c:.1f}×'
          f'   total active frames: {total:,}')


def _item_row(label: str, n_clips: int, n_win: int, n_fc: int) -> None:
    print(f'  {label:40s}  {n_clips:>6}  {n_win:>12}  {n_fc:>10}')


# ── Section ①: split statistics ───────────────────────────────────────────────

def print_split_statistics() -> dict:
    print(f'\n{"█" * W}')
    print('  SPLIT STATISTICS  (annotation-based, no audio loaded)')
    print(f'  MixedSELDDataModule layout (mixed_mode=True for TAU2019):')
    print(f'    Train = STARSS23 train + TAU-NIGENS2021 train + TAU2019 train')
    print(f'    Val   = STARSS23 val   + TAU2019 val')
    print(f'    Test  = STARSS23 test  + TAU2019 test')
    print(f'{"█" * W}')

    print('\n  Building datasets …', end='', flush=True)

    s_tr_w  = _starss('train', WINDOW_S, HOP_S)
    s_vl_w  = _starss('val',   WINDOW_S, HOP_S)
    s_te_w  = _starss('test',  WINDOW_S, HOP_S)
    n_tr_w  = _nigens('train', WINDOW_S, HOP_S)
    t_tr_w  = _tau2019('train', WINDOW_S, HOP_S)
    t_vl_w  = _tau2019('val',   WINDOW_S, HOP_S)
    t_te_w  = _tau2019('test',  WINDOW_S, HOP_S)

    s_tr_fc = _starss('train', None, None)
    s_vl_fc = _starss('val',   None, None)
    s_te_fc = _starss('test',  None, None)
    n_tr_fc = _nigens('train', None, None)
    t_tr_fc = _tau2019('train', None, None)
    t_vl_fc = _tau2019('val',   None, None)
    t_te_fc = _tau2019('test',  None, None)

    mix_tr_w  = _mixed(s_tr_w,  n_tr_w,  t_tr_w)
    mix_vl_w  = _mixed(s_vl_w,  n_tr_w,  t_vl_w)   # nigens has no val split
    mix_te_w  = _mixed(s_te_w,  n_tr_w,  t_te_w)
    mix_tr_fc = _mixed(s_tr_fc, n_tr_fc, t_tr_fc)

    print(' done.')

    print(f'\n  Item counts (windowed: {WINDOW_S:.0f}s / {HOP_S:.0f}s hop)')
    print(_sep())
    print(f'  {"Component":40s}  {"clips":>6}  {"windowed":>12}  {"full-clip":>10}')
    print(_sep())
    _item_row('STARSS23 train',
              len(s_tr_w._events), len(s_tr_w), len(s_tr_fc))
    _item_row('TAU-NIGENS2021 train',
              len(n_tr_w._events), len(n_tr_w), len(n_tr_fc))
    _item_row('TAU2019 train  (mixed_mode splits 2-4)',
              len(t_tr_w._events), len(t_tr_w), len(t_tr_fc))
    _item_row('MIXED train  (all three)',
              len(s_tr_w._events) + len(n_tr_w._events) + len(t_tr_w._events),
              len(mix_tr_w), len(mix_tr_fc))
    print(_sep())
    _item_row('STARSS23 val',  len(s_vl_w._events), len(s_vl_w), len(s_vl_fc))
    _item_row('TAU2019 val  (split 1)',
              len(t_vl_w._events), len(t_vl_w), len(t_vl_fc))
    _item_row('STARSS23 test', len(s_te_w._events), len(s_te_w), len(s_te_fc))
    _item_row('TAU2019 test  (eval/split0)',
              len(t_te_w._events), len(t_te_w), len(t_te_fc))

    # ── Per-class unified distributions ───────────────────────────────────────
    print(f'\n{_sep("═")}')
    print(f'  STARSS23 — train  (→ unified class space)')
    print(_sep('═'))
    c_s_tr = _count_frames_unified(s_tr_w, STARSS_TO_UNIFIED, 13)
    _print_class_table(c_s_tr, 'Per-class frame counts (unified)')

    print(f'\n{_sep("═")}')
    print(f'  TAU-NIGENS2021 — train  (→ unified class space)')
    print(_sep('═'))
    c_n_tr = _count_frames_unified(n_tr_w, NIGENS_TO_UNIFIED, 12)
    _print_class_table(c_n_tr, 'Per-class frame counts (unified)')

    print(f'\n{_sep("═")}')
    print(f'  TAU2019 — train  (→ unified class space, mixed_mode)')
    print(_sep('═'))
    c_t_tr = _count_frames_unified(t_tr_w, TAU2019_TO_UNIFIED, 11)
    _print_class_table(c_t_tr, 'Per-class frame counts (unified)')

    # Combined train
    print(f'\n{_sep("═")}')
    print(f'  COMBINED TRAIN  (STARSS23 + TAU-NIGENS2021 + TAU2019)')
    print(_sep('═'))
    c_all = {c: c_s_tr[c] + c_n_tr[c] + c_t_tr[c] for c in range(N_CLASSES)}
    _print_class_table(c_all, 'Combined per-class frame counts (unified)')

    # Build JSON stats
    stats = {
        'window_s': WINDOW_S,
        'hop_s':    HOP_S,
        'splits': {
            'STARSS23_train':    {'n_clips': len(s_tr_w._events), 'n_windows': len(s_tr_w)},
            'NIGENS_train':      {'n_clips': len(n_tr_w._events), 'n_windows': len(n_tr_w)},
            'TAU2019_train':     {'n_clips': len(t_tr_w._events), 'n_windows': len(t_tr_w)},
            'mixed_train_total': {'n_clips': (len(s_tr_w._events) + len(n_tr_w._events)
                                              + len(t_tr_w._events)),
                                  'n_windows': len(mix_tr_w)},
            'STARSS23_val':      {'n_clips': len(s_vl_w._events), 'n_windows': len(s_vl_w)},
            'TAU2019_val':       {'n_clips': len(t_vl_w._events), 'n_windows': len(t_vl_w)},
            'STARSS23_test':     {'n_clips': len(s_te_w._events), 'n_windows': len(s_te_w)},
            'TAU2019_test':      {'n_clips': len(t_te_w._events), 'n_windows': len(t_te_w)},
        },
        'unified_class_counts': {
            'STARSS23_train':  {CLASS_NAMES[c]: int(c_s_tr[c]) for c in range(N_CLASSES)},
            'NIGENS_train':    {CLASS_NAMES[c]: int(c_n_tr[c]) for c in range(N_CLASSES)},
            'TAU2019_train':   {CLASS_NAMES[c]: int(c_t_tr[c]) for c in range(N_CLASSES)},
            'combined_train':  {CLASS_NAMES[c]: int(c_all[c])  for c in range(N_CLASSES)},
        },
    }
    return stats


# ── Section ②: batch-shape verification ──────────────────────────────────────

def _test_batches(mix: MixedSELDDataset, label: str,
                  n_batches: int, batch_size: int) -> bool:
    print(f'\n{_sep("═")}')
    print(f'  BATCH SHAPE TEST — {label}')
    print(_sep('═'))
    collate = None if mix.windowed else SELDDataset.collate_fn
    print(f'  windowed={mix.windowed}  n_items={len(mix)}'
          f'  batch_size={batch_size}  iterating {n_batches} batches …\n')

    loader = DataLoader(mix, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate, drop_last=True)
    ok = True
    for i, (wav, tgt) in enumerate(loader):
        if i >= n_batches:
            break
        wav_ok   = wav.ndim == 3 and wav.shape[1] == 4
        tgt_ok   = (tgt.ndim == 5 and tgt.shape[2] == N_TRACKS
                    and tgt.shape[3] == N_CLASSES and tgt.shape[4] == 3)
        dtype_ok = wav.dtype == torch.float32 and tgt.dtype == torch.float32
        passed   = wav_ok and tgt_ok and dtype_ok
        ok       = ok and passed
        mark     = '✓' if passed else '✗'
        print(f'  batch {i+1:02d}  {mark}'
              f'  wav={tuple(wav.shape)} {wav.dtype}'
              f'  tgt={tuple(tgt.shape)} {tgt.dtype}')
        if not wav_ok:
            print(f'    [FAIL] wav: expected (B,4,T_samples), got {wav.shape}')
        if not tgt_ok:
            print(f'    [FAIL] tgt: expected (B,T_mel,{N_TRACKS},{N_CLASSES},3),'
                  f' got {tgt.shape}')

    verdict = 'PASS' if ok else 'FAIL'
    print(f'\n  Result: {verdict}')
    return ok


# ── Entry point ────────────────────────────────────────────────────────────────

def _check_roots():
    missing = []
    for name, root in [('STARSS23', STARSS_ROOT),
                        ('TAU-NIGENS2021', NIGENS_ROOT),
                        ('TAU2019', TAU2019_ROOT)]:
        if not root.exists():
            missing.append(f'{name} ({root})')
    return missing


def main() -> None:
    missing = _check_roots()
    if missing:
        for m in missing:
            print(f'[ERROR] Data root not found: {m}')
        sys.exit(1)

    # ── Section ①: annotation stats ──────────────────────────────────────────
    stats = print_split_statistics()

    # Save JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / 'mixed_seld_stats.json'
    with open(json_path, 'w') as fh:
        json.dump(stats, fh, indent=2)
    print(f'\n  JSON → {json_path.relative_to(_ROOT)}')

    # ── Section ②: batch shape tests ─────────────────────────────────────────
    print(f'\n{"█" * W}')
    print(f'  AUDIO BATCH TESTS — WINDOWED  ({WINDOW_S:.0f}s / {HOP_S:.0f}s hop)')
    print(f'{"█" * W}')
    mix_w = _mixed(
        _starss('train',  WINDOW_S, HOP_S),
        _nigens('train',  WINDOW_S, HOP_S),
        _tau2019('train', WINDOW_S, HOP_S),
    )
    ok_w = _test_batches(mix_w, 'WINDOWED', n_batches=3, batch_size=2)

    print(f'\n{"█" * W}')
    print(f'  AUDIO BATCH TESTS — FULL-CLIP')
    print(f'{"█" * W}')
    mix_fc = _mixed(
        _starss('train',  None, None),
        _nigens('train',  None, None),
        _tau2019('train', None, None),
    )
    ok_fc = _test_batches(mix_fc, 'FULL-CLIP', n_batches=2, batch_size=1)

    print(f'\n{_sep("═")}')
    verdict = 'ALL PASS' if (ok_w and ok_fc) else 'SOME TESTS FAILED'
    print(f'  OVERALL: {verdict}')
    print(_sep('═'))
    sys.exit(0 if (ok_w and ok_fc) else 1)


if __name__ == '__main__':
    main()
