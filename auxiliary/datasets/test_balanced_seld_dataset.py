#!/usr/bin/env python3
"""
auxiliary/datasets/test_balanced_seld_dataset.py
=================================================
End-to-end functional test and statistics for BalancedSTARSS23Dataset.

The dataset is instantiated exactly as the training pipeline would use it
(no approximations — actual events scanned, actual audio loaded for batch tests).

Sections
--------
  1. Clip-level statistics
       Per-class item counts by source (starss23 / nigens / tau2019 / mixup),
       totals, % of total items, imbalance ratio.
  2. Windowed statistics  (window=10.0 s, hop=2.5 s)
       Per-class window counts by source, no-event window count, imbalance ratio.
  3. DOA statistics
       For each class and combined: mean / std / min / max of azimuth and
       elevation angles (degrees) collected from all full-clip training events.
  4. Batch-shape verification
       Load 3 windowed batches and 2 full-clip batches; check wav + target shapes
       and dtypes.

Output
------
  results/balanced_starss23_stats.json

Run
---
    python auxiliary/datasets/test_balanced_seld_dataset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datasets.balanced import BalancedSTARSS23Dataset
from datasets.utils    import SELDDataset

# ── Constants (matching standalone SELD training configs) ─────────────────────

STARSS_ROOT  = _ROOT / 'data' / 'STARSS23'
NIGENS_ROOT  = _ROOT / 'data' / 'TAU-NIGENS2021'
TAU2019_ROOT = _ROOT / 'data' / 'TAU2019'
RESULTS_DIR  = _ROOT / 'results'

MODEL_SR     = 24_000
HOP_LENGTH   = 240
N_TRACKS     = 3
N_CLASSES    = 13
VAL_FRACTION = 0.2
VAL_SEED     = 2
WINDOW_S     = 10.0
HOP_S        = 2.5
FRAME_DUR    = 0.1   # seconds per annotation frame (100 ms)

SOURCES = ('starss23', 'nigens', 'tau2019', 'mixup')

CLASS_NAMES = [
    'female_speech',    # 0
    'male_speech',      # 1
    'clapping',         # 2
    'telephone',        # 3
    'laughter',         # 4
    'domestic_sounds',  # 5
    'walk_footsteps',   # 6
    'door_open_close',  # 7
    'music',            # 8
    'musical_instr.',   # 9
    'water_tap',        # 10
    'bell',             # 11
    'knock',            # 12
]

W = 80


def _sep(c='─') -> str:
    return c * W


# ── Event helpers ─────────────────────────────────────────────────────────────

def _classes_in_item(item: tuple, ds: BalancedSTARSS23Dataset) -> Tuple[str, set]:
    """Return (source_name, set_of_active_classes) for a full-clip item."""
    src = item[0]
    if src == 'starss23':
        ci   = item[1]
        evts = ds._starss_ds._events[ci]
    elif src == 'nigens':
        ci   = item[1]
        evts = ds._nigens_proj[ci]
    elif src == 'tau2019':
        ci   = item[1]
        evts = ds._tau2019_proj[ci]
    elif src == 'mixup':
        ci_a, ci_b, ds_b_type = item[1], item[2], item[3]
        evts_a = list(ds._starss_ds._events[ci_a])
        if ds_b_type == 'starss23':
            evts_b = list(ds._starss_ds._events[ci_b])
        elif ds_b_type == 'nigens':
            evts_b = list(ds._nigens_proj[ci_b])
        else:  # 'tau2019'
            evts_b = list(ds._tau2019_proj[ci_b])
        evts = evts_a + evts_b
    else:
        return src, set()
    classes = {cls for (_, cls, _, _, _) in evts if 0 <= cls < N_CLASSES}
    return src, classes


def _classes_in_window(item: tuple, ds: BalancedSTARSS23Dataset) -> Tuple[str, set]:
    """Return (source_name, set_of_active_classes) for a windowed item."""
    src = item[0]
    if src == 'starss23':
        ci, start = item[1], item[2]
        evts     = ds._starss_ds._events[ci]
        orig_sr  = ds._starss_ds.orig_sr
        win_samp = ds._win_samp['starss23']
    elif src == 'nigens':
        ci, start = item[1], item[2]
        evts     = ds._nigens_proj[ci]
        orig_sr  = ds._nigens_ds.orig_sr
        win_samp = ds._win_samp['nigens']
    elif src == 'tau2019':
        ci, start = item[1], item[2]
        evts     = ds._tau2019_proj[ci]
        orig_sr  = ds._tau2019_ds.orig_sr
        win_samp = ds._win_samp['tau2019']
    elif src == 'mixup':
        ci_a, ci_b, ds_b_type, start = item[1], item[2], item[3], item[4]
        evts_a = list(ds._starss_ds._events[ci_a])
        if ds_b_type == 'starss23':
            evts_b = list(ds._starss_ds._events[ci_b])
        elif ds_b_type == 'nigens':
            evts_b = list(ds._nigens_proj[ci_b])
        else:  # 'tau2019'
            evts_b = list(ds._tau2019_proj[ci_b])
        evts     = evts_a + evts_b
        orig_sr  = ds._starss_ds.orig_sr   # start is always in STARSS23 SR
        win_samp = ds._win_samp['starss23']
    else:
        return src, set()

    start_s = start / orig_sr
    end_s   = (start + win_samp) / orig_sr
    start_f = int(start_s / FRAME_DUR)
    end_f   = int(end_s   / FRAME_DUR)
    classes = {cls for (f, cls, _, _, _) in evts
               if start_f <= f < end_f and 0 <= cls < N_CLASSES}
    return src, classes


def _all_events_by_class(ds: BalancedSTARSS23Dataset) -> Dict[int, List[Tuple]]:
    """Collect all (az, el) pairs per class from all full-clip items."""
    by_class: Dict[int, List[Tuple]] = {c: [] for c in range(N_CLASSES)}
    for item in ds._items:
        src = item[0]
        if src == 'starss23':
            evts = ds._starss_ds._events[item[1]]
        elif src == 'nigens':
            evts = ds._nigens_proj[item[1]]
        elif src == 'tau2019':
            evts = ds._tau2019_proj[item[1]]
        elif src == 'mixup':
            ci_a, ci_b, ds_b_type = item[1], item[2], item[3]
            evts_a = list(ds._starss_ds._events[ci_a])
            if ds_b_type == 'starss23':
                evts_b = list(ds._starss_ds._events[ci_b])
            elif ds_b_type == 'nigens':
                evts_b = list(ds._nigens_proj[ci_b])
            else:
                evts_b = list(ds._tau2019_proj[ci_b])
            evts = evts_a + evts_b
        else:
            continue
        for (_, cls, _, az, el) in evts:
            if 0 <= cls < N_CLASSES:
                by_class[cls].append((float(az), float(el)))
    return by_class


# ── Section 1 — Clip-level statistics ─────────────────────────────────────────

def section_clip_stats(ds_fc: BalancedSTARSS23Dataset) -> dict:
    print(f'\n{"█" * W}')
    print('  SECTION 1 — CLIP-LEVEL STATISTICS  (full-clip mode)')
    print(f'{"█" * W}')

    items = ds_fc._items

    # Source counts
    src_counts: Dict[str, int] = {s: 0 for s in SOURCES}
    for it in items:
        src_counts[it[0]] += 1

    total = len(items)
    print(f'\n  Total items: {total}')
    print(_sep())
    print(f'  {"Source":12s}  {"count":>6}  {"% total":>8}')
    print(_sep())
    for src in SOURCES:
        n = src_counts[src]
        print(f'  {src:12s}  {n:>6}  {100*n/total:>7.1f}%')
    print(_sep())

    # Per-class breakdown
    per_cls: Dict[int, Dict[str, int]] = {
        c: {s: 0 for s in SOURCES} for c in range(N_CLASSES)
    }
    for it in items:
        src, classes = _classes_in_item(it, ds_fc)
        for c in classes:
            per_cls[c][src] += 1

    print(f'\n  Per-class item presence by source:')
    print(_sep())
    hdr = f'  {"idx":>3}  {"Class":22s}  {"total":>5}  ' \
          + '  '.join(f'{s:>8s}' for s in SOURCES) + '  {"% total":>8}'
    print(hdr)
    print(_sep())

    totals_by_class = {}
    for c in range(N_CLASSES):
        row = per_cls[c]
        total_c = sum(row.values())
        totals_by_class[c] = total_c
        vals = '  '.join(f'{row[s]:>8d}' for s in SOURCES)
        print(f'  {c:>3}  {CLASS_NAMES[c]:22s}  {total_c:>5}  {vals}  {100*total_c/total:>7.1f}%')
    print(_sep())

    all_totals = list(totals_by_class.values())
    ratio = max(all_totals) / max(min(all_totals), 1)
    print(f'  Imbalance ratio (max/min): {ratio:.2f}×')

    return {
        'source_item_counts': src_counts,
        'total_items':        total,
        'per_class_counts':   {
            CLASS_NAMES[c]: {
                'total':    totals_by_class[c],
                **{s: per_cls[c][s] for s in SOURCES},
            }
            for c in range(N_CLASSES)
        },
        'imbalance_ratio': round(ratio, 3),
    }


# ── Section 2 — Windowed statistics ───────────────────────────────────────────

def section_windowed_stats(ds_win: BalancedSTARSS23Dataset) -> dict:
    print(f'\n{"█" * W}')
    print(f'  SECTION 2 — WINDOWED STATISTICS'
          f'  (window={WINDOW_S:.1f}s, hop={HOP_S:.1f}s)')
    print(f'{"█" * W}')

    items = ds_win._items
    total_w = len(items)

    src_counts: Dict[str, int] = {s: 0 for s in SOURCES}
    per_cls: Dict[int, Dict[str, int]] = {
        c: {s: 0 for s in SOURCES} for c in range(N_CLASSES)
    }
    n_silent = 0

    for it in items:
        src_counts[it[0]] += 1
        src, classes = _classes_in_window(it, ds_win)
        if not classes:
            n_silent += 1
        for c in classes:
            per_cls[c][src] += 1

    print(f'\n  Total windows: {total_w}')
    print(_sep())
    print(f'  {"Source":12s}  {"count":>6}  {"% total":>8}')
    print(_sep())
    for src in SOURCES:
        n = src_counts[src]
        print(f'  {src:12s}  {n:>6}  {100*n/total_w:>7.1f}%')
    print(f'  {"no-event":12s}  {n_silent:>6}  {100*n_silent/total_w:>7.1f}%')
    print(_sep())

    print(f'\n  Per-class window presence by source:')
    print(_sep())
    hdr = f'  {"idx":>3}  {"Class":22s}  {"total":>6}  ' \
          + '  '.join(f'{s:>8s}' for s in SOURCES) + '  {"% windows":>9}'
    print(hdr)
    print(_sep())

    totals_by_class = {}
    for c in range(N_CLASSES):
        row = per_cls[c]
        total_c = sum(row.values())
        totals_by_class[c] = total_c
        vals = '  '.join(f'{row[s]:>8d}' for s in SOURCES)
        print(f'  {c:>3}  {CLASS_NAMES[c]:22s}  {total_c:>6}  {vals}  {100*total_c/total_w:>8.1f}%')
    print(_sep())

    all_totals = list(totals_by_class.values())
    ratio = max(all_totals) / max(min(all_totals), 1)
    print(f'  No-event windows: {n_silent} ({100*n_silent/total_w:.1f}%)')
    print(f'  Imbalance ratio (max/min): {ratio:.2f}×')

    return {
        'window_s':           WINDOW_S,
        'hop_s':              HOP_S,
        'total_windows':      total_w,
        'source_win_counts':  src_counts,
        'n_silent':           n_silent,
        'per_class_windows':  {
            CLASS_NAMES[c]: {
                'total':  totals_by_class[c],
                **{s: per_cls[c][s] for s in SOURCES},
            }
            for c in range(N_CLASSES)
        },
        'imbalance_ratio': round(ratio, 3),
    }


# ── Section 3 — DOA statistics ────────────────────────────────────────────────

def section_doa_stats(ds_fc: BalancedSTARSS23Dataset) -> dict:
    print(f'\n{"█" * W}')
    print('  SECTION 3 — DOA STATISTICS  (all training events, full-clip items)')
    print(f'{"█" * W}')

    by_class = _all_events_by_class(ds_fc)

    all_az, all_el = [], []
    result = {}

    print(f'\n  {"Class":22s}  {"n":>6}  '
          f'{"az_mean":>8}  {"az_std":>7}  {"az_min":>7}  {"az_max":>7}  '
          f'{"el_mean":>8}  {"el_std":>7}  {"el_min":>7}  {"el_max":>7}')
    print(_sep())

    for c in range(N_CLASSES):
        pairs = by_class[c]
        n     = len(pairs)
        if n == 0:
            print(f'  {CLASS_NAMES[c]:22s}  {0:>6}  {"—":>8}  {"—":>7} …')
            result[CLASS_NAMES[c]] = {'n_events': 0}
            continue
        azs = np.array([p[0] for p in pairs])
        els = np.array([p[1] for p in pairs])
        all_az.extend(azs.tolist())
        all_el.extend(els.tolist())
        print(
            f'  {CLASS_NAMES[c]:22s}  {n:>6}  '
            f'{azs.mean():>8.1f}  {azs.std():>7.1f}  {azs.min():>7.1f}  {azs.max():>7.1f}  '
            f'{els.mean():>8.1f}  {els.std():>7.1f}  {els.min():>7.1f}  {els.max():>7.1f}'
        )
        result[CLASS_NAMES[c]] = {
            'n_events': n,
            'az': {'mean': round(float(azs.mean()), 2), 'std': round(float(azs.std()), 2),
                   'min':  round(float(azs.min()),  2), 'max': round(float(azs.max()),  2)},
            'el': {'mean': round(float(els.mean()), 2), 'std': round(float(els.std()), 2),
                   'min':  round(float(els.min()),  2), 'max': round(float(els.max()),  2)},
        }
    print(_sep())

    if all_az:
        azs = np.array(all_az)
        els = np.array(all_el)
        n_all = len(azs)
        print(
            f'  {"ALL CLASSES":22s}  {n_all:>6}  '
            f'{azs.mean():>8.1f}  {azs.std():>7.1f}  {azs.min():>7.1f}  {azs.max():>7.1f}  '
            f'{els.mean():>8.1f}  {els.std():>7.1f}  {els.min():>7.1f}  {els.max():>7.1f}'
        )
        result['_ALL'] = {
            'n_events': n_all,
            'az': {'mean': round(float(azs.mean()), 2), 'std': round(float(azs.std()), 2),
                   'min':  round(float(azs.min()),  2), 'max': round(float(azs.max()),  2)},
            'el': {'mean': round(float(els.mean()), 2), 'std': round(float(els.std()), 2),
                   'min':  round(float(els.min()),  2), 'max': round(float(els.max()),  2)},
        }

    return result


# ── Section 4 — Batch-shape tests ─────────────────────────────────────────────

def _test_batches(ds, label: str, n_batches: int, batch_size: int,
                  exp_n_classes: int, exp_n_tracks: int) -> bool:
    print(f'\n{_sep("═")}')
    print(f'  {label}  (n_items={len(ds)}, batch_size={batch_size}, '
          f'{n_batches} batches)')
    print(_sep('═'))

    windowed = getattr(ds, 'windowed', False)
    collate  = None if windowed else SELDDataset.collate_fn
    loader   = DataLoader(ds, batch_size=batch_size, shuffle=False,
                          num_workers=0, collate_fn=collate, drop_last=True)
    ok = True
    for i, (wav, tgt) in enumerate(loader):
        if i >= n_batches:
            break
        wav_ok  = wav.ndim == 3 and wav.shape[1] == 4
        tgt_ok  = (tgt.ndim == 5
                   and tgt.shape[2] == exp_n_tracks
                   and tgt.shape[3] == exp_n_classes
                   and tgt.shape[4] == 3)
        dt_ok   = wav.dtype == torch.float32 and tgt.dtype == torch.float32
        passed  = wav_ok and tgt_ok and dt_ok
        ok      = ok and passed
        mark    = '✓' if passed else '✗'
        print(f'  batch {i+1:02d}  {mark}'
              f'  wav={tuple(wav.shape)} {wav.dtype}'
              f'  tgt={tuple(tgt.shape)} {tgt.dtype}')
        if not wav_ok:
            print(f'    [FAIL] wav: expected ndim=3, ch=4, got {wav.shape}')
        if not tgt_ok:
            print(f'    [FAIL] tgt: expected (B,T,{exp_n_tracks},{exp_n_classes},3), '
                  f'got {tgt.shape}')
    return ok


def section_batch_tests(ds_win: BalancedSTARSS23Dataset,
                         ds_fc:  BalancedSTARSS23Dataset) -> bool:
    print(f'\n{"█" * W}')
    print('  SECTION 4 — BATCH-SHAPE TESTS')
    print(f'{"█" * W}')

    ok_win = _test_batches(ds_win, 'WINDOWED', n_batches=3, batch_size=2,
                           exp_n_classes=N_CLASSES, exp_n_tracks=N_TRACKS)
    ok_fc  = _test_batches(ds_fc,  'FULL-CLIP', n_batches=2, batch_size=1,
                           exp_n_classes=N_CLASSES, exp_n_tracks=N_TRACKS)

    verdict = 'PASS' if (ok_win and ok_fc) else 'FAIL'
    print(f'\n  Overall: {verdict}')
    return ok_win and ok_fc


# ── Entry point ────────────────────────────────────────────────────────────────

def _check_roots() -> List[str]:
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

    common = dict(
        starss_root  = STARSS_ROOT,
        nigens_root  = NIGENS_ROOT,
        tau2019_root = TAU2019_ROOT,
        model_sr     = MODEL_SR,
        hop_length   = HOP_LENGTH,
        n_tracks     = N_TRACKS,
        val_fraction = VAL_FRACTION,
        val_seed     = VAL_SEED,
        augment      = None,
        balance_seed = 2,
    )

    print('Building BalancedSTARSS23Dataset (full-clip) …')
    ds_fc  = BalancedSTARSS23Dataset(window_s=None,     hop_s=None,    **common)

    print('Building BalancedSTARSS23Dataset (windowed) …')
    ds_win = BalancedSTARSS23Dataset(window_s=WINDOW_S, hop_s=HOP_S,   **common)

    stats = {}
    stats['section1_clip']    = section_clip_stats(ds_fc)
    stats['section2_windowed'] = section_windowed_stats(ds_win)
    stats['section3_doa']     = section_doa_stats(ds_fc)

    all_ok = section_batch_tests(ds_win, ds_fc)

    stats['section4_batch_ok'] = all_ok

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / 'balanced_starss23_stats.json'
    with open(json_path, 'w') as fh:
        json.dump(stats, fh, indent=2)
    print(f'\n  JSON → {json_path.relative_to(_ROOT)}')

    print(f'\n{"═" * W}')
    print(f'  OVERALL: {"PASS" if all_ok else "FAIL"}')
    print(f'{"═" * W}')
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
