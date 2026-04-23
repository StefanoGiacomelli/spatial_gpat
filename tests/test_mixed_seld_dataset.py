"""
tests/test_mixed_seld_dataset.py
=================================
End-to-end functional test for MixedSELDDataset and FinetuneDataModule,
exercising both windowed (10 s / 5 s hop) and full-clip (window_s=None) modes.

What is reported
----------------
  ① Split statistics (annotation-based, no audio loaded — runs in seconds):

       Split layout matching FinetuneDataModule exactly:
         Train  : STARSS23 train  +  TAU-NIGENS2021 train  (MixedSELDDataset)
         Val    : STARSS23 val    only
         Test   : STARSS23 test   only

       For each split:
         • n_clips, n_items (windowed and full-clip counts side by side)
         • Per-class frame counts, % of total, pos_weight (13-class schema)
         • Imbalance ratio

       For the combined train split also:
         • WeightedRandomSampler weight distribution
         • Top-5 rarest classes (highest oversampling factor)

  ② Batch-shape verification (loads actual audio):
       • DataLoader in windowed and/or full-clip mode
       • Checks wav dtype/shape, target dtype/shape (…, 13, 3)

Usage
-----
  # Full report, annotation stats only (fast)
  .venv/bin/python tests/test_mixed_seld_dataset.py --no-audio

  # Full report + 3 audio batches in both modes
  .venv/bin/python tests/test_mixed_seld_dataset.py

  # Audio test — windowed only, 5 batches
  .venv/bin/python tests/test_mixed_seld_dataset.py --mode windowed --n-batches 5

Author
------
Stefano Giacomelli, Ph.D. candidate — DISIM, University of L'Aquila
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datasets.starss23           import STARSS23Dataset
from datasets.tau_nigens         import TAUNIGENSDataset
from datasets.mixed_seld_dataset import MixedSELDDataset, compute_combined_class_counts
from datasets.seld_dataset       import SELDDataset

# ── Constants ──────────────────────────────────────────────────────────────────

STARSS23_ROOT = _ROOT / 'data' / 'STARSS23'
TAU_ROOT      = _ROOT / 'data' / 'TAU-NIGENS2021'

MODEL_SR     = 32_000
HOP_LENGTH   = 320
N_TRACKS     = 2
N_CLASSES    = 13       # unified STARSS23 schema
VAL_FRACTION = 0.2
VAL_SEED     = 42

WINDOW_S_DEFAULT = 10.0
HOP_S_DEFAULT    =  5.0

CLASS_NAMES = [
    'Female speech', 'Male speech', 'Clapping', 'Telephone', 'Laughter',
    'Domestic sounds', 'Walk/footsteps', 'Door', 'Music',
    'Musical instr.', 'Water tap', 'Bell', 'Knock',
]


# ══════════════════════════════════════════════════════════════════════════════
# Dataset factory helpers
# ══════════════════════════════════════════════════════════════════════════════

def _starss(split: str, window_s: Optional[float], hop_s: Optional[float]) -> STARSS23Dataset:
    return STARSS23Dataset(
        root         = STARSS23_ROOT,
        split        = split,
        model_sr     = MODEL_SR,
        hop_length   = HOP_LENGTH,
        n_tracks     = N_TRACKS,
        window_s     = window_s,
        hop_s        = hop_s,
        audio_format = 'foa',
        augment      = None,
        val_fraction = VAL_FRACTION,
        val_seed     = VAL_SEED,
    )


def _tau(split: str, window_s: Optional[float], hop_s: Optional[float]) -> TAUNIGENSDataset:
    return TAUNIGENSDataset(
        root         = TAU_ROOT,
        split        = split,
        model_sr     = MODEL_SR,
        hop_length   = HOP_LENGTH,
        n_tracks     = N_TRACKS,
        window_s     = window_s,
        hop_s        = hop_s,
        audio_format = 'foa',
        augment      = None,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Annotation-based per-class counting
# ══════════════════════════════════════════════════════════════════════════════

def _count_frames(events_per_clip: List, n_classes: int) -> Dict[int, int]:
    """
    Count unique (clip_idx, frame_100ms, class_idx) active frames per class.
    Mirrors the logic in compute_combined_class_counts for a single dataset.
    """
    counts: Dict[int, int] = {c: 0 for c in range(n_classes)}
    for clip_idx, events in enumerate(events_per_clip):
        seen: set = set()
        for frame_100ms, class_idx, *_ in events:
            key = (clip_idx, frame_100ms, class_idx)
            if key not in seen:
                seen.add(key)
                if class_idx < n_classes:
                    counts[class_idx] += 1
    for c in range(n_classes):
        if counts[c] == 0:
            counts[c] = 1   # guard against division by zero
    return counts


# ══════════════════════════════════════════════════════════════════════════════
# Display helpers
# ══════════════════════════════════════════════════════════════════════════════

W = 74   # table width

def _sep(char='─') -> str:
    return char * W


def _print_class_table(counts: Dict[int, int], header: str) -> None:
    total = sum(counts.values())
    print(f'\n  {header}')
    print(_sep())
    print(f'  {"idx":>3}  {"class name":22}  {"frames":>8}  {"% total":>8}  {"pos_weight":>10}')
    print(_sep())
    for idx in range(N_CLASSES):
        c    = counts[idx]
        neg  = total - c
        pw   = neg / c if c > 0 else float('inf')
        pct  = 100.0 * c / total
        name = CLASS_NAMES[idx]
        marker = ' ←' if idx == 12 else ''   # Knock: STARSS23-only
        print(f'  {idx:>3}  {name:22}  {c:>8}  {pct:>7.2f}%  {pw:>10.1f}{marker}')
    print(_sep())
    max_c = max(counts.values())
    min_c = min(counts.values())
    print(f'  Imbalance ratio  max/min frames: {max_c/min_c:.1f}×'
          f'   total active frames: {total:,}')


def _item_row(label: str, n_clips: int, n_win: int, n_fc: int) -> None:
    print(f'  {label:35s}  {n_clips:>6}  {n_win:>16}  {n_fc:>10}')


# ══════════════════════════════════════════════════════════════════════════════
# Section ① — Split statistics
# ══════════════════════════════════════════════════════════════════════════════

def print_split_statistics(window_s: float, hop_s: float) -> None:
    """
    Build all five splits (STARSS23 train/val/test, TAU train, Mixed train)
    and print annotation-based per-class stats + item counts.
    """
    print(f'\n{"█" * W}')
    print(f'  SPLIT STATISTICS  (annotation-based, no audio loaded)')
    print(f'  FinetuneDataModule layout:')
    print(f'    Train = STARSS23 train + TAU-NIGENS2021 train  (MixedSELDDataset)')
    print(f'    Val   = STARSS23 val   only')
    print(f'    Test  = STARSS23 test  only')
    print(f'{"█" * W}')

    # ── Build all datasets in both modes ──────────────────────────────────────
    print('\n  Building datasets…', end='', flush=True)

    s_tr_w  = _starss('train', window_s, hop_s)
    s_val_w = _starss('val',   window_s, hop_s)
    s_te_w  = _starss('test',  window_s, hop_s)
    t_tr_w  = _tau('train',    window_s, hop_s)

    s_tr_fc  = _starss('train', None, None)
    s_val_fc = _starss('val',   None, None)
    s_te_fc  = _starss('test',  None, None)
    t_tr_fc  = _tau('train',    None, None)

    mix_w  = MixedSELDDataset(s_tr_w,  t_tr_w)
    mix_fc = MixedSELDDataset(s_tr_fc, t_tr_fc)

    print(' done.')

    # ── Item count summary ────────────────────────────────────────────────────
    print(f'\n  Item counts')
    print(_sep())
    print(f'  {"Split":35s}  {"clips":>6}  {"windowed items":>16}  {"full-clip":>10}')
    print(f'  {"":35s}  {"":>6}  {"(%.0fs/%.0fs hop)"%( window_s, hop_s):>16}  {"items":>10}')
    print(_sep())
    _item_row('STARSS23     train',
              len(s_tr_w._events),  len(s_tr_w),  len(s_tr_fc))
    _item_row('TAU-NIGENS2021 train',
              len(t_tr_w._events),  len(t_tr_w),  len(t_tr_fc))
    _item_row('MIXED        train  (train split)',
              len(s_tr_w._events) + len(t_tr_w._events),
              len(mix_w), len(mix_fc))
    print(_sep())
    _item_row('STARSS23     val    (val split)',
              len(s_val_w._events), len(s_val_w), len(s_val_fc))
    _item_row('STARSS23     test   (test split)',
              len(s_te_w._events),  len(s_te_w),  len(s_te_fc))

    # ── Per-class tables ──────────────────────────────────────────────────────

    # STARSS23 train
    print(f'\n{_sep("═")}')
    print(f'  STARSS23 — train  (contributes to MixedSELDDataset train)')
    print(_sep('═'))
    c_s_tr = _count_frames(s_tr_w._events, N_CLASSES)
    _print_class_table(c_s_tr, 'Per-class frame counts  (← = STARSS23-only class)')

    # TAU-NIGENS2021 train
    print(f'\n{_sep("═")}')
    print(f'  TAU-NIGENS2021 — train  (contributes to MixedSELDDataset train)')
    print(f'  Note: class 12 (Knock) is absent from TAU-NIGENS → always 0 in targets')
    print(_sep('═'))
    c_t_tr = _count_frames(t_tr_w._events, N_CLASSES)
    _print_class_table(c_t_tr, 'Per-class frame counts  (class 12 = guard 1, never active)')

    # MIXED train combined
    print(f'\n{_sep("═")}')
    print(f'  MIXED TRAIN  (STARSS23 train + TAU-NIGENS2021 train combined)')
    print(f'  This is the distribution seen by the loss during fine-tuning.')
    print(_sep('═'))
    c_mix = compute_combined_class_counts(s_tr_w, t_tr_w)
    _print_class_table(c_mix, 'Combined per-class frame counts')

    # Sampling weights on the combined train set
    weights = mix_w.clip_weights()
    over_1  = (weights > 1.0).float().mean().item() * 100
    med_cnt = max(1, int(torch.tensor(list(c_mix.values()), dtype=torch.float32).median().item()))
    print(f'\n  WeightedRandomSampler  (windowed, {len(weights)} items)')
    print(_sep())
    print(f'  target_count (median class frames) = {med_cnt:,}')
    print(f'  weight  min={weights.min():.3f}  median={weights.median():.3f}'
          f'  mean={weights.mean():.3f}  max={weights.max():.3f}')
    print(f'  Items oversampled (weight > 1.0): {over_1:.1f}%')
    print(f'\n  Top-5 rarest classes (highest oversampling factor):')
    class_w = sorted(
        [(med_cnt / c_mix[i], i, CLASS_NAMES[i]) for i in range(N_CLASSES)],
        reverse=True,
    )
    for w, idx, name in class_w[:5]:
        print(f'    {idx:>2}  {name:22}  ×{w:.2f}  '
              f'({c_mix[idx]:,} frames → target {med_cnt:,})')

    # STARSS23 val
    print(f'\n{_sep("═")}')
    print(f'  STARSS23 — val  (validation split used throughout fine-tuning)')
    print(_sep('═'))
    c_s_val = _count_frames(s_val_w._events, N_CLASSES)
    _print_class_table(c_s_val, 'Per-class frame counts')

    # STARSS23 test
    print(f'\n{_sep("═")}')
    print(f'  STARSS23 — test  (test split; evaluated once after training)')
    print(_sep('═'))
    c_s_te = _count_frames(s_te_w._events, N_CLASSES)
    _print_class_table(c_s_te, 'Per-class frame counts')


# ══════════════════════════════════════════════════════════════════════════════
# Section ② — DataLoader batch-shape verification
# ══════════════════════════════════════════════════════════════════════════════

def _make_loader(mixed: MixedSELDDataset, batch_size: int) -> DataLoader:
    weights = mixed.clip_weights()
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    collate = None if mixed.windowed else SELDDataset.collate_fn
    return DataLoader(
        mixed,
        batch_size  = batch_size,
        sampler     = sampler,
        drop_last   = True,
        num_workers = 0,
        collate_fn  = collate,
        pin_memory  = False,
    )


def _test_batches(
    mixed: MixedSELDDataset,
    label: str,
    n_batches: int,
    batch_size: int,
) -> bool:
    print(f'\n{_sep("═")}')
    print(f'  BATCH SHAPE TEST — {label}')
    print(_sep('═'))
    collate_name = 'default (stack)' if mixed.windowed else 'SELDDataset.collate_fn (pad)'
    print(f'  windowed={mixed.windowed}  collate={collate_name}')
    print(f'  n_items={len(mixed)}  batch_size={batch_size}  iterating {n_batches} batches…\n')

    ok = True
    for i, (wav, tgt) in enumerate(loader := _make_loader(mixed, batch_size)):
        if i >= n_batches:
            break
        wav_ok   = wav.ndim == 3 and wav.shape[1] == 4
        tgt_ok   = (tgt.ndim == 5 and tgt.shape[2] == N_TRACKS
                    and tgt.shape[3] == N_CLASSES and tgt.shape[4] == 3)
        dtype_ok = wav.dtype == torch.float32 and tgt.dtype == torch.float32
        passed   = wav_ok and tgt_ok and dtype_ok
        if not passed:
            ok = False
        mark = '✓' if passed else '✗'
        print(f'  batch {i+1:02d}  {mark}'
              f'  wav={tuple(wav.shape)} {wav.dtype}'
              f'  tgt={tuple(tgt.shape)} {tgt.dtype}')
        if not wav_ok:
            print(f'    [FAIL] wav: expected (B,4,T_samples), got {wav.shape}')
        if not tgt_ok:
            print(f'    [FAIL] tgt: expected (B,T_mel,{N_TRACKS},{N_CLASSES},3),'
                  f' got {tgt.shape}')

    print(f'\n  Result: {"PASS" if ok else "FAIL"}')
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description='Split statistics + functional test for MixedSELDDataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--mode', choices=['windowed', 'fullclip', 'both'],
                   default='both', help='Audio batch test mode(s).')
    p.add_argument('--no-audio', action='store_true',
                   help='Skip audio batch tests (annotation stats only).')
    p.add_argument('--n-batches', type=int, default=3,
                   help='Audio batches to iterate per mode.')
    p.add_argument('--batch-size', type=int, default=1,
                   help='Batch size for windowed audio test.')
    p.add_argument('--window-s', type=float, default=WINDOW_S_DEFAULT,
                   help='Window duration (s) for windowed mode stats.')
    p.add_argument('--hop-s', type=float, default=HOP_S_DEFAULT,
                   help='Hop duration (s) for windowed mode stats.')
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    for name, root in [('STARSS23', STARSS23_ROOT), ('TAU-NIGENS2021', TAU_ROOT)]:
        if not root.exists():
            print(f'[ERROR] {name} root not found: {root}')
            sys.exit(1)

    # ── Section ①: annotation stats for all splits ───────────────────────────
    print_split_statistics(window_s=args.window_s, hop_s=args.hop_s)

    if args.no_audio:
        print(f'\n{"─" * W}')
        print('  (Audio batch tests skipped — use without --no-audio to run them.)')
        return

    # ── Section ②: audio batch tests ────────────────────────────────────────
    all_ok: List[bool] = []

    if args.mode in ('windowed', 'both'):
        print(f'\n{"█" * W}')
        print(f'  AUDIO BATCH TEST — WINDOWED  '
              f'(window_s={args.window_s}  hop_s={args.hop_s})')
        print(f'{"█" * W}')
        mix_w = MixedSELDDataset(
            _starss('train', args.window_s, args.hop_s),
            _tau('train',    args.window_s, args.hop_s),
        )
        all_ok.append(_test_batches(mix_w, 'WINDOWED', args.n_batches, args.batch_size))

    if args.mode in ('fullclip', 'both'):
        print(f'\n{"█" * W}')
        print(f'  AUDIO BATCH TEST — FULL-CLIP  (window_s=None)')
        print(f'{"█" * W}')
        mix_fc = MixedSELDDataset(
            _starss('train', None, None),
            _tau('train',    None, None),
        )
        all_ok.append(_test_batches(mix_fc, 'FULL-CLIP', args.n_batches, 1))

    print(f'\n{_sep("═")}')
    verdict = 'ALL PASS' if all(all_ok) else 'SOME TESTS FAILED'
    print(f'  OVERALL: {verdict}')
    print(_sep('═'))
    sys.exit(0 if all(all_ok) else 1)


if __name__ == '__main__':
    main()
