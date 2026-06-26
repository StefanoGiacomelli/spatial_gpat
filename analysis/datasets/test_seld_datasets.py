#!/usr/bin/env python3
"""
analysis/datasets/test_seld_datasets.py
========================================
Smoke-test, profiling, and dataset analysis script for STARSS23Dataset,
TAUNIGENSDataset, TAU2019Dataset, and MixedSELDDataset.

For each dataset and each split (train / val / test) the script can:
  1. [Smoke test] Build the Dataset (windowed + full-clip), inspect shapes,
     compute class-activity distribution and polyphony histogram, test a
     DataLoader batch, and print a comparison summary table.
  2. [Analysis] Compute and export TABLE 2-style statistics:
       - **native** mode: parsed directly from CSV at 100 ms resolution,
         replicating the paper's TABLE 2 (STARSS23) or equivalent.
       - **model** mode: computed from SELDDataset target tensors at mel-frame
         resolution (captures window/hop quantisation effects).
  3. [Analysis] Generate per-class DOA figures (azimuth histogram / az-el
     scatter density / elevation histogram rotated), styled after Figure 3
     and Appendix B of the STARSS23 paper.

Output directories
------------------
    results/   — CSV tables (TABLE 2 per dataset/split/mode)
    figures/   — SVG figures at 600 DPI (DOA analysis per dataset)

Usage
-----
    # Run everything (smoke tests + analysis)
    python analysis/datasets/test_seld_datasets.py

    # Only dataset analysis + figures (fast, no audio loading for native mode)
    python analysis/datasets/test_seld_datasets.py --analysis-only

    # Only smoke tests
    python analysis/datasets/test_seld_datasets.py --smoke-only

    # Restrict to one dataset
    python analysis/datasets/test_seld_datasets.py --dataset STARSS23 --analysis-only
    python analysis/datasets/test_seld_datasets.py --dataset TAU-NIGENS2021

    # Full-clip mode for smoke tests (no windowing)
    python analysis/datasets/test_seld_datasets.py --smoke-only --no-window

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

import argparse
import csv
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')           # headless — no display required
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datasets import (  # noqa: E402
    STARSS23Dataset,
    TAUNIGENSDataset,
    TAUNIGENS2020Dataset,
    SELDDataset,
)
from datasets.tau2019 import TAU2019Dataset
try:
    from datasets.utils import (MixedSELDDataset, STARSS_TO_UNIFIED,
                                NIGENS_TO_UNIFIED, TAU2019_TO_UNIFIED)
except ImportError:
    MixedSELDDataset = None
    STARSS_TO_UNIFIED = NIGENS_TO_UNIFIED = TAU2019_TO_UNIFIED = None

# ── Dataset roots ─────────────────────────────────────────────────────────────
ROOTS = {
    'STARSS23':       _ROOT / 'data' / 'STARSS23',
    'TAU-NIGENS2021': _ROOT / 'data' / 'TAU-NIGENS2021',
    'TAU-NIGENS2020': _ROOT / 'data' / 'TAU-NIGENS2020',
    'TAU2019':        _ROOT / 'data' / 'TAU2019',
}

# ── Class labels per dataset ──────────────────────────────────────────────────
CLASS_LABELS = {
    'STARSS23': [
        'Female speech', 'Male speech', 'Clapping', 'Telephone', 'Laughter',
        'Domestic sounds', 'Walk/footsteps', 'Door', 'Music',
        'Musical instr.', 'Water tap', 'Bell', 'Knock',
    ],
    'TAU-NIGENS2021': [
        'Alarm', 'Crying baby', 'Crash', 'Barking dog',
        'Female scream', 'Female speech', 'Footsteps', 'Knocking on door',
        'Male scream', 'Male speech', 'Ringing phone', 'Piano',
    ],
    'TAU-NIGENS2020': [
        'Alarm', 'Crying baby', 'Crash', 'Barking dog', 'Running engine',
        'Female scream', 'Female speech', 'Burning fire', 'Footsteps',
        'Knocking on door', 'Male scream', 'Male speech', 'Ringing phone',
        'Piano',
    ],
    'TAU2019': [
        'Clearthroat', 'Cough', 'Doorslam', 'Drawer', 'Keyboard',
        'Keys drop', 'Knock', 'Laughter', 'Page turn', 'Phone', 'Speech',
    ],
}

_TAU2019_CLASS_MAP = {
    'clearthroat': 0, 'cough': 1, 'doorslam': 2, 'drawer': 3,
    'keyboard': 4, 'keysDrop': 5, 'knock': 6, 'laughter': 7,
    'pageturn': 8, 'phone': 9, 'speech': 10,
}

# ── Shared model parameters ───────────────────────────────────────────────────
MODEL_SR   = 24_000
HOP_LENGTH = 240
N_TRACKS   = 3

# ── Official (paper) splits for native-mode analysis ─────────────────────────
# For STARSS23 the "train" here is the full dev-train (Sony+TAU combined),
# as used in TABLE 2 of the paper — NOT the 80/20 model train/val split.
_STARSS23_SPLIT_DIRS = {
    'train': ['dev-train-sony', 'dev-train-tau'],
    'test':  ['dev-test-sony',  'dev-test-tau'],
}
_TAUNIGENS_SPLIT_DIRS = {
    'train': ['dev-train'],
    'val':   ['dev-val'],
    'test':  ['dev-test'],
}
_TAUNIGENS2020_SPLIT_FOLDS = {
    'train': {1, 2, 3, 4},
    'val':   {5},
    'test':  {6},
}
_TAU2019_SPLIT_NUMS = {   # split numbers for mixed_mode
    'train': [2, 3, 4],
    'val':   [1],
    # 'test' uses eval/metadata_eval/split0_*.csv
}
_SPLITS = {
    'STARSS23':       ['train', 'test'],
    'TAU-NIGENS2021': ['train', 'val', 'test'],
    'TAU-NIGENS2020': ['train', 'val', 'test'],
    'TAU2019':        ['train', 'val', 'test'],
}

# ── Output directories ────────────────────────────────────────────────────────
RESULTS_DIR = _ROOT / 'results' / 'datasets'   # CSV tables
FIGURES_DIR = _ROOT / 'figures' / 'datasets'   # SVG figures


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _separator(char='─', width=70):
    print(char * width)


def _make_dataset(dataset_name: str, split: str,
                  window_s=None, hop_s=None) -> SELDDataset:
    kwargs = dict(
        root       = ROOTS[dataset_name],
        split      = split,
        model_sr   = MODEL_SR,
        hop_length = HOP_LENGTH,
        n_tracks   = N_TRACKS,
        window_s   = window_s,
        hop_s      = hop_s,
    )
    if dataset_name == 'STARSS23':
        return STARSS23Dataset(**kwargs, val_fraction=0.2, val_seed=42)
    elif dataset_name == 'TAU-NIGENS2020':
        return TAUNIGENS2020Dataset(**kwargs)
    else:
        return TAUNIGENSDataset(**kwargs)


def _make_tau2019_dataset(split: str, mixed_mode: bool = True,
                          fold: int = 1, window_s=None, hop_s=None) -> TAU2019Dataset:
    return TAU2019Dataset(
        root       = ROOTS['TAU2019'],
        split      = split,
        model_sr   = MODEL_SR,
        hop_length = HOP_LENGTH,
        n_tracks   = N_TRACKS,
        window_s   = window_s,
        hop_s      = hop_s,
        fold       = fold,
        mixed_mode = mixed_mode,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ── SECTION A: SMOKE TEST ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _analyse_split(ds: SELDDataset, label: str, n_sample: int = 50):
    """
    Compute basic statistics over a split without loading the full dataset.
    Samples up to *n_sample* items uniformly at random.
    """
    n_items = len(ds)
    n_cls   = ds.n_classes

    idxs = np.random.default_rng(0).choice(n_items, size=min(n_sample, n_items),
                                           replace=False)

    active_frames_per_class = np.zeros(n_cls, dtype=np.int64)
    polyphony_counts = np.zeros(N_TRACKS + 1, dtype=np.int64)
    total_mel_frames = 0

    for i in idxs:
        _, tgt = ds[i]            # (T_mel, N_tracks, C, 3)
        T = tgt.shape[0]
        total_mel_frames += T

        norm = tgt.norm(dim=-1)   # (T, N_tracks, C)
        cls_active = (norm > 0).any(dim=1)  # (T, C)
        active_frames_per_class += cls_active.sum(dim=0).numpy().astype(np.int64)

        poly = cls_active.sum(dim=1)  # (T,)
        for p in poly.numpy():
            polyphony_counts[min(int(p), N_TRACKS)] += 1

    return {
        'n_items':     n_items,
        'n_sampled':   len(idxs),
        'n_mel_total': total_mel_frames,
        'cls_active':  active_frames_per_class,
        'polyphony':   polyphony_counts,
    }


def _print_split_report(ds: SELDDataset, dataset_name: str, split: str,
                        stats: dict, windowed: bool):
    labels = CLASS_LABELS[dataset_name]
    _separator()
    mode = "windowed" if windowed else "full-clip"
    print(f"  {dataset_name}  |  split={split}  |  mode={mode}")
    _separator()
    print(f"  Items in split   : {stats['n_items']:,}")
    print(f"  Clips backing it : {len(ds.clips)}")
    if windowed:
        print(f"  Windows/clip ~   : {stats['n_items'] / max(len(ds.clips), 1):.1f}")
    print(f"  Sampled          : {stats['n_sampled']} items for statistics")
    print(f"  Total mel frames : {stats['n_mel_total']:,}")

    print(f"\n  Class activity (active mel-frames over sampled items):")
    total_active = stats['cls_active'].sum()
    for ci, cnt in enumerate(stats['cls_active']):
        label = labels[ci] if ci < len(labels) else f'cls_{ci}'
        bar   = '█' * int(30 * cnt / max(total_active, 1))
        print(f"    [{ci:2d}] {label:<20s} {cnt:7,}  {bar}")

    print(f"\n  Polyphony distribution (simultaneously active classes per mel frame):")
    total_frames = stats['polyphony'].sum()
    for p, cnt in enumerate(stats['polyphony']):
        if p == N_TRACKS:
            label = f"{p}+ classes"
        else:
            label = f"{p} class{'es' if p != 1 else ' '}"
        bar = '█' * int(30 * cnt / max(total_frames, 1))
        pct = 100.0 * cnt / max(total_frames, 1)
        print(f"    {label:10s}: {cnt:7,}  ({pct:5.1f}%)  {bar}")


def _test_dataloader(ds: SELDDataset, dataset_name: str, split: str,
                     windowed: bool):
    collate = None if windowed else SELDDataset.collate_fn
    loader  = DataLoader(ds, batch_size=4, shuffle=False,
                         num_workers=0, collate_fn=collate)
    t0 = time.perf_counter()
    wav_batch, tgt_batch = next(iter(loader))
    elapsed = (time.perf_counter() - t0) * 1e3

    _separator('·')
    print(f"  DataLoader batch  ({dataset_name} | {split} | {'windowed' if windowed else 'full-clip'})")
    print(f"    wav  shape : {tuple(wav_batch.shape)}  dtype={wav_batch.dtype}")
    print(f"    tgt  shape : {tuple(tgt_batch.shape)}  dtype={tgt_batch.dtype}")
    norm = tgt_batch.norm(dim=-1)
    active_mask = norm > 0
    if active_mask.any():
        mean_norm = norm[active_mask].mean().item()
        print(f"    Mean ‖xyz‖ on active frames : {mean_norm:.4f}  (expected ≈ 1.0)")
    else:
        print(f"    No active events in this batch.")
    print(f"    First-batch load time : {elapsed:.1f} ms")


def run_smoke_tests(dataset_names: List[str], windowed: bool):
    """Run the run smoke tests step."""

    window_s = 10.0 if windowed else None
    hop_s    = 5.0  if windowed else None
    summary_rows = []

    for dataset_name in dataset_names:
        print(f"\n{'═'*70}")
        print(f"  DATASET: {dataset_name}  ({'windowed 10s/50%hop' if windowed else 'full-clip'})")
        print(f"{'═'*70}")

        for split in ('train', 'val', 'test'):
            print(f"\n  Building {split} split …", flush=True)
            t0 = time.perf_counter()
            ds = _make_dataset(dataset_name, split, window_s, hop_s)
            build_ms = (time.perf_counter() - t0) * 1e3
            print(f"  → {len(ds):,} items in {build_ms:.0f} ms")

            wav, tgt = ds[0]
            print(f"  Sample[0]  wav={tuple(wav.shape)}  tgt={tuple(tgt.shape)}")

            stats = _analyse_split(ds, split, n_sample=min(50, len(ds)))
            _print_split_report(ds, dataset_name, split, stats, windowed)
            _test_dataloader(ds, dataset_name, split, windowed)

            summary_rows.append({
                'dataset':   dataset_name,
                'split':     split,
                'n_items':   len(ds),
                'n_clips':   len(ds.clips),
                'wav_shape': tuple(wav.shape),
                'tgt_shape': tuple(tgt.shape),
            })

    print(f"\n\n{'═'*70}")
    print("  SUMMARY")
    print(f"{'═'*70}")
    header = (f"  {'Dataset':<20s} {'Split':<8s} {'Items':>8s} {'Clips':>7s}"
              f"  {'wav shape':<25s}  {'tgt shape'}")
    print(header)
    _separator()
    for r in summary_rows:
        print(
            f"  {r['dataset']:<20s} {r['split']:<8s} {r['n_items']:>8,} {r['n_clips']:>7,}"
            f"  {str(r['wav_shape']):<25s}  {r['tgt_shape']}"
        )
    print()


# ══════════════════════════════════════════════════════════════════════════════
# ── SECTION A2: TAU2019 SMOKE TEST ────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def run_tau2019_smoke_tests(windowed: bool):
    """Smoke tests for TAU2019Dataset (standalone + mixed mode)."""
    no_tau = not ROOTS['TAU2019'].exists()
    window_s = 10.0 if windowed else None
    hop_s    = 5.0  if windowed else None
    mode     = 'windowed' if windowed else 'full-clip'

    print(f"\n{'═'*70}")
    print(f"  TAU2019 SMOKE TEST  ({'windowed 10s/50%hop' if windowed else 'full-clip'})")
    print(f"{'═'*70}")

    if no_tau:
        print('  [SKIP] TAU2019 data not found at', ROOTS['TAU2019'])
        return

    # ── Mixed mode: shape + resample check ───────────────────────────────────
    for split in ('train', 'val', 'test'):
        print(f"\n  Building mixed_mode split={split} ({mode}) …", flush=True)
        t0 = time.perf_counter()
        ds = _make_tau2019_dataset(split, mixed_mode=True,
                                   window_s=window_s, hop_s=hop_s)
        build_ms = (time.perf_counter() - t0) * 1e3
        print(f"  → {len(ds):,} items ({len(ds.clips)} clips) in {build_ms:.0f} ms")

        wav, tgt = ds[0]
        print(f"  Sample[0]  wav={tuple(wav.shape)}  tgt={tuple(tgt.shape)}")
        assert wav.shape[0] == 4, f"Expected 4 FOA channels, got {wav.shape[0]}"
        assert tgt.shape[1] == N_TRACKS, f"Expected {N_TRACKS} tracks, got {tgt.shape[1]}"
        assert tgt.shape[2] == 11, f"Expected 11 TAU2019 classes, got {tgt.shape[2]}"
        assert tgt.shape[3] == 3, f"Expected 3 xyz DOA dims, got {tgt.shape[3]}"
        assert ds.output_sr == 24000, f"Expected output_sr=24000, got {ds.output_sr}"
        if windowed:
            expected_samples = round(window_s * 24_000)
            expected_mel     = 1 + expected_samples // HOP_LENGTH
            assert wav.shape[-1] == expected_samples, \
                f"wav samples: {wav.shape[-1]} vs {expected_samples}"
            assert tgt.shape[0] == expected_mel, \
                f"T_mel: {tgt.shape[0]} vs {expected_mel}"

        _test_dataloader(ds, 'TAU2019', split, windowed)

    # ── Expected clip counts in mixed mode ────────────────────────────────────
    print(f"\n  Checking expected clip counts (mixed mode, full-clip) …")
    ds_tr = _make_tau2019_dataset('train', mixed_mode=True)
    ds_vl = _make_tau2019_dataset('val',   mixed_mode=True)
    ds_ts = _make_tau2019_dataset('test',  mixed_mode=True)
    assert len(ds_tr.clips) == 300, f"train clips: {len(ds_tr.clips)}"
    assert len(ds_vl.clips) == 100, f"val clips: {len(ds_vl.clips)}"
    assert len(ds_ts.clips) == 100, f"test clips: {len(ds_ts.clips)}"
    print(f"  OK: train={len(ds_tr.clips)} val={len(ds_vl.clips)} test={len(ds_ts.clips)}")

    # ── Standalone mode: fold 1 ───────────────────────────────────────────────
    print(f"\n  Checking standalone fold=1 shapes …")
    for split in ('train', 'val', 'test'):
        ds = _make_tau2019_dataset(split, mixed_mode=False, fold=1,
                                   window_s=window_s, hop_s=hop_s)
        wav, tgt = ds[0]
        assert tgt.shape[2] == 11, f"fold1/{split}: n_classes={tgt.shape[2]}"
        print(f"  fold1/{split}: {len(ds):,} items  tgt={tuple(tgt.shape)}")


# ══════════════════════════════════════════════════════════════════════════════
# ── SECTION A3: MIXED SELD SMOKE TEST ────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def run_mixed_seld_smoke_tests(windowed: bool):
    """Smoke tests for MixedSELDDataset (3 datasets, 15 unified classes)."""
    if MixedSELDDataset is None:
        print("\n[SKIP] MixedSELDDataset is not available in this repository state.")
        return

    mixed_roots = ['STARSS23', 'TAU-NIGENS2021', 'TAU2019']
    no_all = not all(ROOTS[name].exists() for name in mixed_roots)
    window_s = 10.0 if windowed else None
    hop_s    = 5.0  if windowed else None
    mode     = 'windowed' if windowed else 'full-clip'

    print(f"\n{'═'*70}")
    print(f"  MIXED SELD SMOKE TEST  ({'windowed 10s/50%hop' if windowed else 'full-clip'})")
    print(f"{'═'*70}")

    if no_all:
        missing = [name for name in mixed_roots if not ROOTS[name].exists()]
        print(f"  [SKIP] Missing dataset roots: {missing}")
        return

    for split in ('train', 'val', 'test'):
        print(f"\n  Building MixedSELDDataset split={split} ({mode}) …", flush=True)
        t0 = time.perf_counter()

        kw = dict(model_sr=MODEL_SR, hop_length=HOP_LENGTH, n_tracks=N_TRACKS,
                  window_s=window_s, hop_s=hop_s)
        starss  = STARSS23Dataset(root=ROOTS['STARSS23'], split=split,
                                  val_fraction=0.2, val_seed=42, **kw)
        nigens  = TAUNIGENSDataset(root=ROOTS['TAU-NIGENS2021'], split=split, **kw)
        tau2019 = TAU2019Dataset(root=ROOTS['TAU2019'], split=split,
                                 mixed_mode=True, **kw)
        mixed = MixedSELDDataset(
            datasets=[starss, nigens, tau2019],
            mappings=[STARSS_TO_UNIFIED, NIGENS_TO_UNIFIED, TAU2019_TO_UNIFIED],
        )

        build_ms = (time.perf_counter() - t0) * 1e3
        expected = len(starss) + len(nigens) + len(tau2019)
        assert len(mixed) == expected, \
            f"Mixed len: {len(mixed)} vs {expected} (starss={len(starss)}, " \
            f"nigens={len(nigens)}, tau2019={len(tau2019)})"

        print(f"  → {len(mixed):,} items in {build_ms:.0f} ms "
              f"(starss={len(starss)}, nigens={len(nigens)}, tau2019={len(tau2019)})")

        wav, tgt = mixed[0]
        print(f"  Sample[0]  wav={tuple(wav.shape)}  tgt={tuple(tgt.shape)}")
        assert wav.shape[0] == 4, f"Expected 4 FOA channels, got {wav.shape[0]}"
        assert tgt.shape[1] == N_TRACKS, f"Expected {N_TRACKS} tracks, got {tgt.shape[1]}"
        assert tgt.shape[2] == 15, f"Expected 15 unified classes, got {tgt.shape[2]}"
        assert tgt.shape[3] == 3, f"Expected 3 xyz DOA dims, got {tgt.shape[3]}"

        # Unit-vector sanity check on active DOA frames
        import torch
        norms  = tgt.norm(dim=-1)   # (T_mel, N_tracks, 15)
        active = norms > 0
        if active.any():
            err = (norms[active] - 1.0).abs().max().item()
            assert err < 0.01, f"DOA unit-sphere error too large: {err:.4f}"
            print(f"  DOA unit-norm max error: {err:.5f}  (OK)")


# ══════════════════════════════════════════════════════════════════════════════
# ── SECTION B: ANALYSIS — TABLE 2 + DOA FIGURES ──────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# ── B1. Native CSV parsing ────────────────────────────────────────────────────

def _parse_raw(csv_path: Path, n_cols: int = 6
               ) -> List[Tuple[int, int, int, float, float]]:
    """
    Read annotation CSV directly (no SELDDataset).
    Returns list of (frame_100ms, class_idx, source_idx, az_deg, el_deg).
    STARSS23 has 6 cols (includes distance), TAU-NIGENS has 5.
    """
    events = []
    with open(csv_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p = line.split(',')
            events.append((int(p[0]), int(p[1]), int(p[2]),
                           float(p[3]), float(p[4])))
    return events


def _get_native_csv_files(dataset_name: str, split: str) -> List[Path]:
    """Return all annotation CSV paths for the official split."""
    meta_root = ROOTS[dataset_name] / 'metadata_dev'
    if dataset_name == 'STARSS23':
        subdirs = _STARSS23_SPLIT_DIRS[split]
    elif dataset_name == 'TAU-NIGENS2020':
        folds = _TAUNIGENS2020_SPLIT_FOLDS[split]
        paths = []
        for csv_p in sorted(meta_root.glob('fold*.csv')):
            fold_token = csv_p.stem.split('_')[0]
            if not fold_token.startswith('fold'):
                continue
            try:
                fold_num = int(fold_token[4:])
            except ValueError:
                continue
            if fold_num in folds:
                paths.append(csv_p)
        return paths
    else:
        subdirs = _TAUNIGENS_SPLIT_DIRS[split]
    paths = []
    for sub in subdirs:
        paths.extend(sorted((meta_root / sub).glob('*.csv')))
    return paths


def _parse_raw_tau2019(csv_path: Path) -> List[Tuple[int, int, int, float, float]]:
    """
    Read a TAU2019 annotation CSV (onset/offset format with headers).

    CSV columns: sound_event_recording, start_time, end_time, ele, azi, dist
    Returns: list of (frame_100ms, class_idx, source_idx, az_deg, el_deg)
    where frames span [start_time, end_time) at 100 ms resolution.
    """
    import csv as _csv
    events = []
    instance_counter: Dict[int, int] = defaultdict(int)
    with open(csv_path, newline='') as fh:
        for row in _csv.DictReader(fh):
            label = row['sound_event_recording']
            if label not in _TAU2019_CLASS_MAP:
                continue
            cls_idx = _TAU2019_CLASS_MAP[label]
            start_f = int(float(row['start_time']) / 0.1)
            end_f   = int(float(row['end_time'])   / 0.1)
            az      = float(row['azi'])
            el      = float(row['ele'])
            src_idx = instance_counter[cls_idx]
            instance_counter[cls_idx] += 1
            for f in range(start_f, end_f + 1):
                events.append((f, cls_idx, src_idx, az, el))
    return events


def _get_native_csv_files_tau2019(split: str) -> List[Path]:
    """Return annotation CSV paths for TAU2019 mixed_mode split."""
    tau_root  = ROOTS['TAU2019']
    meta_dev  = tau_root / 'dev' / 'metadata_dev'
    meta_eval = tau_root / 'eval' / 'metadata_eval'
    paths: List[Path] = []
    if split == 'test':
        if meta_eval.exists():
            paths.extend(sorted(meta_eval.glob('split0_*.csv')))
    else:
        for sn in _TAU2019_SPLIT_NUMS[split]:
            paths.extend(sorted(meta_dev.glob(f'split{sn}_*.csv')))
    return paths


def _native_stats_tau2019(csv_paths: List[Path]) -> Tuple[Dict, Dict]:
    """
    Compute TABLE 2-style statistics from TAU2019 onset/offset CSVs.
    Reuses the same structure as _native_stats but with _parse_raw_tau2019.
    """
    n_classes = 11
    total_frames_100ms = 0
    cls_frames_active: Dict[int, set] = defaultdict(set)
    cls_frame_srcs: Dict[int, Dict[int, set]] = defaultdict(lambda: defaultdict(set))
    doa_data: Dict[int, Dict[str, list]] = {c: {'az': [], 'el': []} for c in range(n_classes)}
    frame_offset = 0

    for csv_p in csv_paths:
        events = _parse_raw_tau2019(csv_p)
        if not events:
            continue
        clip_n_frames = max(e[0] for e in events) + 1
        total_frames_100ms += clip_n_frames
        for (frame, cls, src, az, el) in events:
            abs_frame = frame_offset + frame
            cls_frames_active[cls].add(abs_frame)
            cls_frame_srcs[cls][abs_frame].add(src)
            doa_data[cls]['az'].append(az)
            doa_data[cls]['el'].append(el)
        frame_offset += clip_n_frames

    # Reuse the per-class stat computation logic
    table2: Dict = {}
    for cls in range(n_classes):
        active_set = cls_frames_active[cls]
        n_active = len(active_set)
        if total_frames_100ms == 0 or n_active == 0:
            table2[cls] = {'frame_coverage_pct': 0.0, 'max_poly': 0,
                           'mean_poly': 0.0, 'poly_dist': [0.0]*4}
            continue
        frame_coverage_pct = 100.0 * n_active / total_frames_100ms
        poly_counts = [len(cls_frame_srcs[cls][f]) for f in active_set]
        max_poly  = int(np.max(poly_counts))
        mean_poly = float(np.mean(poly_counts))
        poly_dist = []
        for k in range(1, 5):
            cnt = (sum(1 for p in poly_counts if p == k)
                   if k < 4 else sum(1 for p in poly_counts if p >= 4))
            poly_dist.append(100.0 * cnt / n_active)
        table2[cls] = {'frame_coverage_pct': frame_coverage_pct, 'max_poly': max_poly,
                       'mean_poly': mean_poly, 'poly_dist': poly_dist}

    all_active_frames: set = set()
    for cls in range(n_classes):
        all_active_frames |= cls_frames_active[cls]
    global_frame_cov = 100.0 * len(all_active_frames) / max(total_frames_100ms, 1)
    global_frame_poly: Dict[int, set] = defaultdict(set)
    frame_offset2 = 0
    for csv_p in csv_paths:
        events = _parse_raw_tau2019(csv_p)
        if not events:
            continue
        clip_n_frames = max(e[0] for e in events) + 1
        for (frame, cls, src, az, el) in events:
            global_frame_poly[frame_offset2 + frame].add((cls, src))
        frame_offset2 += clip_n_frames
    global_poly_counts = [len(v) for v in global_frame_poly.values() if v]
    if global_poly_counts:
        global_max_poly  = int(np.max(global_poly_counts))
        global_mean_poly = float(np.mean(global_poly_counts))
        n_act = len(global_poly_counts)
        global_poly_dist = []
        for k in range(1, 5):
            cnt = (sum(1 for p in global_poly_counts if p == k)
                   if k < 4 else sum(1 for p in global_poly_counts if p >= 4))
            global_poly_dist.append(100.0 * cnt / max(n_act, 1))
    else:
        global_max_poly, global_mean_poly = 0, 0.0
        global_poly_dist = [0.0]*4
    table2['global'] = {'frame_coverage_pct': global_frame_cov, 'max_poly': global_max_poly,
                        'mean_poly': global_mean_poly, 'poly_dist': global_poly_dist}

    return table2, doa_data


def _native_stats(csv_paths: List[Path], n_classes: int, n_cols: int = 6
                  ) -> Tuple[Dict, Dict]:
    """
    Parse CSV files directly and compute TABLE 2-style statistics.

    Returns
    -------
    table2 : dict  {cls_idx → stats_dict}  +  key 'global'
    doa_data : dict  {cls_idx → {'az': list, 'el': list}}
    """
    total_frames_100ms = 0

    cls_frames_active: Dict[int, set] = defaultdict(set)
    # cls_frame_srcs[cls][abs_frame] = set of source_idx seen
    cls_frame_srcs: Dict[int, Dict[int, set]] = defaultdict(lambda: defaultdict(set))
    doa_data: Dict[int, Dict[str, list]] = {c: {'az': [], 'el': []} for c in range(n_classes)}

    frame_offset = 0  # cumulative frame offset across clips

    for csv_p in csv_paths:
        events = _parse_raw(csv_p, n_cols)

        clip_n_frames = (max(e[0] for e in events) + 1) if events else 0
        total_frames_100ms += clip_n_frames

        for (frame, cls, src, az, el) in events:
            abs_frame = frame_offset + frame
            cls_frames_active[cls].add(abs_frame)
            cls_frame_srcs[cls][abs_frame].add(src)
            if cls < n_classes:
                doa_data[cls]['az'].append(az)
                doa_data[cls]['el'].append(el)

        frame_offset += clip_n_frames

    # ── Per-class stats ──────────────────────────────────────────────────────
    table2: Dict = {}

    for cls in range(n_classes):
        active_set = cls_frames_active[cls]
        n_active = len(active_set)

        if total_frames_100ms == 0 or n_active == 0:
            table2[cls] = {
                'frame_coverage_pct': 0.0,
                'max_poly': 0,
                'mean_poly': 0.0,
                'poly_dist': [0.0, 0.0, 0.0, 0.0],
            }
            continue

        frame_coverage_pct = 100.0 * n_active / total_frames_100ms

        poly_counts = [len(cls_frame_srcs[cls][f]) for f in active_set]
        max_poly  = int(np.max(poly_counts))
        mean_poly = float(np.mean(poly_counts))

        poly_dist = []
        for k in range(1, 5):
            cnt = (sum(1 for p in poly_counts if p == k)
                   if k < 4 else
                   sum(1 for p in poly_counts if p >= 4))
            poly_dist.append(100.0 * cnt / n_active)

        table2[cls] = {
            'frame_coverage_pct': frame_coverage_pct,
            'max_poly':           max_poly,
            'mean_poly':          mean_poly,
            'poly_dist':          poly_dist,
        }

    # ── Global row ───────────────────────────────────────────────────────────
    all_active_frames = set()
    for cls in range(n_classes):
        all_active_frames |= cls_frames_active[cls]
    global_frame_cov = 100.0 * len(all_active_frames) / max(total_frames_100ms, 1)

    # Global polyphony: number of distinct (cls, src) pairs active per frame
    global_frame_poly: Dict[int, set] = defaultdict(set)
    frame_offset2 = 0
    for csv_p in csv_paths:
        events = _parse_raw(csv_p, n_cols)
        clip_n_frames = (max(e[0] for e in events) + 1) if events else 0
        for (frame, cls, src, az, el) in events:
            global_frame_poly[frame_offset2 + frame].add((cls, src))
        frame_offset2 += clip_n_frames

    global_poly_counts = [len(v) for v in global_frame_poly.values() if v]
    if global_poly_counts:
        global_max_poly  = int(np.max(global_poly_counts))
        global_mean_poly = float(np.mean(global_poly_counts))
        n_act = len(global_poly_counts)
        global_poly_dist = []
        for k in range(1, 5):
            cnt = (sum(1 for p in global_poly_counts if p == k)
                   if k < 4 else
                   sum(1 for p in global_poly_counts if p >= 4))
            global_poly_dist.append(100.0 * cnt / max(n_act, 1))
    else:
        global_max_poly, global_mean_poly = 0, 0.0
        global_poly_dist = [0.0, 0.0, 0.0, 0.0]

    table2['global'] = {
        'frame_coverage_pct': global_frame_cov,
        'max_poly':           global_max_poly,
        'mean_poly':          global_mean_poly,
        'poly_dist':          global_poly_dist,
    }

    return table2, doa_data


# ── B2. Model-mode stats from SELDDataset tensors ────────────────────────────

def _model_stats(ds: SELDDataset, n_classes: int) -> Dict:
    """
    Compute TABLE 2-style statistics from SELDDataset target tensors.

    Uses full-clip mode (window_s=None) so each clip is one item.
    Polyphony per class = number of active tracks per (mel_frame, class).
    """
    total_mel_frames = 0

    cls_mel_frames_active: Dict[int, int] = defaultdict(int)
    cls_poly_counts: Dict[int, List[int]] = defaultdict(list)

    global_active_mel = 0
    global_poly_counts: List[int] = []

    for idx in range(len(ds)):
        _, tgt = ds[idx]          # (T, N, C, 3)
        T = tgt.shape[0]
        total_mel_frames += T

        norm   = tgt.norm(dim=-1)  # (T, N, C)
        active = norm > 0          # (T, N, C) bool

        for cls in range(n_classes):
            cls_active_t = active[:, :, cls]        # (T, N)
            cls_any_t    = cls_active_t.any(dim=1)  # (T,)
            cls_mel_frames_active[cls] += int(cls_any_t.sum())
            poly_t = cls_active_t.sum(dim=1)        # (T,) — track count
            cls_poly_counts[cls].extend(poly_t[cls_any_t].tolist())

        global_active_t = active.any(dim=2).any(dim=1)   # (T,)
        global_active_mel += int(global_active_t.sum())
        cls_any = active.any(dim=1)                       # (T, C)
        global_poly_t = cls_any.sum(dim=1)                # (T,)
        global_poly_counts.extend(global_poly_t[global_active_t].tolist())

    table2: Dict = {}

    for cls in range(n_classes):
        n_active = cls_mel_frames_active[cls]
        frame_cov = 100.0 * n_active / max(total_mel_frames, 1)

        poly_c = cls_poly_counts[cls]
        if poly_c:
            max_poly  = int(np.max(poly_c))
            mean_poly = float(np.mean(poly_c))
            poly_dist = []
            for k in range(1, 5):
                cnt = (sum(1 for p in poly_c if p == k)
                       if k < 4 else
                       sum(1 for p in poly_c if p >= 4))
                poly_dist.append(100.0 * cnt / max(n_active, 1))
        else:
            max_poly, mean_poly, poly_dist = 0, 0.0, [0.0, 0.0, 0.0, 0.0]

        table2[cls] = {
            'frame_coverage_pct': frame_cov,
            'max_poly':           max_poly,
            'mean_poly':          mean_poly,
            'poly_dist':          poly_dist,
        }

    global_frame_cov = 100.0 * global_active_mel / max(total_mel_frames, 1)
    if global_poly_counts:
        gmax  = int(np.max(global_poly_counts))
        gmean = float(np.mean(global_poly_counts))
        n_act = len(global_poly_counts)
        gpoly = []
        for k in range(1, 5):
            cnt = (sum(1 for p in global_poly_counts if p == k)
                   if k < 4 else
                   sum(1 for p in global_poly_counts if p >= 4))
            gpoly.append(100.0 * cnt / max(n_act, 1))
    else:
        gmax, gmean, gpoly = 0, 0.0, [0.0, 0.0, 0.0, 0.0]

    table2['global'] = {
        'frame_coverage_pct': global_frame_cov,
        'max_poly':           gmax,
        'mean_poly':          gmean,
        'poly_dist':          gpoly,
    }

    return table2


# ── B3. CSV export ────────────────────────────────────────────────────────────

def _save_table2_csv(table2: Dict, class_labels: List[str],
                     output_path: Path):
    """Save TABLE 2-style statistics to a CSV file in analysis/results/."""
    fieldnames = [
        'class',
        'frame_coverage_%',
        'max_polyphony',
        'mean_polyphony',
        'poly1_%',
        'poly2_%',
        'poly3_%',
        'poly4+_%',
    ]

    def _row(name: str, stats: dict) -> dict:
        pd = stats['poly_dist']
        return {
            'class':            name,
            'frame_coverage_%': f"{stats['frame_coverage_pct']:.2f}",
            'max_polyphony':    stats['max_poly'],
            'mean_polyphony':   f"{stats['mean_poly']:.3f}",
            'poly1_%':          f"{pd[0]:.2f}",
            'poly2_%':          f"{pd[1]:.2f}",
            'poly3_%':          f"{pd[2]:.2f}",
            'poly4+_%':         f"{pd[3]:.2f}",
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(_row('Global', table2['global']))
        for ci, label in enumerate(class_labels):
            if ci in table2:
                writer.writerow(_row(label, table2[ci]))

    print(f"    → saved: {output_path.relative_to(_ROOT)}")


# ── B4. DOA figure ────────────────────────────────────────────────────────────

def _plot_doa_analysis(doa_data: Dict[int, Dict[str, list]],
                       class_labels: List[str],
                       dataset_name: str,
                       output_path: Path):
    """
    Generate per-class DOA figure styled after STARSS23 paper Figure 3.
    Saved as SVG at 600 DPI in figures/.

    Layout: one 2×2 panel per class + one "Overall" panel at the end.
    Each panel:
        [az hist (top-left)]   [empty]
        [az-el scatter (center)] [el hist rotated (right)]

    The Overall panel shows:
        - scatter: all classes overlaid, each in its class colour
        - az hist: stacked contributions per class
        - el hist: stacked contributions per class (horizontal)
    """
    n_classes  = len(class_labels)
    N_PER_ROW  = 4
    # Total slots = per-class panels + 1 Overall panel
    n_slots    = n_classes + 1
    n_rows     = math.ceil(n_slots / N_PER_ROW)

    # ── Per-class colour palette ─────────────────────────────────────────────
    cmap   = plt.get_cmap('tab10' if n_classes <= 10 else 'tab20')
    colors = [cmap(i % cmap.N) for i in range(n_classes)]

    fig_w = 14.0
    fig_h = 3.8 * n_rows
    fig = plt.figure(figsize=(fig_w, fig_h))

    outer_gs = GridSpec(
        n_rows, N_PER_ROW,
        figure  = fig,
        hspace  = 0.55,
        wspace  = 0.30,
        left    = 0.05,
        right   = 0.97,
        top     = 0.92,
        bottom  = 0.06,
    )

    def _draw_panel(slot_idx: int, az: np.ndarray, el: np.ndarray,
                    title: str, color, scatter_data=None):
        """
        Draw one 2×2 panel into outer_gs[slot_idx].

        If scatter_data is not None it should be a list of (az, el, color)
        tuples — used for the Overall panel to overlay all classes.
        Otherwise az/el/color are used directly for the single-class panel.
        """
        row = slot_idx // N_PER_ROW
        col = slot_idx % N_PER_ROW

        inner = GridSpecFromSubplotSpec(
            2, 2,
            subplot_spec  = outer_gs[row, col],
            height_ratios = [1, 3],
            width_ratios  = [3, 1],
            hspace        = 0.05,
            wspace        = 0.05,
        )

        ax_az = fig.add_subplot(inner[0, 0])
        ax_sc = fig.add_subplot(inner[1, 0])
        ax_el = fig.add_subplot(inner[1, 1])

        ax_az.sharex(ax_sc)
        ax_el.sharey(ax_sc)

        # ── Scatter (center) ─────────────────────────────────────────────────
        if scatter_data is not None:
            # Overall: overlay all classes
            for (az_c, el_c, col_c) in scatter_data:
                if len(az_c):
                    ax_sc.scatter(
                        az_c, el_c,
                        s         = 1.0,
                        alpha     = 0.15,
                        color     = col_c,
                        linewidths= 0,
                        rasterized= True,
                    )
        else:
            if len(az):
                ax_sc.scatter(
                    az, el,
                    s         = 1.5,
                    alpha     = 0.25,
                    color     = color,
                    linewidths= 0,
                    rasterized= True,
                )

        ax_sc.set_xlim(-180, 180)
        ax_sc.set_ylim(-90, 90)
        ax_sc.set_xlabel('Azimuth (°)', fontsize=6, labelpad=1)
        ax_sc.set_ylabel('Elevation (°)', fontsize=6, labelpad=1)
        ax_sc.tick_params(axis='both', labelsize=5)
        ax_sc.set_xticks([-180, -90, 0, 90, 180])
        ax_sc.set_yticks([-90, -45, 0, 45, 90])

        # ── Azimuth histogram (top) ──────────────────────────────────────────
        n_pts_title = len(az) if scatter_data is None else sum(
            len(d[0]) for d in scatter_data)
        if scatter_data is not None:
            # Stacked: one list per class
            az_lists = [np.array(d[0]) for d in scatter_data]
            col_list  = [d[2] for d in scatter_data]
            non_empty = [(a, c) for a, c in zip(az_lists, col_list) if len(a)]
            if non_empty:
                ax_az.hist(
                    [a for a, _ in non_empty],
                    bins      = 72,
                    range     = (-180, 180),
                    stacked   = True,
                    color     = [c for _, c in non_empty],
                    alpha     = 0.85,
                    linewidth = 0,
                )
        else:
            if len(az):
                ax_az.hist(
                    az,
                    bins      = 72,
                    range     = (-180, 180),
                    color     = color,
                    alpha     = 0.85,
                    linewidth = 0,
                )

        ax_az.set_title(f"{title}\n({n_pts_title:,} frames)", fontsize=7,
                        pad=2, fontweight='bold')
        ax_az.tick_params(labelbottom=False, labelsize=5)
        ax_az.set_yticks([])
        ax_az.spines[['top', 'right', 'left']].set_visible(False)

        # ── Elevation histogram rotated (right) ──────────────────────────────
        if scatter_data is not None:
            el_lists = [np.array(d[1]) for d in scatter_data]
            col_list  = [d[2] for d in scatter_data]
            non_empty = [(e, c) for e, c in zip(el_lists, col_list) if len(e)]
            if non_empty:
                ax_el.hist(
                    [e for e, _ in non_empty],
                    bins        = 36,
                    range       = (-90, 90),
                    stacked     = True,
                    orientation = 'horizontal',
                    color       = [c for _, c in non_empty],
                    alpha       = 0.85,
                    linewidth   = 0,
                )
        else:
            if len(el):
                ax_el.hist(
                    el,
                    bins        = 36,
                    range       = (-90, 90),
                    orientation = 'horizontal',
                    color       = color,
                    alpha       = 0.85,
                    linewidth   = 0,
                )

        ax_el.tick_params(labelleft=False, labelsize=5)
        ax_el.set_xticks([])
        ax_el.spines[['top', 'right', 'bottom']].set_visible(False)

    # ── Draw per-class panels ────────────────────────────────────────────────
    for i, label in enumerate(class_labels):
        az = np.array(doa_data[i]['az'], dtype=np.float32)
        el = np.array(doa_data[i]['el'], dtype=np.float32)
        _draw_panel(i, az, el, label, colors[i])

    # ── Draw Overall panel ───────────────────────────────────────────────────
    scatter_data = [
        (np.array(doa_data[c]['az'], dtype=np.float32),
         np.array(doa_data[c]['el'], dtype=np.float32),
         colors[c])
        for c in range(n_classes)
    ]
    _draw_panel(n_classes, np.array([]), np.array([]),
                'Overall', None, scatter_data=scatter_data)

    # ── Hide remaining empty slots (after Overall) ───────────────────────────
    for j in range(n_slots, n_rows * N_PER_ROW):
        row = j // N_PER_ROW
        col = j % N_PER_ROW
        ax_dummy = fig.add_subplot(outer_gs[row, col])
        ax_dummy.set_visible(False)

    # ── Legend (class → colour) ──────────────────────────────────────────────
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=colors[c], markersize=5, label=class_labels[c])
        for c in range(n_classes)
    ]
    fig.legend(
        handles        = handles,
        loc            = 'lower center',
        ncol           = min(n_classes, 7),
        fontsize       = 6,
        frameon        = False,
        bbox_to_anchor = (0.5, 0.0),
        bbox_transform = fig.transFigure,
    )

    fig.suptitle(
        f"{dataset_name} — DOA distribution per class"
        f" (train split, native 100 ms resolution)",
        fontsize = 10,
        y        = 0.995,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"    → saved: {output_path.relative_to(_ROOT)}")


# ── B5. DOA from windowed SELDDataset ────────────────────────────────────────

def _doa_from_windowed_dataset(ds: SELDDataset, n_classes: int
                               ) -> Dict[int, Dict[str, list]]:
    """
    Extract az/el DOA values from a windowed SELDDataset by inverting
    the Multi-ACCDOA xyz unit vectors stored in the target tensor.

    Frames that appear in multiple overlapping windows are counted multiple
    times, reflecting the actual distribution seen during training.

    xyz → az/el:
        az = atan2(x, y)   (x = sin(az)cos(el),  y = cos(az)cos(el))
        el = asin(z)        (z = sin(el))
    """
    doa: Dict[int, Dict[str, list]] = {c: {'az': [], 'el': []} for c in range(n_classes)}
    rad2deg = 180.0 / math.pi

    for idx in range(len(ds)):
        _, tgt = ds[idx]                        # (T, N, C, 3)
        norm   = tgt.norm(dim=-1)               # (T, N, C)
        active = norm > 0
        if not active.any():
            continue

        xyz = tgt / norm.unsqueeze(-1).clamp(min=1e-8)   # unit vectors
        x = xyz[..., 0]                                   # (T, N, C)
        y = xyz[..., 1]
        z = xyz[..., 2].clamp(-1.0, 1.0)
        az_deg = torch.atan2(x, y).numpy() * rad2deg      # (T, N, C)
        el_deg = torch.asin(z).numpy()      * rad2deg

        for c in range(n_classes):
            mask = active[:, :, c].numpy()   # (T, N) bool
            if mask.any():
                doa[c]['az'].extend(az_deg[:, :, c][mask].tolist())
                doa[c]['el'].extend(el_deg[:, :, c][mask].tolist())

    return doa


# ── B6. Overall comparison figure ────────────────────────────────────────────

def _plot_overall_comparison(
    comparison_data: Dict[str, Dict[str, Dict]],
    figures_dir: Path,
):
    """
    4-panel (or 2-panel if single dataset) figure comparing DOA distributions:
      - native  : parsed directly from CSV at 100 ms resolution (all train clips)
      - windowed: extracted from SELDDataset train split with 10 s / 50 % hop

    Layout: n_datasets rows × 2 columns.
    Each panel is a 2×2 inner grid (az hist top / az-el scatter center /
    el hist rotated right), with all classes overlaid and colour-coded.
    A per-dataset colour legend is embedded in the native scatter axis.
    """
    dataset_names = list(comparison_data.keys())
    n_ds   = len(dataset_names)
    n_cols = 2   # native | windowed

    fig_w = 7.0
    fig_h = 3.8 * n_ds
    fig = plt.figure(figsize=(fig_w, fig_h))

    outer_gs = GridSpec(
        n_ds, n_cols,
        figure  = fig,
        hspace  = 0.50,
        wspace  = 0.28,
        left    = 0.07,
        right   = 0.97,
        top     = 0.90,
        bottom  = 0.06,
    )

    mode_titles = {
        'native':   'Native (CSV, 100 ms)',
        'windowed': 'Windowed (10 s / 50 % hop)',
    }

    for row, ds_name in enumerate(dataset_names):
        n_cls    = len(CLASS_LABELS[ds_name])
        cmap     = plt.get_cmap('tab10' if n_cls <= 10 else 'tab20')
        colors   = [cmap(i % cmap.N) for i in range(n_cls)]
        labels   = CLASS_LABELS[ds_name]

        legend_handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=colors[c], markersize=4,
                       label=labels[c])
            for c in range(n_cls)
        ]

        for col, mode in enumerate(['native', 'windowed']):
            doa_d = comparison_data[ds_name][mode]
            scatter_data = [
                (np.array(doa_d[c]['az'], dtype=np.float32),
                 np.array(doa_d[c]['el'], dtype=np.float32),
                 colors[c])
                for c in range(n_cls)
            ]

            inner = GridSpecFromSubplotSpec(
                2, 2,
                subplot_spec  = outer_gs[row, col],
                height_ratios = [1, 3],
                width_ratios  = [3, 1],
                hspace        = 0.05,
                wspace        = 0.05,
            )
            ax_az = fig.add_subplot(inner[0, 0])
            ax_sc = fig.add_subplot(inner[1, 0])
            ax_el = fig.add_subplot(inner[1, 1])
            ax_az.sharex(ax_sc)
            ax_el.sharey(ax_sc)

            # ── Scatter ──────────────────────────────────────────────────────
            for (az_c, el_c, col_c) in scatter_data:
                if len(az_c):
                    ax_sc.scatter(
                        az_c, el_c,
                        s         = 0.8,
                        alpha     = 0.12,
                        color     = col_c,
                        linewidths= 0,
                        rasterized= True,
                    )
            ax_sc.set_xlim(-180, 180)
            ax_sc.set_ylim(-90, 90)
            ax_sc.set_xlabel('Azimuth (°)', fontsize=6, labelpad=1)
            ax_sc.set_ylabel('Elevation (°)', fontsize=6, labelpad=1)
            ax_sc.tick_params(axis='both', labelsize=5)
            ax_sc.set_xticks([-180, -90, 0, 90, 180])
            ax_sc.set_yticks([-90, -45, 0, 45, 90])

            # Legend only on the native column to avoid duplication
            if col == 0:
                ax_sc.legend(
                    handles    = legend_handles,
                    fontsize   = 4.0,
                    markerscale= 1.2,
                    loc        = 'upper right',
                    frameon    = True,
                    framealpha = 0.6,
                    ncol       = 1,
                    handlelength = 1.0,
                    borderpad  = 0.4,
                )

            # ── Az histogram (top) ───────────────────────────────────────────
            non_empty = [(d[0], d[2]) for d in scatter_data if len(d[0])]
            if non_empty:
                ax_az.hist(
                    [a for a, _ in non_empty],
                    bins      = 72,
                    range     = (-180, 180),
                    stacked   = True,
                    color     = [c for _, c in non_empty],
                    alpha     = 0.85,
                    linewidth = 0,
                )
            total_pts = sum(len(d[0]) for d in scatter_data)
            title = f"{ds_name} — {mode_titles[mode]}\n({total_pts:,} frames)"
            ax_az.set_title(title, fontsize=6.5, pad=2, fontweight='bold')
            ax_az.tick_params(labelbottom=False, labelsize=5)
            ax_az.set_yticks([])
            ax_az.spines[['top', 'right', 'left']].set_visible(False)

            # ── El histogram rotated (right) ─────────────────────────────────
            non_empty_el = [(d[1], d[2]) for d in scatter_data if len(d[1])]
            if non_empty_el:
                ax_el.hist(
                    [e for e, _ in non_empty_el],
                    bins        = 36,
                    range       = (-90, 90),
                    stacked     = True,
                    orientation = 'horizontal',
                    color       = [c for _, c in non_empty_el],
                    alpha       = 0.85,
                    linewidth   = 0,
                )
            ax_el.tick_params(labelleft=False, labelsize=5)
            ax_el.set_xticks([])
            ax_el.spines[['top', 'right', 'bottom']].set_visible(False)

    fig.suptitle(
        'Overall DOA distribution — Native CSV vs Windowed model (train split)',
        fontsize = 9,
        y        = 0.995,
    )

    stem = '_'.join(name.replace('-', '').replace(' ', '') for name in dataset_names)
    out_path = figures_dir / f'{stem}_comparison.svg'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"    → saved: {out_path.relative_to(_ROOT)}")


# ── B7. Console TABLE 2 report ────────────────────────────────────────────────

def _print_table2(table2: Dict, class_labels: List[str],
                  dataset_name: str, split: str, mode: str):
    _separator('═')
    print(f"  TABLE 2  |  {dataset_name}  |  split={split}  |  mode={mode}")
    _separator('═')
    header = (
        f"  {'Class':<22s}  {'Cov%':>6s}  {'MaxP':>4s}  {'MeanP':>6s}"
        f"  {'P1%':>6s}  {'P2%':>6s}  {'P3%':>6s}  {'P4+%':>6s}"
    )
    print(header)
    _separator()

    def _row(name, stats):
        pd = stats['poly_dist']
        print(
            f"  {name:<22s}  {stats['frame_coverage_pct']:>6.2f}  "
            f"{stats['max_poly']:>4d}  {stats['mean_poly']:>6.3f}  "
            f"{pd[0]:>6.2f}  {pd[1]:>6.2f}  {pd[2]:>6.2f}  {pd[3]:>6.2f}"
        )

    _row('Global', table2['global'])
    _separator('·')
    for ci, label in enumerate(class_labels):
        if ci in table2:
            _row(label, table2[ci])
    print()


# ── B6. Orchestrator ──────────────────────────────────────────────────────────

def run_analysis(dataset_names: List[str],
                 results_dir: Path,
                 figures_dir: Path):
    """Run the run analysis step."""

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Accumulated for the overall-comparison figure: {ds_name: {mode: doa_data}}
    comparison_data: Dict[str, Dict[str, Dict]] = {}

    for dataset_name in dataset_names:
        n_classes    = len(CLASS_LABELS[dataset_name])
        n_cols       = 6  if dataset_name == 'STARSS23' else 5
        class_labels = CLASS_LABELS[dataset_name]
        splits       = _SPLITS[dataset_name]

        comparison_data[dataset_name] = {}

        print(f"\n{'═'*70}")
        print(f"  ANALYSIS: {dataset_name}")
        print(f"{'═'*70}")

        # ── Native mode ───────────────────────────────────────────────────────
        doa_data_train: Optional[Dict] = None
        for split in splits:
            print(f"\n  ── Native mode | {split} ──────────────────────────────")
            csv_files = _get_native_csv_files(dataset_name, split)
            print(f"  Parsing {len(csv_files)} CSV files …", flush=True)
            t0 = time.perf_counter()
            table2_nat, doa_data = _native_stats(csv_files, n_classes, n_cols)
            elapsed = (time.perf_counter() - t0) * 1e3
            print(f"  Done in {elapsed:.0f} ms")

            _print_table2(table2_nat, class_labels, dataset_name, split, 'native')

            out_csv = results_dir / f'{dataset_name}_table2_native_{split}.csv'
            _save_table2_csv(table2_nat, class_labels, out_csv)

            if split == 'train':
                doa_data_train = doa_data
                out_fig = figures_dir / f'{dataset_name}_doa_analysis.svg'
                print(f"  Generating DOA figure …", flush=True)
                _plot_doa_analysis(doa_data, class_labels, dataset_name, out_fig)

        comparison_data[dataset_name]['native'] = doa_data_train

        # ── Model mode (full-clip, mel-frame resolution) ─────────────────────
        for split in splits:
            print(f"\n  ── Model mode | {split} ───────────────────────────────")
            print(f"  Building SELDDataset (full-clip) …", flush=True)
            t0 = time.perf_counter()
            ds = _make_dataset(dataset_name, split, window_s=None, hop_s=None)
            build_ms = (time.perf_counter() - t0) * 1e3
            print(f"  → {len(ds):,} clips in {build_ms:.0f} ms")

            print(f"  Computing model stats over {len(ds)} items …", flush=True)
            t0 = time.perf_counter()
            table2_mod = _model_stats(ds, n_classes)
            elapsed = (time.perf_counter() - t0) * 1e3
            print(f"  Done in {elapsed:.0f} ms")

            _print_table2(table2_mod, class_labels, dataset_name, split, 'model')

            out_csv = results_dir / f'{dataset_name}_table2_model_{split}.csv'
            _save_table2_csv(table2_mod, class_labels, out_csv)

        # ── Windowed DOA (train split, for overall comparison) ────────────────
        print(f"\n  ── Windowed DOA extraction | train ──────────────────────")
        print(f"  Building windowed SELDDataset (10 s / 50 % hop) …", flush=True)
        t0 = time.perf_counter()
        ds_win = _make_dataset(dataset_name, 'train', window_s=10.0, hop_s=5.0)
        build_ms = (time.perf_counter() - t0) * 1e3
        print(f"  → {len(ds_win):,} windows in {build_ms:.0f} ms")

        print(f"  Extracting DOA from {len(ds_win)} windows …", flush=True)
        t0 = time.perf_counter()
        doa_windowed = _doa_from_windowed_dataset(ds_win, n_classes)
        elapsed = (time.perf_counter() - t0) * 1e3
        print(f"  Done in {elapsed:.0f} ms")

        comparison_data[dataset_name]['windowed'] = doa_windowed

    # ── Overall comparison figure (all collected datasets) ────────────────────
    print(f"\n  Generating overall comparison figure …", flush=True)
    _plot_overall_comparison(comparison_data, figures_dir)


# ── B6b. TAU2019 analysis orchestrator ───────────────────────────────────────

def run_tau2019_analysis(results_dir: Path, figures_dir: Path):
    """Native CSV + model TABLE 2 analysis and DOA figure for TAU2019."""
    if not ROOTS['TAU2019'].exists():
        print(f"\n[SKIP] TAU2019 data not found at {ROOTS['TAU2019']}")
        return

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    n_classes    = 11
    class_labels = CLASS_LABELS['TAU2019']
    splits       = _SPLITS['TAU2019']

    print(f"\n{'═'*70}")
    print(f"  ANALYSIS: TAU2019")
    print(f"{'═'*70}")

    # ── Native mode ───────────────────────────────────────────────────────────
    doa_data_train: Optional[Dict] = None
    for split in splits:
        print(f"\n  ── Native mode | {split} ──────────────────────────────")
        csv_files = _get_native_csv_files_tau2019(split)
        print(f"  Parsing {len(csv_files)} CSV files …", flush=True)
        t0 = time.perf_counter()
        table2_nat, doa_data = _native_stats_tau2019(csv_files)
        elapsed = (time.perf_counter() - t0) * 1e3
        print(f"  Done in {elapsed:.0f} ms")

        _print_table2(table2_nat, class_labels, 'TAU2019', split, 'native')

        out_csv = results_dir / f'TAU2019_table2_native_{split}.csv'
        _save_table2_csv(table2_nat, class_labels, out_csv)

        if split == 'train':
            doa_data_train = doa_data
            out_fig = figures_dir / 'TAU2019_doa_analysis.svg'
            print(f"  Generating DOA figure …", flush=True)
            _plot_doa_analysis(doa_data, class_labels, 'TAU2019', out_fig)

    # ── Model mode (full-clip, mel-frame resolution) ──────────────────────────
    for split in splits:
        print(f"\n  ── Model mode | {split} ───────────────────────────────")
        print(f"  Building TAU2019Dataset (full-clip, mixed_mode) …", flush=True)
        t0 = time.perf_counter()
        ds = _make_tau2019_dataset(split, mixed_mode=True)
        build_ms = (time.perf_counter() - t0) * 1e3
        print(f"  → {len(ds):,} clips in {build_ms:.0f} ms")

        print(f"  Computing model stats over {len(ds)} items …", flush=True)
        t0 = time.perf_counter()
        table2_mod = _model_stats(ds, n_classes)
        elapsed = (time.perf_counter() - t0) * 1e3
        print(f"  Done in {elapsed:.0f} ms")

        _print_table2(table2_mod, class_labels, 'TAU2019', split, 'model')

        out_csv = results_dir / f'TAU2019_table2_model_{split}.csv'
        _save_table2_csv(table2_mod, class_labels, out_csv)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Run the command-line workflow."""

    parser = argparse.ArgumentParser(
        description='SELD Dataset smoke-test, TABLE 2 analysis and DOA figures'
    )
    parser.add_argument(
        '--dataset',
        choices=['STARSS23', 'TAU-NIGENS2021', 'TAU-NIGENS2020',
                 'TAU2019', 'Mixed', 'all'],
        default='all',
        help='Which dataset to test/analyse (default: all)',
    )
    parser.add_argument(
        '--no-window', action='store_true',
        help='Smoke test: use full-clip mode instead of windowed mode'
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--analysis-only', action='store_true',
        help='Run only TABLE 2 stats and DOA figures (skip smoke tests)'
    )
    mode_group.add_argument(
        '--smoke-only', action='store_true',
        help='Run only smoke tests (skip analysis and figures)'
    )
    args = parser.parse_args()

    run_smoke = not args.analysis_only
    run_anal  = not args.smoke_only
    windowed  = not args.no_window

    ds_filter = args.dataset

    # ── Smoke tests ───────────────────────────────────────────────────────────
    if run_smoke:
        smoke_names = (
            ['STARSS23', 'TAU-NIGENS2021', 'TAU-NIGENS2020']
            if ds_filter in ('STARSS23', 'TAU-NIGENS2021', 'TAU-NIGENS2020')
            else ['STARSS23', 'TAU-NIGENS2021', 'TAU-NIGENS2020']
            if ds_filter == 'all'
            else []
        )
        if ds_filter == 'STARSS23':
            smoke_names = ['STARSS23']
        elif ds_filter == 'TAU-NIGENS2021':
            smoke_names = ['TAU-NIGENS2021']
        elif ds_filter == 'TAU-NIGENS2020':
            smoke_names = ['TAU-NIGENS2020']
        elif ds_filter in ('all', 'TAU2019', 'Mixed'):
            smoke_names = ['STARSS23', 'TAU-NIGENS2021', 'TAU-NIGENS2020']

        if smoke_names:
            run_smoke_tests(smoke_names, windowed)

        if ds_filter in ('TAU2019', 'all'):
            run_tau2019_smoke_tests(windowed)

        if ds_filter in ('Mixed', 'all'):
            run_mixed_seld_smoke_tests(windowed)

    # ── Analysis ──────────────────────────────────────────────────────────────
    if run_anal:
        anal_names = []
        if ds_filter in ('STARSS23', 'all'):
            anal_names.append('STARSS23')
        if ds_filter in ('TAU-NIGENS2021', 'all'):
            anal_names.append('TAU-NIGENS2021')
        if ds_filter in ('TAU-NIGENS2020', 'all'):
            anal_names.append('TAU-NIGENS2020')

        if anal_names:
            run_analysis(anal_names, RESULTS_DIR, FIGURES_DIR)

        if ds_filter in ('TAU2019', 'all'):
            run_tau2019_analysis(RESULTS_DIR, FIGURES_DIR)


if __name__ == '__main__':
    main()
