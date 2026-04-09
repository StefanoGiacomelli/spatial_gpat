#!/usr/bin/env python3
"""
probe_datamodule.py — Real-Time DataModule Label Diagnostics
=============================================================
Instantiates AudioSetStrongDataModule exactly as main_at2sed.py does and
iterates through all batches of the chosen split, printing per-batch and
aggregate statistics about label distribution.

Useful for understanding the actual class imbalance in the DataModule and
for deciding whether WeightedRandomSampler or other rebalancing strategies
are needed.

Usage
-----
    # Train split, window=0.5s, all batches
    .venv/bin/python probe_datamodule.py

    # First 100 batches only, print every 10
    .venv/bin/python probe_datamodule.py --batches 100 --print-every 10

    # Val split, window=2s
    .venv/bin/python probe_datamodule.py --split val --window-s 2.0 --hop-s 2.0

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

import argparse
import csv
import statistics
import sys
import warnings
from collections import Counter
from pathlib import Path

import torch
import yaml

warnings.filterwarnings('ignore', message='.*No positive samples.*')
warnings.filterwarnings('ignore', message='.*elements was resized.*')

_ROOT = Path(__file__).resolve().parent.parent   # project root (tests/ → ../)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datasets.audioset_strong import AudioSetStrongDataModule  # noqa: E402


# ── Constants (mirror main_at2sed.py) ─────────────────────────────────────────
_CONFIGS_DIR = _ROOT / 'configs' / 'experiments_at2sed'
_DATASET_CFG = _ROOT / 'configs' / 'datasets' / 'AudioSetStrong.yaml'
_DEFAULT_EXP = 'at2sed_default'


# ══════════════════════════════════════════════════════════════════════════════
# Config builder (same merge logic as main_at2sed.py)
# ══════════════════════════════════════════════════════════════════════════════

def _build_cfg(exp_yaml: Path, window_s: float, hop_s: float,
               batch_size: int, num_workers: int) -> dict:
    with open(exp_yaml) as fh:
        cfg = yaml.safe_load(fh) or {}

    if _DATASET_CFG.exists():
        with open(_DATASET_CFG) as fh:
            ds_cfg = yaml.safe_load(fh) or {}
        for section in ('data', 'augmentation', 'training', 'semantic'):
            merged = {**ds_cfg.get(section, {}), **cfg.get(section, {})}
            if merged:
                cfg[section] = merged

    for section in ('data', 'semantic'):
        for key, val in cfg.get(section, {}).items():
            if isinstance(val, str) and not Path(val).is_absolute():
                resolved = (_ROOT / val).resolve()
                if resolved.exists():
                    cfg[section][key] = str(resolved)

    cfg.setdefault('data', {}).update({
        'window_s':    window_s,
        'hop_s':       hop_s,
        'num_workers': num_workers,
    })
    cfg.setdefault('training', {}).update({'batch_size': batch_size})
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# Class name loader
# ══════════════════════════════════════════════════════════════════════════════

def _load_class_names(class_map_csv: Path) -> dict:
    """Returns {int_index: display_name}."""
    names = {}
    with open(class_map_csv, newline='') as fh:
        for row in csv.DictReader(fh):
            idx  = int(row['index'])
            name = row.get('display_name', row.get('mid', str(idx)))
            names[idx] = name
    return names


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description='Probe AudioSetStrongDataModule label distribution.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--exp',          type=str,   default=_DEFAULT_EXP,
                   help='Experiment config stem.')
    p.add_argument('--split',        type=str,   default='train',
                   choices=['train', 'val'],
                   help='Which dataloader to probe.')
    p.add_argument('--window-s',     type=float, default=0.5,
                   help='Window duration in seconds.')
    p.add_argument('--hop-s',        type=float, default=0.5,
                   help='Hop duration in seconds.')
    p.add_argument('--batch-size',   type=int,   default=32)
    p.add_argument('--num-workers',  type=int,   default=4)
    p.add_argument('--batches',      type=int,   default=0,
                   help='Max batches to scan (0 = all).')
    p.add_argument('--print-every',  type=int,   default=50,
                   help='Print a batch line every N batches (1 = every batch).')
    p.add_argument('--top-k',        type=int,   default=20,
                   help='Show top-K and bottom-K classes in the final report.')
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args     = _parse_args()
    exp_yaml = _CONFIGS_DIR / f'{args.exp}.yaml'
    if not exp_yaml.exists():
        print(f'[ERROR] Experiment config not found: {exp_yaml}')
        sys.exit(1)

    cfg   = _build_cfg(exp_yaml, args.window_s, args.hop_s,
                       args.batch_size, args.num_workers)
    n_cls = int(cfg.get('model', {}).get('n_cls', 527))

    print(f'\n{"="*64}')
    print(f'  PROBE  split={args.split}  window={args.window_s}s  '
          f'hop={args.hop_s}s  n_cls={n_cls}')
    print(f'{"="*64}\n')

    # ── DataModule setup ──────────────────────────────────────────────────────
    dm = AudioSetStrongDataModule(cfg)
    dm.setup('fit')

    # ── pos_weight snapshot (computed on train items) ─────────────────────────
    pw = dm.pos_weights
    if pw is not None:
        pw_np  = pw.numpy()
        n_cap  = int((pw_np >= dm.pos_weight_max).sum())
        n_zero = int((pw_np <= 1.0).sum())  # classes with so few positives pw ≈ 1
        print(f'[pos_weight from train]')
        print(f'  min    = {pw_np.min():.2f}')
        print(f'  median = {float(pw.median()):.2f}')
        print(f'  mean   = {pw_np.mean():.2f}')
        print(f'  max    = {pw_np.max():.1f}  (cap = {dm.pos_weight_max:.0f})')
        print(f'  classes at cap  : {n_cap}')
        print(f'  classes pw ≤ 1.0: {n_zero}')
        print()

    # ── Class name lookup ─────────────────────────────────────────────────────
    class_map_path = Path(cfg['data']['class_map'])
    class_names    = (_load_class_names(class_map_path)
                      if class_map_path.exists() else {})

    # ── Choose loader ─────────────────────────────────────────────────────────
    loader       = (dm.train_dataloader() if args.split == 'train'
                    else dm.val_dataloader())
    total_batches = len(loader)
    scan_n        = (min(args.batches, total_batches)
                     if args.batches > 0 else total_batches)

    # ── Accumulators ──────────────────────────────────────────────────────────
    total_windows = 0
    pos_windows   = 0
    neg_windows   = 0
    class_counts  = torch.zeros(n_cls, dtype=torch.long)
    labels_per_pos: list[int] = []   # cardinality of label vectors for positive samples

    # ── Real-time scan ────────────────────────────────────────────────────────
    print(f'Scanning {scan_n}/{total_batches} batches  '
          f'(print every {args.print_every})...\n')
    hdr = (f'  {"batch":>6}  {"pos":>5}  {"neg":>5}  '
           f'{"pos%":>5}  {"run_pos%":>8}  active_classes_in_batch')
    print(hdr)
    print(f'  {"-"*6}  {"-"*5}  {"-"*5}  {"-"*5}  {"-"*8}  {"-"*30}')

    for i, (x, y) in enumerate(loader):
        if args.batches > 0 and i >= args.batches:
            break

        B       = y.shape[0]
        has_pos = (y.sum(dim=1) > 0)
        n_pos   = int(has_pos.sum().item())
        n_neg   = B - n_pos

        total_windows += B
        pos_windows   += n_pos
        neg_windows   += n_neg
        class_counts  += y.long().sum(dim=0)

        for lv in y[has_pos]:
            labels_per_pos.append(int(lv.sum().item()))

        if (i + 1) % args.print_every == 0 or i == 0:
            b_pos_rate = n_pos / B if B > 0 else 0.0
            run_rate   = pos_windows / total_windows
            # Unique active classes in this batch
            batch_classes = int((y.sum(dim=0) > 0).sum().item())
            print(
                f'  {i+1:>6}  {n_pos:>5}  {n_neg:>5}  '
                f'{b_pos_rate:>4.0%}  {run_rate:>7.1%}  '
                f'{batch_classes} active classes'
            )

    # ── Final report ──────────────────────────────────────────────────────────
    print(f'\n{"="*64}')
    print(f'  FINAL REPORT — {args.split}  ({total_windows:,} windows scanned)')
    print(f'{"="*64}')

    pos_rate = pos_windows / total_windows if total_windows else 0.0
    neg_rate = neg_windows / total_windows if total_windows else 0.0

    print(f'\n  Window counts')
    print(f'  {"Total":<20}: {total_windows:>10,}')
    print(f'  {"Positive (≥1 label)":<20}: {pos_windows:>10,}   {pos_rate:.2%}')
    print(f'  {"All-negative":<20}: {neg_windows:>10,}   {neg_rate:.2%}')

    # Label cardinality of positive windows
    if labels_per_pos:
        print(f'\n  Labels per positive window')
        print(f'  {"min":<8}: {min(labels_per_pos)}')
        print(f'  {"median":<8}: {statistics.median(labels_per_pos):.1f}')
        print(f'  {"mean":<8}: {sum(labels_per_pos)/len(labels_per_pos):.2f}')
        print(f'  {"max":<8}: {max(labels_per_pos)}')
        print(f'\n  Cardinality histogram (labels per positive window):')
        card_hist = Counter(labels_per_pos)
        max_cnt   = max(card_hist.values())
        for card in sorted(card_hist.keys()):
            bar = '█' * max(1, round(card_hist[card] / max_cnt * 35))
            print(f'    {card:3d} label(s): {card_hist[card]:>7,}  {bar}')

    # Class coverage
    seen   = int((class_counts > 0).sum().item())
    unseen = n_cls - seen
    print(f'\n  Class coverage')
    print(f'  {"Seen classes":<28}: {seen} / {n_cls}')
    print(f'  {"Never-seen classes (0 windows)":<28}: {unseen}')

    # Top-K most frequent
    k = args.top_k
    sorted_desc = class_counts.argsort(descending=True).tolist()
    sorted_asc  = [i for i in reversed(sorted_desc) if class_counts[i] > 0]

    print(f'\n  Top-{k} most frequent classes:')
    print(f'  {"idx":>4}  {"windows":>8}  {"% total":>7}  name')
    print(f'  {"-"*4}  {"-"*8}  {"-"*7}  {"-"*44}')
    for idx in sorted_desc[:k]:
        cnt  = int(class_counts[idx].item())
        pct  = cnt / total_windows * 100
        name = class_names.get(idx, f'class_{idx}')[:44]
        print(f'  {idx:>4}  {cnt:>8,}  {pct:>6.2f}%  {name}')

    print(f'\n  Bottom-{k} least frequent classes (non-zero):')
    print(f'  {"idx":>4}  {"windows":>8}  {"% total":>7}  name')
    print(f'  {"-"*4}  {"-"*8}  {"-"*7}  {"-"*44}')
    for idx in sorted_asc[:k]:
        cnt  = int(class_counts[idx].item())
        pct  = cnt / total_windows * 100
        name = class_names.get(idx, f'class_{idx}')[:44]
        print(f'  {idx:>4}  {cnt:>8,}  {pct:>6.2f}%  {name}')

    # Never-seen classes
    if unseen > 0:
        unseen_names = [
            f'[{i}] {class_names.get(i, f"class_{i}")}'
            for i in range(n_cls) if class_counts[i] == 0
        ]
        print(f'\n  Classes with 0 windows in this scan ({unseen}):')
        for nm in unseen_names:
            print(f'    {nm}')

    # pos_weight vs scan correlation for capped classes
    if pw is not None:
        capped = [i for i in range(n_cls) if pw[i].item() >= dm.pos_weight_max]
        if capped:
            # sort by actual count ascending (rarest first)
            capped.sort(key=lambda i: class_counts[i].item())
            print(f'\n  Classes at pos_weight cap ({dm.pos_weight_max:.0f})'
                  f' — rarest first (showing ≤15):')
            print(f'  {"idx":>4}  {"windows":>8}  {"pw":>6}  name')
            print(f'  {"-"*4}  {"-"*8}  {"-"*6}  {"-"*44}')
            for idx in capped[:15]:
                cnt  = int(class_counts[idx].item())
                name = class_names.get(idx, f'class_{idx}')[:44]
                print(f'  {idx:>4}  {cnt:>8,}  {pw[idx]:>6.1f}  {name}')
            if len(capped) > 15:
                print(f'  ... and {len(capped)-15} more')

    print(f'\n{"="*64}\n')


if __name__ == '__main__':
    main()
