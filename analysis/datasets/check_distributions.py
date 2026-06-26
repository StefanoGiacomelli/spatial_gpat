#!/usr/bin/env python3
"""
analysis/datasets/check_distributions.py
==========================================
Reports class-level item distributions for all three SELD datasets
(STARSS23, TAU-NIGENS2021, TAU2019) across train/val/test splits,
in both full-clip and windowed modes.

Output
------
    results/datasets/seld_dataset_distributions.csv

Run
---
    python analysis/datasets/check_distributions.py

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

DATA    = _ROOT / 'data'
RESULTS = _ROOT / 'results' / 'datasets'

# ── Dataset metadata ──────────────────────────────────────────────────────────

DATASETS = {
    'STARSS23': {
        'root':      DATA / 'STARSS23',
        'n_classes': 13,
        'cls_names': [
            'female_speech', 'male_speech', 'clapping', 'telephone', 'laughter',
            'domestic', 'footsteps', 'door', 'music', 'musical_instr',
            'water_tap', 'bell', 'knock',
        ],
        'splits': ['train', 'val', 'test'],
        'kwargs': {'val_fraction': 0.2, 'val_seed': 42},
    },
    'TAU-NIGENS2021': {
        'root':      DATA / 'TAU-NIGENS2021',
        'n_classes': 12,
        'cls_names': [
            'alarm', 'crying_baby', 'crash', 'barking_dog',
            'female_scream', 'female_speech', 'footsteps', 'knocking',
            'male_scream', 'male_speech', 'ringing_phone', 'piano',
        ],
        'splits': ['train', 'val', 'test'],
        'kwargs': {},
    },
    'TAU2019': {
        'root':      DATA / 'TAU2019',
        'n_classes': 11,
        'cls_names': [
            'clearthroat', 'cough', 'doorslam', 'drawer', 'keyboard',
            'keysDrop', 'knock', 'laughter', 'pageturn', 'phone', 'speech',
        ],
        'splits': ['train', 'val', 'test'],
        'kwargs': {'mixed_mode': True},
    },
}

MODEL_SR   = 24_000
HOP_LENGTH = 240
N_TRACKS   = 3
WINDOW_S   = 10.0
HOP_S      = 5.0


def _build_dataset(name: str, split: str, windowed: bool):
    info = DATASETS[name]
    root = info['root']
    kw   = dict(
        split      = split,
        model_sr   = MODEL_SR,
        hop_length = HOP_LENGTH,
        n_tracks   = N_TRACKS,
        window_s   = WINDOW_S if windowed else None,
        hop_s      = HOP_S    if windowed else None,
        **info['kwargs'],
    )
    if name == 'STARSS23':
        from datasets.starss23 import STARSS23Dataset
        return STARSS23Dataset(root=root, **kw)
    elif name == 'TAU-NIGENS2021':
        from datasets.tau_nigens import TAUNIGENSDataset
        return TAUNIGENSDataset(root=root, **kw)
    elif name == 'TAU2019':
        from datasets.tau2019 import TAU2019Dataset
        return TAU2019Dataset(root=root, **kw)
    raise ValueError(name)


def _count_classes(ds, n_classes: int, windowed: bool):
    """
    Return n_items and per-class item counts.
    An item is counted for class c if the class is active anywhere in that item.
    """
    n_items = len(ds._items)
    counts  = [0] * n_classes

    if not windowed:
        for ci, _ss, _ in ds._items:
            cls_set = {cls for (_, cls, *_) in ds._events[ci] if cls < n_classes}
            for c in cls_set:
                counts[c] += 1
    else:
        samples_per_csv = round(ds.orig_sr * ds.FRAME_DURATION_S)
        for ci, ss, _ in ds._items:
            s_f = ss // samples_per_csv
            e_f = (ss + ds.window_samples) // samples_per_csv
            cls_set = {
                cls for (f, cls, *_) in ds._events[ci]
                if s_f <= f < e_f and cls < n_classes
            }
            for c in cls_set:
                counts[c] += 1

    return n_items, counts


def main():
    """Run the command-line workflow."""

    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []

    for ds_name, info in DATASETS.items():
        if not info['root'].exists():
            print(f'[SKIP] {ds_name} — data directory not found')
            continue

        n_cls  = info['n_classes']
        names  = info['cls_names']

        for split in info['splits']:
            for windowed, mode_label in [(False, 'full-clip'), (True, 'windowed')]:
                try:
                    ds = _build_dataset(ds_name, split, windowed)
                except Exception as exc:
                    print(f'[ERR] {ds_name}/{split}/{mode_label}: {exc}')
                    continue

                n_items, counts = _count_classes(ds, n_cls, windowed)

                for c, (name, cnt) in enumerate(zip(names, counts)):
                    pct = 100 * cnt / max(n_items, 1)
                    rows.append({
                        'dataset':    ds_name,
                        'split':      split,
                        'mode':       mode_label,
                        'n_items':    n_items,
                        'class_idx':  c,
                        'class_name': name,
                        'item_count': cnt,
                        'pct_items':  round(pct, 2),
                    })

                non_zero = [c for c in counts if c > 0]
                max_c    = max(counts) if counts else 0
                min_c    = min(non_zero) if non_zero else 0
                print(f'  {ds_name:20s} {split:5s} {mode_label:10s}'
                      f'  items={n_items:5d}  max/min={max_c}/{min_c}')

    if not rows:
        print('No data collected.')
        return

    out_csv = RESULTS / 'seld_dataset_distributions.csv'
    with open(out_csv, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nCSV → {out_csv.relative_to(_ROOT)}')


if __name__ == '__main__':
    main()
