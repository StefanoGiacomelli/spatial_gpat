#!/usr/bin/env python3
"""
experiments_at2sed/build_strong_label_map.py — Build AudioSet-Strong Label Map
===============================================================================
One-shot script that generates ``configs/datasets/audioset_strong_label_map.csv``,
a 456-row label map aligning AudioSet-Strong MIDs with their AudioSet Standard
counterparts.

Row ordering:
    0–381   : 382 shared classes, sorted by AudioSet Standard index
              (preserves semantic ordering of the pretrained fc_audioset head)
    382–455 : 74 Strong-only classes, sorted alphabetically by display_name

Columns:
    index               — new integer index used by the model (0-455)
    mid                 — MID from AudioSet-Strong
    display_name        — display name from mid_to_display_name.tsv
    as_std_mid          — corresponding MID in AudioSet Standard (or "None")
    as_std_display_name — display name in AudioSet Standard (or "Strong-only")

Usage::

    .venv/bin/python experiments_at2sed/build_strong_label_map.py

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

import csv
from pathlib import Path

import yaml

_ROOT        = Path(__file__).resolve().parent.parent
_DATASET_CFG = _ROOT / 'configs' / 'datasets' / 'AudioSetStrong.yaml'

# ── Output file ───────────────────────────────────────────────────────────────
_OUT_CSV = _ROOT / 'configs' / 'datasets' / 'audioset_strong_label_map.csv'


def main() -> None:
    # ── Load paths from config ────────────────────────────────────────────────
    with open(_DATASET_CFG) as fh:
        cfg = yaml.safe_load(fh)
    _STRONG_TSV = Path(cfg['semantic']['mid_to_display_name_tsv'])
    _AS_STD_CSV = Path(cfg['data']['class_map'])

    # ── Load AudioSet-Strong MIDs ─────────────────────────────────────────────
    strong: dict[str, str] = {}   # mid → display_name
    with open(_STRONG_TSV) as fh:
        for line in fh:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                strong[parts[0].strip()] = parts[1].strip()

    # ── Load AudioSet Standard (527) ──────────────────────────────────────────
    as_std: dict[str, tuple[int, str]] = {}   # mid → (index, display_name)
    with open(_AS_STD_CSV) as fh:
        for row in csv.DictReader(fh):
            as_std[row['mid'].strip()] = (int(row['index']), row['display_name'].strip())

    # ── Build ordered rows ────────────────────────────────────────────────────
    # Shared classes: sorted by AudioSet Standard index (0–381)
    shared = sorted(
        [(as_std[m][0], m, strong[m], m, as_std[m][1])
         for m in strong if m in as_std],
        key=lambda x: x[0],   # sort by old AS Standard index
    )

    # Strong-only classes: sorted alphabetically by display_name (382–455)
    strong_only = sorted(
        [(m, strong[m]) for m in strong if m not in as_std],
        key=lambda x: x[1],
    )

    # ── Write CSV ─────────────────────────────────────────────────────────────
    _OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    # Shared rows
    for new_idx, (_, mid, disp, as_mid, as_disp) in enumerate(shared):
        rows.append({
            'index':               new_idx,
            'mid':                 mid,
            'display_name':        disp,
            'as_std_mid':          as_mid,
            'as_std_display_name': as_disp,
        })

    # Strong-only rows
    base = len(shared)
    for i, (mid, disp) in enumerate(strong_only):
        rows.append({
            'index':               base + i,
            'mid':                 mid,
            'display_name':        disp,
            'as_std_mid':          'None',
            'as_std_display_name': 'Strong-only',
        })

    with open(_OUT_CSV, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            'index', 'mid', 'display_name', 'as_std_mid', 'as_std_display_name',
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f'Written {len(rows)} rows to {_OUT_CSV.relative_to(_ROOT)}')
    print(f'  Shared classes   : {len(shared)} (indices 0–{len(shared)-1})')
    print(f'  Strong-only      : {len(strong_only)} (indices {len(shared)}–{len(rows)-1})')
    print(f'  Total            : {len(rows)}')


if __name__ == '__main__':
    main()
