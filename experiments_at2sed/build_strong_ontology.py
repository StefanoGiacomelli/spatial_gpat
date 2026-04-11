#!/usr/bin/env python3
"""
experiments_at2sed/build_strong_ontology.py
===========================================
One-shot script that builds AudioSet-Strong-specific versions of two ontology
matrices originally defined for AudioSet Standard (527 classes):

  1. ``configs/datasets/audioset_strong_hlp.pt``
     Boolean (456, 456) lookup table for Hierarchical Label Propagation.
     ``table[i, j] = True`` iff class j is an unambiguous ancestor of class i
     in the AudioSet ontology.

  2. ``configs/datasets/audioset_strong_graph_distance.npy``
     Float32 (456, 456) matrix of hop distances in the AudioSet ontology graph.

Both files are produced by **submatrix extraction** from the 527×527 originals:
  - Classes 0–381 (shared with AS-Standard): indices are remapped via
    ``audioset_strong_label_map.csv`` → ``class_labels_indices.csv``.
  - Classes 382–455 (Strong-only, no position in the AS ontology):
    HLP → identity only (no ancestors).
    Distance → 0 on diagonal; ``max_dist + 1`` off-diagonal (maximally far).

Source matrices:
  - HLP:      hierarchical-label-propagation/src/.../HLP_lookup_table.pt
              (pre-built from AudioSet ontology.json by HLP authors)
  - Distance: ontology-aware-audio-tagging/ontology_audio_tagging/
              audioset_graph_distance.py  (hardcoded 527×527)

Usage::

    .venv/bin/python experiments_at2sed/build_strong_ontology.py

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

HLP_SRC    = ROOT / 'hierarchical-label-propagation' / 'src' / \
             'hierarchical_label_propagation' / 'HLP_lookup_table.pt'
LABEL_MAP  = ROOT / 'configs' / 'datasets' / 'audioset_strong_label_map.csv'
AS_STD_CSV = Path('/Volumes/Lexar/audioset-qiuqiangkong/metadata/class_labels_indices.csv')

OUT_HLP    = ROOT / 'configs' / 'datasets' / 'audioset_strong_hlp.pt'
OUT_DIST   = ROOT / 'configs' / 'datasets' / 'audioset_strong_graph_distance.npy'

N_STRONG = 456

# ── Dependency check ──────────────────────────────────────────────────────────
if not HLP_SRC.exists():
    sys.exit(f'[ERROR] HLP source not found: {HLP_SRC}')
if not LABEL_MAP.exists():
    sys.exit(f'[ERROR] Label map not found: {LABEL_MAP}')
if not AS_STD_CSV.exists():
    sys.exit(f'[ERROR] AS-Standard class map not found: {AS_STD_CSV}\n'
             f'        Mount the Lexar volume and re-run.')

# ── Load source matrices ──────────────────────────────────────────────────────
print('[1/4] Loading HLP lookup table (527×527) …')
hlp_std = torch.load(str(HLP_SRC), weights_only=True)   # bool (527, 527)
assert hlp_std.shape == (527, 527), f'Unexpected HLP shape: {hlp_std.shape}'

print('[2/4] Loading AudioSet graph distances (527×527) …')
sys.path.insert(0, str(ROOT / 'ontology-aware-audio-tagging'))
from ontology_audio_tagging.audioset_graph_distance import get_audioset_graph_distance
dist_std = np.array(get_audioset_graph_distance(), dtype=np.float32)   # (527, 527)
assert dist_std.shape == (527, 527), f'Unexpected dist shape: {dist_std.shape}'
max_dist = float(dist_std.max())
print(f'    max distance in AS-Std graph: {max_dist}')

# ── Build AS-Standard MID → index lookup ─────────────────────────────────────
print('[3/4] Building Strong→Standard index mapping …')
as_std_mid_to_idx: dict[str, int] = {}
with open(AS_STD_CSV, newline='') as fh:
    for row in csv.DictReader(fh):
        as_std_mid_to_idx[row['mid'].strip()] = int(row['index'])

# AS-Strong label map: strong_idx → as_std_mid (or "None")
strong_to_std: list[int | None] = [None] * N_STRONG
with open(LABEL_MAP, newline='') as fh:
    for row in csv.DictReader(fh):
        strong_idx = int(row['index'])
        as_mid     = row['as_std_mid'].strip()
        if as_mid != 'None':
            std_idx = as_std_mid_to_idx.get(as_mid)
            if std_idx is not None:
                strong_to_std[strong_idx] = std_idx

n_shared     = sum(1 for v in strong_to_std if v is not None)
n_strong_only = N_STRONG - n_shared
print(f'    {n_shared} shared classes  /  {n_strong_only} Strong-only classes')

# ── Build 456×456 matrices ────────────────────────────────────────────────────
print('[4/4] Building 456×456 submatrices …')

# HLP: default = identity (each class is ancestor of itself only)
hlp_new = torch.eye(N_STRONG, dtype=torch.bool)

# Distance: default = identity 0 + off-diagonal = max_dist + 1
sentinel = max_dist + 1.0
dist_new = np.full((N_STRONG, N_STRONG), sentinel, dtype=np.float32)
np.fill_diagonal(dist_new, 0.0)

for i in range(N_STRONG):
    std_i = strong_to_std[i]
    if std_i is None:
        continue   # Strong-only: keep identity defaults
    for j in range(N_STRONG):
        std_j = strong_to_std[j]
        if std_j is None:
            continue   # no ontology relationship to Strong-only classes
        hlp_new[i, j]  = hlp_std[std_i, std_j]
        dist_new[i, j] = dist_std[std_i, std_j]

# ── Sanity checks ─────────────────────────────────────────────────────────────
assert hlp_new.shape  == (N_STRONG, N_STRONG)
assert dist_new.shape == (N_STRONG, N_STRONG)
assert hlp_new.diagonal().all(),      'HLP diagonal should be all-True'
assert (dist_new.diagonal() == 0).all(), 'Distance diagonal should be all-0'

n_ancestors = int(hlp_new.sum().item()) - N_STRONG  # exclude diagonal
print(f'    HLP: {n_ancestors} non-self ancestor entries  '
      f'({n_ancestors / n_shared:.2f} per shared class on average)')
print(f'    Dist: range [{dist_new[dist_new < sentinel].min():.0f}, '
      f'{dist_new[dist_new < sentinel].max():.0f}] '
      f'(sentinel={sentinel:.0f} for Strong-only pairs)')

# ── Save ──────────────────────────────────────────────────────────────────────
OUT_HLP.parent.mkdir(parents=True, exist_ok=True)
torch.save(hlp_new, str(OUT_HLP))
np.save(str(OUT_DIST), dist_new)

print(f'\nSaved:')
print(f'  {OUT_HLP.relative_to(ROOT)}   shape={tuple(hlp_new.shape)}  dtype={hlp_new.dtype}')
print(f'  {OUT_DIST.relative_to(ROOT)}  shape={dist_new.shape}  dtype={dist_new.dtype}')
print('\nDone.')


if __name__ == '__main__':
    main = lambda: None   # noqa — script is top-level, not callable
