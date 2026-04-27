"""
experiments_at2sed/starss_mapping.py
=====================================
Build the (n_starss, n_strong) soft-prior mapping matrix used to initialise
the STARSS23 projection head (``head_starss``) in ``AT2SEDModel``.

The mapping translates the AudioSet-Standard indices stored in
``configs/labels_map.json`` into AudioSet-Strong indices (0–455) via two
look-up tables:

    class_labels_indices.csv   — Standard index → MID (527 classes)
    audioset_strong_label_map.csv — Strong index, as_std_mid → Standard MID

Usage
-----
::

    from pathlib import Path
    from experiments_at2sed.starss_mapping import build_strong_to_starss_mapping

    M = build_strong_to_starss_mapping(
        labels_map_json      = Path('configs/labels_map.json'),
        class_labels_csv     = Path('semantic_backbone/class_labels_indices.csv'),
        strong_label_map_csv = Path('configs/datasets/audioset_strong_label_map.csv'),
    )
    # M : (13, 456) float32 — M[c, j] > 0 iff Strong index j maps to STARSS23 class c

Author
------
Stefano Giacomelli, Ph.D. candidate — DISIM, University of L'Aquila
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch


# ── Default paths (relative to project root) ─────────────────────────────────

_REPO = Path(__file__).resolve().parent.parent

_DEFAULT_LABELS_MAP      = _REPO / 'configs' / 'labels_map.json'
_DEFAULT_CLASS_LABELS    = _REPO / 'semantic_backbone' / 'class_labels_indices.csv'
_DEFAULT_STRONG_LABEL_MAP = _REPO / 'configs' / 'datasets' / 'audioset_strong_label_map.csv'


# ══════════════════════════════════════════════════════════════════════════════

def build_strong_to_starss_mapping(
    labels_map_json:      Optional[Path] = None,
    class_labels_csv:     Optional[Path] = None,
    strong_label_map_csv: Optional[Path] = None,
    n_strong: int = 456,
    dataset: str = 'STARSS23',
) -> torch.Tensor:
    """
    Build a soft-prior mapping matrix  M : (n_target_classes, n_strong).

    ``M[c, j] = 1 / |{Strong indices for class c}|``
    for each Strong index j that maps to target class c; 0 elsewhere.

    This initialises ``head_starss.weight`` so that the projection starts as
    a simple averaging of the relevant AudioSet-Strong logits — the same
    approach used by ``head_semantic`` in the NAS-SELD model.

    Parameters
    ----------
    labels_map_json : Path, optional
        Path to ``configs/labels_map.json``.
    class_labels_csv : Path, optional
        Path to AudioSet-Standard ``class_labels_indices.csv`` (3 cols:
        ``index, mid, display_name``).
    strong_label_map_csv : Path, optional
        Path to ``audioset_strong_label_map.csv`` (5 cols:
        ``index, mid, display_name, as_std_mid, as_std_display_name``).
    n_strong : int
        Number of AudioSet-Strong classes (default 456).
    dataset : str
        Key in ``labels_map.json`` (default ``'STARSS23'``).

    Returns
    -------
    torch.Tensor
        Float32 tensor of shape ``(n_target_classes, n_strong)``.
    """
    labels_map_json      = Path(labels_map_json      or _DEFAULT_LABELS_MAP)
    class_labels_csv     = Path(class_labels_csv     or _DEFAULT_CLASS_LABELS)
    strong_label_map_csv = Path(strong_label_map_csv or _DEFAULT_STRONG_LABEL_MAP)

    # ── 1. Load target class definitions ─────────────────────────────────────
    with open(labels_map_json) as fh:
        labels_map = json.load(fh)
    if dataset not in labels_map:
        raise KeyError(
            f"Dataset '{dataset}' not found in {labels_map_json}. "
            f"Available: {list(labels_map)}"
        )
    target_classes: List[dict] = labels_map[dataset]['classes']
    n_target = len(target_classes)

    # ── 2. Build Standard-index → MID lookup ─────────────────────────────────
    std_index_to_mid: Dict[int, str] = {}
    with open(class_labels_csv, newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            std_index_to_mid[int(row['index'])] = row['mid'].strip()

    # ── 3. Build Standard-MID → Strong-index lookup ───────────────────────────
    std_mid_to_strong: Dict[str, int] = {}
    with open(strong_label_map_csv, newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            as_std_mid = row['as_std_mid'].strip()
            if as_std_mid and as_std_mid.lower() != 'none':
                std_mid_to_strong[as_std_mid] = int(row['index'])

    # ── 4. Build mapping matrix ───────────────────────────────────────────────
    M = torch.zeros(n_target, n_strong, dtype=torch.float32)

    for c, cls_def in enumerate(target_classes):
        std_indices: List[int] = cls_def.get('audioset_indices', [])
        strong_indices: List[int] = []

        for std_idx in std_indices:
            mid = std_index_to_mid.get(std_idx)
            if mid is None:
                print(
                    f'[starss_mapping] WARNING: Standard index {std_idx} '
                    f'(class "{cls_def["name"]}") not in class_labels_indices.csv — skipped'
                )
                continue
            s_idx = std_mid_to_strong.get(mid)
            if s_idx is None:
                print(
                    f'[starss_mapping] WARNING: Standard MID {mid} '
                    f'(Standard index {std_idx}, class "{cls_def["name"]}") '
                    f'has no AudioSet-Strong equivalent — skipped'
                )
                continue
            if s_idx >= n_strong:
                print(
                    f'[starss_mapping] WARNING: Strong index {s_idx} >= n_strong={n_strong} '
                    f'for class "{cls_def["name"]}" — skipped'
                )
                continue
            strong_indices.append(s_idx)

        if not strong_indices:
            print(
                f'[starss_mapping] WARNING: STARSS23 class {c} "{cls_def["name"]}" '
                f'has no AudioSet-Strong mapping — row {c} will be all-zeros'
            )
            continue

        weight = 1.0 / len(strong_indices)
        for j in strong_indices:
            M[c, j] = weight

    n_mapped = int((M > 0).any(dim=0).sum().item())
    print(
        f'[starss_mapping] Built {dataset} mapping: '
        f'{n_target} target classes → {n_mapped} AudioSet-Strong indices covered'
    )
    return M
