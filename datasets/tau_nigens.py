"""
TAUNIGENSDataset
================
TAU-NIGENS Spatial Sound Events 2021 (DCASE Challenge 2021 Task 3).

Split strategy
--------------
Uses the *official* DCASE2021 fold-based split:

* **train** → ``metadata_dev/dev-train/``  (folds 1 - 4, 400 one-minute clips)
* **val**   → ``metadata_dev/dev-val/``    (fold 5,      100 one-minute clips)
* **test**  → ``metadata_dev/dev-test/``   (fold 6,      100 one-minute clips)

CSV format (5 columns, no header)
-----------------------------------
``frame_100ms, class_idx, source_idx, azimuth_deg, elevation_deg``

No distance field (unlike STARSS23).

References
----------
A. Politis, S. Adavanne, D. Krause, A. Deleforge, P. Srivastava, T. Virtanen,
"A Dataset of Dynamic Reverberant Sound Scenes with Directional Interferers
for Sound Event Localization and Detection," DCASE Workshop 2021.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import torch
from .utils import SELDDataset


class TAUNIGENSDataset(SELDDataset):
    """
    Args:
        root         : path to the TAU-NIGENS2021 dataset root
                       (contains foa_dev/, mic_dev/, metadata_dev/, …)
        split        : 'train' | 'val' | 'test'
        model_sr     : model input sample rate (e.g. 32000)
        hop_length   : hop length at model_sr (e.g. 320)
        n_tracks     : Multi-ACCDOA output tracks (e.g. 5)
        window_s     : window duration in seconds (None = full clip, ~60 s)
        hop_s        : window stride in seconds (None = non-overlapping)
        audio_format : 'foa' (default) | 'mic'
    """

    N_CLASSES = 12
    ORIG_SR   = 24000

    _SPLIT_TO_SUBDIR = {
        'train': 'dev-train',
        'val':   'dev-val',
        'test':  'dev-test',
    }

    def __init__(
        self,
        root: Path,
        split: str,
        model_sr: int,
        hop_length: int,
        n_tracks: int,
        window_s: Optional[float] = None,
        hop_s: Optional[float] = None,
        audio_format: str = 'foa',
        augment=None,
    ):
        self.root = Path(root)
        super().__init__(
            split        = split,
            orig_sr      = self.ORIG_SR,
            model_sr     = model_sr,
            hop_length   = hop_length,
            n_tracks     = n_tracks,
            window_s     = window_s,
            hop_s        = hop_s,
            audio_format = audio_format,
            augment      = augment,
        )

    @property
    def n_classes(self) -> int:
        return self.N_CLASSES

    def _pos_weights(self) -> 'torch.Tensor':
        """Return (12,) pos_weight tensor computed from training CSV annotations."""
        return compute_pos_weights(self.root, n_classes=self.N_CLASSES)

    def _collect_clips(self, split: str) -> List[Tuple[Path, Path]]:
        sub        = self._SPLIT_TO_SUBDIR[split]
        audio_root = self.root / f'{self.audio_format}_dev' / sub
        meta_root  = self.root / 'metadata_dev' / sub

        pairs: List[Tuple[Path, Path]] = []
        for csv_p in sorted(meta_root.glob('*.csv')):
            wav_p = audio_root / (csv_p.stem + '.wav')
            pairs.append((wav_p, csv_p))
        return pairs

    def _parse_csv(
        self, csv_path: Path
    ) -> List[Tuple[int, int, int, float, float]]:
        """Parse TAU-NIGENS2021 CSV (5 columns); return (frame, cls, src, az, el)."""
        events: List[Tuple[int, int, int, float, float]] = []
        with open(csv_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                p = line.split(',')
                try:
                    events.append((int(p[0]), int(p[1]), int(p[2]),
                                   float(p[3]), float(p[4])))
                except (ValueError, IndexError):
                    continue   # skip malformed or header rows
        return events


# ══════════════════════════════════════════════════════════════════════════════
# Class-imbalance utility
# ══════════════════════════════════════════════════════════════════════════════

def compute_pos_weights(root: Path, n_classes: int = 12) -> 'torch.Tensor':
    """
    Compute per-class BCE pos_weight from TAU-NIGENS2021 training CSVs.

    Scans all CSV files in ``metadata_dev/dev-train/``, counts unique
    (clip, frame_100ms) pairs per class, and returns:

        pos_weight[c] = neg_frames[c] / pos_frames[c]

    where ``neg_frames[c] = total_frames - pos_frames[c]``.

    Args:
        root      : TAU-NIGENS2021 dataset root (contains ``metadata_dev/``).
        n_classes : number of classes (12 for TAU-NIGENS2021).

    Returns:
        FloatTensor of shape ``(n_classes,)``.
    """
    import torch

    root = Path(root)
    train_dir = root / 'metadata_dev' / 'dev-train'

    frame_classes: dict = {}
    if train_dir.exists():
        for csv_p in train_dir.glob('*.csv'):
            with open(csv_p) as fh:
                for line in fh:
                    parts = line.strip().split(',')
                    if len(parts) < 2:
                        continue
                    try:
                        key = (csv_p.stem, int(parts[0]))
                    except ValueError:
                        continue   # skip header or malformed rows
                    if key not in frame_classes:
                        frame_classes[key] = set()
                    frame_classes[key].add(int(parts[1]))

    total_frames = len(frame_classes)
    active = [0] * n_classes
    for classes in frame_classes.values():
        for c in classes:
            if c < n_classes:
                active[c] += 1

    weights = []
    for c in range(n_classes):
        pos = max(active[c], 1)
        neg = total_frames - pos
        weights.append(neg / pos)

    return torch.tensor(weights, dtype=torch.float32)
