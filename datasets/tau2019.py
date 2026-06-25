"""
TAU2019Dataset
==============
TAU Spatial Sound Events 2019 (DCASE2019 Task 3).

Audio format  : FOA, 48 kHz → resampled to 24 kHz before returning
Annotations   : onset–offset CSV, columns: sound_event_recording, start_time,
                end_time, ele, azi, dist
Classes       : 11 (clearthroat, cough, doorslam, drawer, keyboard,
                keysDrop, knock, laughter, pageturn, phone, speech)

Standalone mode (default)
-------------------------
4-fold cross-validation over dev/ (400 clips, splits 1–4).

    Fold | Train       | Val    | Test
    -----+-------------+--------+------
      1  | split3,4    | split2 | split1
      2  | split4,1    | split3 | split2
      3  | split1,2    | split4 | split3
      4  | split2,3    | split1 | split4

Config key ``data.fold`` selects the fold (default 1).

Mixed mode (``mixed_mode=True``)
---------------------------------
Fixed split for MixedSELDDataModule — no fold overlap with eval set:

    Train : dev splits 2, 3, 4  (300 clips)
    Val   : dev split 1          (100 clips)
    Test  : eval split0          (100 clips, from eval/proj/…/foa_eval/)

References
----------
A. Politis et al., "A Dataset of Reverberant Spatial Sound Scenes with Moving
Sources for Sound Event Localization and Detection," DCASE2019 Workshop.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .utils import SELDDataset


class TAU2019Dataset(SELDDataset):
    """
    Args:
        root       : path to the TAU2019 dataset root (contains dev/ and eval/)
        split      : 'train' | 'val' | 'test'
        model_sr   : model input sample rate (e.g. 32000)
        hop_length : hop length in samples at model_sr (e.g. 320)
        n_tracks   : Multi-ACCDOA output tracks (e.g. 3)
        window_s   : window duration in seconds (None = full clip)
        hop_s      : window stride in seconds (None = non-overlapping)
        fold       : cross-validation fold 1–4 (standalone mode only)
        mixed_mode : if True, use fixed train/val/test splits ignoring fold
        augment    : optional SELDAugment instance
    """

    N_CLASSES = 11
    ORIG_SR   = 48000
    OUTPUT_SR = 24000   # resample target; passed as output_sr to base class

    CLASS_MAP: Dict[str, int] = {
        'clearthroat': 0,
        'cough':       1,
        'doorslam':    2,
        'drawer':      3,
        'keyboard':    4,
        'keysDrop':    5,
        'knock':       6,
        'laughter':    7,
        'pageturn':    8,
        'phone':       9,
        'speech':      10,
    }

    # Fold → {split: [split_numbers]}
    FOLD_TABLE: Dict[int, Dict[str, List[int]]] = {
        1: {'train': [3, 4], 'val': [2], 'test': [1]},
        2: {'train': [4, 1], 'val': [3], 'test': [2]},
        3: {'train': [1, 2], 'val': [4], 'test': [3]},
        4: {'train': [2, 3], 'val': [1], 'test': [4]},
    }

    # Fixed splits for MixedSELDDataModule (eval set used as test)
    MIXED_SPLITS: Dict[str, List[int]] = {
        'train': [2, 3, 4],
        'val':   [1],
        # 'test' is handled separately (uses eval/)
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
        fold: int = 1,
        mixed_mode: bool = False,
        augment=None,
    ):
        self.root       = Path(root)
        self.fold       = fold
        self.mixed_mode = mixed_mode
        super().__init__(
            split      = split,
            orig_sr    = self.ORIG_SR,
            model_sr   = model_sr,
            hop_length = hop_length,
            n_tracks   = n_tracks,
            window_s   = window_s,
            hop_s      = hop_s,
            output_sr  = self.OUTPUT_SR,
            augment    = augment,
        )

    @property
    def n_classes(self) -> int:
        return self.N_CLASSES

    def _collect_clips(self, split: str) -> List[Tuple[Path, Path]]:
        if self.mixed_mode:
            return self._collect_mixed(split)
        return self._collect_fold(split)

    def _collect_fold(self, split: str) -> List[Tuple[Path, Path]]:
        """Standalone 4-fold CV over dev/ directory."""
        if self.fold not in self.FOLD_TABLE:
            raise ValueError(f"fold must be 1–4, got {self.fold}")
        split_nums = self.FOLD_TABLE[self.fold][split]
        return self._collect_from_dev(split_nums)

    def _collect_mixed(self, split: str) -> List[Tuple[Path, Path]]:
        """Fixed splits for MixedSELDDataModule."""
        if split == 'test':
            return self._collect_eval()
        split_nums = self.MIXED_SPLITS[split]
        return self._collect_from_dev(split_nums)

    def _collect_from_dev(self, split_nums: List[int]) -> List[Tuple[Path, Path]]:
        foa_root  = self.root / 'dev' / 'foa_dev'
        meta_root = self.root / 'dev' / 'metadata_dev'
        pairs: List[Tuple[Path, Path]] = []
        for sn in split_nums:
            for csv_p in sorted(meta_root.glob(f'split{sn}_*.csv')):
                wav_p = foa_root / (csv_p.stem + '.wav')
                pairs.append((wav_p, csv_p))
        return pairs

    def _collect_eval(self) -> List[Tuple[Path, Path]]:
        foa_root  = self.root / 'eval' / 'proj' / 'asignal' / 'DCASE2019' / 'dataset' / 'foa_eval'
        meta_root = self.root / 'eval' / 'metadata_eval'
        pairs: List[Tuple[Path, Path]] = []
        for csv_p in sorted(meta_root.glob('split0_*.csv')):
            wav_p = foa_root / (csv_p.stem + '.wav')
            pairs.append((wav_p, csv_p))
        return pairs

    def _parse_csv(
        self, csv_path: Path
    ) -> List[Tuple[int, int, int, float, float]]:
        """
        Parse TAU2019 onset–offset CSV into per-frame events.

        CSV columns: sound_event_recording, start_time, end_time, ele, azi, dist
        NOTE: column order is (ele, azi) — reversed vs STARSS23/TAU-NIGENS!

        source_idx is assigned per class by order of appearance in the CSV,
        to distinguish simultaneous instances of the same class.

        Returns list of (frame_100ms, class_idx, source_idx, az_deg, el_deg).
        """
        events: List[Tuple[int, int, int, float, float]] = []
        class_instance_counter: Dict[int, int] = defaultdict(int)

        with open(csv_path, newline='') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                label = row['sound_event_recording']
                if label not in self.CLASS_MAP:
                    continue
                cls_idx = self.CLASS_MAP[label]
                start_f = int(float(row['start_time']) / self.FRAME_DURATION_S)
                end_f   = int(float(row['end_time'])   / self.FRAME_DURATION_S)
                az      = float(row['azi'])
                el      = float(row['ele'])   # ele column, not azi!
                src_idx = class_instance_counter[cls_idx]
                class_instance_counter[cls_idx] += 1
                for f in range(start_f, end_f + 1):
                    events.append((f, cls_idx, src_idx, az, el))

        return events
