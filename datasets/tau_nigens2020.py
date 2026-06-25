"""
TAUNIGENS2020Dataset
====================
TAU-NIGENS Spatial Sound Events 2020 (DCASE Challenge 2020 Task 3).

Split strategy
--------------
Uses the official DCASE2020 fold-based split encoded in the filename prefix:

* **train** → folds 1–4  (400 one-minute clips, ``fold[1-4]_*.wav``)
* **val**   → fold  5    (100 one-minute clips, ``fold5_*.wav``)
* **test**  → fold  6    (100 one-minute clips, ``fold6_*.wav``)

All audio files live in a single **flat** directory ``foa_dev/``; the fold
number is the first token of every filename (e.g. ``fold1_room1_mix001_ov1.wav``).
This differs from TAU-NIGENS2021, which uses explicit ``dev-train/`` / ``dev-val/``
/ ``dev-test/`` sub-directories.

CSV format (5 columns, no header)
-----------------------------------
``frame_100ms, class_idx, source_idx, azimuth_deg, elevation_deg``

Classes (14 total — DCASE2020 NIGENS subset)
--------------------------------------------
0: alarm            1: crying baby     2: crash
3: barking dog      4: running engine  5: female scream
6: female speech    7: burning fire    8: footsteps
9: knocking on door 10: male scream   11: male speech
12: ringing phone   13: piano

References
----------
A. Politis et al., "A Dataset of Dynamic Reverberant Sound Scenes with
Directional Interferers for Sound Event Localization and Detection,"
DCASE Challenge 2020.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

from pathlib import Path
from typing import List, Optional, Tuple

from .utils import SELDDataset


class TAUNIGENS2020Dataset(SELDDataset):
    """
    Args:
        root       : path to the TAU-NIGENS2020 dataset root
                     (contains foa_dev/, metadata_dev/, …)
        split      : 'train' | 'val' | 'test'
        model_sr   : model input sample rate (e.g. 32000)
        hop_length : hop length at model_sr (e.g. 320)
        n_tracks   : Multi-ACCDOA output tracks (e.g. 3)
        window_s   : window duration in seconds (None = full clip, ~60 s)
        hop_s      : window stride in seconds (None = non-overlapping)
        augment    : optional SELDAugment instance
    """

    N_CLASSES = 14
    ORIG_SR   = 24000

    # Folds assigned to each logical split
    _SPLIT_TO_FOLDS = {
        'train': {1, 2, 3, 4},
        'val':   {5},
        'test':  {6},
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
        augment=None,
    ):
        self.root = Path(root)
        super().__init__(
            split      = split,
            orig_sr    = self.ORIG_SR,
            model_sr   = model_sr,
            hop_length = hop_length,
            n_tracks   = n_tracks,
            window_s   = window_s,
            hop_s      = hop_s,
            augment    = augment,
        )

    @property
    def n_classes(self) -> int:
        return self.N_CLASSES

    def _collect_clips(self, split: str) -> List[Tuple[Path, Path]]:
        """
        Return (wav_path, csv_path) pairs for all clips in the given split.

        Files in foa_dev/ are named ``fold{N}_room{R}_mix{M}_ov{O}.wav``.
        We select clips whose fold number (first token before '_') belongs
        to the split's fold set.
        """
        valid_folds = self._SPLIT_TO_FOLDS[split]
        audio_root  = self.root / 'foa_dev'
        meta_root   = self.root / 'metadata_dev'

        pairs: List[Tuple[Path, Path]] = []
        for csv_p in sorted(meta_root.glob('fold*.csv')):
            # Filename starts with "fold{N}_" — extract N
            fold_token = csv_p.stem.split('_')[0]   # e.g. "fold1"
            if not fold_token.startswith('fold'):
                continue
            try:
                fold_num = int(fold_token[4:])
            except ValueError:
                continue
            if fold_num not in valid_folds:
                continue
            wav_p = audio_root / (csv_p.stem + '.wav')
            pairs.append((wav_p, csv_p))
        return pairs

    def _parse_csv(
        self, csv_path: Path
    ) -> List[Tuple[int, int, int, float, float]]:
        """Parse TAU-NIGENS2020 CSV (5 columns); return (frame, cls, src, az, el)."""
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
