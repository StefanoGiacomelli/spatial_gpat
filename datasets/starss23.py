"""
STARSS23Dataset
===============
Sony-TAU Realistic Spatial Soundscapes 2023.

Split strategy
--------------
* **test**  → ``metadata_dev/dev-test-sony/`` + ``metadata_dev/dev-test-tau/``
  (78 clips — same as the DCASE2023 challenge evaluation subset)
* **train** → stratified 80 % of ``metadata_dev/dev-train-*/`` (90 clips)
* **val**   → stratified 20 % of ``metadata_dev/dev-train-*/`` (≈18 clips)

Stratification is performed room-by-room within each recording site (Sony/TAU)
so that every room contributes to both train and val.  The val_seed makes the
split reproducible across runs.

CSV format (6 columns, no header)
-----------------------------------
``frame_100ms, class_idx, source_idx, azimuth_deg, elevation_deg, distance_cm``

References
----------
K. Shimada et al., "STARSS23: An Audio-Visual Dataset of Spatial Recordings
of Real Scenes with Spatiotemporal Annotations of Sound Events," 2023.
"""

import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from .seld_dataset import SELDDataset


class STARSS23Dataset(SELDDataset):
    """
    Args:
        root         : path to the STARSS23 dataset root (contains foa_dev/, mic_dev/, …)
        split        : 'train' | 'val' | 'test'
        model_sr     : model input sample rate (e.g. 32000)
        hop_length   : hop length at model_sr (e.g. 320)
        n_tracks     : Multi-ACCDOA output tracks (e.g. 5)
        window_s     : window duration in seconds (None = full clip)
        hop_s        : window stride in seconds (None = non-overlapping)
        audio_format : 'foa' (default) | 'mic'
        val_fraction : fraction of dev-train reserved for validation (default 0.2)
        val_seed     : RNG seed for reproducible train/val split (default 42)
    """

    N_CLASSES = 13
    ORIG_SR   = 24000

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
        val_fraction: float = 0.2,
        val_seed: int = 42,
        augment=None,
    ):
        self.root         = Path(root)
        self.val_fraction = val_fraction
        self.val_seed     = val_seed
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
        """Return (13,) pos_weight tensor computed from training CSV annotations."""
        return compute_pos_weights(self.root, n_classes=self.N_CLASSES)

    def _collect_clips(self, split: str) -> List[Tuple[Path, Path]]:
        audio_root = self.root / f'{self.audio_format}_dev'
        meta_root  = self.root / 'metadata_dev'

        if split == 'test':
            pairs: List[Tuple[Path, Path]] = []
            for sub in ('dev-test-sony', 'dev-test-tau'):
                for csv_p in sorted((meta_root / sub).glob('*.csv')):
                    wav_p = audio_root / sub / (csv_p.stem + '.wav')
                    pairs.append((wav_p, csv_p))
            return pairs

        # Collect all dev-train clips
        all_pairs: List[Tuple[Path, Path]] = []
        for sub in ('dev-train-sony', 'dev-train-tau'):
            for csv_p in sorted((meta_root / sub).glob('*.csv')):
                wav_p = audio_root / sub / (csv_p.stem + '.wav')
                all_pairs.append((wav_p, csv_p))

        # Group by (site, room): e.g. "sony_room21", "tau_room12"
        def _room_key(pair: Tuple[Path, Path]) -> str:
            parent = pair[1].parent.name   # 'dev-train-sony' or 'dev-train-tau'
            site   = 'sony' if 'sony' in parent else 'tau'
            parts  = pair[1].stem.split('_')   # fold3_room21_mix001
            return f'{site}_{parts[1]}'

        groups: dict = {}
        for p in all_pairs:
            groups.setdefault(_room_key(p), []).append(p)

        rng = np.random.default_rng(self.val_seed)
        train_pairs: List[Tuple[Path, Path]] = []
        val_pairs:   List[Tuple[Path, Path]] = []

        for _key, room_clips in sorted(groups.items()):
            n_val = max(1, round(len(room_clips) * self.val_fraction))
            shuffle_idx = rng.permutation(len(room_clips))
            val_set = set(shuffle_idx[:n_val].tolist())
            for i, p in enumerate(room_clips):
                (val_pairs if i in val_set else train_pairs).append(p)

        return train_pairs if split == 'train' else val_pairs

    def _parse_csv(
        self, csv_path: Path
    ) -> List[Tuple[int, int, int, float, float]]:
        """Parse STARSS23 CSV (6 columns); return events as (frame, cls, src, az, el)."""
        events: List[Tuple[int, int, int, float, float]] = []
        with open(csv_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                p = line.split(',')
                # cols: frame, class, source_idx, azimuth, elevation, distance
                try:
                    events.append((int(p[0]), int(p[1]), int(p[2]),
                                   float(p[3]), float(p[4])))
                except (ValueError, IndexError):
                    continue   # skip malformed or header rows
                # distance (p[5]) is not used for ACCDOA targets
        return events

    @staticmethod
    def _parse_csv_with_distance(
        csv_path: Path,
    ) -> List[Tuple[int, int, int, float, float, float]]:
        """Parse STARSS23 CSV returning 6-tuples (frame, cls, src, az_deg, el_deg, dist_cm)."""
        events: List[Tuple[int, int, int, float, float, float]] = []
        with open(csv_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                p = line.split(',')
                try:
                    events.append((int(p[0]), int(p[1]), int(p[2]),
                                   float(p[3]), float(p[4]), float(p[5])))
                except (ValueError, IndexError):
                    continue
        return events


# ══════════════════════════════════════════════════════════════════════════════
# Class-imbalance utility
# ══════════════════════════════════════════════════════════════════════════════

def compute_pos_weights(root: Path, n_classes: int = 13) -> 'torch.Tensor':
    """
    Compute per-class BCE pos_weight from STARSS23 training CSVs.

    Scans all dev-train-sony and dev-train-tau CSV files, counts unique
    (clip, frame_100ms) pairs per class, and returns:

        pos_weight[c] = neg_frames[c] / pos_frames[c]

    where ``neg_frames[c] = total_frames - pos_frames[c]``.

    This weight is passed to ``BCEWithLogitsLoss`` / ``tPITLoss`` to
    up-weight rare classes (e.g. Knock: ~600×) and down-weight frequent
    ones (e.g. Male speech: ~0.1×).

    Args:
        root      : STARSS23 dataset root (contains ``metadata_dev/``).
        n_classes : number of classes (13 for STARSS23).

    Returns:
        FloatTensor of shape ``(n_classes,)``.
    """
    import torch

    root = Path(root)
    train_dirs = [
        root / 'metadata_dev' / 'dev-train-tau',
        root / 'metadata_dev' / 'dev-train-sony',
    ]

    # Map (clip_stem, frame_100ms) → set of active class indices
    frame_classes: dict = {}
    for d in train_dirs:
        if not d.exists():
            continue
        for csv_p in d.glob('*.csv'):
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
        pos = max(active[c], 1)          # guard against unseen class
        neg = total_frames - pos
        weights.append(neg / pos)

    return torch.tensor(weights, dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Class-agnostic spatial dataset (distance-aware)
# ══════════════════════════════════════════════════════════════════════════════

class STARSS23WithDistanceDataset(STARSS23Dataset):
    """
    STARSS23 returning class-agnostic spatial targets alongside Multi-ACCDOA.

    Each ``__getitem__`` yields a 5-tuple:
        wav        : ``(n_mics, T_samples)``              float32 at orig_sr
        accdoa     : ``(T_mel, N_tracks, C_classes, 3)``  Multi-ACCDOA target
        doa_target : ``(T_mel, N_tracks, 3)``             Cartesian unit vectors (0 ⟹ silent)
        act_target : ``(T_mel, N_tracks)``                binary activity  {0, 1}
        dist_target: ``(T_mel, N_tracks)``                distance in cm  (0 ⟹ silent)

    Track slots are filled in ascending ``source_idx`` order; excess sources
    beyond ``n_tracks`` are dropped.  Distance is per-source (not per-class).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._dist_events: List[List[Tuple[int, int, int, float, float, float]]] = [
            self._parse_csv_with_distance(csv_p)
            for _, csv_p in self.clips
        ]

    def __getitem__(self, idx: int):  # type: ignore[override]
        wav, accdoa = super().__getitem__(idx)

        clip_idx, start_sample = self._items[idx]
        n_mel          = accdoa.shape[0]
        start_at_model = round(start_sample * self.model_sr / self.orig_sr)

        doa_t, act_t, dist_t = self._build_spatial_targets(
            self._dist_events[clip_idx], n_mel, start_at_model
        )
        return wav, accdoa, doa_t, act_t, dist_t

    def _build_spatial_targets(
        self,
        events: List[Tuple[int, int, int, float, float, float]],
        n_mel: int,
        start_at_model_sr: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build ``(doa_target, act_target, dist_target)`` from 6-tuple distance events.

        Returns:
            doa_target : ``(T_mel, N_tracks, 3)``
            act_target : ``(T_mel, N_tracks)``
            dist_target: ``(T_mel, N_tracks)`` in cm
        """
        N     = self.n_tracks
        doa_t  = torch.zeros(n_mel, N, 3, dtype=torch.float32)
        act_t  = torch.zeros(n_mel, N,    dtype=torch.float32)
        dist_t = torch.zeros(n_mel, N,    dtype=torch.float32)

        if not events:
            return doa_t, act_t, dist_t

        # Group events by CSV frame: frame → [(source_idx, (x,y,z), dist_cm)]
        csv_data: dict = {}
        for (frame_100ms, _cls, source_idx, az_deg, el_deg, dist_cm) in events:
            az  = math.radians(az_deg)
            el  = math.radians(el_deg)
            xyz = (
                math.sin(az) * math.cos(el),
                math.cos(az) * math.cos(el),
                math.sin(el),
            )
            csv_data.setdefault(frame_100ms, []).append((source_idx, xyz, dist_cm))

        for srcs in csv_data.values():
            srcs.sort(key=lambda s: s[0])   # ascending source_idx

        # Map mel frames → CSV frames (mirrors SELDDataset._build_target)
        t_s_arr   = (
            start_at_model_sr + np.arange(n_mel, dtype=np.float64) * self.hop_length
        ) / self.model_sr
        t_csv_arr = (t_s_arr / self.FRAME_DURATION_S).astype(np.int64)

        prev_csv:    int  = -1
        frame_srcs:  list = []
        for t_mel in range(n_mel):
            t_csv = int(t_csv_arr[t_mel])
            if t_csv != prev_csv:
                frame_srcs = csv_data.get(t_csv, [])
                prev_csv   = t_csv
            for track_idx, (_sid, (x, y, z), dist_cm) in enumerate(frame_srcs[:N]):
                doa_t[t_mel, track_idx, 0] = x
                doa_t[t_mel, track_idx, 1] = y
                doa_t[t_mel, track_idx, 2] = z
                act_t[t_mel,  track_idx]   = 1.0
                dist_t[t_mel, track_idx]   = dist_cm

        return doa_t, act_t, dist_t
