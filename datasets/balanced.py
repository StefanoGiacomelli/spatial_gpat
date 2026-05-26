"""
datasets/balanced.py — BalancedSTARSS23 Training Dataset
=========================================================

BalancedSTARSS23Dataset
    13-class dataset anchored on STARSS23 that mitigates long-tail class
    imbalance via greedy external clip selection from TAU-NIGENS2021 and
    TAU2019, plus on-the-fly FOA mixup for the 3 classes with no external
    coverage (clapping, water_tap, bell).

    Balancing is always at clip level; windowed mode expands each clip into
    fixed-length windows after balancing.  Windowed (windowed=True) and
    full-clip (windowed=False) modes are both supported.

ProjectedSELDDataset
    Thin adapter that wraps any SELDDataset and projects its per-event class
    indices to the 13-class STARSS23 space via a list-of-lists mapping.
    Used by BalancedSELDDataModule for val/test evaluation.

BalancedSELDDataModule
    LightningDataModule for BalancedSTARSS23 training.
"""

from __future__ import annotations

import heapq
import math
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
import lightning as L
from torch.utils.data import DataLoader, Dataset

from .starss23   import STARSS23Dataset
from .tau_nigens import TAUNIGENSDataset
from .tau2019    import TAU2019Dataset
from .utils      import SELDAugment, SELDDataset


# ── Class-space projection mappings ───────────────────────────────────────────
#
# For each source class index, list the STARSS23 class indices to activate.
# [] means the event is unmappable → silent in the 13-class target.

NIGENS_TO_STARSS13: List[List[int]] = [
    [],      # 0  alarm
    [],      # 1  baby_cry
    [],      # 2  crash
    [],      # 3  dog_bark
    [0],     # 4  female_scream  → female_speech
    [0],     # 5  female_speech  → female_speech
    [6],     # 6  footstep       → walk_footsteps
    [12],    # 7  knock          → knock
    [1],     # 8  male_scream    → male_speech
    [1],     # 9  male_speech    → male_speech
    [3],     # 10 phone_ring     → telephone
    [8, 9],  # 11 piano          → music + musical_instrument
]

TAU2019_TO_STARSS13: List[List[int]] = [
    [],      # 0  clearthroat   (respiratory — not speech)
    [],      # 1  cough
    [7],     # 2  doorslam      → door_open_close
    [5],     # 3  drawer        → domestic_sounds
    [5],     # 4  keyboard      → domestic_sounds
    [5],     # 5  keysDrop      → domestic_sounds
    [12],    # 6  knock         → knock
    [4],     # 7  laughter      → laughter
    [5],     # 8  pageturn      → domestic_sounds
    [3],     # 9  phone         → telephone
    [],      # 10 speech        → excluded (gender unknown, weakly annotated)
]

# STARSS23 class indices with no external coverage → handled via FOA mixup
_MIXUP_ONLY: Set[int] = {2, 10, 11}  # clapping, water_tap, bell

_FRAME_DUR: float = 0.1   # 100 ms — annotation resolution for all 3 datasets


# ── Module-level helpers ───────────────────────────────────────────────────────

def _project_events(
    events: List[Tuple],
    mapping: List[List[int]],
) -> List[Tuple]:
    """Project raw events to 13-class STARSS23 space via mapping."""
    projected = []
    for (frame, cls_src, src_idx, az, el) in events:
        if 0 <= cls_src < len(mapping):
            for cls_dst in mapping[cls_src]:
                projected.append((frame, cls_dst, src_idx, az, el))
    return projected


def _build_seld_target(
    events: List[Tuple],
    n_mel: int,
    n_tracks: int,
    n_classes: int,
    model_sr: int,
    hop_length: int,
    start_at_model_sr: int = 0,
) -> torch.Tensor:
    """Build ``(T_mel, N_tracks, C, 3)`` Multi-ACCDOA target tensor."""
    target = torch.zeros(n_mel, n_tracks, n_classes, 3, dtype=torch.float32)
    if not events:
        return target

    csv_data: dict = {}
    for (frame_100ms, class_idx, source_idx, az_deg, el_deg) in events:
        if not (0 <= class_idx < n_classes):
            continue
        az  = math.radians(float(az_deg))
        el  = math.radians(float(el_deg))
        xyz = (
            math.sin(az) * math.cos(el),
            math.cos(az) * math.cos(el),
            math.sin(el),
        )
        (csv_data
         .setdefault(frame_100ms, {})
         .setdefault(class_idx, [])
         .append((source_idx, *xyz)))

    for frame_data in csv_data.values():
        for ci in frame_data:
            frame_data[ci].sort(key=lambda s: s[0])

    t_s_arr   = (
        start_at_model_sr + np.arange(n_mel, dtype=np.float64) * hop_length
    ) / model_sr
    t_csv_arr = (t_s_arr / _FRAME_DUR).astype(np.int64)

    prev_csv   = -1
    frame_data = {}
    for t_mel in range(n_mel):
        t_csv = int(t_csv_arr[t_mel])
        if t_csv != prev_csv:
            frame_data = csv_data.get(t_csv, {})
            prev_csv   = t_csv
        for class_idx, srcs in frame_data.items():
            for track_idx, (_, x, y, z) in enumerate(srcs[:n_tracks]):
                target[t_mel, track_idx, class_idx, 0] = x
                target[t_mel, track_idx, class_idx, 1] = y
                target[t_mel, track_idx, class_idx, 2] = z

    return target


def _load_clip_segment(
    ds: SELDDataset,
    clip_idx: int,
    start_sample: int,        # in ds.orig_sr samples; 0 for full clip
    window_samples: int,      # in ds.orig_sr samples; 0 means full clip
    model_sr: int,
) -> Tuple[torch.Tensor, int]:
    """
    Load a clip segment (or full clip if window_samples == 0).

    Returns:
        wav_t : ``(4, T_samples)`` at ds._eff_sr
        n_mel : mel frame count
    """
    audio_path, _ = ds.clips[clip_idx]

    if window_samples > 0:
        wav, _ = sf.read(
            str(audio_path),
            start     = start_sample,
            stop      = start_sample + window_samples,
            dtype     = 'float32',
            always_2d = True,
        )
        if wav.shape[0] < window_samples:
            pad = window_samples - wav.shape[0]
            wav = np.concatenate(
                [wav, np.zeros((pad, wav.shape[1]), dtype=np.float32)]
            )
    else:
        wav, _ = sf.read(str(audio_path), dtype='float32', always_2d=True)

    wav_t = torch.from_numpy(wav.T.copy())   # (4, T)

    if ds.output_sr is not None:
        wav_t = torchaudio.functional.resample(wav_t, ds.orig_sr, ds.output_sr)

    eff_sr          = ds.output_sr if ds.output_sr is not None else ds.orig_sr
    n_samples_model = round(wav_t.shape[-1] * model_sr / eff_sr)
    n_mel           = 1 + n_samples_model // ds.hop_length
    return wav_t, n_mel


def _union_targets(
    tgt_a: torch.Tensor,
    tgt_b: torch.Tensor,
    n_tracks: int,
) -> torch.Tensor:
    """
    Union two Multi-ACCDOA targets.

    Active DOA vectors from ``tgt_a`` are assigned first (priority → slot 0),
    then from ``tgt_b``.  Excess sources beyond ``n_tracks`` are dropped.

    Args:
        tgt_a, tgt_b : ``(T, N, C, 3)``
        n_tracks     : output track count
    Returns:
        ``(T, n_tracks, C, 3)``
    """
    T = min(tgt_a.shape[0], tgt_b.shape[0])
    C = tgt_a.shape[2]

    combined = torch.cat([tgt_a[:T], tgt_b[:T]], dim=1)   # (T, Na+Nb, C, 3)
    active   = combined.norm(dim=-1) > 0                  # (T, Na+Nb, C)

    out = torch.zeros(T, n_tracks, C, 3, dtype=torch.float32)
    for c in range(C):
        mask_c = active[:, :, c]
        for t in range(T):
            slots = mask_c[t].nonzero(as_tuple=False).view(-1)
            for slot_out, src in enumerate(slots[:n_tracks]):
                out[t, slot_out, c] = combined[t, src, c]

    return out


# ══════════════════════════════════════════════════════════════════════════════
# BalancedSTARSS23Dataset
# ══════════════════════════════════════════════════════════════════════════════

class BalancedSTARSS23Dataset(Dataset):
    """
    Balanced 13-class SELD training dataset.

    Balancing is performed at **clip level** (greedy external selection +
    FOA mixup for uncovered classes).  In windowed mode each clip-level item
    is subsequently expanded to fixed-length windows.

    Item types in ``_items``:
        Full-clip  mode: ``('starss23', ci)``, ``('nigens', ci)``,
                         ``('tau2019', ci)``, ``('mixup', ci_a, ci_b)``
        Windowed   mode: ``('starss23', ci, start)``, ``('nigens', ci, start)``,
                         ``('tau2019', ci, start)``, ``('mixup', ci_a, ci_b, start)``
        where ``start`` is in the dataset's native sample rate.
    """

    N_CLASSES = 13

    def __init__(
        self,
        starss_root:   Path,
        nigens_root:   Path,
        tau2019_root:  Path,
        model_sr:      int   = 24000,
        hop_length:    int   = 240,
        n_tracks:      int   = 3,
        window_s:      Optional[float] = None,
        hop_s:         Optional[float] = None,
        val_fraction:  float = 0.2,
        val_seed:      int   = 42,
        augment              = None,
        balance_seed:  int   = 42,
        target_count:  Optional[int] = None,
    ):
        super().__init__()
        self.model_sr   = model_sr
        self.hop_length = hop_length
        self.n_tracks   = n_tracks
        self.augment    = augment

        self.windowed   = window_s is not None
        self.window_s   = window_s
        self.hop_s      = hop_s if hop_s is not None else window_s

        if self.windowed:
            # T_mel_window is the same for every dataset type after resampling
            self.T_mel_window = 1 + round(window_s * model_sr) // hop_length
        else:
            self.T_mel_window = None

        common = dict(
            model_sr   = model_sr,
            hop_length = hop_length,
            n_tracks   = n_tracks,
            window_s   = None,   # sub-datasets always full-clip; windowing done here
            augment    = None,
        )
        self._starss_ds = STARSS23Dataset(
            root         = Path(starss_root),
            split        = 'train',
            val_fraction = val_fraction,
            val_seed     = val_seed,
            **common,
        )
        self._nigens_ds = TAUNIGENSDataset(
            root  = Path(nigens_root),
            split = 'train',
            **common,
        )
        self._tau2019_ds = TAU2019Dataset(
            root       = Path(tau2019_root),
            split      = 'train',
            mixed_mode = True,
            **common,
        )

        # Pre-project NIGENS and TAU2019 events to 13-class space
        self._nigens_proj  = [
            _project_events(evts, NIGENS_TO_STARSS13)
            for evts in self._nigens_ds._events
        ]
        self._tau2019_proj = [
            _project_events(evts, TAU2019_TO_STARSS13)
            for evts in self._tau2019_ds._events
        ]

        # Window/hop sizes in each dataset's native sample rate
        if self.windowed:
            self._win_samp = {
                'starss23': round(window_s * self._starss_ds.orig_sr),
                'nigens':   round(window_s * self._nigens_ds.orig_sr),
                'tau2019':  round(window_s * self._tau2019_ds.orig_sr),
            }
            self._hop_samp = {
                'starss23': round(self.hop_s * self._starss_ds.orig_sr),
                'nigens':   round(self.hop_s * self._nigens_ds.orig_sr),
                'tau2019':  round(self.hop_s * self._tau2019_ds.orig_sr),
            }

        rng = np.random.default_rng(balance_seed)
        clip_items   = self._build_clip_index(rng, target_count)
        self._items  = (
            self._expand_to_windows(clip_items) if self.windowed else clip_items
        )

        n_s = sum(1 for it in self._items if it[0] == 'starss23')
        n_n = sum(1 for it in self._items if it[0] == 'nigens')
        n_t = sum(1 for it in self._items if it[0] == 'tau2019')
        n_m = sum(1 for it in self._items if it[0] == 'mixup')
        mode = f'windowed {window_s}s/{self.hop_s}s' if self.windowed else 'full-clip'
        print(
            f'[BalancedSTARSS23Dataset] {len(self._items)} items ({mode}): '
            f'{n_s} starss23, {n_n} nigens, {n_t} tau2019, {n_m} mixup'
        )

    # ── Clip-level balancing ──────────────────────────────────────────────────

    def _build_clip_index(
        self,
        rng: np.random.Generator,
        target_count: Optional[int],
    ) -> list:
        """
        Build clip-level item list in two passes.

        Phase 2 — greedy external selection for all non-mixup classes.
        Phase 3 — scored mixup for _MIXUP_ONLY classes using the FULL pool
                   (STARSS23 + all selected NIGENS + TAU2019) as mixer candidates.
                   Mixers are picked via a min-heap scored by sum(current[cls]),
                   so each pair inflates the least-represented classes first.
        """
        starss_pres = [
            {cls for (_, cls, _, _, _) in evts if 0 <= cls < self.N_CLASSES}
            for evts in self._starss_ds._events
        ]
        nigens_pres = [
            {cls for (_, cls, _, _, _) in proj if 0 <= cls < self.N_CLASSES}
            for proj in self._nigens_proj
        ]
        tau2019_pres = [
            {cls for (_, cls, _, _, _) in proj if 0 <= cls < self.N_CLASSES}
            for proj in self._tau2019_proj
        ]

        current = np.zeros(self.N_CLASSES, dtype=int)
        for pres in starss_pres:
            for c in pres:
                current[c] += 1

        if target_count is None:
            target_count = int(current.max())

        items: list = [('starss23', i) for i in range(len(self._starss_ds.clips))]

        ext_pool = (
            [('nigens',   ci, pres) for ci, pres in enumerate(nigens_pres)]
            + [('tau2019', ci, pres) for ci, pres in enumerate(tau2019_pres)]
        )

        selected_nigens:  set = set()
        selected_tau2019: set = set()

        # ── Phase 2: external selection (non-mixup-only classes) ──────────────
        phase2_order = [
            (target_count - current[c], c)
            for c in range(self.N_CLASSES) if c not in _MIXUP_ONLY
        ]
        phase2_order.sort(reverse=True)

        for _, c in phase2_order:
            actual_deficit = target_count - current[c]
            if actual_deficit <= 0:
                continue

            not_sel, already_sel = [], []
            for ds_type, ci, pres in ext_pool:
                if c not in pres:
                    continue
                if ds_type == 'nigens' and ci not in selected_nigens:
                    not_sel.append((ds_type, ci, pres))
                elif ds_type == 'tau2019' and ci not in selected_tau2019:
                    not_sel.append((ds_type, ci, pres))
                else:
                    already_sel.append((ds_type, ci, pres))

            rng.shuffle(not_sel)

            for ds_type, ci, pres in not_sel:
                if current[c] >= target_count:
                    break
                items.append((ds_type, ci))
                for c2 in pres:
                    current[c2] += 1
                if ds_type == 'nigens':
                    selected_nigens.add(ci)
                else:
                    selected_tau2019.add(ci)

            all_covering = not_sel + already_sel
            while current[c] < target_count and all_covering:
                pick = all_covering[rng.integers(len(all_covering))]
                ds_type, ci, pres = pick
                items.append((ds_type, ci))
                for c2 in pres:
                    current[c2] += 1

        # ── Phase 3: scored mixup for _MIXUP_ONLY classes ─────────────────────
        # Build the full pool of unique selected clips (all STARSS23 + external).
        all_sel: list = []
        for ci, pres in enumerate(starss_pres):
            all_sel.append(('starss23', ci, pres))
        for ci in sorted(selected_nigens):
            all_sel.append(('nigens', ci, nigens_pres[ci]))
        for ci in sorted(selected_tau2019):
            all_sel.append(('tau2019', ci, tau2019_pres[ci]))

        phase3_order = [
            (target_count - current[c], c)
            for c in _MIXUP_ONLY
        ]
        phase3_order.sort(reverse=True)

        for _, c in phase3_order:
            actual_deficit = target_count - current[c]
            if actual_deficit <= 0:
                continue

            rare = [ci for ci, pres in enumerate(starss_pres) if c in pres]
            if not rare:
                continue

            # Mixer candidates: clips that do NOT contain the rare class.
            # Heap key = sum(current[cls] for cls in pres) — lower means
            # the mixer inflates classes that are currently under-represented.
            heap: list = []
            for idx, (type_, ci2, pres) in enumerate(all_sel):
                if c in pres:
                    continue
                score = sum(current[cls] for cls in pres)
                heap.append((score, idx, type_, ci2, pres))
            heapq.heapify(heap)

            if not heap:
                continue

            for _ in range(actual_deficit):
                ci_a = rare[rng.integers(len(rare))]
                score, idx, type_b, ci_b, pres_b = heapq.heappop(heap)
                items.append(('mixup', ci_a, ci_b, type_b))
                current[c] += 1
                for cls in pres_b:
                    current[cls] += 1
                new_score = sum(current[cls] for cls in pres_b)
                heapq.heappush(heap, (new_score, idx, type_b, ci_b, pres_b))

        return items

    # ── Windowed expansion ────────────────────────────────────────────────────

    def _expand_to_windows(self, clip_items: list) -> list:
        """Expand clip-level items to window-level items (windowed mode only)."""
        windowed = []
        ws_s = self._win_samp['starss23']
        hs_s = self._hop_samp['starss23']
        ws_n = self._win_samp['nigens']
        hs_n = self._hop_samp['nigens']
        ws_t = self._win_samp['tau2019']
        hs_t = self._hop_samp['tau2019']

        def _windows(audio_path: Path, ws: int, hs: int):
            n = sf.info(str(audio_path)).frames
            start = 0
            # Include window if at least 50% of its span contains real audio.
            # Partial tail is zero-padded by _load_clip_segment.
            while start + ws // 2 <= n:
                yield start
                start += hs

        for item in clip_items:
            t = item[0]
            if t == 'starss23':
                ci = item[1]
                ap = self._starss_ds.clips[ci][0]
                for start in _windows(ap, ws_s, hs_s):
                    windowed.append(('starss23', ci, start))
            elif t == 'nigens':
                ci = item[1]
                ap = self._nigens_ds.clips[ci][0]
                for start in _windows(ap, ws_n, hs_n):
                    windowed.append(('nigens', ci, start))
            elif t == 'tau2019':
                ci = item[1]
                ap = self._tau2019_ds.clips[ci][0]
                for start in _windows(ap, ws_t, hs_t):
                    windowed.append(('tau2019', ci, start))
            elif t == 'mixup':
                ci_a   = item[1]
                ci_b   = item[2]
                type_b = item[3]   # 'starss23', 'nigens', or 'tau2019'

                ap_a = self._starss_ds.clips[ci_a][0]
                n_a  = sf.info(str(ap_a)).frames

                if type_b == 'starss23':
                    ap_b = self._starss_ds.clips[ci_b][0]
                    n_b_starss = sf.info(str(ap_b)).frames
                elif type_b == 'nigens':
                    ap_b = self._nigens_ds.clips[ci_b][0]
                    n_b_starss = sf.info(str(ap_b)).frames  # same orig_sr
                else:  # 'tau2019'
                    ap_b = self._tau2019_ds.clips[ci_b][0]
                    n_b_tau = sf.info(str(ap_b)).frames
                    n_b_starss = int(
                        n_b_tau
                        * self._starss_ds.orig_sr
                        / self._tau2019_ds.orig_sr
                    )

                n_min = min(n_a, n_b_starss)
                start = 0
                # Same 50% threshold as non-mixup items: include if at least
                # half the window span has real audio in the shorter clip.
                while start + ws_s // 2 <= n_min:
                    windowed.append(('mixup', ci_a, ci_b, type_b, start))
                    start += hs_s

        return windowed

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item      = self._items[idx]
        item_type = item[0]

        # Extract start_sample and window_samples for the relevant sub-dataset
        if self.windowed:
            start_orig = item[-1]   # last element
        else:
            start_orig = 0

        if item_type == 'starss23':
            ci  = item[1]
            ds  = self._starss_ds
            ws  = self._win_samp.get('starss23', 0) if self.windowed else 0
            wav, n_mel = _load_clip_segment(ds, ci, start_orig, ws, self.model_sr)
            if self.windowed:
                n_mel = self.T_mel_window
            start_at_model = round(start_orig * self.model_sr / ds.orig_sr)
            events  = ds._events[ci]
            target  = _build_seld_target(
                events, n_mel, self.n_tracks, self.N_CLASSES,
                self.model_sr, self.hop_length, start_at_model,
            )

        elif item_type == 'nigens':
            ci  = item[1]
            ds  = self._nigens_ds
            ws  = self._win_samp.get('nigens', 0) if self.windowed else 0
            wav, n_mel = _load_clip_segment(ds, ci, start_orig, ws, self.model_sr)
            if self.windowed:
                n_mel = self.T_mel_window
            start_at_model = round(start_orig * self.model_sr / ds.orig_sr)
            events  = self._nigens_proj[ci]
            target  = _build_seld_target(
                events, n_mel, self.n_tracks, self.N_CLASSES,
                self.model_sr, self.hop_length, start_at_model,
            )

        elif item_type == 'tau2019':
            ci  = item[1]
            ds  = self._tau2019_ds
            ws  = self._win_samp.get('tau2019', 0) if self.windowed else 0
            wav, n_mel = _load_clip_segment(ds, ci, start_orig, ws, self.model_sr)
            if self.windowed:
                n_mel = self.T_mel_window
            # TAU2019 orig_sr=48kHz → start_at_model at 24kHz
            start_at_model = round(start_orig * self.model_sr / ds.orig_sr)
            events  = self._tau2019_proj[ci]
            target  = _build_seld_target(
                events, n_mel, self.n_tracks, self.N_CLASSES,
                self.model_sr, self.hop_length, start_at_model,
            )

        elif item_type == 'mixup':
            ci_a      = item[1]
            ci_b      = item[2]
            ds_b_type = item[3]
            # start_orig already extracted above (full-clip → 0, windowed → item[-1])

            ds_a = self._starss_ds
            ws_a = self._win_samp.get('starss23', 0) if self.windowed else 0
            wav_a, n_mel_a = _load_clip_segment(ds_a, ci_a, start_orig, ws_a, self.model_sr)

            if ds_b_type == 'starss23':
                ds_b     = self._starss_ds
                ws_b     = self._win_samp.get('starss23', 0) if self.windowed else 0
                start_b  = start_orig
                events_b = ds_a._events[ci_b]
            elif ds_b_type == 'nigens':
                ds_b     = self._nigens_ds
                ws_b     = self._win_samp.get('nigens', 0) if self.windowed else 0
                start_b  = start_orig   # same orig_sr as STARSS23
                events_b = self._nigens_proj[ci_b]
            else:  # 'tau2019'
                ds_b     = self._tau2019_ds
                ws_b     = self._win_samp.get('tau2019', 0) if self.windowed else 0
                start_b  = round(start_orig * ds_b.orig_sr / ds_a.orig_sr)
                events_b = self._tau2019_proj[ci_b]

            wav_b, n_mel_b = _load_clip_segment(ds_b, ci_b, start_b, ws_b, self.model_sr)

            T   = min(wav_a.shape[-1], wav_b.shape[-1])
            wav = (wav_a[:, :T] + wav_b[:, :T]) * 0.5

            n_mel          = self.T_mel_window if self.windowed else min(n_mel_a, n_mel_b)
            start_at_model = round(start_orig * self.model_sr / ds_a.orig_sr)

            tgt_a = _build_seld_target(
                ds_a._events[ci_a], n_mel, self.n_tracks, self.N_CLASSES,
                self.model_sr, self.hop_length, start_at_model,
            )
            tgt_b = _build_seld_target(
                events_b, n_mel, self.n_tracks, self.N_CLASSES,
                self.model_sr, self.hop_length, start_at_model,
            )
            target = _union_targets(tgt_a, tgt_b, self.n_tracks)

        else:
            raise ValueError(f'Unknown item type: {item_type!r}')

        if self.augment is not None:
            wav, target = self.augment(wav, target, force=False)

        return wav, target


# ══════════════════════════════════════════════════════════════════════════════
# ProjectedSELDDataset
# ══════════════════════════════════════════════════════════════════════════════

class ProjectedSELDDataset(Dataset):
    """
    Wraps a SELDDataset and projects per-event class indices to the 13-class
    STARSS23 space.

    Intended for **val/test** only (``augment=None`` on the base dataset).
    Exposes ``windowed`` and ``collate_fn`` so DataLoader dispatch works.
    """

    N_CLASSES = 13

    def __init__(self, base_ds: SELDDataset, mapping: List[List[int]]):
        self.base_ds      = base_ds
        self._proj_events = [
            _project_events(evts, mapping) for evts in base_ds._events
        ]

    @property
    def windowed(self) -> bool:
        return self.base_ds.windowed

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ds                        = self.base_ds
        clip_idx, start_sample, _ = ds._items[idx]
        audio_path, _             = ds.clips[clip_idx]
        proj_events               = self._proj_events[clip_idx]

        if ds.windowed:
            wav, _ = sf.read(
                str(audio_path),
                start     = start_sample,
                stop      = start_sample + ds.window_samples,
                dtype     = 'float32',
                always_2d = True,
            )
            if wav.shape[0] < ds.window_samples:
                pad = ds.window_samples - wav.shape[0]
                wav = np.concatenate(
                    [wav, np.zeros((pad, wav.shape[1]), dtype=np.float32)]
                )
            n_mel = ds.T_mel_window
        else:
            wav, _      = sf.read(str(audio_path), dtype='float32', always_2d=True)
            start_sample = 0

        wav_t = torch.from_numpy(wav.T.copy())

        if ds.output_sr is not None:
            wav_t = torchaudio.functional.resample(wav_t, ds.orig_sr, ds.output_sr)

        if not ds.windowed:
            eff_sr          = ds.output_sr if ds.output_sr is not None else ds.orig_sr
            n_samples_model = round(wav_t.shape[-1] * ds.model_sr / eff_sr)
            n_mel           = 1 + n_samples_model // ds.hop_length

        start_at_model = round(start_sample * ds.model_sr / ds.orig_sr)
        target = _build_seld_target(
            proj_events, n_mel, ds.n_tracks, self.N_CLASSES,
            ds.model_sr, ds.hop_length, start_at_model,
        )

        return wav_t, target


# ══════════════════════════════════════════════════════════════════════════════
# BalancedSELDDataModule
# ══════════════════════════════════════════════════════════════════════════════

class BalancedSELDDataModule(L.LightningDataModule):
    """
    LightningDataModule for BalancedSTARSS23 training.

    Train : BalancedSTARSS23Dataset  (~545 clips / ~4k+ windows in windowed mode)
    Val   : STARSS23Dataset(split='val')
    Test  : 3 DataLoaders
        idx=0 → STARSS23Dataset(split='test')                   → test_starss/*
        idx=1 → ProjectedSELDDataset(NIGENS test, NIGENS map)   → test_tau/*
        idx=2 → ProjectedSELDDataset(TAU2019 eval, TAU2019 map) → test_tau2019/*
    """

    def __init__(self, cfg: dict):
        super().__init__()

        data_cfg  = cfg.get('data', {})
        in_cfg    = cfg['input']
        head_cfg  = cfg.get('heads', {})
        aug_cfg   = cfg.get('augmentation', {})
        train_cfg = cfg.get('training', {})

        self.starss_root  = Path(data_cfg['starss_root'])
        self.nigens_root  = Path(data_cfg['tau_root'])
        self.tau2019_root = Path(data_cfg['tau2019_root'])

        self.model_sr     = in_cfg['sr']
        self.hop_length   = in_cfg.get('hop_length', 320)
        self.n_tracks     = head_cfg.get('n_tracks', 3)
        self.window_s     = data_cfg.get('window_s')
        self.hop_s        = data_cfg.get('hop_s')
        self.val_fraction = data_cfg.get('val_fraction', 0.2)
        self.val_seed     = data_cfg.get('val_seed', 42)
        self.num_workers  = data_cfg.get('num_workers', 4)
        self.batch_size   = train_cfg.get('batch_size', 4)

        self._augment: Optional[SELDAugment] = None
        if aug_cfg.get('enabled', False):
            self._augment = SELDAugment(
                p_16patterns = aug_cfg.get('p_16patterns', 0.0),
                p_gain       = aug_cfg.get('p_gain', 0.0),
                p_noise      = aug_cfg.get('p_noise', 0.0),
                gain_min     = aug_cfg.get('gain_min', 0.5),
                gain_max     = aug_cfg.get('gain_max', 0.5),
                noise_snr_db = aug_cfg.get('noise_snr_db', 30.0),
            )

        self._train_ds: Optional[BalancedSTARSS23Dataset] = None
        self._val_ds:   Optional[SELDDataset]              = None
        self._test_dss: Optional[list]                     = None

    def setup(self, stage: Optional[str] = None) -> None:
        common = dict(
            model_sr   = self.model_sr,
            hop_length = self.hop_length,
            n_tracks   = self.n_tracks,
        )

        if stage in ('fit', None):
            self._train_ds = BalancedSTARSS23Dataset(
                starss_root   = self.starss_root,
                nigens_root   = self.nigens_root,
                tau2019_root  = self.tau2019_root,
                window_s      = self.window_s,
                hop_s         = self.hop_s,
                val_fraction  = self.val_fraction,
                val_seed      = self.val_seed,
                augment       = self._augment,
                **common,
            )
            self._val_ds = STARSS23Dataset(
                root         = self.starss_root,
                split        = 'val',
                window_s     = self.window_s,
                hop_s        = self.hop_s,
                val_fraction = self.val_fraction,
                val_seed     = self.val_seed,
                augment      = None,
                **common,
            )

        if stage in ('test', None):
            test_kw = dict(
                split    = 'test',
                window_s = self.window_s,
                hop_s    = self.hop_s,
                augment  = None,
                **common,
            )
            starss_test  = STARSS23Dataset(root=self.starss_root,  **test_kw)
            nigens_test  = TAUNIGENSDataset(root=self.nigens_root, **test_kw)
            tau2019_test = TAU2019Dataset(
                root       = self.tau2019_root,
                mixed_mode = True,
                **test_kw,
            )
            self._test_dss = [
                starss_test,
                ProjectedSELDDataset(nigens_test,  NIGENS_TO_STARSS13),
                ProjectedSELDDataset(tau2019_test, TAU2019_TO_STARSS13),
            ]

    def _make_loader(self, ds, shuffle: bool = False) -> DataLoader:
        windowed = getattr(ds, 'windowed', False)
        collate  = None if windowed else SELDDataset.collate_fn
        return DataLoader(
            ds,
            batch_size         = self.batch_size,
            shuffle            = shuffle,
            num_workers        = self.num_workers,
            collate_fn         = collate,
            pin_memory         = False,
            persistent_workers = self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self._train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self._val_ds, shuffle=False)

    def test_dataloader(self) -> List[DataLoader]:
        return [self._make_loader(ds, shuffle=False) for ds in self._test_dss]
