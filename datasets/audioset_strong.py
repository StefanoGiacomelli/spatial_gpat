"""
AudioSetStrongDataset — PyTorch Dataset for AudioSet-Strong SED
===============================================================
Windowed multi-label dataset for fine-tuning the GP-AT (E-PANNs) backbone
as a frame-level SED classifier on AudioSet-Strong annotations.

Each audio clip is 10 seconds, mono, 32 kHz.  The dataset slides a window of
``window_s`` seconds with hop ``hop_s`` over each clip, yielding N independent
training items from each clip.  Each item's label vector is derived from the
strong annotations (start_s, end_s, mid) by checking which classes overlap
the window by at least ``overlap_threshold`` fraction.

Windowing example::

    window_s=2s, hop_s=2s   →  5 windows from a 10 s clip
    window_s=10s, hop_s=10s →  1 window  (baseline, same as original clip)
    window_s=1s,  hop_s=0.5s → ~19 windows (with overlap)

Class imbalance::

    pos_weight[c] = neg_windows[c] / pos_windows[c]
    (computed on training items only, capped by pos_weight_max)

Strong annotation formats::

    Train TSV (4 columns):  clip_id  start_s  end_s  mid
    Eval  TSV (5 columns):  clip_id  start_s  end_s  mid  presence
    Only rows with presence='PRESENT' are used from the eval TSV.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

import csv
import math
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import soundfile as sf
import torch
import lightning as L
from scipy.sparse import csr_matrix as sp_csr
from torch.utils.data import DataLoader, WeightedRandomSampler

from experiments_at2sed.hlp import StrongHLP


# ══════════════════════════════════════════════════════════════════════════════
# Augmentation
# ══════════════════════════════════════════════════════════════════════════════

class AudioSetAugment:
    """
    Waveform-level augmentation for AudioSet-Strong mono clips.

    Two techniques applied **mutually exclusively** (one per sample):

    1. **Random gain** — scales waveform by ``g = 10^(U(-gain_db, gain_db)/20)``.
    2. **White noise** — adds Gaussian noise at ``noise_snr_db`` dB SNR.

    If ``class_balance=True`` and ``rare_classes`` is set via
    ``set_rare_classes()``, augmentation is applied **only** to samples
    containing at least one rare class.  All other samples are returned
    unchanged.

    Args:
        p_gain              : probability of applying random gain (default 0.4)
        gain_db             : max gain magnitude in dB, range ±gain_db (default 6.0)
        p_noise             : probability of applying white noise (default 0.3)
        noise_snr_db        : SNR in dB for the additive noise (default 35.0)
        class_balance       : if True, only augment samples with rare classes
        rare_class_threshold: pos_rate below this → class is considered rare (default 0.1)
    """

    def __init__(
        self,
        p_gain: float = 0.4,
        gain_db: float = 6.0,
        p_noise: float = 0.3,
        noise_snr_db: float = 35.0,
        class_balance: bool = False,
        rare_class_threshold: float = 0.1,
        **kwargs,   # absorb extra YAML keys gracefully
    ):
        self.p_gain               = p_gain
        self.gain_db              = gain_db
        self.p_noise              = p_noise
        self.noise_snr_db         = noise_snr_db
        self.class_balance        = class_balance
        self.rare_class_threshold = rare_class_threshold
        self._rare_classes: Set[int] = set()

    def set_rare_classes(self, rare_classes: Set[int]) -> None:
        """Inject the set of rare class indices for targeted augmentation."""
        self._rare_classes = rare_classes

    def __call__(
        self,
        wav: torch.Tensor,
        label: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply augmentation to a single sample.

        Args:
            wav   : ``(T_samples,)`` float32 waveform
            label : ``(n_cls,)`` float32 multi-hot label (not modified)

        Returns:
            Augmented ``(wav, label)`` pair.  ``label`` is always returned
            unchanged; only the waveform may be altered.
        """
        # If class_balance is on, skip samples without any rare class
        if self.class_balance and self._rare_classes:
            active = set(label.nonzero(as_tuple=False).view(-1).tolist())
            if not (active & self._rare_classes):
                return wav, label

        p_total = self.p_gain + self.p_noise
        if p_total <= 0.0 or torch.rand(1).item() >= p_total:
            return wav, label   # no augmentation this sample

        # Mutually exclusive: choose gain or noise proportionally
        if torch.rand(1).item() < self.p_gain / p_total:
            wav = self._apply_gain(wav)
        else:
            wav = self._apply_noise(wav)

        return wav, label

    # ── Private helpers ──────────────────────────────────────────────────────

    def _apply_gain(self, wav: torch.Tensor) -> torch.Tensor:
        log_gain = (torch.rand(1).item() * 2.0 - 1.0) * (self.gain_db / 20.0) * math.log(10)
        return wav * math.exp(log_gain)

    def _apply_noise(self, wav: torch.Tensor) -> torch.Tensor:
        rms = wav.pow(2).mean().sqrt().clamp(min=1e-8)
        noise_rms = rms * (10.0 ** (-self.noise_snr_db / 20.0))
        return wav + torch.randn_like(wav) * noise_rms


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class AudioSetStrongDataset(torch.utils.data.Dataset):
    """
    Windowed multi-label dataset for AudioSet-Strong SED.

    Args:
        split                : ``'train'`` | ``'val'`` | ``'test'``
        audio_dir            : directory containing ``*.wav`` audio files
        strong_tsv           : path to strong annotation TSV (train or eval)
        class_map_csv        : path to ``class_labels_indices.csv`` (MID → int index)
        window_s             : window duration in seconds
        hop_s                : hop between windows; ``None`` → equal to ``window_s``
        n_cls                : number of target classes (default 527)
        overlap_threshold    : label[c]=1 iff event/window overlap ≥ threshold fraction
                               (default 0.0 = any presence triggers label)
        val_fraction         : fraction of balanced clips used for validation (default 0.2)
        val_seed             : random seed for reproducible train/val split (default 42)
        max_negative_fraction: cap on the fraction of negative-only windows in training
                               items (default 1.0 = no filtering).  Training-only.
        augment              : optional :class:`AudioSetAugment` instance; applied only
                               if ``split == 'train'``
    """

    # All AudioSet clips are exactly 10 seconds
    CLIP_DURATION_S: float = 10.0
    SAMPLE_RATE:     int   = 32000   # native SR of the qiuqiangkong download

    def __init__(
        self,
        split: str,
        audio_dir: Path,
        strong_tsv: Path,
        class_map_csv: Path,
        window_s: float,
        hop_s: Optional[float] = None,
        n_cls: int = 527,
        overlap_threshold: float = 0.0,
        val_fraction: float = 0.2,
        val_seed: int = 42,
        max_negative_fraction: float = 1.0,
        augment: Optional[AudioSetAugment] = None,
        extra_audio_dirs: Optional[List[Path]] = None,
        hlp_enabled: bool = False,
        hlp_table_path: Optional[Path] = None,
        balanced_sampler_formula: str = 'maxpos',
    ):
        super().__init__()
        self.split                = split
        self.audio_dir            = Path(audio_dir)
        self.extra_audio_dirs     = [Path(d) for d in extra_audio_dirs] if extra_audio_dirs else []
        self.strong_tsv           = Path(strong_tsv)
        self.n_cls                = n_cls
        self.window_s             = float(window_s)
        self.hop_s                = float(hop_s) if hop_s is not None else self.window_s
        self.window_samples       = round(self.window_s  * self.SAMPLE_RATE)
        self.hop_samples          = round(self.hop_s     * self.SAMPLE_RATE)
        self.overlap_threshold    = float(overlap_threshold)
        self.val_fraction         = float(val_fraction)
        self.val_seed             = int(val_seed)
        self._max_neg_frac        = float(max_negative_fraction)
        # augment is applied only during training
        self.augment              = augment if split == 'train' else None

        # ── HLP (Hierarchical Label Propagation) ──────────────────────────────
        if hlp_enabled:
            self._hlp: Optional[StrongHLP] = StrongHLP(hlp_table_path)
            print(f'[AudioSetStrongDataset] HLP enabled '
                  f'(table: {Path(hlp_table_path) if hlp_table_path else "default"})')
        else:
            self._hlp = None

        # ── Balanced sampler formula ──────────────────────────────────────────
        self._sampler_formula = balanced_sampler_formula   # 'psla' | 'maxpos'

        # ── Load class map: MID → int index ──────────────────────────────────
        self._class_map: Dict[str, int] = self._load_class_map(class_map_csv)

        # ── Load events: clip_id → list[(start_s, end_s, class_idx)] ─────────
        self._events: Dict[str, List[Tuple[float, float, int]]] = self._load_events()

        # ── Collect (clip_id, audio_path) pairs for the requested split ───────
        all_audio_clips = self._collect_clips()   # all clips with matching audio
        if split == 'test':
            self._clips = all_audio_clips
        else:
            self._clips = self._split_clips(all_audio_clips)

        # Fast clip_id → audio_path lookup used in __getitem__
        self._clip_audio: Dict[str, Path] = {cid: p for cid, p in self._clips}

        # ── Build windowed item index ─────────────────────────────────────────
        self._items: List[Tuple[str, float]] = self._build_item_index()

        n_clips = len(self._clips)
        n_items = len(self._items)
        print(
            f'[AudioSetStrongDataset] {split}: {n_clips} clips, {n_items} windows '
            f'(window={self.window_s:.1f}s, hop={self.hop_s:.1f}s)'
        )

    # ── Class map ────────────────────────────────────────────────────────────

    def _load_class_map(self, class_map_csv: Path) -> Dict[str, int]:
        """Load MID → integer index mapping from ``class_labels_indices.csv``."""
        mapping: Dict[str, int] = {}
        with open(class_map_csv, newline='') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                mapping[row['mid'].strip()] = int(row['index'])
        return mapping

    # ── Events loader ─────────────────────────────────────────────────────────

    def _load_events(self) -> Dict[str, List[Tuple[float, float, int]]]:
        """
        Parse the strong annotation TSV.

        Handles both 4-column (train) and 5-column (eval, includes presence flag)
        formats.  Only ``PRESENT`` rows are kept from the 5-column eval format.

        Returns:
            ``dict[clip_id → [(start_s, end_s, class_idx), ...]]``
        """
        events: Dict[str, List[Tuple[float, float, int]]] = {}
        skipped_mid = 0

        with open(self.strong_tsv, newline='') as fh:
            reader = csv.reader(fh, delimiter='\t')
            for row in reader:
                if not row or row[0].startswith('#'):
                    continue   # blank / comment lines
                if row[0] == 'segment_id':
                    continue   # header row

                if len(row) == 4:
                    clip_id, start_s, end_s, mid = row
                    presence = 'PRESENT'
                elif len(row) == 5:
                    clip_id, start_s, end_s, mid, presence = row
                else:
                    continue   # unexpected format

                if presence.strip().upper() != 'PRESENT':
                    continue   # discard NOT_PRESENT rows from eval TSV

                class_idx = self._class_map.get(mid.strip())
                if class_idx is None:
                    skipped_mid += 1
                    continue

                events.setdefault(clip_id.strip(), []).append(
                    (float(start_s), float(end_s), class_idx)
                )

        if skipped_mid > 0:
            warnings.warn(
                f'[AudioSetStrongDataset] {skipped_mid} events skipped '
                f'(MID not found in class_map_csv)',
                stacklevel=2,
            )
        return events

    # ── Audio discovery ───────────────────────────────────────────────────────

    def _collect_clips(self) -> List[Tuple[str, Path]]:
        """
        Match TSV clip_ids to audio files across all audio directories.

        Searches ``audio_dir`` first, then any ``extra_audio_dirs`` in order.
        Handles filenames with or without a leading ``Y`` prefix, and TSV
        clip_ids in ``{ytid}_{start_ms}`` format (strips the suffix to match
        bare ``{ytid}.wav`` filenames).

        Returns:
            Sorted list of ``(clip_id, audio_path)`` for clips that have both
            strong annotations and a matching audio file.
        """
        # Build unified {stem: path} lookup from all directories.
        # Primary dir is searched first; extra dirs add without overwriting.
        audio_lookup: Dict[str, Path] = {}
        for search_dir in [self.audio_dir] + self.extra_audio_dirs:
            if not search_dir.exists():
                continue
            for p in search_dir.glob('*.wav'):
                stem = p.stem
                if stem not in audio_lookup:
                    audio_lookup[stem] = p
                # Also map without the Y prefix so both forms resolve
                if stem.startswith('Y') and stem[1:] not in audio_lookup:
                    audio_lookup[stem[1:]] = p

        found: List[Tuple[str, Path]] = []
        missing = 0
        for clip_id in sorted(self._events.keys()):
            if clip_id in audio_lookup:
                found.append((clip_id, audio_lookup[clip_id]))
            else:
                # TSV clip_ids have format {ytid}_{start_ms}; audio files are
                # named {ytid}.wav (one file per 10 s clip, already extracted).
                # Strip the trailing _{digits} suffix to recover the ytid.
                ytid = re.sub(r'_\d+$', '', clip_id)
                if ytid in audio_lookup:
                    found.append((clip_id, audio_lookup[ytid]))
                else:
                    missing += 1

        if missing > 0:
            all_dirs = [self.audio_dir] + self.extra_audio_dirs
            dirs_str = ', '.join(str(d.name) for d in all_dirs if d.exists())
            warnings.warn(
                f'[AudioSetStrongDataset] {missing} clip(s) in TSV have no matching '
                f'audio file in [{dirs_str}] — skipped.',
                stacklevel=2,
            )
        return found

    # ── Train / val split ─────────────────────────────────────────────────────

    def _split_clips(
        self,
        all_clips: List[Tuple[str, Path]],
    ) -> List[Tuple[str, Path]]:
        """
        Randomly split clips into train/val subsets.

        Uses ``val_seed`` for reproducibility.  The validation set gets
        ``val_fraction`` of clips; the training set gets the remainder.
        """
        rng     = np.random.RandomState(self.val_seed)
        indices = np.arange(len(all_clips))
        rng.shuffle(indices)

        n_val = max(1, int(len(indices) * self.val_fraction))
        if self.split == 'val':
            selected = sorted(indices[:n_val])
        else:   # 'train'
            selected = sorted(indices[n_val:])

        return [all_clips[i] for i in selected]

    # ── Item index ────────────────────────────────────────────────────────────

    def _build_item_index(self) -> List[Tuple[str, float]]:
        """
        Slide a window over every clip and build the flat item list.

        For each clip, windows are generated at positions
        ``[0, hop_s, 2*hop_s, ...]`` while the start is within the clip
        duration.  The last window that extends beyond the clip is still kept;
        ``__getitem__`` zero-pads it.

        For the **training** split, negative-only windows are downsampled to at
        most ``max_negative_fraction`` of total items (if < 1.0).
        """
        items: List[Tuple[str, float]] = []
        clip_dur = self.CLIP_DURATION_S

        for clip_id, _ in self._clips:
            t_start = 0.0
            while t_start < clip_dur:
                items.append((clip_id, t_start))
                t_start += self.hop_s
                t_start = round(t_start, 10)   # avoid float accumulation error

        if self.split == 'train' and self._max_neg_frac < 1.0:
            items = self._downsample_negatives(items)

        return items

    def _downsample_negatives(
        self,
        items: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """
        Subsample negative-only windows so their fraction ≤ max_negative_fraction.

        Preserves clip-level ordering.  Uses ``val_seed`` for reproducibility.
        """
        pos_items: List[Tuple[str, float]] = []
        neg_items: List[Tuple[str, float]] = []

        for item in items:
            clip_id, t_start = item
            t_end  = t_start + self.window_s
            active = False
            for ev_s, ev_e, _ in self._events.get(clip_id, []):
                if min(ev_e, t_end) - max(ev_s, t_start) > 0.0:
                    active = True
                    break
            if active:
                pos_items.append(item)
            else:
                neg_items.append(item)

        n_pos = len(pos_items)
        if n_pos == 0:
            return items   # nothing to balance

        # target_neg = pos / (1 - frac) * frac   but capped to available negatives
        frac       = self._max_neg_frac
        target_neg = int(round(n_pos * frac / max(1.0 - frac, 1e-9)))
        target_neg = min(target_neg, len(neg_items))

        if target_neg < len(neg_items):
            rng     = np.random.RandomState(self.val_seed)
            indices = rng.choice(len(neg_items), size=target_neg, replace=False)
            neg_selected = [neg_items[i] for i in sorted(indices)]
        else:
            neg_selected = neg_items

        all_items = pos_items + neg_selected
        # Restore clip-level ordering for reproducible batch composition
        clip_order = {cid: i for i, (cid, _) in enumerate(self._clips)}
        all_items.sort(key=lambda x: (clip_order.get(x[0], 0), x[1]))

        n_before = len(items)
        n_after  = len(all_items)
        pct_neg  = 100.0 * len(neg_selected) / max(n_after, 1)
        print(
            f'[AudioSetStrongDataset] train: negatives downsampled '
            f'{n_before} → {n_after} items  ({pct_neg:.1f}% negative)'
        )
        return all_items

    # ── Target construction ──────────────────────────────────────────────────

    def _build_target(self, clip_id: str, t_start: float) -> torch.Tensor:
        """
        Build ``(n_cls,)`` multi-hot label for window ``[t_start, t_start + window_s]``.

        Label[c] = 1 iff the fractional overlap of any event of class c with the
        window is ≥ ``overlap_threshold``.  With ``overlap_threshold=0.0`` (default),
        any positive overlap triggers the label.
        """
        target = torch.zeros(self.n_cls, dtype=torch.float32)
        t_end  = t_start + self.window_s

        for ev_s, ev_e, class_idx in self._events.get(clip_id, []):
            overlap = min(ev_e, t_end) - max(ev_s, t_start)
            if overlap <= 0.0:
                continue
            if self.overlap_threshold == 0.0 or overlap / self.window_s >= self.overlap_threshold:
                target[class_idx] = 1.0

        # Apply Hierarchical Label Propagation if enabled
        if self._hlp is not None:
            target = self._hlp.propagate(target)

        return target

    # ── Pos-weights ───────────────────────────────────────────────────────────

    def _compute_pos_weights(self) -> torch.Tensor:
        """
        Compute per-class positive weights from the training item index.

        ``pos_weight[c] = neg_windows[c] / pos_windows[c]``

        Computed on the current ``_items`` (post-windowing, post-negative-downsampling),
        so it reflects exactly what the model sees during training.

        Returns:
            ``(n_cls,)`` float32 Tensor.  Classes with zero positive windows
            get weight 1.0 (neutral).
        """
        pos_counts = np.zeros(self.n_cls, dtype=np.float64)
        n_items    = len(self._items)

        for clip_id, t_start in self._items:
            t_end = t_start + self.window_s
            seen: Set[int] = set()
            for ev_s, ev_e, class_idx in self._events.get(clip_id, []):
                if class_idx in seen:
                    continue
                overlap = min(ev_e, t_end) - max(ev_s, t_start)
                if overlap <= 0.0:
                    continue
                if self.overlap_threshold == 0.0 or overlap / self.window_s >= self.overlap_threshold:
                    seen.add(class_idx)
                    pos_counts[class_idx] += 1.0

        neg_counts = np.maximum(n_items - pos_counts, 0.0)
        pos_counts = np.maximum(pos_counts, 1.0)   # avoid division by zero
        weights    = neg_counts / pos_counts
        return torch.from_numpy(weights).float()

    # ── Weighted sampler support ───────────────────────────────────────────────

    def item_weights(self) -> torch.Tensor:
        """
        Per-item importance weights for ``WeightedRandomSampler``.

        Two formulas are available via ``balanced_sampler_formula``:

        ``'maxpos'`` (default, original behaviour):
            ``w_i = max(pos_weight[c] for active classes c)``
            where ``pos_weight[c] = neg_windows[c] / pos_windows[c]``.
            Negative items get weight 1.0.

        ``'psla'`` (PSLA paper, balanced sampling):
            ``w_i = sum_c( 1000 / (n_c + 0.01) )``
            where ``n_c`` is the number of training items where class c is
            active.  Negative items get weight 1.0.

            Reference: Gong et al., "PSLA: Improving Audio Tagging with
            Pretraining, Sampling, Labeling, and Aggregation," TASLP 2021.

        Returns:
            ``(len(self),)`` float32 Tensor.
        """
        if self._sampler_formula == 'psla':
            return self._item_weights_psla()
        return self._item_weights_maxpos()

    def _item_weights_maxpos(self) -> torch.Tensor:
        """max-pos-weight formula: w_i = max(pos_weight[c]) over active classes."""
        all_active = self._compute_active_classes_fast()
        n = len(self._items)

        pos_counts = np.zeros(self.n_cls, dtype=np.float64)
        for acts in all_active:
            for c in acts:
                pos_counts[c] += 1.0
        neg_counts = np.maximum(n - pos_counts, 0.0)
        pw = neg_counts / np.maximum(pos_counts, 1.0)   # (C,)

        weights = np.ones(n, dtype=np.float32)
        for i, acts in enumerate(all_active):
            if acts:
                weights[i] = float(max(pw[c] for c in acts))
        return torch.from_numpy(weights)

    def _item_weights_psla(self) -> torch.Tensor:
        """PSLA balanced sampling formula: w_i = sum_c 1000 / (n_c + 0.01)."""
        all_active = self._compute_active_classes_fast()

        counts = np.zeros(self.n_cls, dtype=np.float64)
        for acts in all_active:
            for c in acts:
                counts[c] += 1.0

        weights = np.ones(len(self._items), dtype=np.float32)
        for i, acts in enumerate(all_active):
            if acts:
                weights[i] = float(sum(1000.0 / (counts[c] + 0.01) for c in acts))

        wt = torch.from_numpy(weights)
        pos_mask = wt > 1.0
        if pos_mask.any():
            print(
                f'[AudioSetStrongDataset] PSLA weights: '
                f'pos items  min={wt[pos_mask].min():.2f}  '
                f'max={wt[pos_mask].max():.2f}  '
                f'mean={wt[pos_mask].mean():.2f}'
            )
        return wt

    def _compute_active_classes_fast(self) -> List[List[int]]:
        """
        Compute (HLP-propagated) active class indices for every item.

        Uses chunked scipy sparse matrix multiply instead of calling
        ``_build_target()`` per item, avoiding O(N × C²) overhead when
        HLP is enabled.
        """
        n = len(self._items)
        CHUNK = 65536

        hlp_sparse = None
        if self._hlp is not None:
            hlp_np = self._hlp._table.numpy()   # (C, C) float32
            hlp_sparse = sp_csr(hlp_np)

        print(
            f'[AudioSetStrongDataset] Computing sampler weights '
            f'for {n:,} windows'
            + (' + HLP propagation' if hlp_sparse is not None else '')
            + ' …'
        )

        all_active: List[List[int]] = []

        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            chunk_size = end - start
            mat = np.zeros((chunk_size, self.n_cls), dtype=np.float32)

            for i, (clip_id, t_start) in enumerate(self._items[start:end]):
                t_end = t_start + self.window_s
                for ev_s, ev_e, class_idx in self._events.get(clip_id, []):
                    ov = min(ev_e, t_end) - max(ev_s, t_start)
                    if ov <= 0.0:
                        continue
                    if (self.overlap_threshold == 0.0
                            or ov / self.window_s >= self.overlap_threshold):
                        mat[i, class_idx] = 1.0

            if hlp_sparse is not None:
                mat = (sp_csr(mat) @ hlp_sparse).toarray()
                np.clip(mat, 0.0, 1.0, out=mat)

            for i in range(chunk_size):
                all_active.append(list(np.where(mat[i] > 0)[0]))

        return all_active

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            wav   : ``(T_samples,)`` float32 at 32 kHz
            label : ``(n_cls,)``     float32 multi-hot
        """
        clip_id, t_start = self._items[idx]
        audio_path       = self._clip_audio[clip_id]
        start_sample     = round(t_start * self.SAMPLE_RATE)

        # ── Load audio window ─────────────────────────────────────────────────
        wav, _ = sf.read(
            str(audio_path),
            start     = start_sample,
            stop      = start_sample + self.window_samples,
            dtype     = 'float32',
            always_2d = False,
        )
        # Zero-pad last window if the clip ended before window_samples
        if len(wav) < self.window_samples:
            pad = self.window_samples - len(wav)
            wav = np.concatenate([wav, np.zeros(pad, dtype=np.float32)])

        wav_t  = torch.from_numpy(wav.copy())   # (T_samples,) float32

        # ── Build multi-hot label ─────────────────────────────────────────────
        label = self._build_target(clip_id, t_start)

        # ── Augmentation (train split only) ───────────────────────────────────
        if self.augment is not None:
            wav_t, label = self.augment(wav_t, label)

        return wav_t, label


# ══════════════════════════════════════════════════════════════════════════════
# DataModule
# ══════════════════════════════════════════════════════════════════════════════

class AudioSetStrongDataModule(L.LightningDataModule):
    """
    LightningDataModule for AudioSet-Strong SED fine-tuning.

    Wraps :class:`AudioSetStrongDataset` and exposes train/val/test DataLoaders.
    After calling ``setup('fit')``, the ``pos_weights`` property returns the
    class-imbalance correction weights computed on the training set (post-windowing).

    Configuration (``cfg`` dict keys used)::

        data:
          audio_root_train       : path to balanced_train_segments/
          audio_root_unbalanced  : path to unbalanced_strong/ (optional extra dir)
          audio_root_test        : path to eval_segments/
          strong_tsv_train       : path to audioset_train_strong.tsv
          strong_tsv_test        : path to audioset_eval_strong.tsv
          class_map          : path to class_labels_indices.csv
          window_s           : window duration in seconds
          hop_s              : hop between windows (default = window_s)
          val_fraction       : fraction of clips for validation (default 0.2)
          val_seed           : RNG seed for train/val split (default 42)
          overlap_threshold  : fractional overlap threshold (default 0.0)
          num_workers        : DataLoader workers (default 4)

        training:
          batch_size         : batch size (default 32)
          weighted_sampler   : bool — use WeightedRandomSampler (default false)
          pos_weight_max     : cap on pos_weight values (default 20.0)
          max_negative_fraction : max fraction of negative-only windows in train
          class_balance      : bool — augment only rare classes (default false)
          rare_class_threshold : pos_rate below this → rare class (default 0.1)

        augmentation:
          enabled            : bool (default false)
          p_gain / gain_db   : random gain parameters
          p_noise / noise_snr_db : white noise parameters

        model:
          n_cls              : number of target classes (default 527)
    """

    def __init__(self, cfg: dict):
        super().__init__()

        data_cfg  = cfg.get('data', {})
        train_cfg = cfg.get('training', {})
        aug_cfg   = cfg.get('augmentation', {})

        self.audio_root_train = Path(data_cfg['audio_root_train'])
        self.audio_root_test  = Path(data_cfg['audio_root_test'])
        self.strong_tsv_train = Path(data_cfg['strong_tsv_train'])
        self.strong_tsv_test  = Path(data_cfg['strong_tsv_test'])
        self.class_map        = Path(data_cfg['class_map'])

        # Optional extra directory for unbalanced_strong clips (train+val only)
        _unbal = data_cfg.get('audio_root_unbalanced', None)
        self.audio_root_unbalanced: Optional[Path] = Path(_unbal) if _unbal else None

        self.window_s             = float(data_cfg.get('window_s', 10.0))
        self.hop_s                = float(data_cfg.get('hop_s', self.window_s))
        self.n_cls                = int(cfg.get('model', {}).get('n_cls', 527))
        self.overlap_threshold    = float(data_cfg.get('overlap_threshold', 0.0))
        self.val_fraction         = float(data_cfg.get('val_fraction', 0.2))
        self.val_seed             = int(data_cfg.get('val_seed', 42))
        self.num_workers          = int(data_cfg.get('num_workers', 4))

        self.batch_size               = int(train_cfg.get('batch_size', 32))
        self.use_weighted_sampler     = bool(train_cfg.get('weighted_sampler', False))
        self.pos_weight_max           = float(train_cfg.get('pos_weight_max', 20.0))
        self.max_neg_frac             = float(train_cfg.get('max_negative_fraction', 1.0))
        self.balanced_sampler_formula = str(train_cfg.get('balanced_sampler_formula', 'maxpos'))

        # ── HLP ───────────────────────────────────────────────────────────────
        self.hlp_enabled    = bool(data_cfg.get('hlp_enabled', False))
        _hlp_path           = data_cfg.get('hlp_table_path', None)
        self.hlp_table_path = Path(_hlp_path) if _hlp_path else None

        # ── Augmentation (train only) ─────────────────────────────────────────
        if aug_cfg.get('enabled', False):
            self._augment = AudioSetAugment(
                class_balance        = bool(train_cfg.get('class_balance', False)),
                rare_class_threshold = float(train_cfg.get('rare_class_threshold', 0.1)),
                **aug_cfg,
            )
        else:
            self._augment = None

        self._train_ds: Optional[AudioSetStrongDataset] = None
        self._val_ds:   Optional[AudioSetStrongDataset] = None
        self._test_ds:  Optional[AudioSetStrongDataset] = None
        self._pos_weights: Optional[torch.Tensor]       = None

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self, stage: Optional[str] = None) -> None:
        """Instantiate datasets for the requested stage."""
        # Guard against Lightning calling setup() a second time after we already
        # ran it manually in main_at2sed.py (avoids duplicate dataset construction
        # and duplicate prints).
        if stage in ('fit', None) and self._train_ds is not None:
            return
        if stage == 'test' and self._test_ds is not None:
            return

        common_train = dict(
            strong_tsv               = self.strong_tsv_train,
            class_map_csv            = self.class_map,
            window_s                 = self.window_s,
            hop_s                    = self.hop_s,
            n_cls                    = self.n_cls,
            overlap_threshold        = self.overlap_threshold,
            val_fraction             = self.val_fraction,
            val_seed                 = self.val_seed,
            max_negative_fraction    = self.max_neg_frac,
            hlp_enabled              = self.hlp_enabled,
            hlp_table_path           = self.hlp_table_path,
            balanced_sampler_formula = self.balanced_sampler_formula,
        )

        extra_dirs = ([self.audio_root_unbalanced]
                      if self.audio_root_unbalanced and self.audio_root_unbalanced.exists()
                      else [])

        if stage in ('fit', None):
            self._train_ds = AudioSetStrongDataset(
                split            = 'train',
                audio_dir        = self.audio_root_train,
                extra_audio_dirs = extra_dirs,
                augment          = self._augment,
                **common_train,
            )
            self._val_ds = AudioSetStrongDataset(
                split            = 'val',
                audio_dir        = self.audio_root_train,
                extra_audio_dirs = extra_dirs,
                augment          = None,
                **common_train,
            )
            # Compute and store pos_weights from training items
            pw = self._train_ds._compute_pos_weights()
            pw = pw.clamp(max=self.pos_weight_max)

            # If class_balance augmentation is enabled, identify rare classes
            if self._augment is not None and self._augment.class_balance:
                n_items   = len(self._train_ds)
                pos_rates = 1.0 / (1.0 + pw.numpy())   # approximate from weights
                thresh    = self._augment.rare_class_threshold
                rare      = set(int(c) for c in np.where(pos_rates < thresh)[0])
                self._augment.set_rare_classes(rare)
                print(
                    f'[AudioSetStrongDataModule] Class-balance: {len(rare)} rare classes '
                    f'(pos_rate < {thresh})'
                )

            self._pos_weights = pw

        if stage in ('test', None):
            self._test_ds = AudioSetStrongDataset(
                split               = 'test',
                audio_dir           = self.audio_root_test,
                strong_tsv          = self.strong_tsv_test,
                class_map_csv       = self.class_map,
                window_s            = self.window_s,
                hop_s               = self.hop_s,
                n_cls               = self.n_cls,
                overlap_threshold   = self.overlap_threshold,
                augment             = None,
                # val_fraction / val_seed not used for test split
            )

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _make_loader(
        self,
        ds: AudioSetStrongDataset,
        shuffle: bool,
        drop_last: bool = False,
        use_sampler: bool = False,
    ) -> DataLoader:
        sampler = None
        if use_sampler and self.use_weighted_sampler:
            weights = ds.item_weights()
            sampler = WeightedRandomSampler(
                weights,
                num_samples = len(weights),
                replacement = True,
            )
            shuffle = False   # sampler and shuffle are mutually exclusive

        return DataLoader(
            ds,
            batch_size         = self.batch_size,
            shuffle            = shuffle,
            sampler            = sampler,
            drop_last          = drop_last,
            num_workers        = self.num_workers,
            pin_memory         = False,
            persistent_workers = self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        # drop_last=True avoids B=1 batches that crash BatchNorm
        return self._make_loader(self._train_ds, shuffle=True, drop_last=True, use_sampler=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self._val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self._test_ds, shuffle=False)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def pos_weights(self) -> Optional[torch.Tensor]:
        """
        Per-class positive weights (capped at ``pos_weight_max``).

        Available after ``setup('fit')`` is called.  ``None`` before setup.
        """
        return self._pos_weights

    @property
    def train_dataset(self) -> Optional[AudioSetStrongDataset]:
        return self._train_ds

    @property
    def val_dataset(self) -> Optional[AudioSetStrongDataset]:
        return self._val_ds

    @property
    def test_dataset(self) -> Optional[AudioSetStrongDataset]:
        return self._test_ds
