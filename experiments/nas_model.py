"""
NASModel — Spatial × Semantic SELD Pipeline
============================================
Configurable end-to-end model that combines:

  * A spatial front-end (SpectralProcessor / GaborFrontEnd / MaxCorrFilterBank)
  * A semantic backbone (SplitEPANNs — E-PANNs with per-group freeze/init)
  * Two optional MSCA-enhanced CrossStitchBridges (early + late)
  * Spatial feature processing stacks (homologous NAS modules)
  * A spatial track smoother (BiGRU or MHA-MGU) → Multi-ACCDOA head
  * A semantic head using pretrained EPANNs fc1+fc_audioset (EPANNs Splitted)

Data flow
---------
.. code-block::

    x: (B, n_mics, T_samples)
        ↓  resample to cfg.input.sr
    x: (B, n_mics, T_sr)
    x_mono = x.mean(dim=1)  →  (B, T_sr)   # mono for E-PANNs

    SplitEPANNs.forward_early(x_mono) → h_early  (B, 256,  T/8,  F/8)
    InputProcessor(x)                 → S         (B, C_in, T_mel, F_in)
    SpatialEarlyModules(S)            → S_early   (B, C_e,  T_mel, F_e)

    CrossStitchBridge #1 [if enabled]:
        S_early, h_early = bridge_early(S_early, h_early)

    SplitEPANNs.forward_late(h_early) → h_late   (B, 1024, T/32, F/32)
    SpatialLateModules(S_early)        → S_late   (B, C_l,  T_mel, F_l)

    CrossStitchBridge #2 [if enabled]:
        S_late, h_late = bridge_late(S_late, h_late)

    Spatial smoother: mean(F) → (B, T, C_l) → smoother → (B, T, D_s)
        → FC head → reshape → (B, T, N_tracks, C_classes, 3)

    Semantic (EPANNs Splitted):
        h_late mean(F) → (B, T_sem, 1024)
        → sem_fc1  [pretrained fc1]   → F.relu_ → (B, T_sem, 2048)
        → sem_fc_audioset [pretrained fc_audioset] → (B, T_sem, 527)
        → head_semantic [trainable Linear(527→N_cls), prior-init] → (B, T_sem, N_cls)

Stacking rules
--------------
Only homologous stacking within a stage (no type mixing):

  * early features  : ConvBlock ×1  or  ResNetBlock ×1
  * late features   : FormantEnhancement ×1  or  TrackTransformer ×1
  * track smoother  : BiGRU ×1  or  MHA-MGU ×1

EPANNs Splitted
---------------
When ``semantic.dataset`` is present in the config (e.g. ``STARSS23`` or
``TAU-NIGENS2021``), the semantic branch uses pretrained fc1 and fc_audioset
weights (527-class AudioSet) followed by a trainable Linear(527→N_cls) with
soft-prior init loaded from ``configs/labels_map.json``.
Legacy ``semantic.target_labels`` is still accepted (n_cls=len(list)) but
provides no structured prior init.
There is no separate temporal smoother for the semantic branch.

References
----------
S. Adavanne et al., "Sound Event Localization and Detection of Overlapping
Sources Using Convolutional Recurrent Neural Networks," IEEE JSTSP 2019.

Y. Cao et al., "An Improved Event-Independent Network for Polyphonic SELD,"
DCASE Workshop 2021.

A. Singh Boparai et al., "Efficient Pre-trained Audio Neural Networks via
Passive Filter Pruning," DCASE Workshop 2024.

K. Shimada et al., "Multi-ACCDOA: Localizing and Detecting Overlapping
Sounds," ICASSP 2022.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF

from semantic_backbone.epanns import SplitEPANNs
from spatial_modules import (
    CrossStitchBridge,
    SpectralProcessor, ConvBlock, BiGRU,
    ResNetBlock,
    TrackTransformer,
    MHA_MGU,
    GaborFrontEnd, MaxCorrFilterBank, FormantEnhancementModule,
)


# ══════════════════════════════════════════════════════════════════════════════
# AUDIOSET → DATASET LABEL MAPPING
# ══════════════════════════════════════════════════════════════════════════════

_LABELS_MAP_PATH = Path(__file__).resolve().parent.parent / 'configs' / 'labels_map.json'


def _load_label_map(dataset: str) -> List[List[int]]:
    """
    Load AudioSet index groups for a named dataset from configs/labels_map.json.

    Returns a list of length n_cls, where each entry is the list of AudioSet
    indices corresponding to that class (used for soft-prior init of head_semantic).
    """
    with open(_LABELS_MAP_PATH) as f:
        data = json.load(f)
    if dataset not in data:
        raise ValueError(
            f"[NASModel] dataset '{dataset}' not found in {_LABELS_MAP_PATH}. "
            f"Available: {list(data.keys())}"
        )
    return [cls['audioset_indices'] for cls in data[dataset]['classes']]


# ══════════════════════════════════════════════════════════════════════════════
# BUILDER HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _channel_schedule(n_layers: int, out_ch: int) -> List[int]:
    """
    Return per-layer output channel counts for spatial_early bottleneck stacking.

    Policy:
      n=1 → [out_ch]
      n=2 → [2*out_ch, out_ch]
      n≥3 → [3*out_ch, 2*out_ch, out_ch, ..., out_ch]  (last n-2 layers at out_ch)
    The final entry is always out_ch, so bridge_early always receives out_ch channels.
    """
    if n_layers == 1:
        return [out_ch]
    elif n_layers == 2:
        return [2 * out_ch, out_ch]
    else:
        return [3 * out_ch, 2 * out_ch] + [out_ch] * (n_layers - 2)


def _build_spatial_early(cfg: dict, in_channels: int) -> Tuple[nn.ModuleList, int]:
    """
    Build a bottleneck stack of spatial early-feature modules.

    Channel schedule (n=layers, ref=out_channels):
      n=1: ref
      n=2: 2×ref → ref
      n≥3: 3×ref → 2×ref → ref … ref

    Returns:
        modules     : nn.ModuleList of n_layers modules
        out_channels: channel count of the final layer (= out_ch reference)
    """
    typ      = cfg.get('type', 'convblock')
    n_layers = cfg.get('n_layers', 1)
    out_ch   = cfg.get('out_channels', 64)

    schedule = _channel_schedule(n_layers, out_ch)
    modules  = nn.ModuleList()
    c_in     = in_channels

    for ch_out in schedule:
        if typ == 'convblock':
            m = ConvBlock(
                in_channels  = c_in,
                out_channels = ch_out,
                kernel_size  = tuple(cfg.get('kernel_size', [3, 3])),
                pool_size    = cfg.get('pool_size', 1),
                dropout      = cfg.get('dropout', 0.0),
            )
        elif typ == 'resnet_block':
            m = ResNetBlock(
                in_channels    = c_in,
                out_channels   = ch_out,
                stride         = tuple(cfg.get('stride', [1, 1])),
                use_msca       = cfg.get('use_msca', False),
                msca_reduction = cfg.get('msca_reduction', 4),
            )
        else:
            raise ValueError(f"[NASModel] Unknown spatial_early type: '{typ}'")
        modules.append(m)
        c_in = ch_out

    return modules, out_ch  # out_ch = schedule[-1], always the reference width


def _build_spatial_late(
    cfg: dict,
    in_channels: int,
    in_freq: Optional[int],
) -> Tuple[nn.ModuleList, int, bool]:
    """
    Build the spatial late-feature module stack.

    FormantEnhancement: supports n_layers ≥ 1 (stacked, shape-preserving).
    TrackTransformer:   n_layers maps to ``num_layers`` inside the single
                        TransformerEncoder — deeper internal stacking without
                        adding outer module instances.

    For TrackTransformer the input is mean-pooled over frequency then reshaped
    to (B, T, C) before the module; ``in_freq`` is unused (freq axis averaged).

    Returns:
        modules       : nn.ModuleList (1 to n_layers modules for FE; 1 for TT)
        out_channels  : output channel/d_model dim
        is_transformer: True if TrackTransformer (different I/O shape)
    """
    typ      = cfg.get('type', 'formant_enhancement')
    n_layers = cfg.get('n_layers', 1)
    modules  = nn.ModuleList()

    if typ == 'formant_enhancement':
        # Stack n_layers shape-preserving FormantEnhancement modules.
        # Each module sees in_channels input and outputs in_channels (shape-preserving).
        for _ in range(n_layers):
            modules.append(FormantEnhancementModule(
                n_channels  = in_channels,
                kernel_size = tuple(cfg.get('kernel_size', [3, 1])),
            ))
        return modules, in_channels, False

    elif typ == 'track_transformer':
        d_model  = cfg.get('d_model', 128)
        n_heads  = cfg.get('n_heads', 4)
        n_tracks = cfg.get('n_tracks', 5)
        # n_layers controls internal TransformerEncoder depth (not outer stacking).
        modules.append(TrackTransformer(
            d_model   = d_model,
            n_heads   = n_heads,
            n_tracks  = n_tracks,
            num_layers= n_layers,
            d_input   = in_channels,   # freq axis already averaged before forward
        ))
        return modules, n_tracks * d_model, True

    else:
        raise ValueError(f"[NASModel] Unknown spatial_late type: '{typ}'")


def _build_track_smoother(
    cfg: dict,
    in_dim: int,
    in_tracks: Optional[int],
) -> Tuple[nn.Module, int]:
    """
    Build a spatial track smoother.

    Args:
        cfg       : smoother section config dict
        in_dim    : input feature dimension (D or d_model)
        in_tracks : number of input track slots (for MHA_MGU's M_X);
                    None → 1 (treat whole input as single detection set)

    Returns:
        module     : the smoother module
        out_dim    : output feature dimension D
    """
    typ      = cfg.get('type', 'bigru')
    n_layers = cfg.get('n_layers', 1)

    if typ == 'bigru':
        hidden = cfg.get('hidden_size', 128)
        return (
            BiGRU(
                in_size     = in_dim,
                hidden_size = hidden,
                num_layers  = n_layers,
                dropout     = cfg.get('dropout', 0.0),
            ),
            hidden,
        )

    elif typ == 'mhamgu':
        d_model  = cfg.get('d_model', 128)
        n_heads  = cfg.get('n_heads', 4)
        n_tracks = cfg.get('n_tracks', 5)
        mods = nn.ModuleList()
        for i in range(n_layers):
            mods.append(MHA_MGU(
                d_model  = d_model,
                n_heads  = n_heads,
                n_tracks = n_tracks,
                d_input  = in_dim if i == 0 else d_model,
                dropout  = cfg.get('dropout', 0.0),
            ))
        return mods, d_model

    else:
        raise ValueError(f"[NASModel] Unknown track_smoother type: '{typ}'")


# ══════════════════════════════════════════════════════════════════════════════
# NASModel
# ══════════════════════════════════════════════════════════════════════════════

class NASModel(nn.Module):
    """
    Configurable spatial × semantic SELD pipeline.

    Args:
        cfg (dict): full parsed YAML config (nas_base.yaml).

    Forward:
        x : (B, n_mics, T_samples) — raw multichannel waveform at *any* SR.
            NASModel resamples to cfg['input']['sr'] at entry.

    Returns:
        out_spatial  : (B, T_spatial, N_tracks, C_classes, 3)  Multi-ACCDOA
        out_semantic : (B, T_semantic, C_classes)               class-presence logits (BCE)
    """

    def __init__(self, cfg: dict):
        """Initialize NASModel."""
        super().__init__()

        in_cfg  = cfg['input']
        sem_cfg = cfg['semantic']
        cs_cfg  = cfg.get('cross_stitch', {})
        se_cfg  = cfg.get('spatial_early', {})
        sl_cfg  = cfg.get('spatial_late', {})
        tss_cfg = cfg.get('track_smoothing_spatial', {})
        head_cfg= cfg.get('heads', {})

        self.src_sr  = in_cfg['sr']
        self.n_mics  = in_cfg.get('n_mics', 4)
        # Granular CrossStitch flags: cs_early_enabled / cs_late_enabled take priority;
        # fallback to legacy 'enabled' key for backward compatibility.
        _cs_default     = cs_cfg.get('enabled', True)
        self.cs_early_on = cs_cfg.get('cs_early_enabled', _cs_default)
        self.cs_late_on  = cs_cfg.get('cs_late_enabled',  _cs_default)
        c_bridge_early  = cs_cfg.get('C_bridge_early', 64)
        c_bridge_late   = cs_cfg.get('C_bridge_late',  128)
        self.n_tracks= head_cfg.get('n_tracks', 5)

        # n_cls + label_groups: prefer semantic.dataset (structured JSON mapping),
        # fallback to legacy target_labels (list of strings, len only), then default 13.
        dataset_name = sem_cfg.get('dataset', None)
        if dataset_name:
            label_groups  = _load_label_map(dataset_name)
            self.n_cls    = len(label_groups)
        else:
            target_labels = sem_cfg.get('target_labels', None)
            self.n_cls    = len(target_labels) if target_labels else head_cfg.get('n_classes', 13)
            label_groups  = None   # no structured prior; head_semantic weights stay zero

        # ── Input processor ──────────────────────────────────────────────
        itype = in_cfg.get('type', 'spectral')
        if itype == 'spectral':
            self.input_proc = SpectralProcessor(
                n_fft      = in_cfg.get('n_fft', 512),
                hop_length = in_cfg.get('hop_length', 320),
            )
            c_in_spatial = 2 * self.n_mics   # magnitude + phase per channel
        elif itype == 'gabor':
            self.input_proc = GaborFrontEnd(
                n_filters     = in_cfg.get('n_filters', 128),
                kernel_length = in_cfg.get('kernel_length', 1024),
                stride        = in_cfg.get('stride', 320),
                n_mics        = self.n_mics,
                format        = in_cfg.get('format', 'foa'),
                sr            = self.src_sr,
                fmin          = in_cfg.get('fmin', 50.0),
                fmax          = in_cfg.get('fmax', 16000.0),
            )
            fmt = in_cfg.get('format', 'foa')
            c_in_spatial = (1 + 2 * (self.n_mics - 1)) if fmt == 'foa' else self.n_mics
        elif itype == 'maxcorr':
            self.input_proc = MaxCorrFilterBank(
                n_filters     = in_cfg.get('n_filters', 64),
                kernel_length = in_cfg.get('kernel_length', 481),
                stride        = in_cfg.get('stride', 320),
                n_channels    = self.n_mics,
            )
            c_in_spatial = in_cfg.get('n_filters', 64)
        else:
            raise ValueError(f"[NASModel] Unknown input type: '{itype}'")

        self.itype = itype

        # ── Semantic backbone ─────────────────────────────────────────────
        # SplitEPANNs must load fc_audioset at its native AudioSet size (527),
        # regardless of the STARSS23 n_cls (13).  We override n_cls here so
        # _load_checkpoint can copy the full pretrained fc_audioset weights.
        sem_cfg_backbone = {**sem_cfg, 'n_cls': 527}
        self.split_epanns = SplitEPANNs(sem_cfg_backbone)

        # Channel dims from E-PANNs (fixed by architecture + 50% pruning)
        c_early_sem = 256
        c_late_sem  = 1024

        # ── Spatial early feature modules ─────────────────────────────────
        se_mods, c_early_spat = _build_spatial_early(se_cfg, c_in_spatial)
        self.spatial_early = se_mods

        # ── CrossStitchBridge #1 ──────────────────────────────────────────
        if self.cs_early_on:
            self.bridge_early = CrossStitchBridge(c_early_spat, c_early_sem, c_bridge_early)

        # ── Spatial late feature modules ──────────────────────────────────
        # F dimension after spatial early: unknown at build time (depends on
        # input processor and conv pooling). We pass None and rely on forward
        # for TrackTransformer; for FormantEnhancement F is irrelevant.
        sl_mods, c_late_spat, self._sl_is_transformer = _build_spatial_late(
            sl_cfg, c_early_spat, None
        )
        self.spatial_late = sl_mods

        # If TrackTransformer: record n_tracks for shape handling
        self._sl_n_tracks = sl_cfg.get('n_tracks', 5) if self._sl_is_transformer else None
        self._sl_d_model  = sl_cfg.get('d_model', 128) if self._sl_is_transformer else None

        # ── CrossStitchBridge #2 ──────────────────────────────────────────
        if self.cs_late_on:
            self.bridge_late = CrossStitchBridge(c_late_spat, c_late_sem, c_bridge_late)

        # ── Spatial track smoother ────────────────────────────────────────
        # Spatial smoother input dim
        if self._sl_is_transformer:
            # TrackTransformer outputs (B, T, n_tracks, d_model)
            # → MHA_MGU: pass as (B, T, n_tracks, d_model)
            # → BiGRU:   flatten to (B, T, n_tracks * d_model)
            tss_in_dim    = (self._sl_n_tracks or 5) * (self._sl_d_model or 128)
            tss_in_tracks = self._sl_n_tracks
        else:
            # FormantEnhancement outputs (B, C, T, F) → mean(F) → (B, T, C)
            tss_in_dim    = c_late_spat
            tss_in_tracks = 1

        self._tss_type = tss_cfg.get('type', 'bigru')

        # When mhamgu_expand will be used (mhamgu + FE path), the expand maps
        # tss_in_dim → n_tracks * d_model then reshapes to (B, T, n_tracks, d_model).
        # Layer-0 input_proj must therefore be Linear(d_model, d_model), not Linear(tss_in_dim, d_model).
        _mhamgu_expand_active = (self._tss_type == 'mhamgu' and not self._sl_is_transformer)
        effective_tss_in_dim  = tss_cfg.get('d_model', 128) if _mhamgu_expand_active else tss_in_dim

        self.smoother_spatial, d_spat_out = _build_track_smoother(
            tss_cfg, effective_tss_in_dim, tss_in_tracks
        )

        # ── MHA-MGU expand projection (FormantEnhancement → n_tracks detections) ──
        # When spatial_late is FormantEnhancement, _smoother_forward would receive
        # (B, T, C) and unsqueeze to M_X=1 — all n_tracks queries attend to the same
        # single detection, making assignment degenerate.  This projection generates
        # n_tracks distinct detection vectors so each query has a unique input.
        self.mhamgu_expand = None
        if self._tss_type == 'mhamgu' and not self._sl_is_transformer:
            n_t = tss_cfg.get('n_tracks', 5)
            d_m = tss_cfg.get('d_model', 128)
            self.mhamgu_expand = nn.Linear(tss_in_dim, n_t * d_m)
            nn.init.normal_(self.mhamgu_expand.weight, std=0.01)
            nn.init.zeros_(self.mhamgu_expand.bias)

        # ── Spatial output head (Multi-ACCDOA) ────────────────────────────
        spat_out_dim = self.n_tracks * self.n_cls * 3

        if self._tss_type == 'mhamgu':
            n_t   = tss_cfg.get('n_tracks', 5)
            d_m   = tss_cfg.get('d_model', 128)
            d_spat_out_flat = n_t * d_m
        else:
            d_spat_out_flat = d_spat_out

        # DOA regression head — outputs raw Cartesian vectors (unnormalised)
        self.head_spatial = nn.Linear(d_spat_out_flat, spat_out_dim)
        nn.init.xavier_uniform_(self.head_spatial.weight)
        nn.init.zeros_(self.head_spatial.bias)

        # Activity (SED) head — outputs per-track per-class logits for BCE
        act_out_dim = self.n_tracks * self.n_cls
        self.head_activity = nn.Linear(d_spat_out_flat, act_out_dim)
        nn.init.xavier_uniform_(self.head_activity.weight)
        nn.init.zeros_(self.head_activity.bias)

        # ── Semantic head (EPANNs Splitted) ───────────────────────────────
        # fc1 and fc_audioset: fine-tunable copies of pretrained EPANNs weights.
        # head_semantic: trainable Linear(527→n_cls) with soft-prior init.
        fc1_ref, fc_as_ref = self.split_epanns.get_head_modules()

        self.sem_fc1 = nn.Linear(c_late_sem, 2048, bias=True)
        with torch.no_grad():
            self.sem_fc1.weight.copy_(fc1_ref.weight)
            self.sem_fc1.bias.copy_(fc1_ref.bias)

        self.sem_fc_audioset = nn.Linear(2048, 527, bias=True)
        with torch.no_grad():
            self.sem_fc_audioset.weight.copy_(fc_as_ref.weight)
            self.sem_fc_audioset.bias.copy_(fc_as_ref.bias)

        # Trainable projection 527 → n_cls with structured soft-prior init
        self.head_semantic = nn.Linear(527, self.n_cls, bias=True)
        with torch.no_grad():
            self.head_semantic.weight.zero_()
            self.head_semantic.bias.zero_()
            if label_groups is not None:
                for out_idx, grp in enumerate(label_groups):
                    for as_idx in grp:
                        self.head_semantic.weight[out_idx, as_idx] = 1.0 / len(grp)

        # Log config summary
        print(f"[NASModel] input={itype}  sr={self.src_sr}  n_mics={self.n_mics}")
        print(f"[NASModel] cross_stitch_early={'on' if self.cs_early_on else 'off'}"
              f"  C_bridge_early={c_bridge_early if self.cs_early_on else '-'}"
              f"  cross_stitch_late={'on' if self.cs_late_on else 'off'}"
              f"  C_bridge_late={c_bridge_late if self.cs_late_on else '-'}")
        print(f"[NASModel] spatial_early={se_cfg.get('type','convblock')}×{se_cfg.get('n_layers',1)}"
              f"  spatial_late={sl_cfg.get('type','formant_enhancement')}×{sl_cfg.get('n_layers',1)}")
        print(f"[NASModel] smoother_spatial={tss_cfg.get('type','bigru')}×{tss_cfg.get('n_layers',1)}")
        print(f"[NASModel] semantic=EPANNs-Splitted "
              f"(fc1→fc_audioset pretrained + head_semantic trainable 527→{self.n_cls})")
        print(f"[NASModel] spatial head : (B, T, {self.n_tracks}, {self.n_cls}, 3) Multi-ACCDOA")
        print(f"[NASModel] semantic head: (B, T_sem, {self.n_cls}) class-presence logits (BCE)")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_input_processor(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n_mics, T_samples)
        → S: (B, C_in, T_mel, F_in)
        """
        if self.itype == 'spectral':
            return self.input_proc(x)                    # (B, 2*n_mics, T, F)
        elif self.itype == 'gabor':
            return self.input_proc(x)                    # (B, C_spat, T, n_filters)
        else:  # maxcorr
            return self.input_proc(x).permute(0, 1, 2, 3)  # already (B, n_filters, T, n_mics)

    def _run_spatial_late(
        self,
        x: torch.Tensor,
        c_in: int,
    ) -> torch.Tensor:
        """
        Run the spatial late modules.

        FormantEnhancement: (B, C, T, F) → (B, C, T, F) [shape-preserving]
        TrackTransformer:   (B, C, T, F) → reshape to (B, T, C*F)
                            → each layer (B, T, D) → (B, T, n_tracks, d_model)
                            → flatten to (B, T, n_tracks * d_model)
        """
        if not self._sl_is_transformer:
            for mod in self.spatial_late:
                x = mod(x)
            return x                                 # (B, C, T, F)

        # TrackTransformer path: mean-pool frequency axis, then reshape to sequence.
        # input_proj was built with d_input=C (in_freq=None → freq-mean assumed).
        B, C, T, Fq = x.shape
        seq = x.mean(dim=3).permute(0, 2, 1)               # (B, T, C)

        for mod in self.spatial_late:
            out = mod(seq)                           # (B, T, n_tracks, d_model)
            B2, T2, M, D = out.shape
            seq = out.reshape(B2, T2, M * D)         # flatten for next layer / head
        return seq                                   # (B, T, n_tracks * d_model)

    def _smoother_forward(
        self,
        smoother: nn.Module,
        smoother_type: str,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run a track smoother and return a flat (B, T, D) tensor.

        BiGRU   : single module, input (B, T, D) → output (B, T, H)
        MHA_MGU : ModuleList of n layers chained directly.
                  Layer 0 input : (B, T, 1, D_in)   — unsqueeze from (B,T,D)
                  Layer i>0     : (B, T, n_tracks, d_model) — output of prev layer
                  Final output  : (B, T, n_tracks * d_model) — flattened
        """
        if smoother_type == 'bigru':
            return smoother(x)                        # (B, T, H)

        # MHA_MGU stack (ModuleList)
        if x.dim() == 3:
            if self.mhamgu_expand is not None:
                # Generate n_tracks distinct detection vectors from the pooled feature.
                # Each row of mhamgu_expand learns a different "detection profile" so
                # MHA-MGU can perform meaningful assignment across tracks.
                B2, T2, _ = x.shape
                n_t = self.smoother_spatial[0].n_tracks
                d_m = self.smoother_spatial[0].d_model
                x = self.mhamgu_expand(x).view(B2, T2, n_t, d_m)  # (B, T, n_tracks, d_model)
            else:
                x = x.unsqueeze(2)                    # fallback M_X=1 (TrackTransformer path)
        H = x
        for mod in smoother:
            H, _ = mod(H)                             # (B, T, n_tracks, d_model)
        B, T, M, D = H.shape
        return H.reshape(B, T, M * D)                 # (B, T, n_tracks * d_model)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        orig_sr: Optional[int] = None,
        return_components: bool = False,
    ):
        """
        Args:
            x                 : (B, n_mics, T_samples) raw multichannel waveform
            orig_sr           : source sample rate (if None, assumed == cfg.input.sr)
            return_components : if True, also return raw act_logits and doa_raw for
                                use with tPITLoss in the training step.

        Returns (return_components=False):
            out_spatial  : (B, T, N, C, 3)  Multi-ACCDOA  [norm = sigmoid(act) in (0,1)]
            out_semantic : (B, T_sem, C)    class-presence logits

        Returns (return_components=True):
            out_spatial, out_semantic, act_logits (B,T,N,C), doa_raw (B,T,N,C,3)
        """
        B = x.shape[0]

        # ── Resample if needed ──────────────────────────────────────────
        if orig_sr is not None and orig_sr != self.src_sr:
            x = AF.resample(x, orig_sr, self.src_sr)

        # ── Mono for E-PANNs ────────────────────────────────────────────
        x_mono = x.mean(dim=1)                       # (B, T_sr)

        # ── Semantic: early stage ────────────────────────────────────────
        h_early = self.split_epanns.forward_early(x_mono)    # (B,256,T/8,F/8)

        # ── Spatial: input processing ────────────────────────────────────
        S = self._run_input_processor(x)             # (B, C_in, T_mel, F_in)

        # ── Spatial: early modules ───────────────────────────────────────
        for mod in self.spatial_early:
            S = mod(S)                               # (B, C_early, T_mel, F_e)

        # ── CrossStitchBridge #1 ─────────────────────────────────────────
        if self.cs_early_on:
            S, h_early = self.bridge_early(S, h_early)

        # ── Semantic: late stage (from enriched h_early) ─────────────────
        h_late = self.split_epanns.forward_late(h_early)     # (B,1024,T/32,F/32)

        # ── Spatial: late modules ─────────────────────────────────────────
        S_late = self._run_spatial_late(S, S.shape[1])

        # ── CrossStitchBridge #2 ─────────────────────────────────────────
        if self.cs_late_on:
            if self._sl_is_transformer:
                # S_late is (B, T, D) — bridge expects (B, C, T, F)
                S_late_4d = S_late.permute(0, 2, 1).unsqueeze(3)  # (B, D, T, 1)
                S_late_4d, h_late = self.bridge_late(S_late_4d, h_late)
                S_late = S_late_4d.squeeze(3).permute(0, 2, 1)    # (B, T, D)
            else:
                S_late, h_late = self.bridge_late(S_late, h_late)

        # ── Track smoother — spatial ─────────────────────────────────────
        if self._sl_is_transformer:
            seq_spat = S_late                                # (B, T_mel, n_tracks*d_model)
        else:
            if S_late.dim() == 4:
                seq_spat = S_late.mean(dim=3).permute(0, 2, 1)
            else:
                seq_spat = S_late

        spat_out = self._smoother_forward(
            self.smoother_spatial, self._tss_type, seq_spat
        )                                                    # (B, T_mel, D_s)

        # ── Semantic head (EPANNs Splitted) ──────────────────────────────
        # h_late: (B, 1024, T_sem, F_sem) post-CrossStitch#2
        if h_late.dim() == 4:
            seq_sem = h_late.mean(dim=3).permute(0, 2, 1)   # (B, T_sem, 1024)
        else:
            seq_sem = h_late
        fc1_out      = F.relu_(self.sem_fc1(seq_sem))        # (B, T_sem, 2048)
        as_logits    = self.sem_fc_audioset(fc1_out)          # (B, T_sem, 527)
        out_semantic = self.head_semantic(as_logits)          # (B, T_sem, n_cls)

        # ── Spatial output heads ──────────────────────────────────────────
        N, C = self.n_tracks, self.n_cls

        # Raw activity logits (B, T, N*C) → (B, T, N, C)
        act_logits = self.head_activity(spat_out).view(B, -1, N, C)

        # Raw DOA regression (B, T, N*C*3) → (B, T, N, C, 3)
        doa_raw    = self.head_spatial(spat_out).view(B, -1, N, C, 3)

        # Combine into Multi-ACCDOA: norm = sigmoid(act) ∈ (0,1)
        # SELDMetrics act_thresh=0.5 ↔ sigmoid(logit) > 0.5 ↔ logit > 0
        act_prob   = torch.sigmoid(act_logits)                # (B, T, N, C)
        doa_unit   = F.normalize(doa_raw, dim=-1)             # (B, T, N, C, 3)
        out_spatial = act_prob.unsqueeze(-1) * doa_unit       # (B, T, N, C, 3)

        if return_components:
            return out_spatial, out_semantic, act_logits, doa_raw
        return out_spatial, out_semantic
