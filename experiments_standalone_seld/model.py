"""
StandaloneSELDModel — Dual-Branch SELD without GP-AT Backbone
==============================================================
Standalone SELD model derived from the best NAS-SELD architecture
(3× ResNetBlock early, 1× TrackTransformer late, 1× BiGRU smoothing).

Two architecturally symmetric branches process the same spectrogram input:

  * ``sed_branch`` — produces activity logits (B, T, N_tracks, N_classes)
  * ``doa_branch`` — produces tanh-bounded DOA vectors (B, T, N_tracks, N_classes, 3)

Each branch has its own InstanceNorm layers, ResNet/TT/GRU weights, and head.
The ``SpectralProcessor`` is shared (parameter-free STFT).

Optional cross-stitching (configurable) between the two branches at:
  * post-early  — 4D tensors (B, 64, T, F),  CrossStitching(n_channels=64)
  * post-late   — 3D tensors (B, T, 256),     CrossStitching(n_channels=256)
                  (requires .transpose(1,2) wrapper since C is at dim 2)

Data flow
---------
.. code-block::

    x: (B, n_mics, T_samples) → resample → SpectralProcessor → S: (B, 11, T, F)

    SED branch:   S → norm_mag/phase/iv → early (3× ResNet) → [cs_early] →
                  late (TT, freq mean-pool) → [cs_late] → smoother (BiGRU) →
                  head_sed → act_logits: (B, T, N, C)

    DOA branch:   S → norm_mag/phase/iv → early (3× ResNet) → [cs_early] →
                  late (TT, freq mean-pool) → [cs_late] → smoother (BiGRU) →
                  head_doa → tanh → doa_tanh: (B, T, N, C, 3)

Loss: single tPITLoss — act_logits (from SED branch) and doa_tanh (from DOA
branch) passed together; gradient flows to the respective branch only.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torchaudio.functional as AF

from spatial_modules import (
    SpectralProcessor,
    BiGRU,
    ResNetBlock,
    ConvBlock,
    TrackTransformer,
    CrossStitching,
)


# ══════════════════════════════════════════════════════════════════════════════
# BUILDER HELPERS  (adapted from experiments_nas_seld/nas_model.py)
# ══════════════════════════════════════════════════════════════════════════════

def _channel_schedule(n_layers: int, out_ch: int) -> List[int]:
    """Progressive halving schedule for early-stage channel counts.

    n=1 → [out_ch]
    n=2 → [2*out_ch, out_ch]
    n=3 → [4*out_ch, 2*out_ch, out_ch]
    """
    return [out_ch * (2 ** (n_layers - 1 - i)) for i in range(n_layers)]


def _build_spatial_early(cfg: dict, in_channels: int) -> Tuple[nn.ModuleList, int]:
    """Build a bottleneck stack of spatial early-feature modules."""
    typ      = cfg.get('type', 'resnet_block')
    n_layers = cfg.get('n_layers', 3)
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
                dropout        = cfg.get('dropout', 0.0),
            )
        else:
            raise ValueError(f'[StandaloneSELD] Unknown spatial_early type: {typ!r}')
        modules.append(m)
        c_in = ch_out

    return modules, out_ch


def _build_spatial_late(
    cfg: dict,
    in_channels: int,
) -> Tuple[nn.ModuleList, int, bool]:
    """Build the spatial late-feature module stack.

    Returns (modules, out_channels, is_transformer).
    """
    typ      = cfg.get('type', 'track_transformer')
    n_layers = cfg.get('n_layers', 1)
    modules  = nn.ModuleList()

    if typ == 'track_transformer':
        d_model  = cfg.get('d_model', 128)
        n_heads  = cfg.get('n_heads', 4)
        n_tracks = cfg.get('n_tracks', 2)
        modules.append(TrackTransformer(
            d_model    = d_model,
            n_heads    = n_heads,
            n_tracks   = n_tracks,
            num_layers = n_layers,
            d_input    = in_channels,
        ))
        return modules, n_tracks * d_model, True
    else:
        raise ValueError(f'[StandaloneSELD] Unsupported spatial_late type: {typ!r}. '
                         'Only track_transformer is supported in StandaloneSELDModel.')


def _build_track_smoother(cfg: dict, in_dim: int) -> Tuple[nn.Module, int]:
    """Build a BiGRU track smoother."""
    typ = cfg.get('type', 'bigru')
    if typ != 'bigru':
        raise ValueError(f'[StandaloneSELD] Unsupported track_smoothing type: {typ!r}. '
                         'Only bigru is supported in StandaloneSELDModel.')
    hidden   = cfg.get('hidden_size', 256)
    n_layers = cfg.get('n_layers', 1)
    return (
        BiGRU(
            in_size     = in_dim,
            hidden_size = hidden,
            num_layers  = n_layers,
            dropout     = cfg.get('dropout', 0.0),
        ),
        hidden,
    )


# ══════════════════════════════════════════════════════════════════════════════
# StandaloneSELDBranch
# ══════════════════════════════════════════════════════════════════════════════

class StandaloneSELDBranch(nn.Module):
    """
    One complete SELD processing branch (SED or DOA).

    Exposes four stage methods so StandaloneSELDModel can interleave
    cross-stitching between branches:
      * forward_norm(S)       → normalised spectrogram (B, C_in, T, F)
      * forward_early(x)      → (B, C_early, T, F)
      * forward_late(x)       → (B, T, D_late)
      * forward_head(x)       → (B, T, N, C) or (B, T, N, C, 3)

    forward(S) chains all four stages.
    """

    def __init__(
        self,
        in_channels:  int,
        n_mics:       int,
        audio_format: str,
        early_cfg:    dict,
        late_cfg:     dict,
        smoother_cfg: dict,
        head_type:    str,
        n_tracks:     int,
        n_classes:    int,
    ):
        super().__init__()

        self.n_mics       = n_mics
        self.audio_format = audio_format
        self.head_type    = head_type
        self.n_tracks     = n_tracks
        self.n_classes    = n_classes

        # Per-group InstanceNorm — separate for each branch
        self.norm_mag   = nn.InstanceNorm2d(n_mics, affine=True)
        self.norm_phase = nn.InstanceNorm2d(n_mics, affine=True)
        if audio_format == 'foa':
            self.norm_iv = nn.InstanceNorm2d(3, affine=True)
        else:  # mic
            n_pairs = n_mics * (n_mics - 1) // 2
            self.norm_gcc = nn.InstanceNorm2d(n_pairs, affine=True)

        # Early modules
        early_mods, c_early = _build_spatial_early(early_cfg, in_channels)
        self.early = early_mods
        self._c_early = c_early

        # Late modules
        late_mods, c_late, self._late_is_transformer = _build_spatial_late(
            late_cfg, c_early
        )
        self.late  = late_mods
        self._c_late = c_late  # = n_tracks * d_model for TrackTransformer

        # Track smoother
        smoother, d_out = _build_track_smoother(smoother_cfg, c_late)
        self.smoother = smoother
        self._d_out   = d_out

        # Output head
        if head_type == 'sed':
            self.head = nn.Linear(d_out, n_tracks * n_classes)
            nn.init.xavier_uniform_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        elif head_type == 'doa':
            self.head = nn.Linear(d_out, n_tracks * n_classes * 3)
            nn.init.xavier_uniform_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        else:
            raise ValueError(f'[StandaloneSELDBranch] Unknown head_type: {head_type!r}')

    def forward_norm(self, S: torch.Tensor) -> torch.Tensor:
        """Apply per-group InstanceNorm. S: (B, C_in, T, F) → same shape."""
        C = self.n_mics
        S_mag   = self.norm_mag(S[:, :C])
        S_phase = self.norm_phase(S[:, C:2*C])
        parts   = [S_mag, S_phase]
        offset  = 2 * C
        if self.audio_format == 'foa':
            parts.append(self.norm_iv(S[:, offset:offset + 3]))
        else:
            parts.append(self.norm_gcc(S[:, offset:]))
        return torch.cat(parts, dim=1)

    def forward_early(self, x: torch.Tensor) -> torch.Tensor:
        """Run early ResNet stack. Returns (B, C_early, T, F)."""
        for mod in self.early:
            x = mod(x)
        return x

    def forward_late(self, x: torch.Tensor) -> torch.Tensor:
        """Run late TrackTransformer with freq mean-pooling. Returns (B, T, D_late)."""
        # x: (B, C_early, T, F) → mean over F → (B, T, C_early)
        B, C, T, Fq = x.shape
        seq = x.mean(dim=3).permute(0, 2, 1)   # (B, T, C_early)
        for mod in self.late:
            out = mod(seq)                       # (B, T, n_tracks, d_model)
            B2, T2, M, D = out.shape
            seq = out.reshape(B2, T2, M * D)     # (B, T, n_tracks * d_model)
        return seq   # (B, T, D_late = n_tracks * d_model)

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        """BiGRU smoother + head. Returns (B, T, N, C) or (B, T, N, C, 3)."""
        x = self.smoother(x)    # (B, T, D_out)
        B, T, _ = x.shape
        N, C    = self.n_tracks, self.n_classes

        if self.head_type == 'sed':
            return self.head(x).view(B, T, N, C)
        else:  # doa
            return torch.tanh(self.head(x)).view(B, T, N, C, 3)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        x = self.forward_norm(S)
        x = self.forward_early(x)
        x = self.forward_late(x)
        return self.forward_head(x)


# ══════════════════════════════════════════════════════════════════════════════
# StandaloneSELDModel
# ══════════════════════════════════════════════════════════════════════════════

class StandaloneSELDModel(nn.Module):
    """
    Standalone dual-branch SELD model (no GP-AT backbone).

    Args:
        cfg (dict): merged YAML config.

    Forward:
        x       : (B, n_mics, T_samples) raw waveform at orig_sr
        orig_sr : source sample rate (resampled to cfg.input.sr if different)

    Returns:
        act_logits : (B, T, N_tracks, N_classes)       raw SED logits
        doa_tanh   : (B, T, N_tracks, N_classes, 3)    tanh-bounded DOA vectors
    """

    def __init__(self, cfg: dict):
        super().__init__()

        in_cfg      = cfg['input']
        se_cfg      = cfg.get('spatial_early', {})
        sl_cfg      = cfg.get('spatial_late', {})
        tss_cfg     = cfg.get('track_smoothing', {})
        head_cfg    = cfg.get('heads', {})
        cs_cfg      = cfg.get('cross_stitch', {})
        data_cfg    = cfg.get('data', {})

        self.src_sr       = in_cfg['sr']
        self.n_mics       = in_cfg.get('n_mics', 4)
        self.audio_format = data_cfg.get('format', 'foa')

        _iv  = (self.audio_format == 'foa')
        _gcc = (self.audio_format == 'mic')
        n_pairs     = self.n_mics * (self.n_mics - 1) // 2
        c_in_spat   = 2 * self.n_mics + (3 if _iv else n_pairs if _gcc else 0)

        n_tracks  = head_cfg.get('n_tracks', 2)
        n_classes = head_cfg.get('n_classes', 13)

        # ── Shared input processor (parameter-free STFT) ─────────────────────
        self.input_proc = SpectralProcessor(
            n_fft       = in_cfg.get('n_fft', 512),
            hop_length  = in_cfg.get('hop_length', 320),
            compute_iv  = _iv,
            compute_gcc = _gcc,
            tau_max     = in_cfg.get('n_fft', 512) // 4,
        )

        # ── Cross-stitch flags ────────────────────────────────────────────────
        self.cs_early_on = cs_cfg.get('cs_early_enabled', False)
        self.cs_late_on  = cs_cfg.get('cs_late_enabled', False)

        # ── Two branches ──────────────────────────────────────────────────────
        branch_kwargs = dict(
            in_channels  = c_in_spat,
            n_mics       = self.n_mics,
            audio_format = self.audio_format,
            early_cfg    = se_cfg,
            late_cfg     = sl_cfg,
            smoother_cfg = tss_cfg,
            n_tracks     = n_tracks,
            n_classes    = n_classes,
        )
        self.sed_branch = StandaloneSELDBranch(head_type='sed', **branch_kwargs)
        self.doa_branch = StandaloneSELDBranch(head_type='doa', **branch_kwargs)

        # ── Optional CrossStitching layers ───────────────────────────────────
        # post-early: 4D tensors (B, C_early, T, F) — C at dim 1, native
        # post-late:  3D tensors (B, T, D_late)     — requires .transpose(1,2)
        c_early = self.sed_branch._c_early
        d_late  = self.sed_branch._c_late

        if self.cs_early_on:
            self.cs_early = CrossStitching(n_channels=c_early)
        if self.cs_late_on:
            self.cs_late = CrossStitching(n_channels=d_late)

        print(f'[StandaloneSELDModel] format={self.audio_format}  sr={self.src_sr}  '
              f'n_mics={self.n_mics}  c_in={c_in_spat}')
        print(f'[StandaloneSELDModel] early={se_cfg.get("type","resnet_block")}×'
              f'{se_cfg.get("n_layers",3)}  out_ch={c_early}  '
              f'late={sl_cfg.get("type","track_transformer")}×{sl_cfg.get("n_layers",1)}  '
              f'd_late={d_late}')
        print(f'[StandaloneSELDModel] smoother={tss_cfg.get("type","bigru")}×'
              f'{tss_cfg.get("n_layers",1)}  hidden={tss_cfg.get("hidden_size",256)}')
        print(f'[StandaloneSELDModel] cs_early={"on" if self.cs_early_on else "off"}  '
              f'cs_late={"on" if self.cs_late_on else "off"}')
        print(f'[StandaloneSELDModel] n_tracks={n_tracks}  n_classes={n_classes}')

    def forward(
        self,
        x: torch.Tensor,
        orig_sr: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x       : (B, n_mics, T_samples) raw waveform
            orig_sr : source sample rate; resampled to cfg.input.sr if different

        Returns:
            act_logits : (B, T, N, C)
            doa_tanh   : (B, T, N, C, 3)
        """
        if orig_sr is not None and orig_sr != self.src_sr:
            x = AF.resample(x, orig_sr, self.src_sr)

        # Shared spectral input (parameter-free, same tensor for both branches)
        S = self.input_proc(x)   # (B, C_in, T, F)

        # Stage 1: per-branch InstanceNorm
        S_sed = self.sed_branch.forward_norm(S)
        S_doa = self.doa_branch.forward_norm(S)

        # Stage 2: early feature extraction
        feat_sed_e = self.sed_branch.forward_early(S_sed)   # (B, C_early, T, F)
        feat_doa_e = self.doa_branch.forward_early(S_doa)   # (B, C_early, T, F)

        # Optional cross-stitch post-early (4D, C at dim 1 — native)
        if self.cs_early_on:
            feat_sed_e, feat_doa_e = self.cs_early(feat_sed_e, feat_doa_e)

        # Stage 3: late feature extraction (freq pool + TrackTransformer)
        feat_sed_l = self.sed_branch.forward_late(feat_sed_e)   # (B, T, D_late)
        feat_doa_l = self.doa_branch.forward_late(feat_doa_e)   # (B, T, D_late)

        # Optional cross-stitch post-late (3D: transpose to put C at dim 1)
        if self.cs_late_on:
            sed_t, doa_t = self.cs_late(
                feat_sed_l.transpose(1, 2),   # (B, D_late, T)
                feat_doa_l.transpose(1, 2),
            )
            feat_sed_l = sed_t.transpose(1, 2)   # (B, T, D_late)
            feat_doa_l = doa_t.transpose(1, 2)

        # Stage 4: smoother + head
        act_logits = self.sed_branch.forward_head(feat_sed_l)   # (B, T, N, C)
        doa_tanh   = self.doa_branch.forward_head(feat_doa_l)   # (B, T, N, C, 3)

        return act_logits, doa_tanh
