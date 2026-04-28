"""
StandaloneSELDModel v2 — Conformer + TCN Dual-Branch SELD
==========================================================
Standalone SELD model with an improved architecture:

  * Early stage  : CNN freq-reduction frontend (2× ResNetBlock + MSCA, stride=(1,2))
                   followed by a ConformerBlock stack (global self-attention +
                   local depthwise conv) → full temporal context over the clip.
  * Late stage   : TrackTransformer (per-track independent self-attention).
  * Smoother     : Temporal Convolutional Network (TCN), applied per-track with
                   shared weights — faster than BiGRU on all hardware.
  * Optional CrossStitching between SED and DOA branches at:
      - post-early  : (B, T, d_model)
      - post-late   : (B, T, n_tracks, d_model)  [flattened for CS, then restored]

Data flow (per branch)
----------------------
.. code-block::

    x: (B, n_mics, T_samples)
      → [resample if needed]
      → SpectralProcessor (shared, parameter-free)
      → S: (B, C_in, T, F)       C_in = 11 for FOA (8 mag+phase + 3 IV)
      → InstanceNorm (per branch)
      → _ConformerEarlyStage     → (B, T, d_model)
      → [cs_early]
      → TrackTransformer         → (B, T, n_tracks, d_model)
      → [cs_late]
      → per-track TCN smoother   → (B, T, n_tracks, d_smooth)
      → [optional projection]
      → head                     → (B, T, n_tracks, n_classes)    [SED]
                                 → (B, T, n_tracks, n_classes, 3) [DOA]

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchaudio.functional as AF

from spatial_modules import (
    SpectralProcessor,
    BiGRU,
    ResNetBlock,
    ConformerBlock,
    PositionalEncoder,
    TrackTransformer,
    CrossStitching,
    TCN,
)


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL MODULES
# ══════════════════════════════════════════════════════════════════════════════

class _ConformerEarlyStage(nn.Module):
    """CNN frequency-reduction frontend + ConformerBlock stack.

    Pipeline:
        (B, C_in, T, F)
          → ResNetBlock[MSCA, stride=(1,2)] ×2   → (B, d_model, T, F//4)
          → AdaptiveAvgPool2d(None, 4)            → (B, d_model, T, 4)
          → Conv2d depthwise(1×4)                 → (B, d_model, T)
          → PositionalEncoder (sinusoidal, T)     → (B, d_model, T)
          → permute                               → (B, T, d_model)
          → ConformerBlock × n_conf_layers        → (B, T, d_model)
    """

    def __init__(
        self,
        in_channels:    int,
        d_model:        int,
        n_conf_layers:  int,
        n_heads:        int,
        ffn_expansion:  int   = 4,
        conv_kernel:    int   = 31,
        dropout:        float = 0.0,
        use_msca:       bool  = True,
        msca_reduction: int   = 4,
        pe_max_len:     int   = 2048,
    ):
        super().__init__()
        d_mid = max(d_model // 2, 1)

        # Two strided ResNetBlocks collapse frequency: F → F//2 → F//4
        self.cnn = nn.ModuleList([
            ResNetBlock(
                in_channels    = in_channels,
                out_channels   = d_mid,
                stride         = (1, 2),
                use_msca       = use_msca,
                msca_reduction = msca_reduction,
                dropout        = 0.0,
            ),
            ResNetBlock(
                in_channels    = d_mid,
                out_channels   = d_model,
                stride         = (1, 2),
                use_msca       = use_msca,
                msca_reduction = msca_reduction,
                dropout        = 0.0,
            ),
        ])
        # Pool to 4 frequency subbands, then project with learned per-channel weights
        self.freq_pool = nn.AdaptiveAvgPool2d((None, 4))
        self.freq_proj = nn.Conv2d(d_model, d_model, kernel_size=(1, 4),
                                   groups=d_model, bias=True)

        # Sinusoidal PE along T before the Conformer stack
        self.pe = PositionalEncoder(d_model, max_len=pe_max_len, pe_type='t')

        self.conformer = nn.ModuleList([
            ConformerBlock(d_model, n_heads, ffn_expansion, conv_kernel, dropout)
            for _ in range(n_conf_layers)
        ])

        self.out_channels = d_model   # exposed for downstream sizing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T, F)
        for blk in self.cnn:
            x = blk(x)                             # (B, d_model, T, F//4)
        x = self.freq_pool(x)                      # (B, d_model, T, 4)
        x = self.freq_proj(x).squeeze(-1)          # (B, d_model, T)
        x = self.pe(x)                             # (B, d_model, T) — PE along T
        x = x.permute(0, 2, 1)                     # (B, T, d_model)
        for blk in self.conformer:
            x = blk(x)                             # (B, T, d_model)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# BUILDER HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _build_spatial_early(
    cfg: dict,
    in_channels: int,
) -> Tuple[_ConformerEarlyStage, int]:
    """Build the CNN-Conformer early stage. Returns (module, d_model)."""
    typ = cfg.get('type', 'conformer')
    if typ != 'conformer':
        raise ValueError(
            f'[StandaloneSELD] spatial_early.type must be "conformer", got {typ!r}.'
        )
    d_model = cfg.get('d_model', 128)
    module  = _ConformerEarlyStage(
        in_channels    = in_channels,
        d_model        = d_model,
        n_conf_layers  = cfg.get('n_conf_layers', 4),
        n_heads        = cfg.get('n_heads', 4),
        ffn_expansion  = cfg.get('ffn_expansion', 4),
        conv_kernel    = cfg.get('conv_kernel', 31),
        dropout        = cfg.get('dropout', 0.0),
        use_msca       = cfg.get('use_msca', True),
        msca_reduction = cfg.get('msca_reduction', 4),
        pe_max_len     = cfg.get('pe_max_len', 2048),
    )
    return module, d_model


def _build_spatial_late(
    cfg: dict,
    d_in: int,
) -> Tuple[TrackTransformer, int, int]:
    """Build the TrackTransformer late stage. Returns (module, n_tracks, d_model)."""
    typ = cfg.get('type', 'track_transformer')
    if typ != 'track_transformer':
        raise ValueError(
            f'[StandaloneSELD] spatial_late.type must be "track_transformer", got {typ!r}.'
        )
    d_model  = cfg.get('d_model', 128)
    n_tracks = cfg.get('n_tracks', 2)
    module   = TrackTransformer(
        d_model    = d_model,
        n_heads    = cfg.get('n_heads', 4),
        n_tracks   = n_tracks,
        num_layers = cfg.get('n_layers', 2),
        d_input    = d_in,
        dropout    = cfg.get('dropout', 0.0),
        pe_enable  = True,
        pe_max_len = cfg.get('pe_max_len', 2048),
    )
    return module, n_tracks, d_model


def _build_track_smoother(
    cfg: dict,
    d_model: int,
) -> Tuple[nn.Module, int]:
    """Build the temporal smoother. Returns (module, d_out).

    Supported types: 'tcn' (default), 'bigru' (ablation only).
    d_model is the per-track dimension from the TrackTransformer.
    """
    typ = cfg.get('type', 'tcn')

    if typ == 'tcn':
        tcn_d  = cfg.get('d_model', d_model)   # default: same as TT d_model
        module = TCN(
            d_model     = tcn_d,
            n_layers    = cfg.get('n_layers', 6),
            kernel_size = cfg.get('kernel_size', 3),
            dropout     = cfg.get('dropout', 0.0),
        )
        return module, module.out_dim

    elif typ == 'bigru':
        hidden = cfg.get('hidden_size', 256)
        module = BiGRU(
            in_size     = d_model,
            hidden_size = hidden,
            num_layers  = cfg.get('n_layers', 1),
            dropout     = cfg.get('dropout', 0.0),
        )
        return module, hidden

    else:
        raise ValueError(
            f'[StandaloneSELD] track_smoothing.type must be "tcn" or "bigru", got {typ!r}.'
        )


# ══════════════════════════════════════════════════════════════════════════════
# StandaloneSELDBranch
# ══════════════════════════════════════════════════════════════════════════════

class StandaloneSELDBranch(nn.Module):
    """One complete SELD processing branch (SED or DOA).

    Exposes stage methods so StandaloneSELDModel can interleave cross-stitching:
      * forward_norm(S)         → (B, C_in, T, F)   normalised spectrogram
      * forward_early(x)        → (B, T, d_early)    CNN + Conformer (freq pooled)
      * forward_late(x)         → (B, T, N, d_model) TrackTransformer
      * forward_smoother(x)     → (B, T, N, d_smooth) per-track TCN
      * forward_head(x)         → (B, T, N, C) or (B, T, N, C, 3)
    """

    def __init__(
        self,
        in_channels:  int,
        n_mics:       int,
        audio_format: str,
        early_cfg:    dict,
        late_cfg:     dict,
        smoother_cfg: dict,
        head_cfg:     dict,
        head_type:    str,
    ):
        super().__init__()

        n_tracks  = late_cfg.get('n_tracks', 2)
        n_classes = head_cfg.get('n_classes', 13)
        proj_dim  = head_cfg.get('proj_dim', None)

        self.n_tracks     = n_tracks
        self.n_classes    = n_classes
        self.head_type    = head_type
        self.audio_format = audio_format

        # ── Per-group InstanceNorm (separate weights per branch) ──────────────
        self.norm_mag   = nn.InstanceNorm2d(n_mics, affine=True)
        self.norm_phase = nn.InstanceNorm2d(n_mics, affine=True)
        if audio_format == 'foa':
            self.norm_iv = nn.InstanceNorm2d(3, affine=True)
        else:
            n_pairs = n_mics * (n_mics - 1) // 2
            self.norm_gcc = nn.InstanceNorm2d(n_pairs, affine=True)
        self._n_mics = n_mics

        # ── Early stage: CNN frontend + ConformerBlock stack ──────────────────
        self.early, self._d_early = _build_spatial_early(early_cfg, in_channels)

        # ── Late stage: TrackTransformer ──────────────────────────────────────
        self.late, self._n_tracks, self._d_model = _build_spatial_late(
            late_cfg, self._d_early
        )

        # ── Smoother: per-track TCN (shared weights across tracks) ────────────
        self.smoother, self._d_smooth = _build_track_smoother(
            smoother_cfg, self._d_model
        )

        # ── Optional projection before head ───────────────────────────────────
        head_in = self._d_smooth
        if proj_dim is not None and proj_dim != self._d_smooth:
            self.proj = nn.Sequential(nn.Linear(self._d_smooth, proj_dim), nn.GELU())
            head_in = proj_dim
        else:
            self.proj = nn.Identity()

        # ── Output head (applied per-track, per-class independently) ──────────
        if head_type == 'sed':
            self.head = nn.Linear(head_in, n_classes)
        elif head_type == 'doa':
            self.head = nn.Linear(head_in, n_classes * 3)
        else:
            raise ValueError(f'[StandaloneSELDBranch] Unknown head_type: {head_type!r}')
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    # ── Stage methods ─────────────────────────────────────────────────────────

    def forward_norm(self, S: torch.Tensor) -> torch.Tensor:
        """Apply per-group InstanceNorm. S: (B, C_in, T, F) → same shape."""
        C = self._n_mics
        parts = [
            self.norm_mag(S[:, :C]),
            self.norm_phase(S[:, C:2 * C]),
        ]
        offset = 2 * C
        if self.audio_format == 'foa':
            parts.append(self.norm_iv(S[:, offset:offset + 3]))
        else:
            parts.append(self.norm_gcc(S[:, offset:]))
        return torch.cat(parts, dim=1)

    def forward_early(self, x: torch.Tensor) -> torch.Tensor:
        """CNN frontend + Conformer. Returns (B, T, d_early)."""
        return self.early(x)

    def forward_late(self, x: torch.Tensor) -> torch.Tensor:
        """TrackTransformer. x: (B, T, d_early) → (B, T, n_tracks, d_model)."""
        return self.late(x)

    def forward_smoother(self, x: torch.Tensor) -> torch.Tensor:
        """Per-track TCN with shared weights.
        x: (B, T, n_tracks, d_model) → (B, T, n_tracks, d_smooth)
        """
        B, T, N, d = x.shape
        x = x.reshape(B * N, T, d)    # treat each track as independent sequence
        x = self.smoother(x)          # (B*N, T, d_smooth)
        return x.reshape(B, T, N, -1) # (B, T, N, d_smooth)

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        """Optional projection + head.
        x: (B, T, N, d_smooth) → (B, T, N, C) or (B, T, N, C, 3)
        """
        x = self.proj(x)   # Identity if no proj_dim configured
        if self.head_type == 'sed':
            return self.head(x)   # (B, T, N, C)
        else:
            B, T, N, _ = x.shape
            return torch.tanh(self.head(x)).reshape(B, T, N, self.n_classes, 3)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        x = self.forward_norm(S)
        x = self.forward_early(x)
        x = self.forward_late(x)
        x = self.forward_smoother(x)
        return self.forward_head(x)


# ══════════════════════════════════════════════════════════════════════════════
# StandaloneSELDModel
# ══════════════════════════════════════════════════════════════════════════════

class StandaloneSELDModel(nn.Module):
    """
    Standalone dual-branch SELD model v2 (Conformer early stage + TCN smoother).

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

        in_cfg   = cfg['input']
        data_cfg = cfg.get('data', {})
        cs_cfg   = cfg.get('cross_stitch', {})
        tss_cfg  = cfg.get('track_smoothing', {})
        head_cfg = cfg.get('heads', {})

        # Auto-compute PE max_len from sr / hop_length / window_s
        _sr  = in_cfg['sr']
        _hop = in_cfg.get('hop_length', _sr // 100)
        _win = float(data_cfg.get('window_s', 10.0) or 10.0)
        _pe_max_len = int(math.ceil(_win * _sr / _hop)) + 50

        # Shallow copies so we can inject pe_max_len without mutating the original cfg
        se_cfg = dict(cfg.get('spatial_early', {}))
        sl_cfg = dict(cfg.get('spatial_late',  {}))
        se_cfg['pe_max_len'] = _pe_max_len
        sl_cfg['pe_max_len'] = _pe_max_len

        self.src_sr       = in_cfg['sr']
        n_mics            = in_cfg.get('n_mics', 4)
        audio_format      = data_cfg.get('format', 'foa')
        self.audio_format = audio_format

        _iv  = (audio_format == 'foa')
        _gcc = (audio_format == 'mic')
        n_pairs    = n_mics * (n_mics - 1) // 2
        c_in_spat  = 2 * n_mics + (3 if _iv else n_pairs if _gcc else 0)

        # ── Shared spectral frontend (parameter-free STFT) ────────────────────
        n_fft      = in_cfg.get('n_fft', 1024)
        hop_length = in_cfg.get('hop_length', self.src_sr // 100)
        self.input_proc = SpectralProcessor(
            n_fft       = n_fft,
            hop_length  = hop_length,
            compute_iv  = _iv,
            compute_gcc = _gcc,
            tau_max     = n_fft // 4,
        )

        # ── Cross-stitch flags ────────────────────────────────────────────────
        self.cs_early_on = cs_cfg.get('cs_early_enabled', False)
        self.cs_late_on  = cs_cfg.get('cs_late_enabled',  False)

        # ── Two symmetric branches ────────────────────────────────────────────
        branch_kwargs = dict(
            in_channels  = c_in_spat,
            n_mics       = n_mics,
            audio_format = audio_format,
            early_cfg    = se_cfg,
            late_cfg     = sl_cfg,
            smoother_cfg = tss_cfg,
            head_cfg     = head_cfg,
        )
        self.sed_branch = StandaloneSELDBranch(head_type='sed', **branch_kwargs)
        self.doa_branch = StandaloneSELDBranch(head_type='doa', **branch_kwargs)

        # ── CrossStitching layers ─────────────────────────────────────────────
        # CS_EARLY operates on (B, T, d_early) via permute to (B, d_early, T)
        # CS_LATE  operates on (B, T, N*d_model) flattened, via permute to (B, D, T)
        d_early = self.sed_branch._d_early
        n_tracks = self.sed_branch._n_tracks
        d_model  = self.sed_branch._d_model

        if self.cs_early_on:
            self.cs_early = CrossStitching(n_channels=d_early)
        if self.cs_late_on:
            self.cs_late = CrossStitching(n_channels=n_tracks * d_model)

        # ── Diagnostic print ─────────────────────────────────────────────────
        print(
            f'[StandaloneSELDModel v2]'
            f'  format={audio_format}  sr={self.src_sr}'
            f'  n_mics={n_mics}  c_in={c_in_spat}'
            f'  n_fft={n_fft}  hop={hop_length}'
        )
        print(
            f'  early=conformer(CNN×2 + Conf×{se_cfg.get("n_conf_layers",4)})'
            f'  d_model={d_early}'
            f'  late=TrackTF×{sl_cfg.get("n_layers",2)}'
            f'  n_tracks={n_tracks}  d_model={d_model}'
        )
        print(
            f'  smoother={tss_cfg.get("type","tcn")}×{tss_cfg.get("n_layers",6)}'
            f'  d_smooth={self.sed_branch._d_smooth}'
            f'  cs_early={"on" if self.cs_early_on else "off"}'
            f'  cs_late={"on" if self.cs_late_on else "off"}'
        )

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

        # ── Stage 1: per-branch InstanceNorm ─────────────────────────────────
        S_sed = self.sed_branch.forward_norm(S)
        S_doa = self.doa_branch.forward_norm(S)

        # ── Stage 2: CNN frontend + Conformer → (B, T, d_early) ──────────────
        e_sed = self.sed_branch.forward_early(S_sed)
        e_doa = self.doa_branch.forward_early(S_doa)

        # CS_EARLY: permute (B, T, d) → (B, d, T) for CrossStitching → back
        if self.cs_early_on:
            e_sed_p, e_doa_p = self.cs_early(
                e_sed.permute(0, 2, 1),
                e_doa.permute(0, 2, 1),
            )
            e_sed = e_sed_p.permute(0, 2, 1)
            e_doa = e_doa_p.permute(0, 2, 1)

        # ── Stage 3: TrackTransformer → (B, T, n_tracks, d_model) ────────────
        l_sed = self.sed_branch.forward_late(e_sed)
        l_doa = self.doa_branch.forward_late(e_doa)

        # CS_LATE: flatten tracks → permute → CrossStitch → reshape back
        if self.cs_late_on:
            B, T, N, d = l_sed.shape
            D = N * d
            sed_cs, doa_cs = self.cs_late(
                l_sed.reshape(B, T, D).permute(0, 2, 1),   # (B, D, T)
                l_doa.reshape(B, T, D).permute(0, 2, 1),
            )
            l_sed = sed_cs.permute(0, 2, 1).reshape(B, T, N, d)
            l_doa = doa_cs.permute(0, 2, 1).reshape(B, T, N, d)

        # ── Stage 4: per-track TCN → (B, T, n_tracks, d_smooth) ──────────────
        l_sed = self.sed_branch.forward_smoother(l_sed)
        l_doa = self.doa_branch.forward_smoother(l_doa)

        # ── Stage 5: optional projection + head ──────────────────────────────
        act_logits = self.sed_branch.forward_head(l_sed)   # (B, T, N, C)
        doa_tanh   = self.doa_branch.forward_head(l_doa)   # (B, T, N, C, 3)

        return act_logits, doa_tanh
