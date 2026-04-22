"""
NeuralSRPEncoder — Pairwise CRNN encoder for DOA/spatial feature extraction
============================================================================
Provenance: ablated from neural_srp (MIT licence)
  Original classes: NeuralSrp, NeuralSrpFeatureExtractor, GCC, Window,
                    Mlp, ConditionalConvBlock  (neural_srp/models/)
  Original helpers: get_all_pairs, prepare_mic_pos  (neural_srp/datasets/)
Adapted: standalone, no external dependencies, PyTorch-native windowing,
         output (B, T_frames, d_out) framewise embedding for downstream use.

Architecture overview
---------------------
1. Hann windowing:  (B, n_mics, T_samples) → (B, T_frames, n_mics, win_size)
2. GCC-PHAT (all mic pairs i<j):
       rfft → magnitude normalisation (PHAT) → pairwise cross-spectra
       irfft → centre → (B, T_frames, n_pairs, 2×tau_max)
3. Mic-position metadata per pair:
       absolute 3-D positions of mic_i and mic_j → (B, n_pairs, 6)  [cm]
4. Pairwise CRNN — processed over all pairs simultaneously (batch×pairs
   folded into the batch dimension):
       _ConvBlock × n_conv_layers  (Conv2d + BN + PReLU + optional pool/drop
                                    + optional metadata bias)
       → frequency mean → (B×n_pairs, T, C_conv)
       → BiGRU bidirectional (tanh, mul merge_mode)  → (B×n_pairs, T, rnn_h)
       → late metadata concat → _Mlp (PReLU) → (B×n_pairs, T, fnn_pairwise)
5. Pair aggregation (mean): → (B, T, fnn_pairwise)

Output: (B, T_frames, d_out)

References
----------
E. Grinstein, C.M. Hicks, T. van Waterschoot, M. Brookes, and P.A. Naylor, 
“The Neural-SRP Method for Universal Robust Multi-Source Tracking,” 
IEEE Open Journal of Signal Processing, vol. 5, pp. 19–28, 2024, 
DOI: 10.1109/OJSP.2023.3340057.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

import itertools
from typing import List, Optional

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _next_pow2(x: int) -> int:
    """Return the smallest power of 2 ≥ x."""
    return 1 if x <= 1 else 2 ** (x - 1).bit_length()


def _get_all_pairs(n_mics: int, device=None) -> torch.Tensor:
    """Return (n_pairs, 2) long tensor of all unique mic-pair indices (i < j)."""
    pairs = list(itertools.combinations(range(n_mics), 2))
    return torch.tensor(pairs, dtype=torch.long, device=device)


def _prepare_mic_pos(mic_pos: torch.Tensor,
                     pair_idxs: torch.Tensor) -> torch.Tensor:
    """
    Extract and flatten microphone position metadata for each pair.

    Args:
        mic_pos   : (B, n_mics, 3)  mic positions in metres
        pair_idxs : (n_pairs, 2)    mic indices per pair

    Returns:
        (B, n_pairs, 6)  concatenated [pos_i | pos_j] in centimetres
    """
    B       = mic_pos.shape[0]
    n_pairs = pair_idxs.shape[0]
    i_idx   = pair_idxs[:, 0]  # (n_pairs,)
    j_idx   = pair_idxs[:, 1]  # (n_pairs,)
    pos_i   = mic_pos[:, i_idx]  # (B, n_pairs, 3)
    pos_j   = mic_pos[:, j_idx]  # (B, n_pairs, 3)
    return torch.cat([pos_i, pos_j], dim=-1) * 100.0  # m → cm, (B, n_pairs, 6)


# ─────────────────────────────────────────────────────────────────────────────
# Windowing
# ─────────────────────────────────────────────────────────────────────────────

class _Window(nn.Module):
    """
    Hann-windowed framing transform (non-trainable).

    Args:
        win_size : frame length in samples
        hop_size : hop between consecutive frames in samples

    Input:  (B, n_mics, T_samples)
    Output: (B, T_frames, n_mics, win_size)
    """

    def __init__(self, win_size: int, hop_size: int) -> None:
        super().__init__()
        self.win_size = win_size
        self.hop_size = hop_size
        # Hann window registered as buffer so it moves with the module
        self.register_buffer(
            'window', torch.hann_window(win_size), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, n_mics, T = x.shape
        # unfold along time axis: (B, n_mics, T_frames, win_size)
        frames = x.unfold(-1, self.win_size, self.hop_size)  # (B, n_mics, T_f, ws)
        frames = frames * self.window                         # apply Hann
        return frames.permute(0, 2, 1, 3)                    # (B, T_f, n_mics, ws)


# ─────────────────────────────────────────────────────────────────────────────
# GCC-PHAT
# ─────────────────────────────────────────────────────────────────────────────

class _GCCPHAT(nn.Module):
    """
    Generalized Cross-Correlation with Phase Transform (PHAT) for all mic pairs.

    Faithful to the original neural_srp GCC class (center=True, abs=True).
    For each ordered pair (i < j):
        1. rfft both signals, normalise by magnitude (PHAT)
        2. cross-spectrum = X_i × conj(X_j)
        3. irfft → centre lags → keep [0, 2×tau_max)

    Args:
        win_size : frame size (K) used for FFT padding to next power of 2
        tau_max  : maximum lag in samples; output width = 2 × tau_max

    Input:  (B, T_frames, n_mics, win_size)
    Output: (B, T_frames, n_pairs, 2×tau_max)   real, non-negative
    """

    def __init__(self, win_size: int, tau_max: int) -> None:
        super().__init__()
        self.n_dft   = _next_pow2(win_size)
        self.tau_max = tau_max

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, n_mics, _ = x.shape
        tau = self.tau_max

        # rfft: (B, T, n_mics, F)
        X = torch.fft.rfft(x, n=self.n_dft, dim=-1)
        # PHAT normalisation
        X = X / (X.abs() + 1e-8)

        # Build all pair indices on the correct device
        pairs = _get_all_pairs(n_mics, device=x.device)  # (n_pairs, 2)
        n_pairs = pairs.shape[0]

        # Cross-spectra for all pairs
        Xi = X[:, :, pairs[:, 0]]  # (B, T, n_pairs, F)
        Xj = X[:, :, pairs[:, 1]]  # (B, T, n_pairs, F)
        cs = Xi * Xj.conj()        # (B, T, n_pairs, F)

        # irfft → (B, T, n_pairs, n_dft)
        gcc_full = torch.fft.irfft(cs, n=self.n_dft, dim=-1)

        # Centre: negative lags = last tau_max samples, positive = first tau_max
        gcc = torch.cat([
            gcc_full[..., -tau:],   # lags [-tau_max, -1]
            gcc_full[...,  :tau],   # lags [0, tau_max-1]
        ], dim=-1)                  # (B, T, n_pairs, 2×tau_max)

        return gcc.abs()


# ─────────────────────────────────────────────────────────────────────────────
# Conv block with optional metadata bias injection
# ─────────────────────────────────────────────────────────────────────────────

class _ConvBlock(nn.Module):
    """
    Convolutional block: Conv2d → BN → PReLU → [MaxPool] → [Dropout2d].
    Optional metadata bias: linearly projected metadata added to conv output
    before BN, enabling geometry-conditioned feature extraction.

    Args:
        in_channels  : input channels
        out_channels : output channels
        n_metadata   : metadata feature size (0 = disabled)
        pool_size    : (t_pool, f_pool) MaxPool kernel; None = no pooling
        dropout_rate : Dropout2d probability (0.0 = off)
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        n_metadata:   int                = 0,
        pool_size:    Optional[tuple]    = None,
        dropout_rate: float              = 0.0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=1, padding=1)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.act  = nn.PReLU()

        self.pool = nn.MaxPool2d(kernel_size=pool_size) if pool_size else None
        self.drop = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0.0 else None

        if n_metadata > 0:
            self.meta_proj = nn.Linear(n_metadata, out_channels)
        else:
            self.meta_proj = None

    def forward(self,
                x:        torch.Tensor,
                metadata: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.conv(x)
        if metadata is not None and self.meta_proj is not None:
            # metadata: (B_pair, n_metadata) → project → (B_pair, out_ch, 1, 1)
            bias = self.meta_proj(metadata).unsqueeze(-1).unsqueeze(-1)
            out  = out + bias
        out = self.act(self.bn(out))
        if self.pool is not None:
            out = self.pool(out)
        if self.drop is not None:
            out = self.drop(out)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# MLP with PReLU and optional BatchNorm1d
# ─────────────────────────────────────────────────────────────────────────────

class _Mlp(nn.Module):
    """
    Multi-layer perceptron with PReLU activations.
    Faithful to neural_srp Mlp (activation='prelu', optional BN1d).

    Args:
        in_features      : input dimension
        out_features     : output dimension
        hidden_features  : hidden dimension (used for all intermediate layers)
        num_layers       : total number of Linear layers (≥ 1)
        batch_norm       : apply BatchNorm1d after each hidden Linear
        output_activation: nn.Module applied after last layer (None = linear out)
    """

    def __init__(
        self,
        in_features:       int,
        out_features:      int,
        hidden_features:   int,
        num_layers:        int,
        batch_norm:        bool              = False,
        output_activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []

        for i in range(num_layers):
            i_feat = in_features      if i == 0          else hidden_features
            o_feat = out_features     if i == num_layers - 1 else hidden_features
            layers.append(nn.Linear(i_feat, o_feat))
            if i < num_layers - 1:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(o_feat))
                layers.append(nn.PReLU())
            else:
                if output_activation is not None:
                    layers.append(output_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Support (B, T, F): apply BN1d on feature dim via transpose
        needs_transpose = (x.dim() == 3)
        for layer in self.net:
            if isinstance(layer, nn.BatchNorm1d) and needs_transpose:
                x = layer(x.transpose(1, 2)).transpose(1, 2)
            else:
                x = layer(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# NeuralSRPEncoder
# ─────────────────────────────────────────────────────────────────────────────

class NeuralSRPEncoder(nn.Module):
    """
    Pairwise CRNN encoder for geometry-independent spatial feature extraction.

    Processes raw multi-channel audio + microphone positions and produces
    framewise embeddings via the Neural-SRP pairwise architecture.

    Args:
        n_mics           : number of microphones (default 4)
        win_size         : STFT frame size in samples (default 512)
        hop_size         : hop between frames in samples (default 256)
        tau_max          : maximum GCC-PHAT lag in samples (default 32);
                           output GCC width = 2 × tau_max
        n_conv_channels  : Conv2d filter count in each conv block (default 32)
        n_conv_layers    : number of _ConvBlock modules (default 3)
        f_pool_size      : frequency MaxPool factor per layer (list, default [2,2,2])
        t_pool_size      : time MaxPool factor per layer     (list, default [1,1,1])
        dropout_rate     : Dropout2d rate in conv blocks (default 0.0)
        rnn_hidden       : GRU hidden size (default 64)
        n_rnn_layers     : number of stacked GRU layers (default 1)
        fnn_pairwise_size: per-pair MLP output dimension (default 64)
        d_out            : final output embedding dimension (default 128)
        pair_agg_mode    : how to aggregate across pairs ('sum' | 'mean', default 'sum')
        metadata_fusion  : where to inject mic-position metadata
                           'early_conv_bias' — add as bias in first conv block
                           'late_concat'     — concatenate before pairwise MLP (default)
        batch_norm_hidden: BatchNorm1d in pairwise MLP (default False)
        bidirectional    : use dual fwd+bwd GRU stacks with tanh-gated mul merge (default True);
                           False = single unidirectional GRU with tanh output (original default)

    Input:
        x       : (B, n_mics, T_samples)  raw waveform
        mic_pos : (B, n_mics, 3)           microphone positions in metres

    Output:
        (B, T_frames, d_out)  framewise spatial embeddings
    """

    def __init__(
        self,
        n_mics:            int   = 4,
        win_size:          int   = 512,
        hop_size:          int   = 256,
        tau_max:           int   = 32,
        n_conv_channels:   int   = 128,
        n_conv_layers:     int   = 3,
        f_pool_size:       list  = None,
        t_pool_size:       list  = None,
        dropout_rate:      float = 0.0,
        rnn_hidden:        int   = 128,
        n_rnn_layers:      int   = 2,
        fnn_pairwise_size: int   = 128,
        d_out:             int   = 128,
        pair_agg_mode:     str   = 'sum',
        metadata_fusion:   str   = 'late_concat',
        batch_norm_hidden: bool  = False,
        bidirectional:     bool  = True,
    ) -> None:
        super().__init__()

        if f_pool_size is None:
            f_pool_size = [2] * n_conv_layers
        if t_pool_size is None:
            t_pool_size = [1] * n_conv_layers

        assert len(f_pool_size) == n_conv_layers, 'f_pool_size length must match n_conv_layers'
        assert len(t_pool_size) == n_conv_layers, 't_pool_size length must match n_conv_layers'
        assert pair_agg_mode   in ('mean', 'sum'), f"pair_agg_mode must be 'mean' or 'sum'"
        assert metadata_fusion in ('early_conv_bias', 'late_concat'), \
            f"metadata_fusion must be 'early_conv_bias' or 'late_concat'"

        self.n_mics            = n_mics
        self.tau_max           = tau_max
        self.pair_agg_mode     = pair_agg_mode
        self.metadata_fusion   = metadata_fusion
        self.bidirectional     = bidirectional
        self.n_metadata        = 6     # 2 mic positions × 3 coordinates

        # ── Signal processing ─────────────────────────────────────────────
        self.window  = _Window(win_size, hop_size)
        self.gcc     = _GCCPHAT(win_size, tau_max)

        # ── Convolutional stack ───────────────────────────────────────────
        conv_blocks: List[nn.Module] = []
        in_ch = 1
        n_freq_after = 2 * tau_max
        for i in range(n_conv_layers):
            early_meta = (self.n_metadata
                          if metadata_fusion == 'early_conv_bias' and i == 0
                          else 0)
            pool = (t_pool_size[i], f_pool_size[i]) if (t_pool_size[i] > 1 or
                                                          f_pool_size[i] > 1) else None
            conv_blocks.append(_ConvBlock(
                in_channels  = in_ch,
                out_channels = n_conv_channels,
                n_metadata   = early_meta,
                pool_size    = pool,
                dropout_rate = dropout_rate,
            ))
            in_ch = n_conv_channels
            n_freq_after = n_freq_after // max(f_pool_size[i], 1)
        self.conv_blocks = nn.ModuleList(conv_blocks)

        # ── GRU ──────────────────────────────────────────────────────────────
        # frequency axis collapsed via mean → input dim = n_conv_channels
        in_gru = n_conv_channels
        if bidirectional:
            # Dual-stack fwd+bwd with tanh-gated mul merge (each unidirectional)
            self.gru_fwd = nn.ModuleList([
                nn.GRU(
                    input_size  = in_gru if j == 0 else rnn_hidden,
                    hidden_size = rnn_hidden,
                    num_layers  = 1,
                    batch_first = True,
                ) for j in range(n_rnn_layers)
            ])
            self.gru_bwd = nn.ModuleList([
                nn.GRU(
                    input_size  = in_gru if j == 0 else rnn_hidden,
                    hidden_size = rnn_hidden,
                    num_layers  = 1,
                    batch_first = True,
                ) for j in range(n_rnn_layers)
            ])
        else:
            # Single unidirectional GRU (original neural_srp default)
            self.gru = nn.GRU(
                input_size  = in_gru,
                hidden_size = rnn_hidden,
                num_layers  = n_rnn_layers,
                batch_first = True,
            )

        # ── Pairwise MLP (per pair) ───────────────────────────────────────
        fnn_in = rnn_hidden + (self.n_metadata if metadata_fusion == 'late_concat' else 0)
        self.fnn_pairwise = _Mlp(
            in_features     = fnn_in,
            out_features    = fnn_pairwise_size,
            hidden_features = fnn_pairwise_size,
            num_layers      = 2,
            batch_norm      = batch_norm_hidden,
            output_activation = nn.PReLU(),
        )

        # ── Output projection ─────────────────────────────────────────────
        self.fnn_out = (
            nn.Linear(fnn_pairwise_size, d_out)
            if fnn_pairwise_size != d_out else nn.Identity()
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self,
                x:       torch.Tensor,
                mic_pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x       : (B, n_mics, T_samples)
            mic_pos : (B, n_mics, 3)  microphone positions in metres

        Returns:
            (B, T_frames, d_out)
        """
        B = x.shape[0]

        # 1. Window
        frames = self.window(x)               # (B, T, n_mics, win_size)

        # 2. GCC-PHAT
        gcc = self.gcc(frames)                # (B, T, n_pairs, 2×tau_max)
        T, n_pairs = gcc.shape[1], gcc.shape[2]

        # 3. Prepare metadata (B, n_pairs, 6)
        pairs = _get_all_pairs(self.n_mics, device=x.device)
        metadata = _prepare_mic_pos(mic_pos, pairs)  # (B, n_pairs, 6)

        # 4. Pairwise CRNN — fold (B, n_pairs) into batch dim
        # gcc: (B, T, n_pairs, 2τ) → (B×n_pairs, 1, T, 2τ)
        feat = gcc.permute(0, 2, 1, 3).reshape(B * n_pairs, 1, T, 2 * self.tau_max)
        meta_flat = metadata.reshape(B * n_pairs, self.n_metadata)  # (B×n_pairs, 6)

        for i, blk in enumerate(self.conv_blocks):
            m = meta_flat if (self.metadata_fusion == 'early_conv_bias' and i == 0) else None
            feat = blk(feat, metadata=m)       # (B×n_pairs, C, T', F')

        # Flatten frequency axis
        feat = feat.mean(dim=-1)               # (B×n_pairs, C, T')
        feat = feat.permute(0, 2, 1)           # (B×n_pairs, T', C)

        # GRU with tanh activation
        if self.bidirectional:
            h = feat
            for fwd, bwd in zip(self.gru_fwd, self.gru_bwd):
                fwd_out, _ = fwd(h)
                bwd_out, _ = bwd(h.flip(1))
                h = torch.tanh(fwd_out) * torch.tanh(bwd_out.flip(1))
        else:
            h, _ = self.gru(feat)
            h = torch.tanh(h)

        # Late metadata concat
        if self.metadata_fusion == 'late_concat':
            T_out = h.shape[1]
            meta_exp = meta_flat.unsqueeze(1).expand(-1, T_out, -1)  # (B×n_pairs, T', 6)
            h = torch.cat([h, meta_exp], dim=-1)

        # Per-pair MLP
        h = self.fnn_pairwise(h)               # (B×n_pairs, T', fnn_pairwise)

        # 5. Pair aggregation → (B, T', fnn_pairwise)
        h = h.reshape(B, n_pairs, h.shape[1], -1)  # (B, n_pairs, T', fnn_p)
        if self.pair_agg_mode == 'mean':
            h = h.mean(dim=1)
        else:
            h = h.sum(dim=1)

        return self.fnn_out(h)                 # (B, T', d_out)


# ─────────────────────────────────────────────────────────────────────────────
# SpatialSRPModel — encoder + three prediction heads
# ─────────────────────────────────────────────────────────────────────────────

class SpatialSRPModel(nn.Module):
    """
    Standalone spatial SELD model built on top of NeuralSRPEncoder.

    Three heads are applied to the framewise encoder embedding:
        DOA      : Linear → Tanh   → (B, T, N_tracks, 3)  Cartesian unit vectors
        Activity : Linear          → (B, T, N_tracks)      logits for BCE
        Distance : Linear → Softplus → (B, T, N_tracks)   positive-valued cm

    Args:
        encoder_cfg : dict of NeuralSRPEncoder constructor kwargs
        n_tracks    : number of simultaneous source tracks (default 2)

    Input:
        wav     : (B, n_mics, T_samples)
        mic_pos : (B, n_mics, 3)  mic positions in metres

    Output dict:
        "doa"      : (B, T, N_tracks, 3)
        "activity" : (B, T, N_tracks)   raw logits
        "distance" : (B, T, N_tracks)   cm, ≥ 0
    """

    def __init__(self, encoder_cfg: dict, n_tracks: int = 2) -> None:
        super().__init__()
        self.encoder  = NeuralSRPEncoder(**encoder_cfg)
        d             = encoder_cfg.get('d_out', 128)
        self.n_tracks = n_tracks

        self.doa_head  = nn.Sequential(nn.Linear(d, n_tracks * 3), nn.Tanh())
        self.act_head  = nn.Linear(d, n_tracks)
        self.dist_head = nn.Sequential(nn.Linear(d, n_tracks), nn.Softplus())

    def forward(self,
                wav:     torch.Tensor,
                mic_pos: torch.Tensor) -> dict:
        h   = self.encoder(wav, mic_pos)               # (B, T, d)
        B, T, _ = h.shape
        N   = self.n_tracks

        doa  = self.doa_head(h).view(B, T, N, 3)       # (B, T, N, 3)
        act  = self.act_head(h)                        # (B, T, N)
        dist = self.dist_head(h)                       # (B, T, N)

        return {"doa": doa, "activity": act, "distance": dist}
