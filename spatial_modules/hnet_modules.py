"""
HNetGRU — Hungarian Network for Soft Source Assignment
=======================================================
Ported 1:1 from the hungarian-net architecture (MIT licence):
  https://github.com/sharathadavanne/hungarian-net

Original paper:
  S. Adavanne, A. Politis, T. Virtanen, "A multi-source sound
  field decomposition inspired by the cocktail party problem," 2021.

Placement in this project
--------------------------
This file is a self-contained port kept in ``spatial_modules/`` for
complete independence from the ``neural_srp/`` clone directory.
No runtime dependency on the ``neural_srp`` package.

Usage
-----
    from spatial_modules.hnet_modules import HNetGRU

    hnet = HNetGRU(max_len=2)          # max_len = N_tracks
    hnet.load_state_dict(torch.load(ckpt_path))
    hnet.eval()

    # dist_mat : (batch_frames, N_tracks, N_tracks)  L2 distance matrix
    out1, out2, out3 = hnet(dist_mat)
    da_mat = out1.sigmoid().view(dist_mat.shape)    # soft assignment (0–1)

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

import torch
import torch.nn as nn


class _AttentionLayer(nn.Module):
    """Single-head key-value attention over the time/sequence axis."""

    def __init__(self, in_channels: int, out_channels: int, key_channels: int) -> None:
        super().__init__()
        self.conv_Q = nn.Conv1d(in_channels, key_channels,  kernel_size=1, bias=False)
        self.conv_K = nn.Conv1d(in_channels, key_channels,  kernel_size=1, bias=False)
        self.conv_V = nn.Conv1d(in_channels, out_channels,  kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.conv_Q(x)                              # (B, key, seq)
        K = self.conv_K(x)                              # (B, key, seq)
        V = self.conv_V(x)                              # (B, out, seq)
        A = Q.permute(0, 2, 1).matmul(K).softmax(2)    # (B, seq, seq)
        return A.matmul(V.permute(0, 2, 1)).permute(0, 2, 1)   # (B, out, seq)

    def __repr__(self) -> str:
        return (
            f'{self._get_name()}'
            f'(in={self.conv_Q.in_channels}, '
            f'out={self.conv_V.out_channels}, '
            f'key={self.conv_K.out_channels})'
        )


class HNetGRU(nn.Module):
    """
    Hungarian Network GRU (HNetGRU).

    Learns soft permutation matrices from pairwise L2 distance matrices
    between predicted and target source DOA unit vectors.

    Architecture (faithful to original):
        GRU(max_len → 128) → AttentionLayer(128 → 128) → FC(128 → max_len)

    Args:
        max_len : number of source tracks (= N_tracks, typically 2)

    Input:
        query : ``(batch_frames, N_tracks, N_tracks)``
                L2 distance matrix between predicted and target DOA unit vectors.
                ``query[t, i, j]`` = distance from predicted track ``i`` to
                target track ``j`` at frame ``t``.

    Output (three views of the same FC output, matching the original API):
        out1  : ``(batch_frames, N_tracks × N_tracks)`` — flattened (squeezed if batch=1)
        out2  : ``(batch_frames, N_tracks)``            — max over feature dim
        out3  : ``(batch_frames, N_tracks)``            — max over sequence dim

    Typical usage (caller)::

        out1, _, _ = hnet(dist_mat)
        da_mat = out1.sigmoid().view(dist_mat.shape)      # (BT, N, N)
        da_mat = (da_mat > 0.5) * da_mat                 # threshold + weight
        da_activity = da_mat.max(-1)[0]                  # (BT, N) per-track activity
    """

    def __init__(self, max_len: int = 4) -> None:
        super().__init__()
        hidden_size = 128
        self.nb_gru_layers = 1
        self.gru  = nn.GRU(max_len, hidden_size, self.nb_gru_layers, batch_first=True)
        self.attn = _AttentionLayer(hidden_size, hidden_size, hidden_size)
        self.fc1  = nn.Linear(hidden_size, max_len)

    def forward(self, query: torch.Tensor):
        out, _ = self.gru(query)              # (B, seq, hidden)

        out = out.permute(0, 2, 1)            # (B, hidden, seq)
        out = self.attn(out)                  # (B, hidden, seq)
        out = out.permute(0, 2, 1)            # (B, seq, hidden)
        out = torch.tanh(out)

        out = self.fc1(out)                   # (B, seq, max_len)

        out1 = out.view(out.shape[0], -1).squeeze()     # (B, seq*max_len)  or (seq*max_len,)
        out2 = torch.max(out, dim=-1)[0].squeeze()      # (B, seq)          or (seq,)
        out3 = torch.max(out, dim=-2)[0].squeeze()      # (B, max_len)      or (max_len,)

        return out1, out2, out3
