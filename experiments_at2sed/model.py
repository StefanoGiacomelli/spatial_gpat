"""
AT2SEDModel — AudioSet-Tagging to SED Classifier
=================================================
Fine-tunes the GP-AT (E-PANNs) backbone as a multi-label SED classifier.

Two operating modes, selected via ``experiment.dataset`` in the config:

``audioset_strong`` (default)
    Standard AT2SED mode.  Output: ``(B, 456)`` raw logits.
    Backbone ``fc_audioset`` is the final layer.

``starss23``
    STARSS23 classification mode.  A trainable projection head
    ``head_starss : Linear(456, 13)`` is appended after ``fc_audioset``.
    Output: ``(B, 13)`` raw logits.  ``fc_audioset`` weights are preserved
    from the pretrained checkpoint; ``head_starss`` is soft-prior initialised
    from the AudioSet-Strong → STARSS23 mapping matrix.
    ``model.n_cls`` must be set to 13 in the experiment YAML.

Architecture::

    x : (B, T_samples)   mono waveform @ 32 kHz
        ↓ SplitEPANNs.forward_early()
          spectrogram → logmel → BN0 → SpecAugment(train) → ConvBlock1-3
    h_early : (B, 256, T/8, F/8)
        ↓ SplitEPANNs.forward_late()
          ConvBlock4-6
    h_late : (B, 1024, T/32, F/32)
        ↓ global pool: mean(F) → max(T) + mean(T)
    pooled : (B, 1024)
        ↓ Dropout(0.5) → fc1(1024→2048) → ReLU → Dropout(0.5)
        ↓ fc_audioset(2048→456)           [always trained]
    logits_456 : (B, 456)  ← returned in audioset_strong mode
        ↓ head_starss(456→13)             [starss23 mode only]
    logits_13  : (B, 13)   ← returned in starss23 mode

Freezing / initialization is fully configurable via the ``semantic`` section
of the config YAML (same keys as in the NAS pipeline)::

    semantic:
      checkpoint  : path/to/pretrained.pt
      early_group : {init: pretrained, frozen: false}
      late_group  : {init: pretrained, frozen: false}
      head_group  : {init: pretrained}

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Ensure repo root is on sys.path when running as a script ─────────────────
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from semantic_backbone.epanns.split_model import SplitEPANNs
from experiments_at2sed.starss_mapping import build_strong_to_starss_mapping


class AT2SEDModel(nn.Module):
    """
    E-PANNs backbone fine-tuned as a multi-label SED classifier.

    Args:
        cfg (dict): config dict.  Must contain a ``semantic`` section with
            ``checkpoint``, ``early_group``, ``late_group``, ``head_group``.
            ``model.n_cls`` sets the number of output classes:
            - 456 for AudioSet-Strong mode (default)
            - 13  for STARSS23 mode (set ``experiment.dataset: starss23``)

    Attributes:
        n_cls (int): number of output classes.
        starss_mode (bool): True when operating in STARSS23 classification mode.
        epanns (SplitEPANNs): backbone with per-group init/freeze support.
        head_starss (nn.Linear | None): 456→13 projection head (starss23 mode only).
    """

    # n_cls used by fc_audioset when backbone is in STARSS23 mode.
    # fc_audioset is always kept at its pretrained size (456) so that
    # the pretrained weights are preserved regardless of mode.
    _BACKBONE_N_CLS = 456

    def __init__(self, cfg: dict):
        super().__init__()

        dataset = cfg.get('experiment', {}).get('dataset', 'audioset_strong').lower()
        self.starss_mode = (dataset == 'starss23')

        # In STARSS23 mode the backbone fc_audioset stays at 456 (pretrained size)
        # and the 13-class projection is handled by head_starss.
        backbone_n_cls = self._BACKBONE_N_CLS if self.starss_mode else int(
            cfg.get('model', {}).get('n_cls', 527)
        )
        self.n_cls = int(cfg.get('model', {}).get('n_cls', backbone_n_cls))

        # ── Backbone ─────────────────────────────────────────────────────────
        sem_cfg          = dict(cfg.get('semantic', {}))
        sem_cfg['n_cls'] = backbone_n_cls
        self.epanns      = SplitEPANNs(sem_cfg)

        # Plain Python references (not re-registered) for use in forward().
        self._fc1, self._fc_audioset = self.epanns.get_head_modules()

        # ── STARSS23 projection head ──────────────────────────────────────────
        self.head_starss: nn.Linear | None = None
        if self.starss_mode:
            M = build_strong_to_starss_mapping(n_strong=self._BACKBONE_N_CLS)
            # M: (13, 456) soft-prior weight matrix
            self.head_starss = nn.Linear(self._BACKBONE_N_CLS, self.n_cls, bias=True)
            with torch.no_grad():
                self.head_starss.weight.copy_(M[:self.n_cls])
                self.head_starss.bias.zero_()

        # ── Parameter count ───────────────────────────────────────────────────
        _trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        _total     = sum(p.numel() for p in self.parameters())
        mode_str   = f'starss23({self.n_cls})' if self.starss_mode else f'as_strong({self.n_cls})'
        print(
            f'[AT2SEDModel] mode={mode_str}  '
            f'params: {_total:,} total / {_trainable:,} trainable'
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x : ``(B, T_samples)`` mono waveform at 32 kHz.

        Returns:
            logits : ``(B, n_cls)`` raw (pre-sigmoid) logits.
                     Shape is ``(B, 456)`` in audioset_strong mode,
                     ``(B, 13)`` in starss23 mode.
        """
        # ── Feature extraction ────────────────────────────────────────────────
        h_early = self.epanns.forward_early(x)        # (B, 256, T/8, F/8)
        h_late  = self.epanns.forward_late(h_early)   # (B, 1024, T/32, F/32)

        # ── Global pooling ────────────────────────────────────────────────────
        x_tf     = torch.mean(h_late, dim=3)           # (B, 1024, T/32)
        x_max, _ = torch.max(x_tf, dim=2)              # (B, 1024)
        x_mean   = torch.mean(x_tf, dim=2)             # (B, 1024)
        pooled   = x_max + x_mean                      # (B, 1024)

        # ── Head ─────────────────────────────────────────────────────────────
        out    = F.dropout(pooled, p=0.5, training=self.training)
        out    = F.relu_(self._fc1(out))               # (B, 2048)
        out    = F.dropout(out, p=0.5, training=self.training)
        logits = self._fc_audioset(out)                # (B, 456) always

        if self.starss_mode:
            logits = self.head_starss(logits)          # (B, 13)

        return logits
