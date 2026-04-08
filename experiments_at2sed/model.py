"""
AT2SEDModel — AudioSet-Tagging to SED Classifier
=================================================
Fine-tunes the GP-AT (E-PANNs) backbone as a multi-label SED classifier on
AudioSet-Strong.  The spatial pipeline and cross-stitch bridges are omitted;
only the semantic backbone is used.

Architecture::

    x : (B, T_samples)   mono waveform @ 32 kHz
        ↓ SplitEPANNs.forward_early()
          spectrogram → logmel → BN0 → SpecAugment(train) → ConvBlock1-3
    h_early : (B, 256, T/8, F/8)
        ↓ SplitEPANNs.forward_late()
          ConvBlock4-6
    h_late : (B, 1024, T/32, F/32)
        ↓ global pool: mean(F) → max(T) + mean(T)   (same as Cnn14Pruned.forward)
    pooled : (B, 1024)
        ↓ Dropout(0.5) → fc1(1024→2048) → ReLU → Dropout(0.5)
        ↓ fc_audioset(2048→n_cls)
    logits : (B, n_cls)   raw logits — use BCEWithLogitsLoss

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


class AT2SEDModel(nn.Module):
    """
    E-PANNs backbone fine-tuned as a multi-label SED classifier.

    Args:
        cfg (dict): config dict.  Must contain a ``semantic`` section with
            ``checkpoint``, ``early_group``, ``late_group``, ``head_group``.
            The ``model.n_cls`` key sets the number of output classes (default 527).

    Attributes:
        n_cls (int): number of output classes.
        epanns (SplitEPANNs): backbone with per-group init/freeze support.
    """

    def __init__(self, cfg: dict):
        super().__init__()

        self.n_cls = int(cfg.get('model', {}).get('n_cls', 527))

        # Build backbone — n_cls is injected so fc_audioset has the right size.
        # SplitEPANNs handles checkpoint loading and per-group init/frozen config.
        sem_cfg        = dict(cfg.get('semantic', {}))
        sem_cfg['n_cls'] = self.n_cls
        self.epanns    = SplitEPANNs(sem_cfg)

        # fc1 and fc_audioset are already registered inside self.epanns.backbone.
        # We keep plain Python references here so that forward() can call them
        # without double-registering the parameters.
        self._fc1, self._fc_audioset = self.epanns.get_head_modules()

        _trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        _total     = sum(p.numel() for p in self.parameters())
        print(
            f'[AT2SEDModel] n_cls={self.n_cls}  '
            f'params: {_total:,} total / {_trainable:,} trainable'
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x : ``(B, T_samples)`` mono waveform at 32 kHz.

        Returns:
            logits : ``(B, n_cls)`` raw (pre-sigmoid) logits.
                     Apply ``torch.sigmoid`` at inference or use
                     ``BCEWithLogitsLoss`` during training.
        """
        # ── Feature extraction ────────────────────────────────────────────────
        h_early = self.epanns.forward_early(x)   # (B, 256, T/8, F/8)
        h_late  = self.epanns.forward_late(h_early)  # (B, 1024, T/32, F/32)

        # ── Global pooling (identical to Cnn14Pruned.forward) ─────────────────
        # Step 1: collapse frequency dimension (mean over dim=3)
        x_tf = torch.mean(h_late, dim=3)          # (B, 1024, T/32)
        # Step 2: max-over-time + mean-over-time, element-wise sum
        x_max, _ = torch.max(x_tf, dim=2)         # (B, 1024)
        x_mean   = torch.mean(x_tf, dim=2)        # (B, 1024)
        pooled   = x_max + x_mean                 # (B, 1024)

        # ── Head ─────────────────────────────────────────────────────────────
        out = F.dropout(pooled, p=0.5, training=self.training)
        out = F.relu_(self._fc1(out))              # (B, 2048)
        out = F.dropout(out, p=0.5, training=self.training)
        logits = self._fc_audioset(out)            # (B, n_cls)

        return logits
