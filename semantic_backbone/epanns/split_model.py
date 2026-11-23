"""
SplitEPANNs — Per-Group Freeze/Init Wrapper for E-PANNs
=========================================================
Wraps ``Cnn14Pruned`` and exposes two intermediate feature maps needed by the
NAS cross-stitch pipeline:

  h_early : (B, 256, T/8,  F/8)   — after conv_block3
  h_late  : (B, 1024, T/32, F/32) — after conv_block6 (50 % pruning)

Three parameter groups can be independently configured:

  early_group : spectrogram extraction + bn0 + conv_block1-3
  late_group  : conv_block4-6
  head_group  : fc1 + fc_audioset

Each group is controlled by two YAML keys:

  init   : pretrained | random   (default pretrained; random re-initialises)
  frozen : true | false           (default false; true freezes requires_grad)

References
----------
A. Singh Boparai, K. Drossos, and T. Heittola,
"Efficient Pre-trained Audio Neural Networks via Passive Filter Pruning,"
Inter-Noise 2023.
https://github.com/Arshdeep-Singh-Boparai/E-PANNs

A. Singh and M. D. Plumbley
"Efficient CNNs via Passive Filter Pruning"
IEEE TASLP 2025.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Cnn14Pruned, PRUNING_RATIOS, init_layer, init_bn


# ============================================================
# HELPERS
# ============================================================

def _random_init_module(m: nn.Module) -> None:
    """Apply random (Xavier/constant) re-initialization to a single layer.

    Supports ``nn.Conv2d``, ``nn.BatchNorm2d``, and ``nn.Linear``.
    Layers of other types are silently skipped.
    """
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.)
        m.bias.data.fill_(0.)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.)


# ============================================================
# SplitEPANNs
# ============================================================

class SplitEPANNs(nn.Module):
    """
    E-PANNs backbone split into three independently configurable groups.

    Args:
        cfg (dict): parsed from the ``semantic`` section of an experiment configuration
            (configs/ YAML or a stored hparams.yaml ``cfg`` block).
            Expected keys:
              checkpoint   (str)  : path to pretrained weights (optional)
              n_cls        (int)  : number of output classes
              early_group  (dict) : {init: pretrained|random, frozen: bool}
              late_group   (dict) : {init: pretrained|random, frozen: bool}
              head_group   (dict) : {init: pretrained|random, frozen: bool}

    Forward input:
        x : (B, T_samples) — mono waveform at 32 kHz

    Forward output:
        h_early : (B, 256,  T_mel/8,  F_mel/8)
        h_late  : (B, 1024, T_mel/32, F_mel/32)
    """

    def __init__(self, cfg: dict):
        """Initialize SplitEPANNs and apply per-group init/freeze configs."""
        super().__init__()

        n_cls      = cfg.get('n_cls', 10)
        ckpt_path  = cfg.get('checkpoint', None)

        # ── Build backbone ─────────────────────────────────────────────
        self.backbone = Cnn14Pruned(classes_num=n_cls, **PRUNING_RATIOS)

        # ── Define group → module mapping ──────────────────────────────
        bb = self.backbone
        self._group_modules: Dict[str, List[nn.Module]] = {
            'early_group': [bb.bn0, bb.conv_block1, bb.conv_block2, bb.conv_block3],
            'late_group':  [bb.conv_block4, bb.conv_block5, bb.conv_block6],
            'head_group':  [bb.fc1, bb.fc_audioset],
        }
        # spectrogram_extractor / logmel_extractor / spec_augmenter are
        # freeze_parameters=True internally; no learnable weights to manage.

        # ── Load pretrained weights if a checkpoint path is given ──────
        # Load unconditionally first; random-init groups are overwritten below.
        pretrained_class_map = cfg.get('pretrained_class_map', None)
        model_class_map      = cfg.get('model_class_map', None)
        if ckpt_path is not None and os.path.isfile(ckpt_path):
            self._load_checkpoint(ckpt_path, n_cls, pretrained_class_map, model_class_map)

        # ── Apply per-group init / freeze config ───────────────────────
        for group_key in ('early_group', 'late_group', 'head_group'):
            gcfg = cfg.get(group_key, {})
            self._apply_group_config(group_key, gcfg)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_checkpoint(
        self,
        path: str,
        n_cls: int,
        pretrained_class_map: str = None,
        model_class_map: str = None,
    ) -> None:
        state = torch.load(path, map_location='cpu', weights_only=True)
        ckpt_cls = state.get('fc_audioset.weight')
        model_cls_shape = self.backbone.fc_audioset.weight.shape

        if ckpt_cls is not None and ckpt_cls.shape != model_cls_shape:
            # Load everything except fc_audioset (shape mismatch)
            filtered = {k: v for k, v in state.items()
                        if not k.startswith('fc_audioset')}
            self.backbone.load_state_dict(filtered, strict=False)
            print(f"[SplitEPANNs] Loaded backbone (partial) from {path}")
            print(f"  ckpt classes={ckpt_cls.shape[0]}  model classes={model_cls_shape[0]}")

            # Selective weight transfer for shared classes if both maps are provided
            if (pretrained_class_map and model_class_map
                    and os.path.isfile(pretrained_class_map)
                    and os.path.isfile(model_class_map)):
                n_transferred = self._transfer_fc_audioset(
                    state, pretrained_class_map, model_class_map
                )
                print(
                    f"  fc_audioset: {n_transferred} neurons transferred from pretrained, "
                    f"{model_cls_shape[0] - n_transferred} randomly initialised"
                )
            else:
                print("  fc_audioset: randomly initialised (no class maps provided)")
        else:
            self.backbone.load_state_dict(state)
            print(f"[SplitEPANNs] Loaded pretrained weights from {path}")

    def _transfer_fc_audioset(
        self,
        ckpt_state: dict,
        pretrained_class_map_csv: str,
        model_class_map_csv: str,
    ) -> int:
        """
        Copy fc_audioset rows from the checkpoint for shared classes.

        Reads the AudioSet Standard map (mid → old_idx) and the Strong model
        map (new_idx → mid), then copies the corresponding weight/bias rows
        from the checkpoint into the backbone's fc_audioset layer.

        Classes not found in the checkpoint are left with their Xavier init.

        Args:
            ckpt_state             : raw state_dict from ``torch.load``
            pretrained_class_map_csv : path to ``class_labels_indices.csv``
                                       (AudioSet Standard, 527 classes)
            model_class_map_csv    : path to ``audioset_strong_label_map.csv``
                                       (456 classes)

        Returns:
            Number of neurons successfully transferred from the checkpoint.
        """
        import csv as _csv

        # AudioSet Standard: mid → index in the pretrained checkpoint
        std_mid2idx: dict = {}
        with open(pretrained_class_map_csv) as fh:
            for row in _csv.DictReader(fh):
                std_mid2idx[row['mid'].strip()] = int(row['index'])

        # Strong model map: new_idx → mid
        new_idx2mid: dict = {}
        with open(model_class_map_csv) as fh:
            for row in _csv.DictReader(fh):
                new_idx2mid[int(row['index'])] = row['mid'].strip()

        ckpt_w = ckpt_state.get('fc_audioset.weight')
        ckpt_b = ckpt_state.get('fc_audioset.bias')
        fc     = self.backbone.fc_audioset

        n_transferred = 0
        with torch.no_grad():
            for new_idx, mid in new_idx2mid.items():
                old_idx = std_mid2idx.get(mid)
                if old_idx is None:
                    continue   # Strong-only class → keep Xavier init
                if ckpt_w is not None and old_idx < ckpt_w.shape[0]:
                    fc.weight[new_idx].copy_(ckpt_w[old_idx])
                if ckpt_b is not None and old_idx < ckpt_b.shape[0]:
                    fc.bias[new_idx].copy_(ckpt_b[old_idx])
                n_transferred += 1

        return n_transferred

    def _apply_group_config(self, group_key: str, gcfg: dict) -> None:
        modules = self._group_modules[group_key]

        # Random re-init (overrides loaded weights for this group)
        if gcfg.get('init', 'pretrained') == 'random':
            for m in modules:
                m.apply(_random_init_module)

        # Freeze
        if gcfg.get('frozen', False):
            for m in modules:
                for p in m.parameters():
                    p.requires_grad = False

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_early(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run spectrogram preprocessing + conv_block1-3.

        Args:
            x : (B, T_samples) mono waveform at 32 kHz

        Returns:
            h_early : (B, 256, T_mel/8, F_mel/8)
        """
        bb = self.backbone

        x = bb.spectrogram_extractor(x)   # (B, 1, T_mel, freq_bins)
        x = bb.logmel_extractor(x)        # (B, 1, T_mel, mel_bins=64)

        x = x.transpose(1, 3)
        x = bb.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            try:
                x = bb.spec_augmenter(x)
            except RuntimeError:
                pass  # spectrogram too short for time/freq masking — skip augmentation

        x = bb.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = bb.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = bb.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        return x                          # (B, 256, T/8, F/8)

    def get_head_modules(self) -> Tuple[nn.Linear, nn.Linear]:
        """Return (fc1, fc_audioset) for weight-copying into NASModel."""
        return self.backbone.fc1, self.backbone.fc_audioset

    def forward_late(self, h_early: torch.Tensor) -> torch.Tensor:
        """
        Run conv_block4-6 from an intermediate h_early tensor.

        This allows CrossStitch bridge #1 to enrich h_early before feeding
        it into the late group, which is the correct bidirectional design.

        Args:
            h_early : (B, 256, T_mel/8, F_mel/8)

        Returns:
            h_late : (B, 1024, T_mel/32, F_mel/32)
        """
        bb = self.backbone
        x = h_early

        x = bb.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = bb.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = bb.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        return x                          # (B, 1024, T/32, F/32)

