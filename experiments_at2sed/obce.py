"""
experiments_at2sed/obce.py — Ontology-Based Cross-Entropy Loss
==============================================================
Provenance
----------
Ablated from:
  ontology-aware-audio-tagging (Apache 2.0 licence)
  Original functions: ``ontology_binary_cross_entropy()``,
                      ``ontology_loss_weight()``

Adaptations
-----------
  * Accepts **logits** as input (sigmoid applied internally) to match the
    interface of ``SEDMultiLabelLoss`` and the rest of the AT2SED codebase.
  * Ontology weight matrix is encapsulated in the ``OBCELoss`` instance
    (no global state).
  * Works with the 456×456 AudioSet-Strong distance matrix produced by
    ``experiments_at2sed/build_strong_ontology.py``.

Formula
-------
.. math::

    OBCE = \\frac{BCE_{weighted}(\\sigma(logits), y) + BCE(\\sigma(logits), y)}{2}

where the per-element weight for position ``(b, c)`` is proportional to the
minimum ontological distance between class ``c`` and any positive class in
sample ``b``.  Positive classes receive weight 1; classes far from all
positives receive weight close to 0.

References
----------
H. Liu et al., "Ontology-aware Learning and Evaluation for Audio Tagging,"
arXiv 2206.03491, 2022.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_DEFAULT_DIST = (
    Path(__file__).resolve().parent.parent
    / 'configs' / 'datasets' / 'audioset_strong_graph_distance.npy'
)


class OBCELoss(nn.Module):
    """
    Ontology-Based Binary Cross-Entropy loss for AudioSet-Strong.

    Args:
        dist_path: Path to the ``(456, 456)`` float32 distance matrix
            produced by ``build_strong_ontology.py``.  Defaults to
            ``configs/datasets/audioset_strong_graph_distance.npy``.
        beta: Exponent applied to the normalised graph weights (default 1).
            Higher values make the weighting more aggressive.
    """

    def __init__(
        self,
        dist_path: Optional[Union[str, Path]] = None,
        beta: float = 1.0,
    ) -> None:
        super().__init__()

        p = Path(dist_path) if dist_path else _DEFAULT_DIST
        if not p.exists():
            raise FileNotFoundError(
                f'Distance matrix not found: {p}\n'
                f'Run:  .venv/bin/python experiments_at2sed/build_strong_ontology.py'
            )

        dist = torch.tensor(
            np.load(str(p)).astype(np.float32), dtype=torch.float32
        )
        dist_norm = dist / dist.max().clamp(min=1e-9)   # normalise to [0, 1]
        # Register as a buffer so it follows the module to the correct device
        self.register_buffer('graph_weight', dist_norm ** beta)   # (456, 456)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        logits:  torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute OBCE loss.

        Args:
            logits  : ``(B, C)`` raw (pre-sigmoid) model outputs.
            targets : ``(B, C)`` float32 multi-hot label tensors ∈ {0, 1}.

        Returns:
            Scalar OBCE loss.
        """
        probs  = torch.sigmoid(logits)
        weight = self._compute_weight(targets)   # (B, C)

        # Weighted BCE (per-element), then standard BCE
        bce_w = F.binary_cross_entropy(probs, targets, reduction='none')
        bce   = F.binary_cross_entropy(probs, targets)

        return (torch.mean(bce_w * weight) + bce) / 2.0

    # ── Weight computation ────────────────────────────────────────────────────

    def _compute_weight(self, target: torch.Tensor) -> torch.Tensor:
        """
        Compute per-element ontology weights for the batch.

        For each sample b and class c:
          ``weight[b, c] = min_{k ∈ positives(b)}( graph_weight[k, c] )``

        Classes with no positive in the sample get weight 0.
        Positive classes always receive weight 1 (not discounted by distance).
        The weights are normalised so their mean across the batch is 1.

        Args:
            target: ``(B, C)`` float32 multi-hot labels ∈ {0, 1}.

        Returns:
            ``(B, C)`` float32 weight tensor.
        """
        gw = self.graph_weight   # (C, C)
        weight_rows = []

        for i in range(target.shape[0]):
            # target[i] ∈ {0, 1}^C — multiply each class weight column by
            # the label: effectively selects weights for positive classes only.
            # result shape: (1, C) * (C, C) → (C, C)
            # Column j of result: distances from all positive classes to class j
            res = target[i : i + 1] * gw           # (1, C) broadcast → (C, C)
            res[res == 0] = float('inf')            # zeroes = not positive → ignore
            w = res.min(dim=0).values               # (C,) min dist from any positive
            weight_rows.append(w.unsqueeze(0))

        weight = torch.cat(weight_rows, dim=0)      # (B, C)

        # Samples where the target is all-zero would have all-inf → zero out
        weight[weight == float('inf')] = 0.0

        # Row-normalise: within each sample, scale to [0, 1]
        row_max = weight.max(dim=1, keepdim=True).values.clamp(min=1e-9)
        weight  = weight / row_max

        # Positive labels always get weight 1
        weight[target > 0] = 1.0

        # Global normalisation: mean weight = 1 (matches original)
        mean_w = weight.mean().clamp(min=1e-9)
        weight = weight / mean_w

        return weight
