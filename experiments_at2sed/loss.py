"""
SED Multi-Label Loss and Metrics — AudioSet-Strong Fine-tuning
==============================================================
Two components for training and evaluating AT2SEDModel:

  SEDMultiLabelLoss      — BCEWithLogitsLoss with per-class positive weights
                           for class-imbalance correction.

  SEDMultiLabelMetrics   — torchmetrics-backed accumulator for multi-label SED.
                           Reports mAP (macro), AUROC (macro), F1 (macro and micro),
                           Precision (macro), and Recall (macro).

Loss design
-----------
.. math::

    L = \\text{BCE}_{\\text{logits}}(\\hat{y}, y, \\text{pos\\_weight})

where ``logits`` :math:`\\in \\mathbb{R}^{B \\times C}` are raw (pre-sigmoid) outputs
and ``targets`` :math:`\\in \\{0, 1\\}^{B \\times C}` are multi-hot label vectors.

Metric design
-------------
Primary checkpoint metric: **mAP macro** (``val/map_macro``, higher is better).
This is the official AudioSet evaluation metric and does not depend on a
classification threshold, making it well-suited for the imbalanced multi-label
setting.

Micro-averaged F1 is also reported because it naturally accounts for class
frequency, giving a complementary view alongside the class-equitable macro-F1.

References
----------
J. F. Gemmeke et al., "Audio Set: An ontology and human-labeled dataset for
audio events," ICASSP 2017.

Q. Kong et al., "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio
Pattern Recognition," IEEE/ACM TASLP 2020.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import numpy as np
from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelAUROC,
    MultilabelAveragePrecision,
    MultilabelConfusionMatrix,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)


# ══════════════════════════════════════════════════════════════════════════════
# SEDMultiLabelLoss
# ══════════════════════════════════════════════════════════════════════════════

class SEDMultiLabelLoss(nn.Module):
    """
    Multi-label binary cross-entropy loss for SED.

    Wraps ``BCEWithLogitsLoss`` and optionally applies per-class positive
    weights for class-imbalance correction.

    Args:
        pos_weight (Tensor, optional): ``(n_cls,)`` positive-class weights.
            Passed directly to ``BCEWithLogitsLoss(pos_weight=...)``.
            Registered as a buffer so it follows the module to the correct device.
    """

    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        # Always register as a buffer (None is valid) so that a subsequent
        # register_buffer call in set_pos_weight() can update it without
        # raising KeyError("attribute 'pos_weight' already exists").
        self.register_buffer('pos_weight', pos_weight)

    def forward(
        self,
        logits:  torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute mean BCE loss over batch and classes.

        Args:
            logits  : ``(B, C)`` raw (pre-sigmoid) model outputs.
            targets : ``(B, C)`` float32 multi-hot label tensors ∈ {0, 1}.

        Returns:
            Scalar mean BCE loss.
        """
        return nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight = self.pos_weight,
        )

    def set_pos_weight(self, pos_weight: torch.Tensor) -> None:
        """
        Inject or update the positive-class weights after construction.

        Called by ``main_sed_only.py`` once the DataModule has computed
        the training-set statistics.

        Args:
            pos_weight : ``(n_cls,)`` float32 Tensor.
        """
        if hasattr(self, 'pos_weight') and self.pos_weight is not None:
            self.pos_weight.copy_(pos_weight)
        else:
            self.register_buffer('pos_weight', pos_weight)


# ══════════════════════════════════════════════════════════════════════════════
# SEDMultiLabelMetrics
# ══════════════════════════════════════════════════════════════════════════════

class SEDMultiLabelMetrics(nn.Module):
    """
    torchmetrics-backed accumulator for multi-label SED evaluation.

    Being an ``nn.Module`` ensures Lightning moves the internal torchmetrics
    objects to the correct device automatically.

    Reported metrics
    ----------------
    =====================  ======  =========================================
    Key                    Avg     Notes
    =====================  ======  =========================================
    ``map_macro``          macro   Primary AudioSet metric; threshold-free
    ``auroc_macro``        macro   Threshold-free; robust to class imbalance
    ``f1_macro``           macro   Equal weight per class; fairer for rare
    ``f1_micro``           micro   Frequency-weighted; complementary view
    ``precision_macro``    macro   |
    ``recall_macro``       macro   |
    ``accuracy``           micro   Exact-match multi-label accuracy
    ``confusion_matrix``   —       (n_cls, 2, 2) per-class TP/FP/TN/FN
    =====================  ======  =========================================

    The ``confusion_matrix`` key is only present in the dict returned by
    :meth:`compute` when ``include_cm=True`` is passed.  It is a
    ``(n_cls, 2, 2)`` int64 NumPy array, not logged to TensorBoard.

    Args:
        n_cls (int): number of target classes (e.g. 527).
        threshold (float): sigmoid threshold for binary predictions in
            precision/recall/F1 (default 0.5).

    Usage::

        metrics = SEDMultiLabelMetrics(n_cls=527)
        metrics.update(logits, targets)   # accumulate over batches
        results = metrics.compute()       # flush and return dict
        metrics.reset()
    """

    def __init__(self, n_cls: int, threshold: float = 0.5):
        super().__init__()
        kw_thr = dict(num_labels=n_cls, threshold=threshold)
        kw_raw = dict(num_labels=n_cls)

        # Threshold-free metrics (operate on probabilities)
        self.map_macro        = MultilabelAveragePrecision(average='macro',  **kw_raw)
        self.auroc_macro      = MultilabelAUROC(average='macro',             **kw_raw)

        # Threshold-dependent metrics
        self.f1_macro         = MultilabelF1Score(average='macro',           **kw_thr)
        self.f1_micro         = MultilabelF1Score(average='micro',           **kw_thr)
        self.precision_macro  = MultilabelPrecision(average='macro',         **kw_thr)
        self.recall_macro     = MultilabelRecall(average='macro',            **kw_thr)
        self.accuracy         = MultilabelAccuracy(average='micro',          **kw_thr)
        self.confusion_matrix = MultilabelConfusionMatrix(                   **kw_thr)

    def reset(self) -> None:
        """Reset all internal accumulators."""
        for m in (
            self.map_macro, self.auroc_macro,
            self.f1_macro, self.f1_micro,
            self.precision_macro, self.recall_macro,
            self.accuracy, self.confusion_matrix,
        ):
            m.reset()

    @torch.no_grad()
    def update(
        self,
        logits:  torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """
        Accumulate predictions for a batch.

        Args:
            logits  : ``(B, C)`` raw (pre-sigmoid) model outputs.
            targets : ``(B, C)`` float32 multi-hot labels ∈ {0, 1}.
        """
        probs   = torch.sigmoid(logits)
        targets = targets.long()    # torchmetrics expects integer targets

        self.map_macro.update(probs, targets)
        self.auroc_macro.update(probs, targets)
        self.f1_macro.update(probs, targets)
        self.f1_micro.update(probs, targets)
        self.precision_macro.update(probs, targets)
        self.recall_macro.update(probs, targets)
        self.accuracy.update(probs, targets)
        self.confusion_matrix.update(probs, targets)

    def compute(self, include_cm: bool = False) -> Dict[str, object]:
        """
        Flush all accumulators and return a metrics dict.

        Args:
            include_cm: if True, adds ``'confusion_matrix'`` key (numpy array,
                not suitable for TensorBoard logging).

        Returns:
            dict with float keys ``map_macro``, ``auroc_macro``, ``f1_macro``,
            ``f1_micro``, ``precision_macro``, ``recall_macro``, ``accuracy``.
            Optionally ``confusion_matrix`` → ``(n_cls, 2, 2)`` int64 ndarray.
        """
        results: Dict[str, object] = {
            'map_macro':       float(self.map_macro.compute()),
            'auroc_macro':     float(self.auroc_macro.compute()),
            'f1_macro':        float(self.f1_macro.compute()),
            'f1_micro':        float(self.f1_micro.compute()),
            'precision_macro': float(self.precision_macro.compute()),
            'recall_macro':    float(self.recall_macro.compute()),
            'accuracy':        float(self.accuracy.compute()),
        }
        if include_cm:
            results['confusion_matrix'] = self.confusion_matrix.compute().cpu().numpy()
        return results
