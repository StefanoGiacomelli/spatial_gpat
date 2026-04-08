"""
experiments_at2sed — AudioSet-Strong SED Fine-tuning
=====================================================
Pipeline for fine-tuning the GP-AT backbone (E-PANNs) as a multi-label classifier
on AudioSet-Strong.  Designed to study the degradation of performance as the 
input window length varies, with a perspective of pseudo-real-time usage.    

Modules
-------
AT2SEDModel              (model.py) — E-PANNs → global pool → logits (B, n_cls)
AT2SEDLightningModule    (train.py) — training/val/test loop with running logging
SEDMultiLabelLoss        (loss.py)  — BCEWithLogitsLoss withpos_weight per class
SEDMultiLabelMetrics     (loss.py)  — mAP, AUROC, F1 macro/micro, Precision, Recall

Usage
-----
    from experiments_at2sed import AT2SEDModel, AT2SEDLightningModule

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

from .model import AT2SEDModel
from .loss  import SEDMultiLabelLoss, SEDMultiLabelMetrics
from .train import AT2SEDLightningModule

__all__ = [
    'AT2SEDModel',
    'SEDMultiLabelLoss',
    'SEDMultiLabelMetrics',
    'AT2SEDLightningModule',
]
