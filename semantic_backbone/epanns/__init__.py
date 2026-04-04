"""
semantic_backbone.epanns — E-PANNs Public API
==============================================
Exports the two classes needed by external consumers of this sub-package:

  SplitEPANNs  — pipeline backbone (three independently configurable groups)
  Cnn14Pruned  — underlying pruned CNN14 (useful for inspection/transfer)

``ConvBlock`` and ``ConvBlockPruned`` are internal building blocks and are
not re-exported here to avoid name collision with
``spatial_modules.seldnet_modules.ConvBlock``.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

from .split_model import SplitEPANNs
from .model import Cnn14Pruned

__all__ = ["SplitEPANNs", "Cnn14Pruned"]
