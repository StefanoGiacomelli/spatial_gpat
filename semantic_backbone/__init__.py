"""
semantic_backbone — Semantic Feature Extraction Public API
==========================================================
Re-exports the two public classes from the ``epanns`` sub-package so that
external code can import them with a single, clean path:

    from semantic_backbone import SplitEPANNs, Cnn14Pruned

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

from .epanns import SplitEPANNs, Cnn14Pruned

__all__ = ["SplitEPANNs", "Cnn14Pruned"]
