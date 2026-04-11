"""
experiments_at2sed/hlp.py — Hierarchical Label Propagation for AudioSet-Strong
===============================================================================
Provenance
----------
Ablated from:
  hierarchical-label-propagation (MIT licence)
  Original class: ``HLP``, method: ``propagate()``
  Original formula: ``(targets.unsqueeze(-1) * table).max(dim=1).values``

Adapted for AudioSet-Strong 456 classes using a pre-built 456×456 submatrix
of the AudioSet Standard 527×527 lookup table.  See
``experiments_at2sed/build_strong_ontology.py`` for how the submatrix is built.

Usage
-----
The lookup table is loaded lazily: the file must be present before the first
call to ``propagate()``.  Build it once with::

    .venv/bin/python experiments_at2sed/build_strong_ontology.py

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

from pathlib import Path
from typing import Optional, Union

import torch

_DEFAULT_TABLE = (
    Path(__file__).resolve().parent.parent
    / 'configs' / 'datasets' / 'audioset_strong_hlp.pt'
)


class StrongHLP:
    """
    Hierarchical Label Propagation for AudioSet-Strong (456 classes).

    Given a multi-hot label vector, propagates activity upward through the
    AudioSet ontology: if class ``i`` is active, all unambiguous ancestors
    ``j`` of ``i`` are also set to active (or, for continuous scores, raised
    to the maximum score of any descendant).

    Args:
        table_path: Path to the ``(456, 456)`` bool tensor produced by
            ``build_strong_ontology.py``.  Defaults to
            ``configs/datasets/audioset_strong_hlp.pt`` relative to the
            project root.
    """

    def __init__(self, table_path: Optional[Union[str, Path]] = None) -> None:
        p = Path(table_path) if table_path else _DEFAULT_TABLE
        if not p.exists():
            raise FileNotFoundError(
                f'HLP lookup table not found: {p}\n'
                f'Run:  .venv/bin/python experiments_at2sed/build_strong_ontology.py'
            )
        # Store as float32 so it can multiply with float targets without casting
        self._table: torch.Tensor = torch.load(str(p), weights_only=True).float()
        assert self._table.shape == (456, 456), \
            f'Expected (456, 456) table, got {tuple(self._table.shape)}'

    # ── Public API ────────────────────────────────────────────────────────────

    def propagate(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Apply HLP to a label vector or a batch of label vectors.

        For binary targets this sets every ancestor class to 1 whenever a
        descendant is active.  For continuous scores it raises each ancestor
        to the maximum score of any of its active descendants — useful as a
        post-processing step on model outputs.

        Args:
            targets: ``(456,)`` or ``(B, 456)`` float32 tensor.

        Returns:
            Propagated tensor of the same shape and dtype as ``targets``.
        """
        if targets.dim() not in (1, 2):
            raise ValueError(
                f'Expected 1D or 2D targets, got {targets.dim()}D'
            )

        # Ensure the table is on the same device as targets
        table = self._table.to(targets.device)

        was_1d = targets.dim() == 1
        t = targets.unsqueeze(0) if was_1d else targets   # (B, 456)

        # Original HLP formula: for each class j, take the max score among
        # all classes i for which j is an ancestor (table[i, j] = 1).
        # targets_HLP[b, j] = max_i( t[b, i] * table[i, j] )
        t_out = (t.unsqueeze(-1) * table).max(dim=1).values   # (B, 456)

        return t_out.squeeze(0) if was_1d else t_out
