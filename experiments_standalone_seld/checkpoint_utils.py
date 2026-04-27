"""
checkpoint_utils — NAS-SELD → StandaloneSELD Weight Extraction
===============================================================
``filter_checkpoint`` loads a NAS-SELD checkpoint (NASLightningModule)
and produces a state_dict compatible with ``StandaloneSELDModel``.

Filtering rules
---------------
1. Drop: semantic backbone + bridge keys (GP-AT components).
2. Strip: ``model.`` prefix from remaining keys.
3. Rename (symmetric — both sed_branch and doa_branch get copies):
     spatial_early.{i}.{w}  → sed_branch.early.{i}.{w}
                               doa_branch.early.{i}.{w}
     spatial_late.{i}.{w}   → sed_branch.late.{i}.{w}
                               doa_branch.late.{i}.{w}
     smoother_spatial.{w}   → sed_branch.smoother.{w}
                               doa_branch.smoother.{w}
     input_norm_mag.{w}     → sed_branch.norm_mag.{w}
                               doa_branch.norm_mag.{w}
     input_norm_phase.{w}   → sed_branch.norm_phase.{w}
                               doa_branch.norm_phase.{w}
     input_norm_iv.{w}      → sed_branch.norm_iv.{w}
                               doa_branch.norm_iv.{w}
     input_norm_gcc.{w}     → sed_branch.norm_gcc.{w}
                               doa_branch.norm_gcc.{w}
4. Rename (asymmetric — each branch gets its own head):
     head_sed.{w}           → sed_branch.head.{w}
     head_doa.{w}           → doa_branch.head.{w}
5. Keep as-is: input_proc.{w}

Usage
-----
    from experiments_standalone_seld.checkpoint_utils import filter_checkpoint
    sd = filter_checkpoint('runs/foa/deep_spec/checkpoints/exp_e3_l1_s1/98_0.3570.ckpt')
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Expected missing: cs_early.stitch / cs_late.stitch (when cross-stitch enabled)
    # Expected unexpected: none

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

from collections import OrderedDict
from pathlib import Path
from typing import Union

import torch


# Keys to drop (prefix match, after stripping the outer Lightning 'model.' prefix)
_DROP_PREFIXES = (
    'split_epanns.',
    'sem_fc1.',
    'sem_fc_audioset.',
    'head_semantic.',
    'bridge_early.',
    'bridge_late.',
    'mhamgu_expand.',
)

# Symmetric renames: one NAS key → two StandaloneSELD keys (one per branch)
_SYMMETRIC_PREFIXES = [
    ('spatial_early.',   'sed_branch.early.',   'doa_branch.early.'),
    ('spatial_late.',    'sed_branch.late.',    'doa_branch.late.'),
    ('smoother_spatial.','sed_branch.smoother.','doa_branch.smoother.'),
    ('input_norm_mag.',  'sed_branch.norm_mag.','doa_branch.norm_mag.'),
    ('input_norm_phase.','sed_branch.norm_phase.','doa_branch.norm_phase.'),
    ('input_norm_iv.',   'sed_branch.norm_iv.', 'doa_branch.norm_iv.'),
    ('input_norm_gcc.',  'sed_branch.norm_gcc.','doa_branch.norm_gcc.'),
]

# Asymmetric renames: one NAS key → one StandaloneSELD key
_ASYMMETRIC_PREFIXES = {
    'head_sed.': 'sed_branch.head.',
    'head_doa.': 'doa_branch.head.',
    'input_proc.': 'input_proc.',   # kept as-is (top-level attribute)
}


def filter_checkpoint(
    ckpt_path: Union[str, Path],
    map_location: str = 'cpu',
    verbose: bool = True,
) -> OrderedDict:
    """
    Load a NAS-SELD checkpoint and return a StandaloneSELDModel-compatible state_dict.

    Args:
        ckpt_path    : path to the NAS-SELD ``.ckpt`` file.
        map_location : torch device for loading (default 'cpu').
        verbose      : print key statistics when True.

    Returns:
        OrderedDict ready for ``StandaloneSELDModel.load_state_dict(strict=False)``.
    """
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=True)

    # NASLightningModule saves under 'state_dict' key
    raw: dict = ckpt.get('state_dict', ckpt)

    out = OrderedDict()
    n_dropped  = 0
    n_kept     = 0
    n_sym      = 0
    n_asym     = 0
    n_unknown  = 0

    for raw_key, tensor in raw.items():
        # Only process 'model.*' keys; skip optimizer/scheduler state
        if not raw_key.startswith('model.'):
            continue

        # Strip Lightning wrapper prefix 'model.'
        key = raw_key[len('model.'):]

        # 1. Drop GP-AT backbone and bridge keys
        if any(key.startswith(pfx) for pfx in _DROP_PREFIXES):
            n_dropped += 1
            continue

        # 2. Symmetric rename (early/late/smoother/norms → both branches)
        matched_sym = False
        for nas_pfx, sed_pfx, doa_pfx in _SYMMETRIC_PREFIXES:
            if key.startswith(nas_pfx):
                suffix = key[len(nas_pfx):]
                out[sed_pfx + suffix] = tensor
                out[doa_pfx + suffix] = tensor.clone()
                n_sym += 2
                matched_sym = True
                break
        if matched_sym:
            continue

        # 3. Asymmetric rename (heads, input_proc)
        matched_asym = False
        for nas_pfx, new_pfx in _ASYMMETRIC_PREFIXES.items():
            if key.startswith(nas_pfx):
                suffix = key[len(nas_pfx):]
                out[new_pfx + suffix] = tensor
                n_asym += 1
                matched_asym = True
                break
        if matched_asym:
            n_kept += 1
            continue

        # 4. Unknown key — warn but skip (e.g. tpit_loss buffers)
        n_unknown += 1
        if verbose:
            print(f'[filter_checkpoint] SKIP unmatched key: {key}')

    if verbose:
        print(f'[filter_checkpoint] dropped={n_dropped}  '
              f'sym_keys={n_sym}  asym_keys={n_asym}  '
              f'skipped_unknown={n_unknown}  '
              f'total_output_keys={len(out)}')

    return out
