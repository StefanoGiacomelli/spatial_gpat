#!/usr/bin/env python3
"""
analysis/profiling/param_accounting.py
==============================================================================
Parameter accounting of an AT2SELD configuration by instantiation (CPU, no data)
==============================================================================

Instantiates ``experiments_nas_seld.nas_model.NASModel`` from a configuration
(a stored ``hparams.yaml`` or a YAML under ``configs/``) and prints/exports the
breakdown reported for the final model:

    total parameters
      = semantic branch  (E-PANNs backbone as stored in ``split_epanns`` *plus* the
                          fine-tunable copies ``sem_fc1`` / ``sem_fc_audioset`` and
                          ``head_semantic``)
      + spatial SELD path (front end, early/late modules, smoother, SED/DOA heads)
      + cross-stitch bridge(s)
    task-facing ("effective SELD") = spatial + bridge
    non-trainable parameters (``requires_grad = False``: torchlibrosa STFT bases and
    mel filterbank, which PyTorch counts as parameters)

The numbers are properties of the architecture only.  For the ExpA configuration
the expected values are (``tests/test_param_accounting.py``):

    total 30,986,080 = 27,475,182 (24,289,295 E-PANNs + 3,179,023 copied heads
    + 6,864 semantic head) + 2,932,082 spatial + 578,816 bridge; non-trainable 1,083,456.

Usage
-----
    python analysis/profiling/param_accounting.py \\
        --config configs/loss_experiments/expA_nas_seld_balanced_starss23.yaml \\
        [--json out.json]

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPATIAL_PREFIXES = (
    "input_proc", "input_norm_mag", "input_norm_phase", "input_norm_iv",
    "input_norm_gcc", "spatial_early", "spatial_late", "smoother_spatial",
    "mhamgu_expand", "head_sed", "head_doa",
)
SEMANTIC_PREFIXES = ("split_epanns", "sem_fc1", "sem_fc_audioset", "head_semantic")
BRIDGE_PREFIXES = ("bridge_early", "bridge_late")


def load_cfg(path: Path) -> dict:
    with Path(path).open() as fh:
        raw = yaml.safe_load(fh)
    return raw.get("cfg", raw)


def build_model(cfg: dict) -> nn.Module:
    """Instantiate ``NASModel`` (pretrained E-PANNs initialisation, CPU, eval).

    The semantic checkpoint path of the configuration is resolved relative to
    the repository root, as the training runner does.
    """
    from experiments_nas_seld.nas_model import NASModel

    cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = NASModel(cfg)
    finally:
        os.chdir(cwd)
    return model.eval()


def breakdown(model: nn.Module) -> Dict[str, int]:
    """Unique-parameter counts by ownership group (and trainable / non-trainable)."""
    out = {
        "total": 0, "trainable": 0, "non_trainable": 0,
        "semantic_total": 0, "semantic_epanns_backbone": 0,
        "semantic_sem_fc1": 0, "semantic_sem_fc_audioset": 0, "semantic_head": 0,
        "spatial_seld": 0, "bridge": 0, "other": 0,
    }
    seen = set()
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        n = p.numel()
        out["total"] += n
        out["trainable" if p.requires_grad else "non_trainable"] += n
        if name.startswith("split_epanns"):
            out["semantic_epanns_backbone"] += n
            out["semantic_total"] += n
        elif name.startswith("sem_fc1"):
            out["semantic_sem_fc1"] += n
            out["semantic_total"] += n
        elif name.startswith("sem_fc_audioset"):
            out["semantic_sem_fc_audioset"] += n
            out["semantic_total"] += n
        elif name.startswith("head_semantic"):
            out["semantic_head"] += n
            out["semantic_total"] += n
        elif name.startswith(SPATIAL_PREFIXES):
            out["spatial_seld"] += n
        elif name.startswith(BRIDGE_PREFIXES):
            out["bridge"] += n
        else:
            out["other"] += n
    out["task_facing_seld"] = out["spatial_seld"] + out["bridge"]
    out["task_facing_share_percent"] = round(100.0 * out["task_facing_seld"] / out["total"], 2)
    buf = sum(b.numel() * b.element_size() for b in model.buffers())
    par = sum(p.numel() * p.element_size() for p in model.parameters())
    out["memory_MB_params_plus_buffers"] = round((par + buf) / 2**20, 3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("Usage")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    model = build_model(cfg)
    res = breakdown(model)
    width = max(len(k) for k in res)
    print(f"configuration: {args.config}")
    for k, v in res.items():
        print(f"  {k:<{width}}  {v:>14,}" if isinstance(v, int) else f"  {k:<{width}}  {v:>14}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"config": str(args.config), **res}, indent=1))


if __name__ == "__main__":
    main()
