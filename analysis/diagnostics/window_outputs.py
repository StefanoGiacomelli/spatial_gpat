#!/usr/bin/env python3
"""
analysis/diagnostics/window_outputs.py
==============================================================================
Window-level inference cache for AT2SELD checkpoints (inference only)
==============================================================================

Runs one NAS-SELD checkpoint over every 10-s window (hop 2.5 s) of one split
of one dataset and stores the *raw network outputs* of every window on disk:

    logits.npy   float32  (W, T, N, C)      SED logits
    doa.npy      float32  (W, T, N, C, 3)   tanh-bounded DOA vectors
    perm.npy     int8     (W, T, C, N)      optional, ``--tpit-perm``: best
                                            tPIT permutation w.r.t. the
                                            reference (slot of the prediction
                                            matched to target slot n; the
                                            same convention as
                                            ``tPITLoss.last_best_perm``)
    meta.json    provenance (checkpoint, config, dataset, split, seeds,
                 window items, device, timing, machine load)

The cache is the common input of the additional analyses: metric variants
(``run_metric_variants.py``), subset-matched native references
(``native_subset_references.py``) and threshold-policy tables.  Nothing is
trained or updated; targets are *not* stored (they are a deterministic
function of the dataset and are rebuilt with ``window_target``).

Usage
-----
    python analysis/diagnostics/window_outputs.py \\
        --name expA_starss23_test \\
        --config runs/at2seld_nas/loss_experiments/balanced_starss23_expA/checkpoints/expA_nas_seld_balanced_starss23/version_0/hparams.yaml \\
        --checkpoint runs/at2seld_nas/loss_experiments/balanced_starss23_expA/checkpoints/expA_nas_seld_balanced_starss23/34_0.3375.ckpt \\
        --dataset starss23 --split test --data-root data/STARSS23 \\
        --cache-dir results/cache/window_outputs --device mps --tpit-perm

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CACHE_DIR = ROOT / "results" / "cache" / "window_outputs"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers shared with the evaluation scripts
# ──────────────────────────────────────────────────────────────────────────────

def machine_load() -> str:
    """Return the 1/5/15-min load averages as a string (``uptime`` equivalent)."""
    try:
        one, five, fifteen = os.getloadavg()
        return f"{one:.2f} {five:.2f} {fifteen:.2f}"
    except OSError:  # pragma: no cover - platform without getloadavg
        return "n/a"


def resolve_device(value: str) -> torch.device:
    """Resolve ``auto|cpu|cuda|mps`` into a ``torch.device``."""
    if value == "auto":
        if torch.cuda.is_available():
            value = "cuda"
        elif torch.backends.mps.is_available():
            value = "mps"
        else:
            value = "cpu"
    return torch.device(value)


def window_start_frame(ds, item_idx: int) -> int:
    """Global model-frame index of the first frame of window ``item_idx``.

    ``start_at_model = round(start_sample * model_sr / orig_sr)`` is the offset
    used by ``SELDDataset.__getitem__`` for the label construction; the window
    hop (2.5 s) is an integer number of model frames (250), which is asserted.
    """
    _, start_sample, _ = ds._items[item_idx]
    start_at_model = round(start_sample * ds.model_sr / ds.orig_sr)
    frame, rem = divmod(start_at_model, ds.hop_length)
    if rem != 0:
        raise ValueError(
            f"window {item_idx}: start {start_at_model} is not a multiple of "
            f"the model hop {ds.hop_length}"
        )
    return int(frame)


def window_target(ds, item_idx: int) -> torch.Tensor:
    """Rebuild the target of window ``item_idx`` without reading audio.

    Identical to ``ds[item_idx][1]`` for a dataset built with ``augment=None``
    (``SELDDataset.__getitem__`` calls the same ``_build_target``).
    """
    clip_idx, start_sample, _ = ds._items[item_idx]
    start_at_model = round(start_sample * ds.model_sr / ds.orig_sr)
    return ds._build_target(ds._events[clip_idx], ds.T_mel_window, start_at_model)


def load_module(config: Path, checkpoint: Path, device: torch.device):
    """Load a ``NASLightningModule`` from a stored ``hparams.yaml`` + checkpoint."""
    import yaml
    from experiments_nas_seld.train import NASLightningModule

    with open(config) as fh:
        raw = yaml.safe_load(fh)
    cfg = raw.get("cfg", raw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module = NASLightningModule.load_from_checkpoint(
            str(checkpoint), cfg=cfg, map_location="cpu"
        )
    module.eval().to(device)
    return module, cfg


def build_split_dataset(dataset: str, split: str, root: Path, cfg: dict, val_seed: int):
    """Build the (windowed, un-augmented) dataset of one split (matrix convention)."""
    from analysis.diagnostics.dataset_generalization_analysis import build_dataset

    return build_dataset(dataset, split, root, cfg, val_seed=val_seed)


class WindowCache:
    """Read access to a cache written by :func:`infer_windows` (memory-mapped)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text())
        self.logits = np.load(self.path / "logits.npy", mmap_mode="r")
        self.doa = np.load(self.path / "doa.npy", mmap_mode="r")
        perm = self.path / "perm.npy"
        self.perm = np.load(perm, mmap_mode="r") if perm.exists() else None

    @property
    def n_windows(self) -> int:
        return int(self.logits.shape[0])


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

def infer_windows(
    module,
    cfg: dict,
    ds,
    out_dir: Path,
    device: torch.device,
    batch_size: int = 4,
    tpit_perm: bool = False,
    limit: Optional[int] = None,
    resume: bool = True,
    log=print,
    extra_meta: Optional[Dict] = None,
) -> Path:
    """Run ``module.model`` over all windows of ``ds`` and store the outputs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_total = len(ds) if limit is None else min(int(limit), len(ds))
    N = int(cfg["heads"]["n_tracks"])
    C = int(module.n_classes)
    T = int(ds.T_mel_window)
    shape_l = (n_total, T, N, C)
    shape_d = (n_total, T, N, C, 3)
    shape_p = (n_total, T, C, N)

    progress_file = out_dir / "progress.json"
    done = 0
    mode = "w+"
    if resume and progress_file.exists() and (out_dir / "logits.npy").exists():
        prog = json.loads(progress_file.read_text())
        if prog.get("n_total") == n_total and not prog.get("finished"):
            done = int(prog["n_done"])
            mode = "r+"
            log(f"[window_outputs] resuming at window {done}/{n_total}")
        elif prog.get("finished") and prog.get("n_total") == n_total:
            log("[window_outputs] cache already complete; nothing to do")
            return out_dir
    logits_mm = np.lib.format.open_memmap(out_dir / "logits.npy", mode=mode,
                                          dtype=np.float32, shape=shape_l)
    doa_mm = np.lib.format.open_memmap(out_dir / "doa.npy", mode=mode,
                                       dtype=np.float32, shape=shape_d)
    perm_mm = None
    if tpit_perm:
        perm_mm = np.lib.format.open_memmap(out_dir / "perm.npy", mode=mode,
                                            dtype=np.int8, shape=shape_p)

    meta = {
        "name": out_dir.name,
        "n_windows": n_total,
        "frames_per_window": T,
        "n_tracks": N,
        "n_classes": C,
        "dataset_class": type(ds).__name__,
        "split": ds.split,
        "model_sr": ds.model_sr,
        "dataset_orig_sr": ds.orig_sr,
        "module_orig_sr": module.orig_sr,
        "hop_length": ds.hop_length,
        "window_samples": ds.window_samples,
        "hop_samples": ds.hop_samples,
        "items": [[int(ci), int(ss)] for ci, ss, _ in ds._items[:n_total]],
        "clips": [str(p) for p, _ in ds.clips],
        "device": str(device),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "batch_size": batch_size,
        "tpit_perm": bool(tpit_perm),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "load_at_start": machine_load(),
    }
    meta.update(extra_meta or {})
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))

    t_start = time.time()
    t_last = t_start
    with torch.inference_mode():
        i = done
        while i < n_total:
            j = min(i + batch_size, n_total)
            wavs, tgts = [], []
            for k in range(i, j):
                w, t = ds[k]
                wavs.append(w)
                tgts.append(t)
            x = torch.stack(wavs).to(device)
            act_logits, doa_tanh, _ = module.model(x, orig_sr=module.orig_sr)
            Tm = min(act_logits.shape[1], T)
            if Tm != T:
                raise RuntimeError(f"unexpected frame count {act_logits.shape[1]} != {T}")
            logits_mm[i:j] = act_logits[:, :T].float().cpu().numpy()
            doa_mm[i:j] = doa_tanh[:, :T].float().cpu().numpy()
            if perm_mm is not None:
                tgt = torch.stack(tgts).to(device)
                module.tpit_loss(act_logits[:, :T], doa_tanh[:, :T], tgt[:, :T])
                bp = module.tpit_loss.last_best_perm            # (B*T*C, N)
                perm_mm[i:j] = (
                    bp.reshape(j - i, T, C, N).to(torch.int8).cpu().numpy()
                )
            i = j
            now = time.time()
            if now - t_last > 60 or i == n_total:
                logits_mm.flush()
                doa_mm.flush()
                if perm_mm is not None:
                    perm_mm.flush()
                progress_file.write_text(json.dumps(
                    {"n_total": n_total, "n_done": i, "finished": i == n_total}))
                rate = (i - done) / max(now - t_start, 1e-9)
                log(f"[window_outputs] {i}/{n_total} windows "
                    f"({rate:.2f} win/s, load {machine_load()})")
                t_last = now
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["load_at_end"] = machine_load()
    meta["elapsed_s"] = round(time.time() - t_start, 1)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("Usage")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="cache sub-folder name")
    p.add_argument("--config", type=Path, required=True,
                   help="stored hparams.yaml (the executed configuration)")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset", required=True,
                   choices=["starss23", "tau_nigens2021", "tau_nigens2020", "tau2019"])
    p.add_argument("--split", required=True, choices=["val", "test"])
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--val-seed", type=int, default=2,
                   help="STARSS23 validation-split seed (2 = matrix/diagnostic "
                        "convention; 42 = the training split)")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--tpit-perm", action="store_true",
                   help="also store the reference-based tPIT permutation per frame")
    p.add_argument("--limit-windows", type=int, default=None)
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()

    device = resolve_device(args.device)
    module, cfg = load_module(args.config, args.checkpoint, device)
    ds = build_split_dataset(args.dataset, args.split, args.data_root, cfg, args.val_seed)
    print(f"[window_outputs] {args.name}: {len(ds)} windows, {len(ds.clips)} clips, "
          f"device={device}", flush=True)
    out = infer_windows(
        module, cfg, ds, args.cache_dir / args.name, device,
        batch_size=args.batch_size, tpit_perm=args.tpit_perm,
        limit=args.limit_windows, resume=not args.no_resume,
        extra_meta={
            "config": str(args.config), "checkpoint": str(args.checkpoint),
            "dataset": args.dataset, "val_seed": args.val_seed,
        },
    )
    print(f"[window_outputs] done: {out}", flush=True)


if __name__ == "__main__":
    main()
