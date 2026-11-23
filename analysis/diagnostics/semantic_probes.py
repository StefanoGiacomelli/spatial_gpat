#!/usr/bin/env python3
"""
analysis/diagnostics/semantic_probes.py
==============================================================================
T6 — Inference-time probes of the semantic prior of a trained AT2SELD checkpoint
==============================================================================

**What this is.**  A *sensitivity analysis of the trained system to its semantic
input*: the checkpoint is evaluated unchanged except for one intervention on the
semantic stream (or on the late bridge) at inference time.  It is **not** a
training-time ablation, it does **not** measure what pretraining contributes (a
network trained without the prior might compensate), and must be worded
accordingly ("sensitivity of the trained system ...").

In AT2SELD the semantic stream (E-PANNs, fed with the 4-channel mean of the FOA
signal, ``NASModel.forward``: ``x_mono = x.mean(dim=1)``) reaches the SELD path
only through the late cross-stitch bridge (``cross_stitch.cs_early_enabled: false``
in ExpA).  The spatial trunk (front end, ResNet blocks, TrackTransformer) is
therefore identical in every probe and is computed once per batch; every probe
re-runs only the semantic path, the bridge, the smoother and the SED/DOA heads.

Probes (exact intervention code is in ``semantic_input`` / ``head_from``):

  baseline           no intervention (must reproduce the stored evaluation)
  zero_semantic      ``x_mono = 0`` (silent input to E-PANNs)
  time_shuffled      ``x_mono`` cut into 1-s blocks, blocks permuted with a fixed
                     seed per window (derangement: no block stays in place);
                     content statistics kept, temporal alignment destroyed
  w_only             ``x_mono = x[:, 0]`` (omnidirectional W channel instead of the
                     4-channel mean; quantifies the QM06 mono-reduction choice)
  other_recording    ``x_mono`` of a window of a *different* recording
                     (window ``(i + shift) mod W``, ``shift`` = about half the split)
  bridge_zeroed      the late bridge is bypassed (``S_out = S``): its additive
                     contribution is set to zero, the semantic stream cannot
                     influence the SELD path at all

For every probe the repository protocol (``SELDMetrics`` semantics, gated 20 deg,
windows pooled; fast kernel verified against ``SELDMetrics``) is reported at a
list of thresholds, together with the oracle-activity DOA statistics
(tPIT-aligned, as ``doa_oracle_test.py``), the frame-level SED F1 and the mean
relative norm of the bridge residual ``||S_out - S|| / ||S||``.

Outputs (``--out-dir``): ``probes_long.csv``, ``probes_summary.json``,
``provenance.json``.

Usage
-----
    python analysis/diagnostics/semantic_probes.py \\
        --config runs/.../expA.../version_0/hparams.yaml --checkpoint runs/.../34_0.3375.ckpt \\
        --dataset starss23 --split test --data-root data/STARSS23 \\
        --out-dir results/diagnostics/semantic_probes_phase2/expA_starss23 --device mps

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as AF

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.diagnostics.seld_variants import GatedCounts, angular_error_array  # noqa: E402
from analysis.diagnostics.window_outputs import (  # noqa: E402
    build_split_dataset, load_module, machine_load, resolve_device,
)
from utils.doa_metrics import align_doa_to_target  # noqa: E402

PROBES = ["baseline", "zero_semantic", "time_shuffled", "w_only", "other_recording",
          "bridge_zeroed"]
DEFAULT_THRESHOLDS = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
BLOCK_S = 1.0


# ── intervention code ─────────────────────────────────────────────────────────

def derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random permutation of ``range(n)`` without fixed points (n >= 2)."""
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def semantic_input(probe: str, x_res: torch.Tensor, x_partner: Optional[torch.Tensor],
                   sr: int, win_ids: Sequence[int]) -> torch.Tensor:
    """Mono signal fed to E-PANNs under ``probe`` (x_res: (B, 4, T) resampled to model sr)."""
    if probe in ("baseline", "bridge_zeroed"):
        return x_res.mean(dim=1)
    if probe == "zero_semantic":
        return torch.zeros_like(x_res[:, 0])
    if probe == "w_only":
        return x_res[:, 0]
    if probe == "other_recording":
        assert x_partner is not None
        return x_partner.mean(dim=1)
    if probe == "time_shuffled":
        mono = x_res.mean(dim=1)
        blk = int(round(BLOCK_S * sr))
        n_blk = mono.shape[-1] // blk
        out = mono.clone()
        for b, wid in enumerate(win_ids):
            rng = np.random.default_rng(10_000 + int(wid))
            perm = derangement(n_blk, rng)
            for dst, src in enumerate(perm):
                out[b, dst * blk:(dst + 1) * blk] = mono[b, src * blk:(src + 1) * blk]
        return out
    raise ValueError(probe)


def spatial_trunk(model, x_res: torch.Tensor) -> torch.Tensor:
    """Spatial path up to the late bridge input (identical to ``NASModel.forward``)."""
    S = model._run_input_processor(x_res)
    if model.itype == "spectral":
        C = model.n_mics
        S_mag = model.input_norm_mag(S[:, :C])
        S_phase = model.input_norm_phase(S[:, C:2 * C])
        parts, offset = [S_mag, S_phase], 2 * C
        if hasattr(model, "input_norm_iv"):
            parts.append(model.input_norm_iv(S[:, offset:offset + 3]))
            offset += 3
        if hasattr(model, "input_norm_gcc"):
            parts.append(model.input_norm_gcc(S[:, offset:]))
        S = torch.cat(parts, dim=1)
    for mod in model.spatial_early:
        S = mod(S)
    return model._run_spatial_late(S, S.shape[1])


def semantic_late(model, x_mono: torch.Tensor) -> torch.Tensor:
    """E-PANNs semantic features entering the late bridge (``cs_early`` is off)."""
    h_early = model.split_epanns.forward_early(x_mono)
    return model.split_epanns.forward_late(h_early)


def head_from(model, S_late: torch.Tensor, h_late: torch.Tensor, bypass_bridge: bool):
    """Late bridge (optional), smoother and heads -- the rest of ``NASModel.forward``.

    Returns ``(act_logits, doa_tanh, bridge_relative_residual)``.
    """
    B = S_late.shape[0]
    rel = torch.zeros(B, device=S_late.device)
    S_in = S_late
    if model.cs_late_on and not bypass_bridge:
        if model._sl_is_transformer:
            S4 = S_late.permute(0, 2, 1).unsqueeze(3)
            S4, h_late = model.bridge_late(S4, h_late)
            S_late = S4.squeeze(3).permute(0, 2, 1)
        else:
            S_late, h_late = model.bridge_late(S_late, h_late)
        num = (S_late - S_in).flatten(1).norm(dim=1)
        den = S_in.flatten(1).norm(dim=1).clamp_min(1e-12)
        rel = num / den
    if model._sl_is_transformer:
        seq_spat = S_late
    else:
        seq_spat = S_late.mean(dim=3).permute(0, 2, 1) if S_late.dim() == 4 else S_late
    spat_out = model._smoother_forward(model.smoother_spatial, model._tss_type, seq_spat)
    N, C = model.n_tracks, model.n_cls
    act = model.head_sed(spat_out).view(B, -1, N, C)
    doa = torch.tanh(model.head_doa(spat_out).view(B, -1, N, C, 3))
    return act, doa, rel


def check_equivalence(module, x: torch.Tensor, device) -> float:
    """Max abs difference between ``NASModel.forward`` and the split forward (baseline)."""
    model = module.model
    with torch.inference_mode():
        a0, d0, _ = model(x, orig_sr=module.orig_sr)
        xr = AF.resample(x, module.orig_sr, model.src_sr) if module.orig_sr != model.src_sr else x
        S = spatial_trunk(model, xr)
        h = semantic_late(model, xr.mean(dim=1))
        a1, d1, _ = head_from(model, S, h, False)
    return float(max((a0 - a1).abs().max(), (d0 - d1).abs().max()))


# ── evaluation ────────────────────────────────────────────────────────────────

class ProbeAccumulator:
    def __init__(self, n_classes: int, thresholds: Sequence[float]):
        self.th = list(thresholds)
        self.seld = {t: GatedCounts(n_classes) for t in self.th}
        self.sed = {t: np.zeros(3, np.int64) for t in self.th}      # tp, fp, fn (any-track)
        self.ae: List[np.ndarray] = []
        self.norm: List[np.ndarray] = []
        self.rel_sum, self.n_win = 0.0, 0

    def update(self, module, act, doa, tgt, rel):
        probs = torch.sigmoid(act).cpu().numpy()
        doa_np = doa.cpu().numpy()
        tgt_np = tgt.cpu().numpy()
        B, T, N, C = probs.shape
        ra = (np.linalg.norm(tgt_np, axis=-1) > 0.5).reshape(-1, N, C)
        raw = (doa_np.reshape(-1, N, C, 3), tgt_np.reshape(-1, N, C, 3), ra)
        for t in self.th:
            pa = (probs > t).reshape(-1, N, C)
            self.seld[t].update(pa, raw[0], raw[2], raw[1])
            pr_any, rf_any = pa.any(axis=1), ra.any(axis=1)
            self.sed[t] += [int((pr_any & rf_any).sum()), int((pr_any & ~rf_any).sum()),
                            int((~pr_any & rf_any).sum())]
        # oracle activity, tPIT alignment (as doa_oracle_test.py)
        with torch.inference_mode():
            module.tpit_loss(act, doa, tgt)
            aligned = align_doa_to_target(doa, module.tpit_loss.last_best_perm, B, T, N, C)
        al = aligned.cpu().numpy()
        act_mask = np.linalg.norm(tgt_np, axis=-1) > 0.5                 # (B,T,N,C)
        p_act, t_act = al[act_mask], tgt_np[act_mask]
        self.ae.append(angular_error_array(p_act, t_act).astype(np.float32))
        self.norm.append(np.linalg.norm(p_act, axis=-1).astype(np.float32))
        self.rel_sum += float(rel.sum())
        self.n_win += B

    def rows(self, probe: str) -> List[dict]:
        ae, nrm = np.concatenate(self.ae), np.concatenate(self.norm)
        oracle = {
            "oracle_active_slots": int(ae.size),
            "oracle_acc_10deg": float((ae <= 10).mean()), "oracle_acc_20deg": float((ae <= 20).mean()),
            "oracle_acc_30deg": float((ae <= 30).mean()), "oracle_acc_45deg": float((ae <= 45).mean()),
            "oracle_median_deg": float(np.median(ae)), "oracle_mean_deg": float(ae.mean()),
            "oracle_p90_deg": float(np.percentile(ae, 90)),
            "oracle_pred_norm_median": float(np.median(nrm)),
            "oracle_near_null_lt_0.10": float((nrm < 0.1).mean()),
            "bridge_relative_residual_mean": self.rel_sum / max(self.n_win, 1),
        }
        out = []
        for t in self.th:
            m = self.seld[t].compute()
            tp, fp, fn = (int(v) for v in self.sed[t])
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            out.append({"probe": probe, "threshold": t, "SELD": m["SELD_score"], "ER": m["ER_20"],
                        "F": m["F_20"], "LE": m["LE_CD"], "LR": m["LR_CD"], "sed_f1": f1,
                        **oracle})
        return out


def run_probes(module, ds, device, probes: Sequence[str], thresholds, batch_size: int,
               limit: Optional[int], shift: int, log=print) -> Dict[str, ProbeAccumulator]:
    model = module.model
    if model.cs_early_on:
        raise NotImplementedError("probes assume the early bridge is off (ExpA configuration)")
    C = module.n_classes
    n_total = len(ds) if limit is None else min(limit, len(ds))
    accs = {p: ProbeAccumulator(C, thresholds) for p in probes}
    sr = model.src_sr
    t0 = time.time()
    with torch.inference_mode():
        i = 0
        while i < n_total:
            j = min(i + batch_size, n_total)
            items = [ds[k] for k in range(i, j)]
            x = torch.stack([w for w, _ in items]).to(device)
            tgt = torch.stack([t for _, t in items]).to(device)
            partner = None
            if "other_recording" in probes:
                xp = torch.stack([ds[(k + shift) % len(ds)][0] for k in range(i, j)]).to(device)
                partner = AF.resample(xp, module.orig_sr, sr) if module.orig_sr != sr else xp
            xr = AF.resample(x, module.orig_sr, sr) if module.orig_sr != sr else x
            S = spatial_trunk(model, xr)
            for probe in probes:
                if probe == "bridge_zeroed":
                    h_late = None            # the bridge is bypassed: semantic features unused
                else:
                    mono = semantic_input(probe, xr, partner, sr, list(range(i, j)))
                    h_late = semantic_late(model, mono)
                act, doa, rel = head_from(model, S, h_late, probe == "bridge_zeroed")
                Tm = min(act.shape[1], tgt.shape[1])
                accs[probe].update(module, act[:, :Tm], doa[:, :Tm], tgt[:, :Tm], rel)
            i = j
            if (i // batch_size) % 50 == 0 or i == n_total:
                log(f"[probes] {i}/{n_total} windows ({time.time() - t0:.0f} s, load {machine_load()})")
    return accs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("Outputs")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--dataset", required=True,
                    choices=["starss23", "tau_nigens2021", "tau_nigens2020", "tau2019"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--val-seed", type=int, default=2)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--probes", nargs="+", default=PROBES, choices=PROBES)
    ap.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    ap.add_argument("--limit-windows", type=int, default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    module, cfg = load_module(args.config, args.checkpoint, device)
    ds = build_split_dataset(args.dataset, args.split, args.data_root, cfg, args.val_seed)
    print(f"[probes] {len(ds)} windows, device={device}", flush=True)
    x0 = torch.stack([ds[k][0] for k in range(min(2, len(ds)))]).to(device)
    diff = check_equivalence(module, x0, device)
    print(f"[probes] split forward vs NASModel.forward: max abs diff {diff:.2e}", flush=True)
    if diff > 1e-3:
        raise RuntimeError("split forward does not reproduce NASModel.forward")
    shift = (len(ds) // 2) | 1
    load0, t0 = machine_load(), time.time()
    accs = run_probes(module, ds, device, args.probes, args.thresholds, args.batch_size,
                      args.limit_windows, shift)
    rows = []
    for p in args.probes:
        rows += accs[p].rows(p)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "probes_long.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for p in args.probes:
        pr = [r for r in rows if r["probe"] == p]
        best = min(pr, key=lambda r: r["SELD"])
        at05 = next((r for r in pr if abs(r["threshold"] - 0.5) < 1e-9), None)
        summary[p] = {"at_0.5": at05, "test_optimal": best}
    (args.out_dir / "probes_summary.json").write_text(json.dumps(summary, indent=1))
    (args.out_dir / "provenance.json").write_text(json.dumps({
        "script": "analysis/diagnostics/semantic_probes.py", "argv": sys.argv[1:],
        "config": str(args.config), "checkpoint": str(args.checkpoint),
        "dataset": args.dataset, "split": args.split, "n_windows": len(ds)
        if args.limit_windows is None else args.limit_windows,
        "device": str(device), "torch": torch.__version__,
        "split_forward_max_abs_diff": diff, "other_recording_shift": shift,
        "time_shuffle_block_s": BLOCK_S, "probes": args.probes,
        "thresholds": args.thresholds, "elapsed_s": round(time.time() - t0, 1),
        "load_at_start": load0, "load_at_end": machine_load(),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wording": "sensitivity of the trained system to its semantic input; not a "
                   "training-time ablation and not a measure of the contribution of pretraining",
    }, indent=1))
    print(f"[probes] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
