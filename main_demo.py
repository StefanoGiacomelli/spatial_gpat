#!/usr/bin/env python3
"""
Generate clean NAS-SELD ExpA demo figures.

The script resolves the best ExpA checkpoint for each dataset, ranks candidate
test recordings by DOA quality, track continuity, and trajectory dynamics, runs
10-second windowed inference across the selected full recording, stitches the
window outputs, and writes one SVG figure per dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent
_CACHE_ROOT = Path(os.environ.get("TMPDIR", "/private/tmp")) / "spatial_gpat_demo_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore", message=".*LeafSpec.*deprecated.*")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update(
    {
        "svg.fonttype": "none",
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    }
)

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import yaml
from matplotlib.lines import Line2D

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datasets.starss23 import STARSS23Dataset
from datasets.tau2019 import TAU2019Dataset
from datasets.tau_nigens import TAUNIGENSDataset
from datasets.tau_nigens2020 import TAUNIGENS2020Dataset
from experiments_nas_seld.train import NASLightningModule
from utils.doa_metrics import align_doa_to_target, angular_error_deg


DATASET_ROOTS = {
    "STARSS23": _ROOT / "data" / "STARSS23",
    "BalancedSTARSS23": _ROOT / "data" / "STARSS23",
    "TAU2019": _ROOT / "data" / "TAU2019",
    "TAU-NIGENS2020": _ROOT / "data" / "TAU-NIGENS2020",
    "TAU-NIGENS2021": _ROOT / "data" / "TAU-NIGENS2021",
}


@dataclass(frozen=True)
class DemoSpec:
    key: str
    dataset: str
    plot_dataset: str
    config: Path
    checkpoint_dir_hint: str


DEFAULT_SPECS = (
    DemoSpec(
        key="balanced_starss23_expA",
        dataset="BalancedSTARSS23",
        plot_dataset="STARSS23",
        config=_ROOT / "configs" / "loss_experiments" / "expA_nas_seld_balanced_starss23.yaml",
        checkpoint_dir_hint="balanced_starss23_expA",
    ),
    DemoSpec(
        key="tau2019_expA",
        dataset="TAU2019",
        plot_dataset="TAU2019",
        config=_ROOT / "configs" / "loss_experiments" / "expA_nas_seld_tau2019.yaml",
        checkpoint_dir_hint="tau2019_expA",
    ),
    DemoSpec(
        key="tau_nigens2020_expA",
        dataset="TAU-NIGENS2020",
        plot_dataset="TAU-NIGENS2020",
        config=_ROOT / "configs" / "loss_experiments" / "expA_nas_seld_tau_nigens2020.yaml",
        checkpoint_dir_hint="tau_nigens2020_expA",
    ),
    DemoSpec(
        key="tau_nigens2021_expA",
        dataset="TAU-NIGENS2021",
        plot_dataset="TAU-NIGENS2021",
        config=_ROOT / "configs" / "loss_experiments" / "expA_nas_seld_tau_nigens2021.yaml",
        checkpoint_dir_hint="tau_nigens2021_expA",
    ),
)


@dataclass
class InferenceResult:
    clip_idx: int
    duration_s: float
    audio_path: Path
    wav: torch.Tensor
    target: torch.Tensor
    act_prob: np.ndarray
    pred_doa: np.ndarray
    score: float
    median_ae: float
    p90_ae: float
    active_slots: int
    dynamic_span: float
    visible_keys: List[Tuple[int, int]]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create four NAS-SELD ExpA demo SVG figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", type=Path, default=_ROOT / "figures" / "demo")
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--max-candidates", type=int, default=8)
    p.add_argument("--window-s", type=float, default=10.0)
    p.add_argument("--hop-s", type=float, default=2.5)
    p.add_argument("--act-thresh", type=float, default=None)
    p.add_argument("--min-active-slots", type=int, default=35)
    p.add_argument("--max-events", type=int, default=5)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(_ROOT))
    except ValueError:
        return str(path)


def _device(name: str) -> torch.device:
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    if name == "mps" or (
        name == "auto"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def _load_cfg(spec: DemoSpec, window_s: float, hop_s: float) -> dict:
    with open(spec.config) as fh:
        cfg = yaml.safe_load(fh)

    cfg.setdefault("semantic", {})["dataset"] = spec.dataset
    cfg.setdefault("augmentation", {})["enabled"] = False
    cfg.setdefault("training", {})["batch_size"] = 1
    cfg.setdefault("data", {})
    cfg["data"].update(
        {
            "window_s": window_s,
            "hop_s": hop_s,
            "val_fraction": 0.2,
            "val_seed": 42,
            "num_workers": 0,
        }
    )
    if spec.dataset == "BalancedSTARSS23":
        cfg["data"].update(
            {
                "starss_root": str((_ROOT / "data" / "STARSS23").resolve()),
                "tau_root": str((_ROOT / "data" / "TAU-NIGENS2021").resolve()),
                "tau2019_root": str((_ROOT / "data" / "TAU2019").resolve()),
            }
        )
    else:
        cfg["data"]["root_path"] = str(DATASET_ROOTS[spec.dataset].resolve())
    return cfg


def _ckpt_score(path: Path) -> float:
    if path.name.startswith("last"):
        return float("inf")
    nums = re.findall(r"(\d+(?:\.\d+)?)", path.stem)
    return float(nums[-1]) if nums else float("inf")


def _find_best_checkpoint(spec: DemoSpec, exp_name: str) -> Path:
    root = _ROOT / "runs" / "at2seld_nas"
    patterns = (
        f"**/{spec.checkpoint_dir_hint}/checkpoints/{exp_name}/*.ckpt",
        f"**/checkpoints/{exp_name}/*.ckpt",
    )
    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend(root.glob(pattern))
    candidates = sorted({p.resolve() for p in candidates if p.is_file()})
    candidates = [p for p in candidates if not p.name.startswith("last")]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found for {exp_name} under {root}")
    return min(candidates, key=_ckpt_score)


def _make_dataset(
    spec: DemoSpec,
    cfg: dict,
    window_s: Optional[float],
    hop_s: Optional[float],
):
    data = cfg["data"]
    common = dict(
        split="test",
        model_sr=cfg["input"]["sr"],
        hop_length=cfg["input"].get("hop_length", 320),
        n_tracks=cfg["heads"].get("n_tracks", 3),
        window_s=window_s,
        hop_s=hop_s,
        augment=None,
    )
    if spec.plot_dataset == "STARSS23":
        return STARSS23Dataset(
            root=DATASET_ROOTS["STARSS23"],
            val_fraction=data.get("val_fraction", 0.2),
            val_seed=data.get("val_seed", 42),
            **common,
        )
    if spec.plot_dataset == "TAU2019":
        return TAU2019Dataset(
            root=DATASET_ROOTS["TAU2019"],
            fold=data.get("fold", 1),
            mixed_mode=True,
            **common,
        )
    if spec.plot_dataset == "TAU-NIGENS2020":
        return TAUNIGENS2020Dataset(root=DATASET_ROOTS["TAU-NIGENS2020"], **common)
    if spec.plot_dataset == "TAU-NIGENS2021":
        return TAUNIGENSDataset(root=DATASET_ROOTS["TAU-NIGENS2021"], **common)
    raise ValueError(f"Unsupported plot dataset: {spec.plot_dataset}")


def _class_names(dataset: str) -> List[str]:
    with open(_ROOT / "configs" / "labels_map.json") as fh:
        data = json.load(fh)
    return [c["name"].split(",")[0] for c in data[dataset]["classes"]]


def _align_activity(
    act_logits: torch.Tensor,
    best_perm: torch.Tensor,
    B: int,
    T: int,
    N: int,
    C: int,
) -> torch.Tensor:
    act = act_logits.permute(0, 1, 3, 2).reshape(B * T * C, N)
    aligned = torch.gather(act, 1, best_perm)
    return aligned.reshape(B, T, C, N).permute(0, 1, 3, 2)


def _xyz_to_azel(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    norm = np.linalg.norm(xyz, axis=-1)
    safe = np.maximum(norm, 1e-8)
    az = np.degrees(np.arctan2(xyz[..., 0], xyz[..., 1]))
    el = np.degrees(np.arcsin(np.clip(xyz[..., 2] / safe, -1.0, 1.0)))
    return az, el


def _trajectory_jitter(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 5:
        return 50.0
    unwrapped = np.degrees(np.unwrap(np.radians(values)))
    return float(np.nanmedian(np.abs(np.diff(unwrapped, n=2))))


def _track_span(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 4:
        return 0.0
    unwrapped = np.degrees(np.unwrap(np.radians(values)))
    return float(np.nanpercentile(unwrapped, 95) - np.nanpercentile(unwrapped, 5))


def _full_target_dynamic_span(target: np.ndarray, keys: List[Tuple[int, int]]) -> float:
    gt_az, gt_el = _xyz_to_azel(target)
    active = np.linalg.norm(target, axis=-1) > 0.5
    spans: List[float] = []
    for n, c in keys:
        mask = active[:, n, c]
        if int(mask.sum()) < 6:
            continue
        spans.append(max(_track_span(gt_az[mask, n, c]), _track_span(gt_el[mask, n, c])))
    return float(max(spans)) if spans else 0.0


def _gt_candidate_score(dataset, clip_idx: int, spec: DemoSpec) -> float:
    events = dataset._events[clip_idx]
    if not events:
        return -float("inf")

    by_key = {}
    for frame, cls, src, az, el in events:
        by_key.setdefault((src, cls), []).append((int(frame), float(az), float(el)))

    counts = sorted((len(v) for v in by_key.values()), reverse=True)
    if not counts:
        return -float("inf")

    active_frames = len({int(e[0]) for e in events})
    n_events = len(counts)
    longest = counts[0]
    spans = []
    for vals in by_key.values():
        if len(vals) < 5:
            continue
        az = np.asarray([v[1] for v in vals], dtype=np.float64)
        el = np.asarray([v[2] for v in vals], dtype=np.float64)
        spans.append(max(_track_span(az), _track_span(el)))

    dynamic_span = max(spans) if spans else 0.0
    dynamic_weight = 0.0 if spec.plot_dataset == "TAU2019" else 1.0
    coverage_score = min(active_frames / 120.0, 1.0) * 25.0
    duration_score = min(longest / 80.0, 1.0) * 20.0
    dynamic_score = min(dynamic_span / 80.0, 1.0) * 30.0 * dynamic_weight
    event_penalty = max(0, n_events - 6) * 4.0
    static_penalty = 18.0 if dynamic_weight and dynamic_span < 12.0 else 0.0
    return coverage_score + duration_score + dynamic_score - event_penalty - static_penalty


def _candidate_clip_indices(dataset, spec: DemoSpec, max_candidates: int) -> List[int]:
    scored = [
        (_gt_candidate_score(dataset, clip_idx, spec), clip_idx)
        for clip_idx in range(len(dataset.clips))
    ]
    scored = [(score, idx) for score, idx in scored if math.isfinite(score)]
    scored.sort(reverse=True)
    if not scored:
        return list(range(min(max_candidates, len(dataset.clips))))
    return [idx for _, idx in scored[:max_candidates]]


def _visible_event_keys(
    target: np.ndarray,
    act_prob: np.ndarray,
    act_thresh: float,
    max_events: int,
) -> List[Tuple[int, int]]:
    gt_active = np.linalg.norm(target, axis=-1) > 0.5
    pred_active = act_prob > act_thresh
    keys = []
    T, N, C = gt_active.shape
    for n in range(N):
        for c in range(C):
            gt_count = int(gt_active[:, n, c].sum())
            pred_count = int(pred_active[:, n, c].sum())
            if gt_count == 0 and pred_count == 0:
                continue
            keys.append((gt_count + 0.35 * pred_count, gt_count, n, c))
    keys.sort(reverse=True)
    return [(n, c) for _, gt_count, n, c in keys[:max_events] if gt_count > 0]


def _score_result(
    target: np.ndarray,
    pred_doa: np.ndarray,
    act_prob: np.ndarray,
    act_thresh: float,
    min_active_slots: int,
    max_events: int,
    dynamic_required: bool,
) -> Tuple[float, float, float, int, float, List[Tuple[int, int]]]:
    active = np.linalg.norm(target, axis=-1) > 0.5
    active_slots = int(active.sum())
    if active_slots < min_active_slots:
        return float("inf"), float("inf"), float("inf"), active_slots, 0.0, []

    pred_active = pred_doa[active]
    tgt_active = target[active]
    ae = angular_error_deg(pred_active, tgt_active)
    finite_ae = ae[np.isfinite(ae)]
    if finite_ae.size == 0:
        return float("inf"), float("inf"), float("inf"), active_slots, 0.0, []

    keys = _visible_event_keys(target, act_prob, act_thresh, max_events=max_events)
    if not keys:
        return float("inf"), float("inf"), float("inf"), active_slots, 0.0, []

    dynamic_span = _full_target_dynamic_span(target, keys)
    if dynamic_required and dynamic_span < 10.0:
        return float("inf"), float("inf"), float("inf"), active_slots, dynamic_span, []

    jitter_terms: List[float] = []
    gt_az, gt_el = _xyz_to_azel(target)
    pr_az, pr_el = _xyz_to_azel(pred_doa)
    for n, c in keys:
        mask = active[:, n, c]
        if int(mask.sum()) < 5:
            continue
        jitter_terms.extend(
            [
                _trajectory_jitter(gt_az[mask, n, c]),
                _trajectory_jitter(gt_el[mask, n, c]),
                _trajectory_jitter(pr_az[mask, n, c]),
                _trajectory_jitter(pr_el[mask, n, c]),
            ]
        )

    median_ae = float(np.median(finite_ae))
    p90_ae = float(np.percentile(finite_ae, 90))
    jitter = float(np.median(jitter_terms)) if jitter_terms else 50.0
    event_penalty = max(0, len(keys) - 3) * 2.5
    missed_penalty = float(((act_prob <= act_thresh) & active).sum()) / max(active_slots, 1)
    dynamic_bonus = 0.08 * min(dynamic_span, 100.0) if dynamic_required else 0.0
    score = (
        median_ae
        + 0.15 * p90_ae
        + 1.6 * jitter
        + event_penalty
        + 8.0 * missed_penalty
        - dynamic_bonus
    )
    return score, median_ae, p90_ae, active_slots, dynamic_span, keys


def _run_window_inference(
    module: NASLightningModule,
    dataset,
    item_idx: int,
    device: torch.device,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    wav, target = dataset[item_idx]
    x = wav.unsqueeze(0).to(device)
    tgt = target.unsqueeze(0).to(device)

    with torch.inference_mode():
        act_logits, doa_tanh, _ = module.model(x, orig_sr=module.orig_sr)
        T = min(act_logits.shape[1], tgt.shape[1])
        act_logits = act_logits[:, :T]
        doa_tanh = doa_tanh[:, :T]
        tgt = tgt[:, :T]
        module.tpit_loss(act_logits, doa_tanh, tgt)
        B, T2, N, C, _ = doa_tanh.shape
        pred_aligned = align_doa_to_target(
            doa_tanh, module.tpit_loss.last_best_perm, B, T2, N, C
        )
        act_aligned = _align_activity(
            act_logits, module.tpit_loss.last_best_perm, B, T2, N, C
        )

    target_np = tgt[0].detach().cpu().float().numpy()
    pred_np = pred_aligned[0].detach().cpu().float().numpy()
    act_np = torch.sigmoid(act_aligned[0]).detach().cpu().float().numpy()

    _, start_sample, _ = dataset._items[item_idx]
    start_at_model = round(start_sample * dataset.model_sr / dataset.orig_sr)
    start_frame = int(round(start_at_model / dataset.hop_length))
    return start_frame, target_np, pred_np, act_np


def _infer_full_clip(
    module: NASLightningModule,
    full_dataset,
    windowed_dataset,
    clip_idx: int,
    device: torch.device,
    act_thresh: float,
    min_active_slots: int,
    max_events: int,
    dynamic_required: bool,
) -> InferenceResult:
    full_wav, full_target = full_dataset[clip_idx]
    target_np = full_target.detach().cpu().float().numpy()
    T, N, C, _ = target_np.shape
    pred_sum = np.zeros((T, N, C, 3), dtype=np.float64)
    act_sum = np.zeros((T, N, C), dtype=np.float64)
    counts = np.zeros(T, dtype=np.float64)

    item_indices = [
        i for i, item in enumerate(windowed_dataset._items)
        if int(item[0]) == int(clip_idx)
    ]
    if not item_indices:
        raise RuntimeError(f"No windows found for clip index {clip_idx}")

    for item_idx in item_indices:
        start_frame, _, pred_np, act_np = _run_window_inference(
            module, windowed_dataset, item_idx, device
        )
        end_frame = min(T, start_frame + pred_np.shape[0])
        win_len = max(0, end_frame - start_frame)
        if win_len == 0:
            continue
        pred_sum[start_frame:end_frame] += pred_np[:win_len]
        act_sum[start_frame:end_frame] += act_np[:win_len]
        counts[start_frame:end_frame] += 1.0

    valid = counts > 0
    pred = np.zeros_like(pred_sum, dtype=np.float32)
    act_prob = np.zeros_like(act_sum, dtype=np.float32)
    pred[valid] = (pred_sum[valid] / counts[valid, None, None, None]).astype(np.float32)
    act_prob[valid] = (act_sum[valid] / counts[valid, None, None]).astype(np.float32)

    score, median_ae, p90_ae, active_slots, dynamic_span, visible_keys = _score_result(
        target_np,
        pred,
        act_prob,
        act_thresh=act_thresh,
        min_active_slots=min_active_slots,
        max_events=max_events,
        dynamic_required=dynamic_required,
    )

    audio_path, _ = full_dataset.clips[clip_idx]
    duration_s = float(full_wav.shape[-1] / getattr(full_dataset, "_eff_sr", full_dataset.orig_sr))
    return InferenceResult(
        clip_idx=clip_idx,
        duration_s=duration_s,
        audio_path=audio_path,
        wav=full_wav.detach().cpu(),
        target=torch.from_numpy(target_np),
        act_prob=act_prob,
        pred_doa=pred,
        score=score,
        median_ae=median_ae,
        p90_ae=p90_ae,
        active_slots=active_slots,
        dynamic_span=dynamic_span,
        visible_keys=visible_keys,
    )


def _select_best_clip(
    module: NASLightningModule,
    full_dataset,
    windowed_dataset,
    spec: DemoSpec,
    device: torch.device,
    max_candidates: int,
    act_thresh: float,
    min_active_slots: int,
    max_events: int,
) -> InferenceResult:
    dynamic_required = spec.plot_dataset != "TAU2019"
    candidates = _candidate_clip_indices(full_dataset, spec, max_candidates)
    best: Optional[InferenceResult] = None
    for clip_idx in candidates:
        result = _infer_full_clip(
            module,
            full_dataset,
            windowed_dataset,
            clip_idx,
            device,
            act_thresh,
            min_active_slots,
            max_events,
            dynamic_required,
        )
        if math.isfinite(result.score) and (best is None or result.score < best.score):
            best = result

    if best is not None:
        return best

    # Fallback for datasets where the test subset has weak/no dynamic tracks.
    fallback: Optional[InferenceResult] = None
    for clip_idx in candidates:
        result = _infer_full_clip(
            module,
            full_dataset,
            windowed_dataset,
            clip_idx,
            device,
            act_thresh,
            min_active_slots=1,
            max_events=max_events,
            dynamic_required=False,
        )
        if math.isfinite(result.score) and (
            fallback is None or result.score < fallback.score
        ):
            fallback = result
    if fallback is None:
        raise RuntimeError("No plottable candidate recording found.")
    return fallback


def _spectrogram_db(wav: torch.Tensor, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_fft = 1024
    hop = 256
    window = torch.hann_window(n_fft)
    spec = torch.stft(
        wav.float(),
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )
    mag = spec.abs().mean(dim=0).numpy()
    db = 20.0 * np.log10(np.maximum(mag, 1e-10))
    db -= float(np.nanmax(db))
    db = np.clip(db, -80.0, 0.0)
    freqs = np.linspace(0.0, sr / 2.0, db.shape[0])
    times = np.arange(db.shape[1]) * hop / sr
    return db, times, freqs


def _series_with_gaps(
    times: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
    max_gap_s: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    t = times[mask]
    v = values[mask]
    if t.size == 0:
        return t, v
    out_t = [t[0]]
    out_v = [v[0]]
    for i in range(1, t.size):
        if (t[i] - t[i - 1]) > max_gap_s or abs(v[i] - v[i - 1]) > 120:
            out_t.append(np.nan)
            out_v.append(np.nan)
        out_t.append(t[i])
        out_v.append(v[i])
    return np.asarray(out_t), np.asarray(out_v)


def _blend_with_white(color, amount: float):
    rgb = np.asarray(mpl.colors.to_rgb(color), dtype=np.float64)
    return tuple(rgb + (1.0 - rgb) * amount)


def _darken(color, amount: float = 0.72):
    rgb = np.asarray(mpl.colors.to_rgb(color), dtype=np.float64)
    return tuple(np.clip(rgb * amount, 0.0, 1.0))


def _plot_doa_axis(
    ax,
    times: np.ndarray,
    target: np.ndarray,
    pred_doa: np.ndarray,
    act_prob: np.ndarray,
    keys: List[Tuple[int, int]],
    class_names: List[str],
    act_thresh: float,
    component: str,
    ylabel: str,
) -> None:
    gt_az, gt_el = _xyz_to_azel(target)
    pr_az, pr_el = _xyz_to_azel(pred_doa)
    gt_vals = gt_az if component == "az" else gt_el
    pr_vals = pr_az if component == "az" else pr_el
    gt_active = np.linalg.norm(target, axis=-1) > 0.5
    pred_active = act_prob > act_thresh
    palette = plt.get_cmap("tab10")

    for i, (n, c) in enumerate(keys):
        base_color = palette(i % 10)
        gt_color = _blend_with_white(base_color, 0.38)
        pred_color = _darken(base_color, 0.62)
        gt_mask = gt_active[:, n, c]
        pred_mask = pred_active[:, n, c] | gt_mask
        label = f"{class_names[c]} / tr{n}"
        t_gt, y_gt = _series_with_gaps(times, gt_vals[:, n, c], gt_mask)
        t_pr, y_pr = _series_with_gaps(times, pr_vals[:, n, c], pred_mask)
        if t_gt.size:
            ax.plot(
                t_gt,
                y_gt,
                color=gt_color,
                lw=2.4,
                alpha=0.98,
                solid_capstyle="round",
                label=label,
                zorder=2,
            )
        if t_pr.size:
            mark_every = max(1, len(t_pr) // 80)
            ax.plot(
                t_pr,
                y_pr,
                color=pred_color,
                lw=0.9,
                ls="--",
                marker="o",
                markersize=2.2,
                markevery=mark_every,
                alpha=0.92,
                zorder=4,
            )

    ax.set_ylabel(ylabel)
    ax.set_xlim(times[0], times[-1] if len(times) else 0.0)
    if component == "az":
        ax.set_ylim(-185, 185)
        ax.set_yticks([-180, -90, 0, 90, 180])
    else:
        ax.set_ylim(-95, 95)
        ax.set_yticks([-90, -45, 0, 45, 90])


def _plot_figure(
    result: InferenceResult,
    spec: DemoSpec,
    cfg: dict,
    output_dir: Path,
    act_thresh: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = _class_names(spec.dataset)
    target = result.target.numpy()
    pred = result.pred_doa
    hop = cfg["input"].get("hop_length", 320)
    model_sr = cfg["input"]["sr"]
    times = np.arange(target.shape[0]) * hop / model_sr
    eff_sr = 24000
    if spec.plot_dataset == "TAU2019":
        eff_sr = 24000
    db, spec_t, freqs = _spectrogram_db(result.wav, eff_sr)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12.8, 8.4),
        dpi=600,
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0, 1.0], "hspace": 0.22},
        constrained_layout=False,
    )

    ax0, ax1, ax2 = axes
    im = ax0.imshow(
        db,
        origin="lower",
        aspect="auto",
        extent=[spec_t[0], spec_t[-1], freqs[0] / 1000.0, freqs[-1] / 1000.0],
        cmap="magma",
        vmin=-80,
        vmax=0,
        interpolation="nearest",
    )
    ax0.set_ylabel("Freq. (kHz)")
    ax0.set_title(
        f"{spec.dataset} - {result.audio_path.stem} | "
        f"AE med {result.median_ae:.1f} deg, p90 {result.p90_ae:.1f} deg, "
        f"dyn {result.dynamic_span:.0f} deg",
        loc="left",
        fontsize=10,
        pad=8,
    )
    cbar = fig.colorbar(im, ax=ax0, pad=0.01, fraction=0.035)
    cbar.set_label("dB", rotation=270, labelpad=12)

    _plot_doa_axis(
        ax1,
        times,
        target,
        pred,
        result.act_prob,
        result.visible_keys,
        class_names,
        act_thresh,
        component="az",
        ylabel="Azimuth (deg)",
    )
    _plot_doa_axis(
        ax2,
        times,
        target,
        pred,
        result.act_prob,
        result.visible_keys,
        class_names,
        act_thresh,
        component="el",
        ylabel="Elevation (deg)",
    )
    ax2.set_xlabel("Time (s)")

    event_handles, event_labels = ax1.get_legend_handles_labels()
    style_handles = [
        Line2D([0], [0], color="#9ecae1", lw=2.4, label="Ground truth"),
        Line2D(
            [0],
            [0],
            color="#08519c",
            lw=0.9,
            ls="--",
            marker="o",
            markersize=2.5,
            label="Prediction",
        ),
    ]
    leg_events = ax1.legend(
        handles=event_handles,
        labels=event_labels,
        loc="upper right",
        title="Events",
        frameon=True,
        framealpha=0.88,
        borderpad=0.45,
        labelspacing=0.35,
    )
    leg_events.get_title().set_fontsize(7)
    ax2.legend(
        handles=style_handles,
        loc="upper right",
        frameon=True,
        framealpha=0.88,
        borderpad=0.45,
        labelspacing=0.35,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.94, bottom=0.085)

    out_path = output_dir / f"demo_{spec.key}_{result.audio_path.stem}_full.svg"
    fig.savefig(out_path, format="svg", dpi=600, metadata={"Creator": "main_demo.py"})
    plt.close(fig)
    return out_path


def main() -> None:
    args = _parse_args()
    device = _device(args.device)
    print(f"[main_demo] device={device}")

    outputs = []
    for spec in DEFAULT_SPECS:
        cfg = _load_cfg(spec, window_s=args.window_s, hop_s=args.hop_s)
        exp_name = cfg.get("experiment", {}).get("name", spec.config.stem)
        ckpt = _find_best_checkpoint(spec, exp_name)
        full_dataset = _make_dataset(spec, cfg, window_s=None, hop_s=None)
        windowed_dataset = _make_dataset(
            spec,
            cfg,
            window_s=args.window_s,
            hop_s=args.hop_s,
        )
        act_thresh = float(
            args.act_thresh
            if args.act_thresh is not None
            else cfg.get("metrics", {}).get("act_thresh", 0.5)
        )
        print(
            f"[main_demo] {spec.dataset}: ckpt={ckpt.relative_to(_ROOT)} | "
            f"test clips={len(full_dataset)} | "
            f"windows={len(windowed_dataset)} | act_thresh={act_thresh:.2f}"
        )
        if args.dry_run:
            continue

        module = NASLightningModule.load_from_checkpoint(
            str(ckpt), cfg=cfg, map_location="cpu"
        )
        module.eval().to(device)
        best = _select_best_clip(
            module,
            full_dataset,
            windowed_dataset,
            spec,
            device=device,
            max_candidates=args.max_candidates,
            act_thresh=act_thresh,
            min_active_slots=args.min_active_slots,
            max_events=args.max_events,
        )
        out = _plot_figure(best, spec, cfg, args.output_dir, act_thresh)
        outputs.append(out)
        print(
            f"[main_demo] wrote {_display_path(out)} "
            f"(clip={best.clip_idx}, active={best.active_slots}, "
            f"dynamic={best.dynamic_span:.1f}, score={best.score:.2f})"
        )

    if args.dry_run:
        print("[main_demo] dry run complete; no SVG files written.")
    else:
        print("[main_demo] SVG outputs:")
        for path in outputs:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
