#!/usr/bin/env python3
"""
analysis/datasets/test_augmentations.py
=========================================
Correctness tests for SELDAugment (16 Patterns, Gain, Noise) followed by
diagnostic plots saved to figures/ (SVG, 600 DPI).

Run
---
    python analysis/datasets/test_augmentations.py

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

FIGURES_DIR = _ROOT / 'figures' / 'datasets'
STARSS_ROOT = _ROOT / 'data' / 'STARSS23'

# ── Minimal test runner ───────────────────────────────────────────────────────

_PASS = 0
_FAIL = 0


def _ok(name):
    global _PASS
    _PASS += 1
    print(f'  PASS  {name}')


def _fail(name, msg):
    global _FAIL
    _FAIL += 1
    print(f'  FAIL  {name}')
    print(f'        {msg}')


def _run(name: str, fn):
    try:
        fn()
        _ok(name)
    except AssertionError as exc:
        _fail(name, str(exc))
    except Exception as exc:
        _fail(name, f'{type(exc).__name__}: {exc}')


def _make_augment(**kwargs):
    from datasets.utils import SELDAugment
    defaults = dict(enabled=True, p_16patterns=0.34, p_gain=0.33, p_noise=0.33)
    defaults.update(kwargs)
    return SELDAugment(**defaults)


# ── Synthetic FOA signal at a single DOA (DC carrier for test correctness) ────

def _synthetic_foa(az_deg: float, el_deg: float, T: int = 200):
    """
    (4, T) FOA tensor using a DC (constant) carrier so that channel
    means directly equal the direction codes — required for correctness tests.

    FOA encoding: X channel (wav[1]) = code_y = cos(az)*cos(el)
                  Y channel (wav[2]) = code_x = sin(az)*cos(el)
                  Z channel (wav[3]) = code_z = sin(el)
    Target: target[..., 0]=code_x, target[..., 1]=code_y, target[..., 2]=code_z
    """
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    code_x = math.sin(az) * math.cos(el)
    code_y = math.cos(az) * math.cos(el)
    code_z = math.sin(el)

    wav = torch.stack([
        torch.ones(T),
        torch.full((T,), code_y),
        torch.full((T,), code_x),
        torch.full((T,), code_z),
    ])  # (4, T)

    target = torch.zeros(T, 3, 5, 3)   # (T_mel, N_tracks, n_cls, 3)
    target[:, 0, 0, 0] = code_x
    target[:, 0, 0, 1] = code_y
    target[:, 0, 0, 2] = code_z

    return wav, target, (code_x, code_y, code_z)


def _synthetic_foa_sinusoid(az_deg: float, el_deg: float,
                             T: int = 4800, freq: float = 440.0,
                             sr: float = 24000.0):
    """
    (4, T) FOA tensor using a sinusoidal carrier — for visualisation only.

    W = sin(2π f t)
    X = code_y * sin(2π f t)
    Y = code_x * sin(2π f t)
    Z = code_z * sin(2π f t)
    """
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    code_x = math.sin(az) * math.cos(el)
    code_y = math.cos(az) * math.cos(el)
    code_z = math.sin(el)

    t       = torch.arange(T, dtype=torch.float32) / sr
    carrier = torch.sin(2 * math.pi * freq * t)

    wav = torch.stack([
        carrier,
        code_y * carrier,
        code_x * carrier,
        code_z * carrier,
    ])  # (4, T)

    target = torch.zeros(T, 3, 5, 3)
    target[:, 0, 0, 0] = code_x
    target[:, 0, 0, 1] = code_y
    target[:, 0, 0, 2] = code_z

    return wav, target, (code_x, code_y, code_z)


def _decode_from_wav(wav):
    cy = float(wav[1].mean())
    cx = float(wav[2].mean())
    cz = float(wav[3].mean())
    az = math.atan2(cx, cy)
    el = math.asin(float(np.clip(cz, -1.0, 1.0)))
    return az, el


def _decode_from_target(tgt):
    cx = tgt[:, 0, 0, 0].mean().item()
    cy = tgt[:, 0, 0, 1].mean().item()
    cz = tgt[:, 0, 0, 2].mean().item()
    az = math.atan2(cx, cy)
    el = math.asin(float(np.clip(cz, -1.0, 1.0)))
    return az, el


# ══════════════════════════════════════════════════════════════════════════════
# 16 Patterns
# ══════════════════════════════════════════════════════════════════════════════

def test_patterns():
    """Run the patterns diagnostic check."""

    from datasets.utils import SELDAugment

    augment = SELDAugment(enabled=True, p_16patterns=1.0, p_gain=0.0, p_noise=0.0)
    WAZ, EL = 35.0, 20.0
    wav, target, _ = _synthetic_foa(WAZ, EL)
    tol = 0.02   # radians ≈ 1.1°

    for p_i, pat in enumerate(augment._PATTERNS):
        def _check(pi=p_i, p=pat):
            ch1_s, ch1_f, ch2_s, ch2_f, ch3_s, cx_s, cx_f, cy_s, cy_f, cz_s = p

            wt = wav.clone()
            tt = target.clone()

            x0, y0, z0 = wt[1].clone(), wt[2].clone(), wt[3].clone()
            wt[1] = ch1_s * (x0 if ch1_f == 1 else y0)
            wt[2] = ch2_s * (x0 if ch2_f == 1 else y0)
            wt[3] = ch3_s * z0

            cx0 = tt[..., 0].clone()
            cy0 = tt[..., 1].clone()
            cz0 = tt[..., 2].clone()
            tt[..., 0] = cx_s * (cx0 if cx_f == 0 else cy0)
            tt[..., 1] = cy_s * (cx0 if cy_f == 0 else cy0)
            tt[..., 2] = cz_s * cz0

            az_wav, el_wav = _decode_from_wav(wt)
            az_tgt, el_tgt = _decode_from_target(tt)

            assert abs(az_wav - az_tgt) < tol, \
                f"az mismatch: wav={math.degrees(az_wav):.2f}° tgt={math.degrees(az_tgt):.2f}°"
            assert abs(el_wav - el_tgt) < tol, \
                f"el mismatch: wav={math.degrees(el_wav):.2f}° tgt={math.degrees(el_tgt):.2f}°"

        _run(f'patterns/pattern_{p_i+1:02d}/doa_consistency', _check)

    def _distinct():
        results = set()
        for pat in augment._PATTERNS:
            ch1_s, ch1_f, ch2_s, ch2_f, ch3_s, *_ = pat
            x0, y0, z0 = wav[1].clone(), wav[2].clone(), wav[3].clone()
            wt = wav.clone()
            wt[1] = ch1_s * (x0 if ch1_f == 1 else y0)
            wt[2] = ch2_s * (x0 if ch2_f == 1 else y0)
            wt[3] = ch3_s * z0
            az, el = _decode_from_wav(wt)
            results.add((round(az, 3), round(el, 3)))
        assert len(results) == 15, f"Expected 15 distinct DOAs, got {len(results)}"

    _run('patterns/all_15_distinct', _distinct)


# ══════════════════════════════════════════════════════════════════════════════
# Gain
# ══════════════════════════════════════════════════════════════════════════════

def test_gain():

    """Run the gain diagnostic check."""

    def _fixed_gain():
        aug = _make_augment(p_16patterns=0, p_gain=1.0, p_noise=0,
                            gain_min=0.4, gain_max=0.4)
        wav, tgt, _ = _synthetic_foa(30, 15)
        wav_aug, _  = aug(wav.clone(), tgt.clone())
        g = (wav_aug[1] / wav[1].clamp(min=1e-9)).mean().abs().item()
        assert abs(g - 0.4) < 0.01, f"Expected gain=0.4, got {g:.4f}"
        for ch in [1, 2, 3]:
            ratio = (wav_aug[ch] / wav[ch].clamp(min=1e-9)).std().item()
            assert ratio < 1e-4, f"Channel {ch}: gain not uniform (std={ratio:.2e})"

    def _gain_range():
        aug = _make_augment(p_16patterns=0, p_gain=1.0, p_noise=0,
                            gain_min=0.3, gain_max=0.9)
        wav, tgt, _ = _synthetic_foa(45, 10)
        for _ in range(200):
            wav_aug, _ = aug(wav.clone(), tgt.clone())
            g = (wav_aug[1] / wav[1]).mean().abs().item()
            assert 0.29 < g < 0.91, f"Gain {g:.3f} outside [0.3, 0.9]"

    def _target_invariant():
        aug = _make_augment(p_16patterns=0, p_gain=1.0, p_noise=0)
        wav, tgt, _ = _synthetic_foa(30, 10)
        _, tgt_aug  = aug(wav.clone(), tgt.clone())
        assert torch.allclose(tgt, tgt_aug), "Gain modified target"

    _run('gain/fixed_factor',     _fixed_gain)
    _run('gain/range_check',      _gain_range)
    _run('gain/target_invariant', _target_invariant)


# ══════════════════════════════════════════════════════════════════════════════
# Noise
# ══════════════════════════════════════════════════════════════════════════════

def test_noise():

    """Run the noise diagnostic check."""

    def _snr_check():
        snr_db = 35.0
        aug    = _make_augment(p_16patterns=0, p_gain=0, p_noise=1.0,
                               noise_snr_db=snr_db)
        wav, tgt, _ = _synthetic_foa(20, 10, T=8000)
        wav_aug, _  = aug(wav.clone(), tgt.clone())
        noise     = wav_aug - wav
        sig_rms   = wav.pow(2).mean().sqrt().item()
        noise_rms = noise.pow(2).mean().sqrt().item()
        expected  = sig_rms / (10.0 ** (snr_db / 20.0))
        rel_err   = abs(noise_rms - expected) / (expected + 1e-9)
        assert rel_err < 0.10, \
            f"SNR mismatch: noise_rms={noise_rms:.4f}, expected={expected:.4f} (err={rel_err:.1%})"

    def _all_channels():
        aug = _make_augment(p_16patterns=0, p_gain=0, p_noise=1.0)
        wav, tgt, _ = _synthetic_foa(30, 10, T=4000)
        wav_aug, _  = aug(wav.clone(), tgt.clone())
        for ch in range(4):
            assert not torch.allclose(wav[ch], wav_aug[ch]), f"Channel {ch}: noise not added"

    def _target_invariant():
        aug = _make_augment(p_16patterns=0, p_gain=0, p_noise=1.0)
        wav, tgt, _ = _synthetic_foa(30, 10)
        _, tgt_aug  = aug(wav.clone(), tgt.clone())
        assert torch.allclose(tgt, tgt_aug), "Noise modified target"

    _run('noise/snr_check',       _snr_check)
    _run('noise/all_channels',    _all_channels)
    _run('noise/target_invariant', _target_invariant)


# ══════════════════════════════════════════════════════════════════════════════
# Mutual exclusion + force
# ══════════════════════════════════════════════════════════════════════════════

def test_mutual_exclusion():

    """Run the mutual exclusion diagnostic check."""

    def _force_always_augments():
        aug = _make_augment(p_16patterns=0.34, p_gain=0.33, p_noise=0.33)
        wav, tgt, _ = _synthetic_foa(30, 10, T=500)
        for i in range(200):
            wav_aug, tgt_aug = aug(wav.clone(), tgt.clone(), force=True)
            changed = (not torch.allclose(wav, wav_aug) or
                       not torch.allclose(tgt, tgt_aug))
            assert changed, f"force=True did not augment on iteration {i}"

    def _disabled_noop():
        from datasets.utils import SELDAugment
        aug = SELDAugment(enabled=False)
        wav, tgt, _ = _synthetic_foa(30, 10)
        wav_aug, tgt_aug = aug(wav.clone(), tgt.clone())
        assert torch.allclose(wav, wav_aug), "Disabled augment changed wav"
        assert torch.allclose(tgt, tgt_aug), "Disabled augment changed target"

    def _p_none_respected():
        aug = _make_augment(p_16patterns=0.01, p_gain=0.01, p_noise=0.01)
        wav, tgt, _ = _synthetic_foa(30, 10, T=1000)
        noop = 0
        for _ in range(500):
            wav_aug, tgt_aug = aug(wav.clone(), tgt.clone())
            if torch.allclose(wav, wav_aug) and torch.allclose(tgt, tgt_aug):
                noop += 1
        assert noop > 300, f"Expected many no-ops (p_none≈0.97), got {noop}/500"

    def _invalid_probs():
        from datasets.utils import SELDAugment
        try:
            SELDAugment(enabled=True, p_16patterns=0.5, p_gain=0.4, p_noise=0.3)
            assert False, "Should raise ValueError for sum > 1"
        except ValueError:
            pass

    _run('mutual_exclusion/force_always_augments', _force_always_augments)
    _run('mutual_exclusion/disabled_noop',         _disabled_noop)
    _run('mutual_exclusion/p_none_respected',      _p_none_respected)
    _run('mutual_exclusion/invalid_probs_raise',   _invalid_probs)


# ══════════════════════════════════════════════════════════════════════════════
# Balancing oversampling (requires real data)
# ══════════════════════════════════════════════════════════════════════════════

def test_balancing():
    """Run the balancing diagnostic check."""

    if not STARSS_ROOT.exists():
        print(f'  SKIP  balancing  [STARSS23 missing at {STARSS_ROOT}]')
        return

    def _item_counts_per_class(ds, n_cls=13):
        counts = [0] * n_cls
        for ci, _ss, _ in ds._items:
            cls_set = {cls for _, cls, *_ in ds._events[ci] if cls < n_cls}
            for c in cls_set:
                counts[c] += 1
        return counts

    def _class_balance():
        from datasets.utils    import SELDAugment
        from datasets.starss23 import STARSS23Dataset
        kw  = dict(model_sr=24000, hop_length=240, n_tracks=3)
        aug = SELDAugment(enabled=True, p_16patterns=0.20, p_gain=0.15, p_noise=0.10)

        ds_over = STARSS23Dataset(root=STARSS_ROOT, split='train', augment=aug,  **kw)
        ds_base = STARSS23Dataset(root=STARSS_ROOT, split='train', augment=None, **kw)

        counts_base = _item_counts_per_class(ds_base)
        counts_over = _item_counts_per_class(ds_over)

        pop_base = [c for c in counts_base if c > 0]
        pop_over = [c for c in counts_over if c > 0]

        ratio_base = max(pop_base) / min(pop_base)
        ratio_over = max(pop_over) / min(pop_over)

        assert ratio_over < ratio_base, (
            f"Oversampling didn't reduce imbalance: "
            f"ratio_over={ratio_over:.2f} >= ratio_base={ratio_base:.2f}"
        )

    _run('balancing/oversample_reduces_imbalance', _class_balance)


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostic plots
# ══════════════════════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Lookup table of rotation equation labels for each of the 15 patterns
_PATTERN_TITLES = [
    "φ′=φ+π/2, θ′=θ",    "φ′=φ−π/2, θ′=θ",
    "φ′=φ+π, θ′=θ",       "φ′=−φ, θ′=θ",
    "φ′=−φ+π/2, θ′=θ",    "φ′=−φ−π/2, θ′=θ",
    "φ′=−φ+π, θ′=θ",      "φ′=φ, θ′=−θ",
    "φ′=φ+π/2, θ′=−θ",    "φ′=φ−π/2, θ′=−θ",
    "φ′=φ+π, θ′=−θ",      "φ′=−φ, θ′=−θ",
    "φ′=−φ+π/2, θ′=−θ",   "φ′=−φ−π/2, θ′=−θ",
    "φ′=−φ+π, θ′=−θ",
]


def _apply_pattern_to_xyz(pat, cx0, cy0, cz0):
    """Apply one pattern's DOA transform to (cx, cy, cz) scalars."""
    _, _, _, _, _, cx_s, cx_f, cy_s, cy_f, cz_s = pat
    cx1 = cx_s * (cx0 if cx_f == 0 else cy0)
    cy1 = cy_s * (cx0 if cy_f == 0 else cy0)
    cz1 = cz_s * cz0
    return cx1, cy1, cz1


def _draw_unit_sphere(ax, alpha=0.08):
    """Draw a faint unit sphere wireframe."""
    u = np.linspace(0, 2 * np.pi, 24)
    v = np.linspace(0, np.pi, 13)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color='lightgrey', alpha=alpha, linewidth=0)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_zlim(-1.1, 1.1)
    ax.set_xlabel('cx', fontsize=5); ax.set_ylabel('cy', fontsize=5)
    ax.set_zlabel('cz', fontsize=5)
    ax.tick_params(labelsize=4)


def _plot_patterns_single_point():
    """Figure 1: single source point transformed by each of the 15 patterns."""
    from datasets.utils import SELDAugment
    augment = SELDAugment(enabled=True, p_16patterns=1.0, p_gain=0.0, p_noise=0.0)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    AZ_DEG, EL_DEG = 35.0, 20.0
    az = math.radians(AZ_DEG)
    el = math.radians(EL_DEG)
    cx0 = math.sin(az) * math.cos(el)
    cy0 = math.cos(az) * math.cos(el)
    cz0 = math.sin(el)

    fig = plt.figure(figsize=(20, 12))
    for p_i, pat in enumerate(augment._PATTERNS):
        ax = fig.add_subplot(3, 5, p_i + 1, projection='3d')
        _draw_unit_sphere(ax)

        cx1, cy1, cz1 = _apply_pattern_to_xyz(pat, cx0, cy0, cz0)

        ax.scatter([cx0], [cy0], [cz0], c='steelblue', s=60, zorder=5, label='orig')
        ax.scatter([cx1], [cy1], [cz1], c='tomato',    s=60, zorder=5, label='aug')
        ax.quiver(cx0, cy0, cz0, cx1 - cx0, cy1 - cy0, cz1 - cz0,
                  color='dimgrey', arrow_length_ratio=0.3, linewidth=0.8)

        ax.set_title(f'P{p_i+1}: {_PATTERN_TITLES[p_i]}', fontsize=6.5, pad=2)

    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
                   markersize=7, label='Original'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='tomato',
                   markersize=7, label='Augmented'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        f'FOA Symmetry Patterns — single source (az={AZ_DEG:.0f}°, el={EL_DEG:.0f}°)',
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = FIGURES_DIR / 'aug_patterns3d_point.svg'
    fig.savefig(out, dpi=600, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f'  [fig] {out.relative_to(_ROOT)}')


def _plot_patterns_trajectory():
    """Figure 2: a smooth trajectory transformed by each of the 15 patterns."""
    from datasets.utils import SELDAugment
    augment = SELDAugment(enabled=True, p_16patterns=1.0, p_gain=0.0, p_noise=0.0)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    N = 50
    t = np.linspace(0, 1, N)
    az_deg = -90 + 180 * t + rng.normal(0, 5, N)
    el_deg = 15 * np.sin(2 * np.pi * t) + rng.normal(0, 3, N)
    az_rad = np.radians(az_deg)
    el_rad = np.radians(el_deg)
    cx0 = np.sin(az_rad) * np.cos(el_rad)
    cy0 = np.cos(az_rad) * np.cos(el_rad)
    cz0 = np.sin(el_rad)

    fig = plt.figure(figsize=(20, 12))
    for p_i, pat in enumerate(augment._PATTERNS):
        ax = fig.add_subplot(3, 5, p_i + 1, projection='3d')
        _draw_unit_sphere(ax)

        cx1_arr, cy1_arr, cz1_arr = zip(*[
            _apply_pattern_to_xyz(pat, cx0[i], cy0[i], cz0[i])
            for i in range(N)
        ])
        cx1 = np.array(cx1_arr)
        cy1 = np.array(cy1_arr)
        cz1 = np.array(cz1_arr)

        ax.plot(cx0, cy0, cz0, color='steelblue', lw=1.2, label='orig')
        ax.plot(cx1, cy1, cz1, color='tomato',    lw=1.2, ls='--', label='aug')
        ax.scatter([cx0[0]], [cy0[0]], [cz0[0]], c='steelblue', s=25, zorder=5)
        ax.scatter([cx1[0]], [cy1[0]], [cz1[0]], c='tomato',    s=25, zorder=5)

        ax.set_title(f'P{p_i+1}: {_PATTERN_TITLES[p_i]}', fontsize=6.5, pad=2)

    handles = [
        plt.Line2D([0], [0], color='steelblue', lw=1.5, label='Original trajectory'),
        plt.Line2D([0], [0], color='tomato',    lw=1.5, ls='--',
                   label='Augmented trajectory'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle('FOA Symmetry Patterns — trajectory (50 points)', fontsize=12)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = FIGURES_DIR / 'aug_patterns3d_trajectory.svg'
    fig.savefig(out, dpi=600, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f'  [fig] {out.relative_to(_ROOT)}')


def _plot_gain():
    """Gain augmentation on sinusoidal FOA signal — 4 channels."""
    aug = _make_augment(p_16patterns=0, p_gain=1.0, p_noise=0,
                        gain_min=0.5, gain_max=0.5)
    wav, tgt, _ = _synthetic_foa_sinusoid(30, 15, T=4800)
    wav_aug, _  = aug(wav.clone(), tgt.clone())

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(11, 6), sharex=True)
    t_ms = np.arange(4800) / 24.0   # time axis in ms
    for i, ax in enumerate(axes):
        ax.plot(t_ms, wav[i].numpy(),     lw=1.0, label='original', color='steelblue')
        ax.plot(t_ms, wav_aug[i].numpy(), lw=1.0, ls='--', label='gain×0.5',
                color='tomato', alpha=0.85)
        ax.set_ylabel(['W', 'X (code_y)', 'Y (code_x)', 'Z (code_z)'][i],
                      fontsize=8)
        ax.legend(fontsize=7, loc='upper right')
        ax.set_ylim(-1.2, 1.2)
        ax.grid(True, alpha=0.3, linewidth=0.5)
    axes[-1].set_xlabel('Time (ms)', fontsize=8)
    fig.suptitle('Gain augmentation (×0.5) — 4 FOA channels  (440 Hz sinusoid, az=30°, el=15°)',
                 fontsize=9)
    plt.tight_layout()
    out = FIGURES_DIR / 'aug_gain_temporal.svg'
    fig.savefig(out, dpi=600, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f'  [fig] {out.relative_to(_ROOT)}')


def _plot_noise():
    """White noise injection — spectrograms per channel."""
    aug = _make_augment(p_16patterns=0, p_gain=0, p_noise=1.0, noise_snr_db=20.0)
    wav, tgt, _ = _synthetic_foa_sinusoid(30, 15, T=8000)
    wav_aug, _  = aug(wav.clone(), tgt.clone())

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(16, 5))
    for ch in range(4):
        for row, (sig, label) in enumerate([
            (wav,     'Original'),
            (wav_aug, 'Noisy (SNR=20 dB)'),
        ]):
            axes[row, ch].specgram(sig[ch].numpy(), Fs=24000, cmap='magma')
            axes[row, ch].set_title(f'Ch{ch}  {label}', fontsize=8)
    fig.suptitle('White Noise Injection — Spectrograms', fontsize=10)
    plt.tight_layout()
    out = FIGURES_DIR / 'aug_noise_spectrogram.svg'
    fig.savefig(out, dpi=600, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f'  [fig] {out.relative_to(_ROOT)}')


def _plot_balancing():
    """Class distribution before vs after oversampling (STARSS23 train)."""
    if not STARSS_ROOT.exists():
        print('  [skip] aug_balancing — STARSS23 missing')
        return

    from datasets.utils    import SELDAugment
    from datasets.starss23 import STARSS23Dataset

    aug     = SELDAugment(enabled=True, p_16patterns=0.20, p_gain=0.15, p_noise=0.10)
    ds_with = STARSS23Dataset(root=STARSS_ROOT, split='train',
                              model_sr=24000, hop_length=240, n_tracks=3,
                              augment=aug)
    ds_base = STARSS23Dataset(root=STARSS_ROOT, split='train',
                              model_sr=24000, hop_length=240, n_tracks=3,
                              augment=None)

    def _item_counts(ds):
        c = [0] * 13
        for ci, _ss, _ in ds._items:
            for cls in {cls for _, cls, *_ in ds._events[ci] if cls < 13}:
                c[cls] += 1
        return c

    before = _item_counts(ds_base)
    after  = _item_counts(ds_with)

    CLASS_NAMES_STARSS = [
        'Fem.spch', 'Male spch', 'Clap', 'Tel', 'Laugh',
        'Domestic', 'Footstep', 'Door', 'Music', 'Instr',
        'Water', 'Bell', 'Knock',
    ]

    x = np.arange(13)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.bar(x - 0.2, before, 0.4, label='base (no aug)',  color='steelblue')
    ax.bar(x + 0.2, after,  0.4, label='oversampled',    color='tomato', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES_STARSS, rotation=30, ha='right', fontsize=8)
    ax.set_xlabel('Class')
    ax.set_ylabel('Items containing class')
    ax.set_title('STARSS23 train — class distribution before vs after oversampling')
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    out = FIGURES_DIR / 'aug_balancing.svg'
    fig.savefig(out, dpi=600, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f'  [fig] {out.relative_to(_ROOT)}')


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Run the command-line workflow."""

    print('\n── 16 Patterns ──')
    test_patterns()

    print('\n── Gain ──')
    test_gain()

    print('\n── White Noise ──')
    test_noise()

    print('\n── Mutual Exclusion / force ──')
    test_mutual_exclusion()

    print('\n── Class Balancing ──')
    test_balancing()

    print(f'\n{"=" * 50}')
    print(f'  Tests: {_PASS} passed, {_FAIL} failed')
    print(f'{"=" * 50}')

    print('\n── Plots ──')
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _plot_patterns_single_point()
    _plot_patterns_trajectory()
    _plot_gain()
    _plot_noise()
    _plot_balancing()
    print(f'\nAll figures saved to {FIGURES_DIR.relative_to(_ROOT)}')

    sys.exit(0 if _FAIL == 0 else 1)


if __name__ == '__main__':
    main()
