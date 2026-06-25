#!/usr/bin/env python3
"""
NAS Configuration Profiler
==========================
For every config in a target directory (default: ``configs/experiments_nas-seld/foa/shallow_spec``):

  1. Instantiate ``NASModel(cfg)``
  2. Warm-up N_WARMUP forward passes (not timed)
  3. Time 1 clean forward pass
  4. Verify output shapes: ``act_logits (B,T,N,C)``, ``doa_tanh (B,T,N,C,3)``,
     ``out_semantic (B,T_sem,C)``
  5. Count total / trainable parameters
  6. Estimate GFLOPs via ``torchinfo``

Metadata (input type, module types, n_layers per stage) is read directly
from the loaded YAML — not inferred from the filename — so it works for
both shallow (``exp_spe_rn_tt_bg``) and deep naming schemes.

Results are saved to ``results/profiling/nas_configs_profile_{slug}_{timestamp}.csv``.
The ``{slug}`` encodes the config sub-group (e.g. ``foa_shallow_spec``,
``foa_shallow_gab``, ``foa_deep_reg``); the timestamp guarantees no previous
CSV is ever overwritten.

Usage::

    # default: all shallow_spec configs under configs/experiments_nas-seld/foa/shallow_spec/
    python analysis/profiling/test_seld_nas_configs.py

    # all NAS configs (all sub-groups, recursive)
    python analysis/profiling/test_seld_nas_configs.py --configs-dir configs/experiments_nas-seld

    # shallow_gab only
    python analysis/profiling/test_seld_nas_configs.py --configs-dir configs/experiments_nas-seld/foa/shallow_gab

    # deep_reg only
    python analysis/profiling/test_seld_nas_configs.py --configs-dir configs/experiments_nas-seld/foa/deep_reg

    # deep_spec only
    python analysis/profiling/test_seld_nas_configs.py --configs-dir configs/experiments_nas-seld/foa/deep_spec

Tune DEVICE / BATCH_SIZE / DURATION_S at the top of this file.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

import argparse
import csv
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch
import yaml

# ── Constants ─────────────────────────────────────────────────────────────────
DEVICE      = 'cpu'    # change to 'mps' or 'cuda' as needed
N_WARMUP    = 2        # warm-up forward passes (not timed)
BATCH_SIZE  = 1
SR          = 24000    # STARSS23 native SR; NASModel resamples to 32 kHz
DURATION_S  = 10.0

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT       = Path(__file__).resolve().parents[2]
CONFIGS_ROOT    = REPO_ROOT / 'configs' / 'experiments_nas-seld' / 'foa' / 'shallow_spec'
RESULTS_DIR     = REPO_ROOT / 'results' / 'profiling'

# Add repo root to path so that local packages are importable
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments_nas_seld.nas_model import NASModel  # noqa: E402  (after sys.path setup)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _group_slug(configs_dir: Path) -> str:
    """Derive a readable identifier from the config directory path.

    Computes the path relative to ``configs/experiments_nas-seld/``, replaces
    ``/`` with ``_``, and falls back to the directory name if the path is not
    under that root.

    Examples
    --------
    ``configs/experiments_nas-seld/foa/shallow_spec``  →  ``foa_shallow_spec``
    ``configs/experiments_nas-seld/foa/deep_reg``      →  ``foa_deep_reg``
    ``configs/experiments_nas-seld``                   →  ``all``
    """
    _nas_seld_root = REPO_ROOT / 'configs' / 'experiments_nas-seld'
    try:
        rel  = configs_dir.resolve().relative_to(_nas_seld_root.resolve())
        slug = str(rel).replace('/', '_').replace('\\', '_')
        return slug if slug else 'all'
    except ValueError:
        return configs_dir.name or 'custom'


def _count_params(model: torch.nn.Module):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _memory_mb(model: torch.nn.Module) -> float:
    """Rough parameter memory footprint in MB (float32 assumed)."""
    total  = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return total / (1024 ** 2)


def _gflops(model: torch.nn.Module, x: torch.Tensor) -> float:
    """GFLOPs for a single forward pass via torchinfo (returns nan on failure)."""
    try:
        from torchinfo import summary
        stats = summary(model, input_data=x, verbose=0, device=DEVICE)
        return stats.total_mult_adds / 1e9
    except Exception:
        return float('nan')


def _meta_from_cfg(cfg: dict) -> dict:
    """Extract architecture metadata from the loaded YAML config dict.

    Works for both shallow configs (exp_spe_rn_tt_bg style) and deep configs
    (exp_e2_l1_s1 style) because it reads values directly from the YAML rather
    than trying to parse the filename.
    """
    se  = cfg.get('spatial_early', {})
    sl  = cfg.get('spatial_late', {})
    tss = cfg.get('track_smoothing_spatial', {})
    return {
        'input_type'    : cfg.get('input', {}).get('type', ''),
        'spatial_early' : se.get('type', ''),
        'early_layers'  : se.get('n_layers', 1),
        'spatial_late'  : sl.get('type', ''),
        'late_layers'   : sl.get('n_layers', 1),
        'smoother'      : tss.get('type', ''),
        'smooth_layers' : tss.get('n_layers', 1),
    }


# ── Core profiler ─────────────────────────────────────────────────────────────

def run(configs_dir: Path) -> None:
    """Execute one configured profiling or evaluation run."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config_paths = sorted(configs_dir.rglob('exp_*.yaml'))
    if not config_paths:
        print(f"[ERROR] No exp_*.yaml configs found under {configs_dir}")
        sys.exit(1)

    slug        = _group_slug(configs_dir)
    timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path    = RESULTS_DIR / f'nas_configs_profile_{slug}_{timestamp}.csv'

    fieldnames = [
        'config_group', 'config_name',
        'input_type', 'spatial_early', 'early_layers',
        'spatial_late', 'late_layers',
        'smoother', 'smooth_layers',
        'total_params', 'trainable_params', 'memory_mb', 'gflops', 't_ms',
        'out_act_shape', 'out_doa_shape', 'out_semantic_shape', 'status',
    ]

    n_ok   = 0
    n_fail = 0

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for cfg_path in config_paths:
            name = cfg_path.stem
            row  = {'config_group': slug, 'config_name': name}

            print(f"\n{'─'*60}")
            print(f"  Config : {cfg_path.relative_to(REPO_ROOT)}")

            try:
                # ── Load config ───────────────────────────────────────────
                with open(cfg_path) as f:
                    cfg = yaml.safe_load(f)

                # ── Architecture metadata from YAML ───────────────────────
                row.update(_meta_from_cfg(cfg))
                print(f"  Arch   : early={row['spatial_early']}×{row['early_layers']}"
                      f"  late={row['spatial_late']}×{row['late_layers']}"
                      f"  smooth={row['smoother']}×{row['smooth_layers']}")

                # ── Build model ───────────────────────────────────────────
                model = NASModel(cfg).to(DEVICE).eval()
                total, trainable = _count_params(model)
                mem_mb           = _memory_mb(model)

                print(f"  Params : {total:,} total  /  {trainable:,} trainable")
                print(f"  Memory : {mem_mb:.1f} MB (params + buffers)")

                row['total_params']     = total
                row['trainable_params'] = trainable
                row['memory_mb']        = f'{mem_mb:.2f}'

                # ── Dummy input — duration from config, fallback to DURATION_S ──
                window_s = (
                    cfg.get('data', {}).get('window_s')
                    or cfg.get('training', {}).get('window_s')
                    or DURATION_S
                )
                n_samples = int(SR * window_s)
                x = torch.randn(BATCH_SIZE, 4, n_samples, device=DEVICE)
                print(f"  Window : {window_s:.1f} s  →  {n_samples} samples @ {SR} Hz")

                # ── GFLOPs (before warm-up to avoid caching bias) ─────────
                with torch.no_grad():
                    gf = _gflops(model, x)
                row['gflops'] = f'{gf:.3f}' if not (gf != gf) else 'nan'
                print(f"  GFLOPs : {row['gflops']}")

                # ── Warm-up ───────────────────────────────────────────────
                with torch.no_grad():
                    for _ in range(N_WARMUP):
                        model(x, orig_sr=SR)

                # ── Timed clean run ───────────────────────────────────────
                with torch.no_grad():
                    t0 = time.perf_counter()
                    act_logits, doa_tanh, out_semantic = model(x, orig_sr=SR)
                    t_ms = (time.perf_counter() - t0) * 1e3

                row['t_ms'] = f'{t_ms:.1f}'
                print(f"  Time   : {t_ms:.1f} ms")

                # ── Shape check ───────────────────────────────────────────
                act_shape = tuple(act_logits.shape)
                doa_shape = tuple(doa_tanh.shape)
                sem_shape = tuple(out_semantic.shape)

                row['out_act_shape']      = str(act_shape)
                row['out_doa_shape']      = str(doa_shape)
                row['out_semantic_shape'] = str(sem_shape)

                N = cfg.get('heads', {}).get('n_tracks', 5)
                C = model.n_cls

                # act_logits: (B, T, N, C)
                assert act_shape[0]  == BATCH_SIZE, f"act batch: {act_shape[0]} != {BATCH_SIZE}"
                assert act_shape[-2] == N,           f"act N_tracks: {act_shape[-2]} != {N}"
                assert act_shape[-1] == C,           f"act C: {act_shape[-1]} != {C}"
                # doa_tanh: (B, T, N, C, 3)
                assert doa_shape[0]  == BATCH_SIZE, f"doa batch: {doa_shape[0]} != {BATCH_SIZE}"
                assert doa_shape[-3] == N,           f"doa N_tracks: {doa_shape[-3]} != {N}"
                assert doa_shape[-2] == C,           f"doa C: {doa_shape[-2]} != {C}"
                assert doa_shape[-1] == 3,           f"doa last dim: {doa_shape[-1]} != 3"
                # out_semantic: (B, T_sem, C)
                assert sem_shape[0]  == BATCH_SIZE, f"semantic batch: {sem_shape[0]} != {BATCH_SIZE}"
                assert sem_shape[-1] == C,           f"semantic C: {sem_shape[-1]} != {C}"

                print(f"  Shapes : act={act_shape}  doa={doa_shape}  sem={sem_shape}  [OK]")
                row['status'] = 'OK'
                n_ok += 1

            except Exception as e:
                row['status'] = f'FAIL: {type(e).__name__}: {e}'
                for k in ('total_params', 'trainable_params', 'memory_mb', 'gflops',
                          't_ms', 'out_act_shape', 'out_doa_shape', 'out_semantic_shape',
                          'input_type', 'spatial_early', 'early_layers',
                          'spatial_late', 'late_layers', 'smoother', 'smooth_layers'):
                    row.setdefault(k, '')
                print(f"  [FAIL] {e}")
                traceback.print_exc()
                n_fail += 1

            writer.writerow(row)
            csvfile.flush()

    print(f"\n{'='*60}")
    print(f"  DONE  {n_ok}/{n_ok+n_fail} configs OK")
    print(f"  CSV   {csv_path}")
    print(f"{'='*60}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Run the command-line workflow."""

    parser = argparse.ArgumentParser(
        description='Profile NAS configurations.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analysis/profiling/test_seld_nas_configs.py
      → shallow_spec configs (default: configs/experiments_nas-seld/foa/shallow_spec/)

  python analysis/profiling/test_seld_nas_configs.py --configs-dir configs/experiments_nas-seld
      → all NAS configs (all sub-groups, recursive)

  python analysis/profiling/test_seld_nas_configs.py --configs-dir configs/experiments_nas-seld/foa/shallow_gab
      → shallow_gab group only

  python analysis/profiling/test_seld_nas_configs.py --configs-dir configs/experiments_nas-seld/foa/deep_reg
      → deep_reg group only

  python analysis/profiling/test_seld_nas_configs.py --configs-dir configs/experiments_nas-seld/foa/deep_spec
      → deep_spec group only
        """,
    )
    parser.add_argument(
        '--configs-dir', type=Path, default=None,
        help=(
            'Directory to search for exp_*.yaml configs (recursive). '
            'Defaults to configs/experiments_nas-seld/foa/shallow_spec/.'
        ),
    )
    args = parser.parse_args()

    configs_dir = (
        args.configs_dir.resolve()
        if args.configs_dir
        else CONFIGS_ROOT
    )
    run(configs_dir)


if __name__ == '__main__':
    main()
