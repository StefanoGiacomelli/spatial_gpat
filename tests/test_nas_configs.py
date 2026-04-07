"""
NAS Configuration Profiler
==========================
For every config in a target directory (default: ``configs/experiments/**``):

  1. Instantiate ``NASModel(cfg)``
  2. Warm-up N_WARMUP forward passes (not timed)
  3. Time 1 clean forward pass
  4. Verify output shapes: ``act_logits (B,T,N,C)``, ``doa_tanh (B,T,N,C,3)``,
     ``out_semantic (B,T_sem,C)``
  5. Count total / trainable parameters
  6. Estimate GFLOPs via ``torchinfo``

Results are saved to ``results/nas_configs_profile_{group_slug}_{timestamp}.csv``,
where ``group_slug`` is derived from the config directory path relative to
``configs/experiments/`` (e.g. ``foa_shallow_spec``).

Usage::

    # all configs under configs/experiments/ (recursive)
    python tests/test_nas_configs.py

    # only a specific sub-group
    python tests/test_nas_configs.py --configs-dir configs/experiments/foa/shallow_spec

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
DURATION_S  = 30.0

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT       = Path(__file__).resolve().parent.parent
CONFIGS_ROOT    = REPO_ROOT / 'configs' / 'experiments'
RESULTS_DIR     = REPO_ROOT / 'results'

# Add repo root to path so that local packages are importable
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.nas_model import NASModel  # noqa: E402  (after sys.path setup)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _group_slug(configs_dir: Path) -> str:
    """Derive a short identifier from the config directory.

    Computes the path relative to ``configs/experiments/``, replaces ``/`` with
    ``_``, and falls back to the directory name if the path is not under the
    expected root.

    Examples
    --------
    ``configs/experiments/foa/shallow_spec``  →  ``foa_shallow_spec``
    ``configs/experiments/foa``               →  ``foa``
    ``configs/experiments``                   →  ``all``
    """
    try:
        rel = configs_dir.resolve().relative_to(CONFIGS_ROOT.resolve())
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
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return total / (1024 ** 2)


def _gflops(model: torch.nn.Module, x: torch.Tensor) -> float:
    """GFLOPs for a single forward pass via torchinfo (returns nan on failure)."""
    try:
        from torchinfo import summary
        stats = summary(
            model,
            input_data=x,
            verbose=0,
            device=DEVICE,
        )
        return stats.total_mult_adds / 1e9
    except Exception:
        return float('nan')


def _parse_config_meta(name: str) -> dict:
    """Extract input / early / late / smoother tokens from the filename."""
    # expected pattern: exp_{inp}_{early}_{late}_{smooth}
    parts = name.split('_')
    if len(parts) == 5 and parts[0] == 'exp':
        return {
            'input_type'    : parts[1],
            'spatial_early' : parts[2],
            'spatial_late'  : parts[3],
            'smoother'      : parts[4],
        }
    return {'input_type': '', 'spatial_early': '', 'spatial_late': '', 'smoother': ''}


# ── Main ──────────────────────────────────────────────────────────────────────

def run(configs_dir: Path) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config_paths = sorted(configs_dir.rglob('exp_*.yaml'))
    if not config_paths:
        print(f"[ERROR] No exp_*.yaml configs found under {configs_dir}")
        sys.exit(1)

    slug        = _group_slug(configs_dir)
    timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path    = RESULTS_DIR / f'nas_configs_profile_{slug}_{timestamp}.csv'

    fieldnames  = [
        'config_group', 'config_name',
        'input_type', 'spatial_early', 'spatial_late', 'smoother',
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
            meta = _parse_config_meta(name)
            row  = {'config_group': slug, 'config_name': name, **meta}

            print(f"\n{'─'*60}")
            print(f"  Config : {cfg_path.relative_to(REPO_ROOT)}")

            try:
                # ── Load config ───────────────────────────────────────────
                with open(cfg_path) as f:
                    cfg = yaml.safe_load(f)

                # ── Build model ───────────────────────────────────────────
                model = NASModel(cfg).to(DEVICE).eval()
                total, trainable = _count_params(model)
                mem_mb           = _memory_mb(model)

                print(f"  Params : {total:,} total  /  {trainable:,} trainable")
                print(f"  Memory : {mem_mb:.1f} MB (params + buffers)")

                row['total_params']     = total
                row['trainable_params'] = trainable
                row['memory_mb']        = f'{mem_mb:.2f}'

                # ── Dummy input ───────────────────────────────────────────
                n_samples = int(SR * DURATION_S)
                x = torch.randn(BATCH_SIZE, 4, n_samples, device=DEVICE)

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
                          't_ms', 'out_act_shape', 'out_doa_shape', 'out_semantic_shape'):
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


def main() -> None:
    parser = argparse.ArgumentParser(description='Profile NAS configurations.')
    parser.add_argument(
        '--configs-dir', default=None,
        help=(
            'Directory to search for exp_*.yaml configs (recursive). '
            'Defaults to configs/experiments/ (all groups).'
        ),
    )
    args = parser.parse_args()

    configs_dir = Path(args.configs_dir).resolve() if args.configs_dir else CONFIGS_ROOT
    run(configs_dir)


if __name__ == '__main__':
    main()
