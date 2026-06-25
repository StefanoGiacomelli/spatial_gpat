#!/usr/bin/env python3
"""
finish_incomplete_runs.py — Resume / complete interrupted SELD experiments
==========================================================================

Usage
-----
python analysis/maintenance/finish_incomplete_runs.py --experiment-path <path> [OPTIONS]

<path> can be either:
  * The experiment run directory (e.g. runs/at2seld_nas/loss_experiments/balanced_starss23_expA)
  * The experiment checkpoints sub-directory (…/checkpoints/<exp_name>)
  * A specific *.ckpt file

Options
-------
--mode          auto | fit | test          (default: auto)
--checkpoint    auto | last | best | PATH  (default: auto)
--family        auto | nas | standalone    (default: auto)
--device        auto | cuda | mps | cpu    (default: auto)
--dry-run       Print plan without running anything.
--skip-test     Run fit only (no test phase).
--test-only     Run test only (no fit phase).

Notes
-----
* sed_eval and DOA oracle are NEVER computed here (flags default to False).
* Data root paths in hparams.yaml are remapped from HPC lustre paths to local paths.
* A completion_report.json is written to the experiment run directory after success.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── repo root ──────────────────────────────────────────────────────────────────
_AUX_DIR = Path(__file__).resolve().parent
_ROOT    = _AUX_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── local dataset roots (mirrors main_nas-seld.py / main_standalone_seld.py) ──
_LOCAL_DATASET_ROOTS = {
    'STARSS23':       _ROOT / 'data' / 'STARSS23',
    'TAU-NIGENS2021': _ROOT / 'data' / 'TAU-NIGENS2021',
    'TAU-NIGENS2020': _ROOT / 'data' / 'TAU-NIGENS2020',
    'TAU2019':        _ROOT / 'data' / 'TAU2019',
}
# Keys that may appear in data cfg for single-dataset or mixed-dataset runs
_MIXED_ROOT_KEYS = {
    'starss_root':  _LOCAL_DATASET_ROOTS['STARSS23'],
    'tau_root':     _LOCAL_DATASET_ROOTS['TAU-NIGENS2021'],
    'tau2019_root': _LOCAL_DATASET_ROOTS['TAU2019'],
}
_SINGLE_ROOT_KEY = 'root_path'


# ══════════════════════════════════════════════════════════════════════════════
# Path utilities
# ══════════════════════════════════════════════════════════════════════════════

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')


def _resolve_exp_dir(path_arg: Path) -> Path:
    """
    Given any of:
      runs/…/balanced_starss23_expA              (run dir)
      runs/…/balanced_starss23_expA/checkpoints  (checkpoints subdir)
      runs/…/checkpoints/expA_name              (checkpoints/exp_name subdir)
      runs/…/checkpoints/expA_name/34_0.3375.ckpt  (specific ckpt)
    Return the *run* directory (parent that contains experiment_state.json or checkpoints/).
    """
    p = path_arg.resolve()
    if p.suffix == '.ckpt':
        p = p.parent  # strip filename
    # Walk up until we find a directory that has 'checkpoints' or 'experiment_state.json'
    candidate = p
    for _ in range(6):
        if (candidate / 'checkpoints').exists() or (candidate / 'experiment_state.json').exists():
            return candidate
        # Also handle: we were given the exp_name subdir inside checkpoints/
        if candidate.parent.name == 'checkpoints':
            grandparent = candidate.parent.parent
            if (grandparent / 'experiment_state.json').exists():
                return grandparent
        candidate = candidate.parent
    return p  # best effort


def _find_hparams(exp_dir: Path, given_path: Path | None = None) -> Path | None:
    """
    Search for hparams.yaml produced by Lightning's TensorBoardLogger.
    Expected: <exp_dir>/checkpoints/<exp_name>/version_0/hparams.yaml

    If given_path is provided (the original --experiment-path argument), first
    look for hparams.yaml along that path so that multi-experiment group
    directories resolve to the correct per-experiment hparams rather than the
    alphabetically-first one.
    """
    # Prefer hparams found along the originally-given path (handles multi-exp groups)
    if given_path:
        search = given_path.resolve()
        for candidate in [search] + list(search.parents):
            hp = candidate / 'version_0' / 'hparams.yaml'
            if hp.exists():
                return hp
            if candidate == exp_dir:
                break

    ckpt_dir = exp_dir / 'checkpoints'
    if not ckpt_dir.exists():
        return None
    for version_dir in sorted(ckpt_dir.rglob('version_0')):
        hp = version_dir / 'hparams.yaml'
        if hp.exists():
            return hp
    return None


def _find_checkpoint(exp_dir: Path, mode: str) -> Path | None:
    """
    Locate a checkpoint to resume / test from.

    mode: 'auto' | 'last' | 'best'
      auto : prefer last.ckpt, then highest-epoch *.ckpt
      last : last.ckpt only
      best : highest-epoch (or highest-score) *.ckpt, ignoring last.ckpt
    """
    ckpt_dir = exp_dir / 'checkpoints'
    if not ckpt_dir.exists():
        return None

    # Flatten search: look one level deep (exp_name subdir)
    all_ckpts = list(ckpt_dir.rglob('*.ckpt'))
    last_ckpts = [c for c in all_ckpts if c.name == 'last.ckpt']
    scored     = sorted(
        [c for c in all_ckpts if c.name != 'last.ckpt'],
        key=lambda p: p.name,
    )

    if mode == 'last':
        return last_ckpts[0] if last_ckpts else None
    if mode == 'best':
        return scored[-1] if scored else None
    # auto
    if last_ckpts:
        return last_ckpts[0]
    return scored[-1] if scored else None


# ══════════════════════════════════════════════════════════════════════════════
# hparams / cfg helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_hparams_cfg(hparams_path: Path) -> dict:
    import yaml
    with open(hparams_path) as fh:
        hp = yaml.safe_load(fh)
    cfg = hp.get('cfg', hp)  # Lightning saves {'cfg': full_dict}
    return cfg


def _remap_data_roots(cfg: dict) -> dict:
    """
    Replace any HPC lustre paths in cfg['data'] with local paths.
    Works for both single-dataset (root_path) and mixed-dataset (starss_root etc.) configs.
    """
    import copy
    cfg = copy.deepcopy(cfg)
    data_cfg = cfg.setdefault('data', {})

    # single-dataset
    if _SINGLE_ROOT_KEY in data_cfg:
        remote = data_cfg[_SINGLE_ROOT_KEY]
        dataset = cfg.get('semantic', {}).get('dataset', 'STARSS23')
        local   = _LOCAL_DATASET_ROOTS.get(dataset)
        if local and remote != str(local):
            print(f'  [remap] data.root_path: {remote!r} → {str(local)!r}')
            data_cfg[_SINGLE_ROOT_KEY] = str(local)

    # mixed-dataset
    for key, local in _MIXED_ROOT_KEYS.items():
        if key in data_cfg:
            remote = data_cfg[key]
            if remote != str(local):
                print(f'  [remap] data.{key}: {remote!r} → {str(local)!r}')
            data_cfg[key] = str(local)

    return cfg


def _disable_heavy_diagnostics(cfg: dict) -> dict:
    """Force sed_eval and DOA oracle off in the metrics section."""
    import copy
    cfg = copy.deepcopy(cfg)
    cfg.setdefault('metrics', {})['compute_doa_oracle_in_training_test'] = False
    cfg.setdefault('metrics', {})['compute_sed_eval_in_training_test']   = False
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# Completeness check
# ══════════════════════════════════════════════════════════════════════════════

def _has_test_row(exp_dir: Path) -> bool:
    """Return True if metrics_summary.csv already contains a 'test' row."""
    csv_candidates = list((exp_dir / 'metrics').glob('metrics_summary.csv'))
    if not csv_candidates:
        csv_candidates = list(exp_dir.rglob('metrics_summary.csv'))
    for csv in csv_candidates:
        try:
            text = csv.read_text()
            for line in text.splitlines()[1:]:
                if line.startswith('test,') or ',test,' in line:
                    return True
        except Exception:
            pass
    return False


def _is_state_completed(exp_dir: Path, exp_name: str) -> bool:
    state_file = exp_dir / 'experiment_state.json'
    if not state_file.exists():
        return False
    try:
        state = json.loads(state_file.read_text())
        return state.get(exp_name, {}).get('status') == 'completed'
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Family detection
# ══════════════════════════════════════════════════════════════════════════════

def _detect_family(exp_dir: Path, hparams_path: Path | None) -> str:
    """Return 'nas' or 'standalone'."""
    path_str = str(exp_dir)
    if 'at2seld_nas' in path_str or 'nas_seld' in path_str:
        return 'nas'
    if 'standalone_seld' in path_str or 'standalone' in path_str:
        return 'standalone'
    # Try reading hparams
    if hparams_path:
        hp_text = hparams_path.read_text()
        if 'NASLightningModule' in hp_text or 'cross_stitch' in hp_text:
            return 'nas'
        if 'StandaloneSELDModule' in hp_text or 'conformer' in hp_text:
            return 'standalone'
    raise RuntimeError(
        f"Cannot auto-detect family from path: {exp_dir}\n"
        "Pass --family nas or --family standalone explicitly."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Build Lightning objects
# ══════════════════════════════════════════════════════════════════════════════

def _build_objects(cfg: dict, family: str, exp_dir: Path, exp_name: str):
    """Return (model, dm, metrics_path)."""
    import json as _json

    if family == 'nas':
        from experiments_nas_seld.train import NASLightningModule
        from datasets.datamodule import SELDDataModule, BalancedSELDDataModule, FoldCyclingDataModule
        ModuleCls = NASLightningModule
    else:
        from experiments_standalone_seld.train import StandaloneSELDModule
        from datasets.datamodule import SELDDataModule, BalancedSELDDataModule, FoldCyclingDataModule
        ModuleCls = StandaloneSELDModule

    dataset_name = cfg.get('semantic', {}).get('dataset', 'STARSS23')

    def _make_dm(c):
        if dataset_name == 'BalancedSTARSS23':
            return BalancedSELDDataModule(c)
        if dataset_name == 'TAU2019':
            return FoldCyclingDataModule(c)
        return SELDDataModule(c)

    train_dm = _make_dm(cfg)
    test_dm  = _make_dm(cfg) if dataset_name == 'BalancedSTARSS23' else train_dm

    metrics_dir  = exp_dir / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f'{exp_name}.json'

    model = ModuleCls(cfg)
    model.set_metrics_path(metrics_path)

    # Restore existing metrics log to avoid gaps in the JSON
    if metrics_path.exists():
        try:
            model._epoch_metrics_log = _json.loads(metrics_path.read_text())
            print(f'  Metrics log restored: {len(model._epoch_metrics_log)} entries')
        except Exception as exc:
            print(f'  [WARN] Could not restore metrics log: {exc}')

    return model, train_dm, test_dm, metrics_path


def _build_trainer(cfg: dict, exp_dir: Path, exp_name: str, family: str,
                   device: str, do_fit: bool, do_test: bool):
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
    from lightning.pytorch.loggers  import TensorBoardLogger

    ckpt_dir_path = exp_dir / 'checkpoints' / exp_name
    ckpt_dir_path.mkdir(parents=True, exist_ok=True)

    tb_logger = TensorBoardLogger(
        save_dir = str(exp_dir / 'checkpoints'),
        name     = exp_name,
        version  = 0,
    )

    callbacks = []
    if do_fit:
        ckpt_cb = ModelCheckpoint(
            dirpath                 = str(ckpt_dir_path),
            filename                = '{epoch:02d}_{val_seld_score:.4f}',
            monitor                 = 'val_seld_score',
            mode                    = 'min',
            save_top_k              = 1,
            save_last               = True,
            verbose                 = True,
            auto_insert_metric_name = False,
        )
        callbacks.append(ckpt_cb)
        patience = cfg.get('training', {}).get('patience', 20)
        if patience > 0:
            callbacks.append(EarlyStopping(
                monitor  = 'val_seld_score',
                mode     = 'min',
                patience = patience,
                verbose  = True,
            ))
    else:
        ckpt_cb = None

    dataset_name = cfg.get('semantic', {}).get('dataset', 'STARSS23')

    trainer_kwargs: dict = dict(
        max_epochs           = cfg.get('training', {}).get('epochs', 40),
        callbacks            = callbacks,
        accelerator          = device,
        devices              = 1,
        gradient_clip_val    = 1.0,
        log_every_n_steps    = 5,
        logger               = tb_logger,
        num_sanity_val_steps = 0,
    )
    if dataset_name == 'TAU2019' and do_fit:
        trainer_kwargs['reload_dataloaders_every_n_epochs'] = 1

    trainer = L.Trainer(**trainer_kwargs)
    return trainer, ckpt_cb


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description='Resume / complete interrupted SELD experiments.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--experiment-path', required=True, type=Path,
                   metavar='PATH',
                   help='Run dir, checkpoints subdir, or *.ckpt file.')
    p.add_argument('--mode', choices=['auto', 'fit', 'test'], default='auto',
                   help='auto=detect missing steps; fit=train only; test=test only.')
    p.add_argument('--checkpoint', default='auto',
                   metavar='auto|last|best|PATH',
                   help='Checkpoint to resume/test from.')
    p.add_argument('--family', choices=['auto', 'nas', 'standalone'], default='auto',
                   help='Experiment family (auto=detect from path).')
    p.add_argument('--device', choices=['auto', 'cuda', 'mps', 'cpu'], default='auto',
                   help='Accelerator to use.')
    p.add_argument('--dry-run', action='store_true',
                   help='Print plan without executing anything.')
    p.add_argument('--skip-test', action='store_true',
                   help='Run fit only, skip test phase.')
    p.add_argument('--test-only', action='store_true',
                   help='Run test phase only (no fit).')
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Run the command-line workflow."""

    import torch

    args = _parse_args()

    if args.skip_test and args.test_only:
        print('[ERROR] --skip-test and --test-only are mutually exclusive.')
        sys.exit(1)

    # ── Resolve paths ──────────────────────────────────────────────────────────
    given_path = args.experiment_path.resolve()
    exp_dir    = _resolve_exp_dir(given_path)
    print(f'Experiment dir : {exp_dir}')

    # ── Load hparams ───────────────────────────────────────────────────────────
    hparams_path = _find_hparams(exp_dir, given_path=given_path)
    if hparams_path is None:
        print('[ERROR] Could not find hparams.yaml under', exp_dir)
        sys.exit(1)
    print(f'hparams.yaml   : {hparams_path}')

    cfg      = _load_hparams_cfg(hparams_path)
    exp_name = cfg.get('experiment', {}).get('name', exp_dir.name)
    print(f'Experiment name: {exp_name}')

    # ── Family ────────────────────────────────────────────────────────────────
    family = args.family
    if family == 'auto':
        family = _detect_family(exp_dir, hparams_path)
    print(f'Family         : {family}')

    # ── Remap paths + disable heavy diagnostics ────────────────────────────────
    cfg = _remap_data_roots(cfg)
    cfg = _disable_heavy_diagnostics(cfg)

    # ── Determine what to run ─────────────────────────────────────────────────
    if args.test_only or args.mode == 'test':
        do_fit, do_test = False, True
    elif args.skip_test or args.mode == 'fit':
        do_fit, do_test = True, False
    elif args.mode == 'auto':
        has_test = _has_test_row(exp_dir)
        is_done  = _is_state_completed(exp_dir, exp_name)
        if is_done and has_test:
            print(f'\n[INFO] {exp_name} is already completed with test metrics. Nothing to do.')
            print('       Pass --test-only or --mode fit to force a re-run.')
            sys.exit(0)
        do_fit  = not (is_done and not has_test)
        do_test = not has_test
        print(f'Auto-detected  : fit={do_fit}, test={do_test}')
    else:
        do_fit, do_test = True, True

    # ── Device ────────────────────────────────────────────────────────────────
    device = args.device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    print(f'Device         : {device}')

    # ── Checkpoint ────────────────────────────────────────────────────────────
    ckpt_path: Path | None = None
    if args.checkpoint not in ('auto', 'last', 'best'):
        ckpt_path = Path(args.checkpoint).resolve()
        if not ckpt_path.exists():
            print(f'[ERROR] Checkpoint not found: {ckpt_path}')
            sys.exit(1)
    else:
        ckpt_mode = args.checkpoint  # 'auto' | 'last' | 'best'
        if do_fit:
            ckpt_path = _find_checkpoint(exp_dir, ckpt_mode)
        elif do_test and not do_fit:
            # test-only: prefer best (non-last) checkpoint
            ckpt_path = _find_checkpoint(exp_dir, 'best' if ckpt_mode == 'auto' else ckpt_mode)

    if ckpt_path:
        print(f'Checkpoint     : {ckpt_path}')
    else:
        print('Checkpoint     : (none — starting fresh)')

    # ── Dry-run summary ───────────────────────────────────────────────────────
    print()
    print(f'Plan: fit={do_fit}, test={do_test}, dry_run={args.dry_run}')
    if args.dry_run:
        print('[DRY RUN] Exiting without running anything.')
        return

    # ── Build model, datamodules, trainer ────────────────────────────────────
    model, train_dm, test_dm, metrics_path = _build_objects(cfg, family, exp_dir, exp_name)
    trainer, ckpt_cb = _build_trainer(cfg, exp_dir, exp_name, family, device, do_fit, do_test)

    dataset_name = cfg.get('semantic', {}).get('dataset', 'STARSS23')
    best_ckpt_path: str | None = None

    # ── Fit ───────────────────────────────────────────────────────────────────
    if do_fit:
        print('\n=== FIT ===')
        resume = str(ckpt_path) if ckpt_path else None
        trainer.fit(model, train_dm, ckpt_path=resume)
        best_ckpt_path = ckpt_cb.best_model_path if ckpt_cb else None

        # Clean up last.ckpt (resume target, not needed post-fit)
        last_path = exp_dir / 'checkpoints' / exp_name / 'last.ckpt'
        if last_path.exists():
            last_path.unlink()

    # ── Test ──────────────────────────────────────────────────────────────────
    if do_test:
        print('\n=== TEST ===')
        if do_fit:
            test_ckpt = 'best'
        else:
            test_ckpt = str(ckpt_path) if ckpt_path else None

        if dataset_name == 'BalancedSTARSS23':
            trainer.test(model, test_dm, ckpt_path=test_ckpt)
        else:
            trainer.test(model, train_dm, ckpt_path=test_ckpt)

    # ── Write completion report ───────────────────────────────────────────────
    report = {
        'exp_name':          exp_name,
        'family':            family,
        'device':            device,
        'do_fit':            do_fit,
        'do_test':           do_test,
        'resume_ckpt':       str(ckpt_path) if ckpt_path else None,
        'best_ckpt':         best_ckpt_path,
        'metrics_file':      str(metrics_path),
        'completed_at':      _now_iso(),
        'sed_eval_computed': False,
        'doa_oracle_computed': False,
    }
    report_path = exp_dir / 'completion_report.json'
    report_path.write_text(json.dumps(report, indent=2))
    print(f'\nCompletion report written to: {report_path}')

    # ── Update experiment_state.json ──────────────────────────────────────────
    state_file = exp_dir / 'experiment_state.json'
    try:
        state = json.loads(state_file.read_text()) if state_file.exists() else {}
    except Exception:
        state = {}
    state[exp_name] = {
        'status':       'completed',
        'best_ckpt':    best_ckpt_path or state.get(exp_name, {}).get('best_ckpt', ''),
        'metrics_file': str(metrics_path),
        'completed_at': _now_iso(),
    }
    state_file.write_text(json.dumps(state, indent=2))
    print(f'State file updated: {state_file}')


if __name__ == '__main__':
    main()
