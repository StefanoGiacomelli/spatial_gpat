#!/usr/bin/env python3
"""
analysis/experiments/export_run_configs.py
==============================================================================
Effective-configuration ledger: one row per executed run (what actually ran)
==============================================================================

The *stored* ``hparams.yaml`` of every run (``runs/**/checkpoints/<exp>/version_0/``)
is the ground truth of the configuration that was executed; the YAML files under
``configs/`` are templates that were edited after many runs (see
``docs/REPRODUCIBILITY.md``).  This script exports, for every run, the hyper-
parameters that matter for the reported tables together with the best epoch and
the validation/test metrics logged by the run itself, to

    results/experiments/run_configs.csv

Nothing under ``runs/`` is modified.  Missing checkpoints/metrics are tolerated
(the corresponding columns stay empty), so the ledger can be regenerated on a
machine that only holds the ``hparams.yaml``/``metrics/*.json`` part of the
OneDrive archive.

Columns (selection)
-------------------
``run_id`` = ``<family>/<group>/<exp>``; architecture (``early_type`` ...
``n_tracks``); semantic group (``sem_init``, ``sem_frozen``); loss/optimisation
(``lambda_spatial`` ... ``scheduler``); ``batch_size`` and
``effective_batch_default_rule`` (= batch x 16 // batch, the runner default rule:
a CLI ``--accum-steps`` override is not stored in ``hparams.yaml``);
``optimizer_inferred_from_code`` (Adam for the AT2SELD runner, AdamW for the standalone
runner; the optimizer, the learning-rate schedule and the early-stopping patience of the
Stage 1-3 runs are *not* stored in their hparams: the runner defaults of the day applied);
augmentation block; ``best_epoch`` / ``best_val_*`` (minimum logged validation
SELD); ``epochs_logged`` vs ``epochs_unique`` (> means replayed epochs after a
resume from ``last.ckpt``); test metrics of the run (training-time evaluation at
the configured threshold; ``test_*`` = ``test/`` or, for BalancedSTARSS23 runs,
the ``test_starss/`` loader).

Usage
-----
    python analysis/experiments/export_run_configs.py            # writes the CSV
    python analysis/experiments/export_run_configs.py --check    # compare with the tracked CSV

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
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.paths import RUNS_DIR, EXPERIMENT_RESULTS_DIR  # noqa: E402

DEFAULT_OUT = EXPERIMENT_RESULTS_DIR / "run_configs.csv"
EFFECTIVE_BATCH = 16   # main_nas-seld.py / main_standalone_seld.py constant


def _g(d: Any, path: str, default: Any = None) -> Any:
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _first(*values: Any) -> Any:
    for v in values:
        if v is not None:
            return v
    return None


def _round(v: Any, nd: int = 6) -> Any:
    return round(v, nd) if isinstance(v, float) else v


def find_hparams(runs: Path) -> List[Path]:
    found = sorted(runs.glob("*/*/checkpoints/*/version_0/hparams.yaml"))
    found += sorted(runs.glob("*/*/*/checkpoints/*/version_0/hparams.yaml"))
    return found


def best_val(metrics: List[dict]) -> Optional[dict]:
    vals = [e for e in metrics if isinstance(e.get("epoch"), int) and "val/SELD_score" in e]
    return min(vals, key=lambda e: e["val/SELD_score"]) if vals else None


def run_row(hp: Path, runs: Path) -> Dict[str, Any]:
    rel = hp.relative_to(runs).parts
    idx = rel.index("checkpoints")
    family = rel[0]
    group = "/".join(rel[:idx])
    exp = rel[idx + 1]
    group_dir = runs.joinpath(*rel[:idx])
    ckdir = runs.joinpath(*rel[: idx + 2])
    cfg = yaml.safe_load(hp.read_text()).get("cfg", {})
    ckpts = sorted(f for f in ckdir.glob("*.ckpt"))
    best_ckpts = [c for c in ckpts if not c.name.startswith("last")]
    mpath = group_dir / "metrics" / f"{exp}.json"
    metrics = json.loads(mpath.read_text()) if mpath.exists() else []
    b = best_val(metrics)
    n_val = sum(1 for e in metrics if isinstance(e.get("epoch"), int) and "val/SELD_score" in e)
    n_uniq = len({e["epoch"] for e in metrics
                  if isinstance(e.get("epoch"), int) and "val/SELD_score" in e})
    test_rows = [e for e in metrics if e.get("epoch") == "test"]
    state: Dict[str, Any] = {}
    sp = group_dir / "experiment_state.json"
    if sp.exists():
        sd = json.loads(sp.read_text())
        state = sd.get(exp, {}) if isinstance(sd, dict) else {}
    dl = _g(cfg, "training.doa_loss") or {}
    batch = _g(cfg, "training.batch_size")
    row: Dict[str, Any] = {
        "run_id": f"{group}/{exp}", "family": family, "group": group, "exp_name": exp,
        "model": "AT2SELD" if family == "at2seld_nas" else
                 f"standalone_{_g(cfg, 'spatial_early.type')}",
        "dataset": _first(_g(cfg, "semantic.dataset"), _g(cfg, "training.dataset_mode")),
        "early_type": _g(cfg, "spatial_early.type"),
        "early_layers": _first(_g(cfg, "spatial_early.n_layers"),
                               _g(cfg, "spatial_early.n_conf_layers")),
        "use_msca": _g(cfg, "spatial_early.use_msca"),
        "late_type": _g(cfg, "spatial_late.type"),
        "late_layers": _g(cfg, "spatial_late.n_layers"),
        "smoother_type": _first(_g(cfg, "track_smoothing_spatial.type"),
                                _g(cfg, "track_smoothing.type")),
        "smoother_layers": _first(_g(cfg, "track_smoothing_spatial.n_layers"),
                                  _g(cfg, "track_smoothing.n_layers")),
        "dropout_early": _g(cfg, "spatial_early.dropout"),
        "dropout_late": _g(cfg, "spatial_late.dropout"),
        "dropout_smoother": _first(_g(cfg, "track_smoothing_spatial.dropout"),
                                   _g(cfg, "track_smoothing.dropout")),
        "cs_early": _g(cfg, "cross_stitch.cs_early_enabled"),
        "cs_late": _g(cfg, "cross_stitch.cs_late_enabled"),
        "input_type": _g(cfg, "input.type"),
        "n_tracks": _g(cfg, "heads.n_tracks"),
        "n_classes": _g(cfg, "heads.n_classes"),
        "sem_init": ",".join(str(_g(cfg, f"semantic.{k}.init"))
                             for k in ("early_group", "late_group", "head_group")),
        "sem_frozen": ",".join(str(_g(cfg, f"semantic.{k}.frozen"))
                               for k in ("early_group", "late_group", "head_group")),
        "lambda_spatial": _g(cfg, "training.lambda_spatial"),
        "lambda_semantic": _g(cfg, "training.lambda_semantic"),
        "beta_sed": _g(cfg, "training.beta_sed"),
        "focal_gamma": _g(cfg, "training.focal_gamma"),
        "doa_loss_mode": dl.get("mode"),
        "inactive_weight": dl.get("inactive_weight"),
        "norm_weight": dl.get("norm_weight"),
        "smooth_weight": dl.get("smooth_weight"),
        "use_sed_pos_weight": _g(cfg, "training.use_sed_pos_weight"),
        "pos_weight_max": _g(cfg, "training.pos_weight_max"),
        "optimizer_inferred_from_code": "Adam" if family == "at2seld_nas" else "AdamW",
        "epochs_cfg": _g(cfg, "training.epochs"),
        "batch_size": batch,
        "effective_batch_default_rule": (batch * max(1, EFFECTIVE_BATCH // batch)
                                         if isinstance(batch, int) and batch > 0 else None),
        "lr": _g(cfg, "training.lr"),
        "scheduler": _g(cfg, "training.scheduler"),
        "lr_patience": _g(cfg, "training.lr_patience"),
        "lr_reduce_factor": _g(cfg, "training.lr_reduce_factor"),
        "patience": _g(cfg, "training.patience"),
        "window_s": _first(_g(cfg, "data.window_s"), _g(cfg, "training.window_s")),
        "hop_s": _first(_g(cfg, "data.hop_s"), _g(cfg, "training.hop_s")),
        "val_fraction": _g(cfg, "data.val_fraction"),
        "val_seed": _g(cfg, "data.val_seed"),
        "seed": _g(cfg, "experiment.seed"),
        "aug_enabled": _g(cfg, "augmentation.enabled"),
        "aug_p_16patterns": _g(cfg, "augmentation.p_16patterns"),
        "aug_p_gain": _g(cfg, "augmentation.p_gain"),
        "aug_p_noise": _g(cfg, "augmentation.p_noise"),
        "aug_gain_min": _g(cfg, "augmentation.gain_min"),
        "aug_gain_max": _g(cfg, "augmentation.gain_max"),
        "aug_noise_snr_db": _g(cfg, "augmentation.noise_snr_db"),
        "act_thresh_cfg": _g(cfg, "metrics.act_thresh"),
        "stored_data_root": _first(_g(cfg, "data.root_path"), _g(cfg, "data.starss_root")),
        "epochs_logged": n_val, "epochs_unique": n_uniq,
        "replayed_epochs": n_val != n_uniq,
        "best_epoch": b["epoch"] if b else None,
        "best_val_SELD": _round(b["val/SELD_score"]) if b else None,
        "best_val_ER": _round(b.get("val/ER_20")) if b else None,
        "best_val_F": _round(b.get("val/F_20")) if b else None,
        "best_val_LE": _round(b.get("val/LE_CD")) if b else None,
        "best_val_LR": _round(b.get("val/LR_CD")) if b else None,
        "checkpoints_present": ";".join(c.name for c in ckpts),
        "best_checkpoint": best_ckpts[0].name if len(best_ckpts) == 1 else
                           ";".join(c.name for c in best_ckpts),
        "state_status": state.get("status"),
        "state_completed_at": state.get("completed_at"),
        "completion_report": (group_dir / "completion_report.json").exists(),
        "metrics_json": str(mpath.relative_to(runs)) if mpath.exists() else None,
    }
    # headline test row (training-time test, configured threshold, CUDA cluster runs)
    t = test_rows[-1] if test_rows else {}
    key = "test" if any(k.startswith("test/") for k in t) else "test_starss"
    row["test_key"] = key if t else None
    for short, long in (("SELD", "SELD_score"), ("ER", "ER_20"), ("F", "F_20"),
                        ("LE", "LE_CD"), ("LR", "LR_CD")):
        row[f"test_{short}"] = _round(t.get(f"{key}/{long}")) if t else None
    for short in ("sem_accuracy", "sem_f1", "sem_auroc"):
        row[f"test_{short}"] = _round(t.get(f"{key}/{short}")) if t else None
    for short in ("acc_20deg", "angular_error_median", "pred_norm_active_median",
                  "zero_norm_active_lt_0.10", "evaluated_active_slots"):
        row[f"test_doa_{short}"] = _round(_first(t.get(f"{key}/doa_{short}"),
                                                 t.get(f"{key}/doa_oracle_{short}")))
    for other in ("test_tau", "test_tau2019"):
        row[f"{other}_SELD"] = _round(t.get(f"{other}/SELD_score")) if t else None
    return row


def build_rows(runs: Path) -> List[Dict[str, Any]]:
    return [run_row(hp, runs) for hp in find_hparams(runs)]


def write_csv(rows: List[Dict[str, Any]], out: Path) -> None:
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("Columns")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true",
                    help="regenerate in memory and compare with the existing CSV")
    args = ap.parse_args()
    rows = build_rows(args.runs_dir)
    if args.check:
        import io
        tmp = io.StringIO()
        fields: List[str] = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
        w = csv.DictWriter(tmp, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
        same = args.out.exists() and args.out.read_text() == tmp.getvalue()
        print(f"{len(rows)} runs; ledger {'identical' if same else 'DIFFERS'}: {args.out}")
        sys.exit(0 if same else 1)
    write_csv(rows, args.out)
    print(f"{len(rows)} runs -> {args.out}")


if __name__ == "__main__":
    main()
