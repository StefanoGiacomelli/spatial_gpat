#!/usr/bin/env python3
"""
analysis/experiments/export_ablation_table.py
==============================================================================
T7 — The existing ablation / control family as one tracked table
==============================================================================

The repository contains controls and ablations that reports can cite truthfully:
this script lists them in one table, joining the effective-configuration
ledger (``results/experiments/run_configs.csv``) with the oracle-activity DOA statistics.

Alignment matters.  The NAS-SELD oracle logs (``results/diagnostics/doa_oracle_test/nas-seld/``) are
tPIT-aligned.  The stored oracle logs of the *standalone* baselines (``.../doa_oracle_test/conformer/``) were
written by ``doa_oracle_test.py`` **without** alignment although these models are trained with the same
tPITLoss: their slot order is arbitrary, so those statistics (acc@20 3.9-13.6 %, median 67-91 deg) are not
comparable with the NAS-SELD ones and are listed in separate ``raw_*`` columns only.  Comparable (tPIT-aligned)
standalone values: the offline re-inference ``results/diagnostics/doa_oracle_test_phase2/*_tpit_aligned/``
(``doa_oracle_test.py --align-standalone``) and the training-time test of the two matched-loss controls
(``test_doa_*`` columns of the ledger, CUDA).

Families
--------
``bridge_arms_stage3``        Stage-3 cross-stitch arms cs00 / cs01 / cs10 / cs11 (native STARSS23,
                              N = 2 tracks, no augmentation, legacy all-entry loss; physical batch
                              4/4/2/2 -- BatchNorm statistics differ between the arms)
``stage4_data_loss_arms``     BCE vs focal reference, ExpA/ExpB/ExpB-beta.7/ExpC/ExpA-posw (data,
                              loss domain and lambda_SED differ; see the confound matrix)
``context_temporal_at2seld``  20-s / 40-s windows and larger temporal capacity (not reported before)
``standalone_baselines``      25 Conformer / ResNet SELD baselines *without* the semantic branch,
                              including the matched-loss ExpA controls (Balanced + augmentation,
                              active-only loss)
``cross_dataset_native``      the native AT2SELD checkpoints of the other corpora

What is *not* in the repository (documented here so that nobody searches for it): a model
with random-initialised or frozen E-PANNs, a semantic-input ablation trained from scratch
and a no-bridge control with the same capacity (all 44 AT2SELD runs have ``init: pretrained``,
``frozen: false``).  Inference-time probes of the trained ExpA are in
``results/diagnostics/semantic_probes_phase2/``.

Outputs: ``results/experiments/ablation_family.csv`` and ``ablation_family.md``.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.paths import DIAGNOSTIC_RESULTS_DIR, EXPERIMENT_RESULTS_DIR  # noqa: E402

LEDGER = EXPERIMENT_RESULTS_DIR / "run_configs.csv"
ORACLE = DIAGNOSTIC_RESULTS_DIR / "doa_oracle_test"
PHASE2_ORACLE = DIAGNOSTIC_RESULTS_DIR / "doa_oracle_test_phase2"
OUT_CSV = EXPERIMENT_RESULTS_DIR / "ablation_family.csv"
OUT_MD = EXPERIMENT_RESULTS_DIR / "ablation_family.md"

# experiment name (as in the ledger) -> oracle-result folder under doa_oracle_test/
ORACLE_DIRS: Dict[str, str] = {
    "exp_ft_mixed_bce": "nas-seld/exp_ft_mixed_bce",
    "exp_ft_mixed_focal": "nas-seld/exp_ft_mixed_focal",
    "exp_ft_tau2019_focal": "nas-seld/exp_ft_tau2019_focal",
    "exp_ft_tau_nigens_focal": "nas-seld/exp_ft_tau_nigens_focal",
    "exp_topt_BiGRU_n2_h512_merge_sum": "nas-seld/exp_topt_BiGRU_n2_h512_merge_sum",
    "exp_topt_TT_n1_ffn8_pe_4tracks": "nas-seld/exp_topt_TT_n1_ffn8_pe_4tracks",
    "expA_nas_seld_balanced_starss23": "nas-seld/expA_balanced_starss23",
    "expB_nas_seld_balanced_starss23": "nas-seld/balanced_starss23_expB",
    "expB_nas_seld_balanced_starss23_beta07_thr03": "nas-seld/balanced_starss23_expB_beta07_thr03",
    "expC_nas_seld_balanced_starss23": "nas-seld/balanced_starss23_expC",
    "exp_posw_nas_seld_balanced_starss23_thr03": "nas-seld/posw_balanced_expA",
    "expA_nas_seld_tau2019": "nas-seld/tau2019_expA",
    "expA_nas_seld_tau_nigens2021": "nas-seld/tau_nigens2021_expA",
    "expA_standalone_conformer_balanced_starss23": "conformer/conformer_loss_experiments",
}
# tPIT-aligned oracle statistics of the standalone matched-loss controls, computed by offline re-inference (inference only)
PHASE2_ORACLE_DIRS: Dict[str, str] = {
    "expA_standalone_conformer_balanced_starss23": "conformer_expA_standalone_tpit_aligned",
    "expA_standalone_resnet_balanced_starss23": "resnet_expA_standalone_tpit_aligned",
    "conformer_starss23": "conformer_starss23_allentry_tpit_aligned",   # all-entry loss, native STARSS23
}
# historical / diagnostic raw-slot (NOT PIT-aligned) logs of standalone models
PHASE2_RAW_DIRS: Dict[str, str] = {
    "expA_standalone_resnet_balanced_starss23": "resnet_expA_standalone_raw_unaligned",
}
for _n in ("conformer_starss23", "conformer_starss23_cs", "conformer_starss23_cse",
           "conformer_starss23_csl", "conformer_starss23_specaug", "conformer_starss23_tpit",
           "conformer_topt_TCN_n8_k5_globskip", "conformer_topt_TT_ffn6_n_tracks4",
           "conformer_topt_conv127_ffn6"):
    ORACLE_DIRS[_n] = f"conformer/{_n}"


def family_of(row: dict) -> str:
    g, e = row["group"], row["exp_name"]
    if row["family"] == "standalone_seld":
        return "standalone_baselines"
    if g == "at2seld_nas/deep_reg":
        return "bridge_arms_stage3"
    if g.startswith("at2seld_nas/input_context") or "topt" in e:
        return "context_temporal_at2seld"
    if e in ("exp_ft_tau2019_focal", "exp_ft_tau_nigens_focal", "expA_nas_seld_tau2019",
             "expA_nas_seld_tau_nigens2021", "expA_nas_seld_tau_nigens2020"):
        return "cross_dataset_native"
    if g.startswith("at2seld_nas/finetune") or g.startswith("at2seld_nas/loss_experiments"):
        return "stage4_data_loss_arms"
    return "stage1_2_architecture_search"


NOTES = {
    "exp_r3_t1_cs00": "no bridge; physical batch 4",
    "exp_r3_t1_cs01": "late bridge; physical batch 4",
    "exp_r3_t1_cs10": "early bridge; physical batch 2 (BatchNorm sees different batches)",
    "exp_r3_t1_cs11": "early + late bridge; physical batch 2",
    "expA_nas_seld_balanced_starss23": "headline model: native STARSS23, augmentation OFF (stored hparams)",
    "expA_standalone_conformer_balanced_starss23": "matched-loss control: Balanced + augmentation, active_mse + 0.005",
    "expA_standalone_resnet_balanced_starss23": "matched-loss control: Balanced + augmentation, active_mse + 0.005",
    "conformer_starss23_tpit": "checkpoint saved at epoch 3 (monitored 0.0279) although the log shows 0.6187 -- anomaly, keep out of tables",
    "exp_ictx_starss23_w30": "1-epoch smoke run",
}


ORACLE_KEYS = ("acc_20deg", "median_AE", "pred_norm_active_median", "zero_norm_active_lt_0.10")


def _read_summary(path: Path) -> Optional[Dict[str, str]]:
    if not path.exists():
        return None
    with path.open() as fh:
        r = next(csv.DictReader(fh))
    return {k: r[k] for k in ORACLE_KEYS}


def oracle_row(exp: str, ledger_row: dict) -> Dict[str, Optional[str]]:
    """tPIT-aligned statistics (NAS-SELD stored logs; standalone: offline aligned run, else the ledger's
    training-time test values) plus the separate raw-slot (unaligned) statistics of standalone models."""
    out: Dict[str, Optional[str]] = {k: None for k in ORACLE_KEYS}
    out["source"] = ""
    out["raw"] = {k: None for k in ORACLE_KEYS}
    stored = _read_summary(ORACLE / ORACLE_DIRS[exp] / "doa_oracle_summary.csv") if exp in ORACLE_DIRS else None
    is_standalone = ORACLE_DIRS.get(exp, "").startswith("conformer/") or exp in PHASE2_ORACLE_DIRS
    if stored is not None and not is_standalone:
        out.update(stored)
        out["source"] = "stored offline log, tPIT-aligned"
    elif stored is not None:
        out["raw"] = stored                      # historical standalone log: raw slot order
    if exp in PHASE2_RAW_DIRS:
        raw = _read_summary(PHASE2_ORACLE / PHASE2_RAW_DIRS[exp] / "doa_oracle_summary.csv")
        if raw is not None:
            out["raw"] = raw
    if exp in PHASE2_ORACLE_DIRS:
        ph2 = _read_summary(PHASE2_ORACLE / PHASE2_ORACLE_DIRS[exp] / "doa_oracle_summary.csv")
        if ph2 is not None:
            out.update(ph2)
            out["source"] = "offline re-inference, tPIT-aligned (--align-standalone)"
        elif ledger_row.get("test_doa_acc_20deg"):
            out.update({"acc_20deg": ledger_row["test_doa_acc_20deg"], "median_AE": ledger_row["test_doa_angular_error_median"],
                        "pred_norm_active_median": ledger_row["test_doa_pred_norm_active_median"],
                        "zero_norm_active_lt_0.10": ledger_row["test_doa_zero_norm_active_lt_0.10"]})
            out["source"] = "training-time test (CUDA), tPIT-aligned"
    return out


def main() -> None:
    with LEDGER.open() as fh:
        runs = list(csv.DictReader(fh))
    rows: List[dict] = []
    for r in runs:
        fam = family_of(r)
        if fam == "stage1_2_architecture_search":
            continue
        o = oracle_row(r["exp_name"], r)
        rows.append({
            "ablation_family": fam, "run_id": r["run_id"], "model": r["model"],
            "early/late/smoother": f"{r['early_type']}/{r['late_type']}/{r['smoother_type']}",
            "cs_early/cs_late": f"{r['cs_early']}/{r['cs_late']}",
            "data": r["dataset"], "augmentation": r["aug_enabled"], "n_tracks": r["n_tracks"],
            "beta_sed": r["beta_sed"], "focal_gamma": r["focal_gamma"],
            "doa_loss_mode": r["doa_loss_mode"], "batch_size": r["batch_size"],
            "epochs_logged/cfg": f"{r['epochs_unique']}/{r['epochs_cfg']}",
            "best_epoch": r["best_epoch"], "best_val_SELD": r["best_val_SELD"],
            "test_SELD_training_time": r["test_SELD"],
            "oracle_acc20_tPIT": o["acc_20deg"], "oracle_median_deg": o["median_AE"],
            "oracle_pred_norm_median": o["pred_norm_active_median"],
            "oracle_near_null_lt_0.10": o["zero_norm_active_lt_0.10"],
            "oracle_source": o["source"],
            "raw_unaligned_acc20_NOT_COMPARABLE": o["raw"]["acc_20deg"],
            "raw_unaligned_median_deg_NOT_COMPARABLE": o["raw"]["median_AE"],
            "note": NOTES.get(r["exp_name"], ""),
        })
    order = ["bridge_arms_stage3", "stage4_data_loss_arms", "context_temporal_at2seld",
             "cross_dataset_native", "standalone_baselines"]
    rows.sort(key=lambda x: (order.index(x["ablation_family"]), x["run_id"]))
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    def f(x, nd=4):
        try:
            return f"{float(x):.{nd}f}"
        except (TypeError, ValueError):
            return "" if x in (None, "") else str(x)

    lines = ["| family | run | early/late/smoother | data | aug | beta_SED | gamma | DOA loss | val SELD | test SELD (train-time) | oracle acc@20 (tPIT) | median (deg) | norm | oracle source | raw-slot acc@20 / median (NOT comparable) | note |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| " + " | ".join([
            r["ablation_family"], r["run_id"].split("/", 1)[1], r["early/late/smoother"], r["data"],
            str(r["augmentation"]), str(r["beta_sed"]), str(r["focal_gamma"]), str(r["doa_loss_mode"]),
            f(r["best_val_SELD"]), f(r["test_SELD_training_time"]), f(r["oracle_acc20_tPIT"], 3),
            f(r["oracle_median_deg"], 2), f(r["oracle_pred_norm_median"], 3), r["oracle_source"],
            (f(r["raw_unaligned_acc20_NOT_COMPARABLE"], 3) + " / " + f(r["raw_unaligned_median_deg_NOT_COMPARABLE"], 1))
            if r["raw_unaligned_acc20_NOT_COMPARABLE"] else "", r["note"]]) + " |")
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"{len(rows)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
