# Spatial GP-AT / AT2SELD

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-AT2SELD-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-Training-792EE5?logo=lightning&logoColor=white)](https://lightning.ai/)
[![Research](https://img.shields.io/badge/Research-GP--AT%20%7C%20SED%20%7C%20DOA%20%7C%20SELD-0A7C66)](#research-context)
[![Status](https://img.shields.io/badge/Status-Research%20Code-orange)](#reproducibility-and-repository-policy)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE)

<p align="center">
  <img src="assets/univaq_logo.png" alt="University of L'Aquila" height="72">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/disim_logo.png" alt="Department of Information Engineering, Computer Science and Mathematics (DISIM), University of L'Aquila" height="72">
</p>

<p align="center">
  <strong>Stefano Giacomelli</strong> &middot;
  <a href="https://orcid.org/0009-0009-0438-1748">ORCID 0009-0009-0438-1748</a><br>
  Department of Information Engineering, Computer Science and Mathematics (DISIM) &middot;
  University of L'Aquila, Italy
</p>

This repository contains the research code for **AT2SELD**, a semantic-to-spatial framework that extends General Purpose Audio Tagging (GP-AT) backbones toward **Sound Event Localization and Detection (SELD)**. The main idea is to reuse high-level semantic representations learned by large-scale Audio Tagging (AT) models as acoustic priors, and to connect them with trainable multichannel spatial processing, temporal modeling, and track-wise SED/DOA output regressors.

It is the code base of the AT2SELD **technical report** (arXiv:2606.27751, see [Related publications](#related-publications)).

<p align="center">
  <img src="figures/demo/demo_tau2019_expA_split0_61_full.svg" alt="Inference_Example" width="900">
</p>

## Highlights

| Area | What this repository provides |
| --- | --- |
| Semantic-to-spatial SELD | AT2SELD models combining E-PANNs semantic priors with a spatio-temporal SELD processing branch. |
| Standalone SELD baselines | Conformer and ResNet-based SELD systems without the GP-AT semantic branch, used as ablations and references. |
| Track-wise representation | Multi-ACCDOA inputs, SED logits and Cartesian DOA vectors outputs and permutation-Invariant Training (tPIT). |
| Informed NAS workflow | Staged architecture search over early/late modules, smoothers, depth allocation, and cross-branch interaction. |
| Diagnostics | activity threshold calibration, DOA-oracle analysis, activity-loss diagnostics, cross-dataset projection, architectural profiling, and dataset statistics. |
| Metric-variant and control analyses | stitched / DCASE-style metric variants, subset-matched native references, inference-time probes of the semantic prior, effective-configuration ledger, recording-level bootstrap |

## Research Context

GP-AT models pretrained on large audio corpora provide robust semantic evidence, but they do not directly estimate source direction, source identity, or track continuity. SELD requires exactly those missing dimensions: framewise sound event activity, spatial direction of active sources, and assignment of concurrent events to output tracks.

AT2SELD addresses this transition by treating semantic AT as one component of a dual-branch Neural Networks architecture. The framework combines:

- a pretrained AT backbone used as an acoustic prior;
- multichannel spatial front-ends (STFT Magnitude & Phase spectra + Intensity Vectors) for FOA inputs;
- customizable early processing (ResNet, Conformer) and late tracking (TrackTransformer, MHA-MCA) spatial modules;
- temporal smoothers such as BiGRU, TCN modules, and related variants;
- track-wise SED and DOA heads;
- tPIT and diagnostic evaluation for multi-source scenes.

The repository includes experiments on **STARSS23**, **TAU-NIGENS2021**, **TAU-NIGENS2020**, **TAU2019**, and **BalancedSTARSS23**. It also contains dataset-projection diagnostics for evaluating how models trained in one label space behave when tested on another.

## Main Results

For further insights, implementation details, and extended analyses, please refer to the AT2SELD technical report:
- S. Giacomelli, S. Damiano, C. Rinaldi, F. Graziosi, and T. van Waterschoot, “From General-Purpose Audio Tagging to Spatially Grounded Sound Event Localization and Detection,” Technical Report, arXiv:2606.27751, 2026. [[Online]](https://doi.org/10.48550/arXiv.2606.27751).

---

## Repository Structure

| Path | Role |
| --- | --- |
| `main_nas-seld.py` | Resumable AT2SELD training entry point. |
| `main_standalone_seld.py` | Resumable standalone SELD training entry point for Conformer/ResNet baselines. |
| `main_demo.py` | Qualitative ExpA demo figures (windowed inference with stitching; the recordings are *ranked and selected* by DOA quality, continuity and dynamics). |
| `configs/` | YAML configurations for datasets, NAS experiments, standalone SELD, loss experiments, and profiling tests. These are templates: the configuration that was *executed* is the stored `hparams.yaml` of each run (see the ledger below). |
| `datasets/` | Dataset implementations, target construction, augmentation, projection wrappers, and Lightning DataModules. |
| `semantic_backbone/` | E-PANNs semantic audio backbone and architecture split/freeze wrappers. |
| `spatial_modules/` | Candidate spatial/temporal modules used by NAS. |
| `experiments_nas_seld/` | AT2SELD model, Lightning module, loss functions, metrics, and training logic. |
| `experiments_standalone_seld/` | Standalone SELD model definitions, checkpoint filtering, SpecAugment, and training logic. |
| `analysis/datasets/` | Dataset validation, class balance, distribution analysis, and related figures. |
| `analysis/experiments/` | Experiment parsers and plotting scripts for NAS/fine-tuning phases; effective-configuration ledger and ablation-family table. |
| `analysis/diagnostics/` | Threshold calibration, DOA-oracle analysis, cross-dataset matrices, metric variants, semantic probes, native subset references and aggregate summaries. |
| `analysis/profiling/` | NAS config profiling, standalone profiling, spatial-module profiling, parameter accounting, and minimum-window checks. |
| `analysis/maintenance/` | Helpers for resuming or completing experiment and diagnostic matrices. |
| `tests/` | Data-free unit and regression tests (`pytest`; tests that need a dataset skip with a reason). |
| `assets/` | Logos (University of L'Aquila, DISIM, funding). |
| `results/` | Generated artifacts: CSV, JSON, NPZ, and caches; `results/diagnostics/terminal_outputs_backup.txt` is the human-readable log of the Stage-4 diagnostics. |
| `figures/` | Generated figures grouped by dataset, experiment, diagnostic, and profiling domain. |
| `runs/` | Training checkpoints, TensorBoard logs, metrics, and experiment state (not tracked; see `runs/README.md`). |
| `data/` | Local dataset roots and data setup notes. |

## Installation

Create a Python environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt        # unpinned top-level names
# or the exact working environment:
pip install -r requirements-lock.txt   # pip freeze of the tested .venv
pip install -r requirements-dev.txt    # pytest (tests only)
```

Core dependencies include PyTorch, TorchAudio, PyTorch Lightning, NumPy, SciPy, numba, pandas, scikit-learn, soundfile, torchlibrosa, torchmetrics, torchinfo, TensorBoard, matplotlib, seaborn, and DCASE/SELD evaluation utilities.

**Tested environment** (inference, analyses and tests): Python 3.14.6, PyTorch 2.11.0, Lightning 2.6.1, NumPy 2.4.4, SciPy 1.17.1, numba 0.64.0, on macOS 26.5.2 (Apple M4 Pro, 24 GB, MPS). The training runs were executed on cluster nodes (CUDA) and cannot be repeated on a laptop.

**Weights & Biases is optional.** The runners log to TensorBoard and, by default, to W&B (historical behaviour, honours `WANDB_MODE`/`WANDB_API_KEY`). Use `--wandb offline` (local files, no account) or `--wandb disabled` (TensorBoard only). `python-dotenv` is optional.

## Quick Start

**1. Data-light checks (no datasets, no checkpoints).**

```bash
pytest                                  # data-free tests; dataset-dependent tests skip with a reason
.venv/bin/python analysis/profiling/param_accounting.py \
  --config configs/loss_experiments/expA_nas_seld_balanced_starss23.yaml
# -> total 30,986,080 = 27,475,182 semantic + 2,932,082 spatial + 578,816 bridge; task-facing 3,510,898 (11.33 %)
```

**2. Analyses from the stored results (no model, seconds; the ledger check needs the restored `runs/` metadata).**

```bash
.venv/bin/python analysis/diagnostics/threshold_policy_table.py        # fixed / validation-selected / post-hoc policies
.venv/bin/python analysis/diagnostics/bootstrap_oracle_recording.py    # recording-level bootstrap of oracle-DOA accuracy
.venv/bin/python analysis/experiments/export_run_configs.py --check    # effective-configuration ledger vs runs/**/hparams.yaml
.venv/bin/python analysis/experiments/export_ablation_table.py         # ablation / control family table
```

**3. Inference only (needs the datasets under `data/` and the checkpoints restored into `runs/`).**

```bash
# window-level outputs of a checkpoint (cache), then every protocol variant from the same predictions
.venv/bin/python analysis/diagnostics/window_outputs.py --name expA_test \
  --config runs/at2seld_nas/loss_experiments/balanced_starss23_expA/checkpoints/expA_nas_seld_balanced_starss23/version_0/hparams.yaml \
  --checkpoint runs/at2seld_nas/loss_experiments/balanced_starss23_expA/checkpoints/expA_nas_seld_balanced_starss23/34_0.3375.ckpt \
  --dataset starss23 --split test --data-root data/STARSS23 --device mps --tpit-perm
.venv/bin/python analysis/diagnostics/run_metric_variants.py --cache results/cache/window_outputs/expA_test \
  --config <same hparams.yaml> --dataset starss23 --split test --data-root data/STARSS23 \
  --out-dir results/diagnostics/metric_variants_phase2/expA_starss23_test
# qualitative figures of ExpA (written to a git-ignored folder; the default output folder holds the tracked figures)
.venv/bin/python main_demo.py --device auto --output-dir results/cache/demo
```

## Data Layout

The repository expects local datasets under `data/`:

```text
data/STARSS23
data/TAU-NIGENS2020
data/TAU-NIGENS2021
data/TAU2019
```

Use the setup notes and helper script inside `data/`:

```bash
bash data/setup_data.sh
```

Dataset code supports full-clip and windowed modes. STARSS23 validation splitting is controlled by seed and fraction parameters in the relevant scripts/configs (training used seed 42). TAU2019 uses fold-aware handling where applicable. BalancedSTARSS23 builds a 13-class STARSS23-anchored training space by combining STARSS23 with projected TAU-NIGENS2021 and TAU2019 material.

## Training Workflows

### NAS AT2SELD

List experiment status:

```bash
.venv/bin/python main_nas-seld.py --list
```

Run a full configuration group:

```bash
.venv/bin/python main_nas-seld.py \
  --config-dir configs/experiments_nas-seld/foa/shallow_spec \
  --epochs 20 \
  --batch-size 4 \
  --patience 5
```

Run a single YAML configuration:

```bash
.venv/bin/python main_nas-seld.py \
  --config-file configs/experiments_nas-seld/foa/deep_spec/exp_e3_l1_s1.yaml
```

Run on a specific dataset mode:

```bash
.venv/bin/python main_nas-seld.py \
  --dataset TAU-NIGENS2021 \
  --config-dir configs/experiments_nas-seld/foa/finetune
```

Run a short smoke test (no W&B account needed):

```bash
.venv/bin/python main_nas-seld.py \
  --config-file configs/experiments_nas-seld/foa/shallow_spec/exp_spe_rn_tt_bg.yaml \
  --epochs 1 \
  --limit-batches 5 \
  --wandb disabled
```

### Standalone SELD Baselines

List standalone experiment status:

```bash
.venv/bin/python main_standalone_seld.py --list
```

Run the Conformer baseline group:

```bash
.venv/bin/python main_standalone_seld.py \
  --config-dir configs/experiments_standalone_seld/foa/preliminary_conformer
```

Run a single standalone configuration:

```bash
.venv/bin/python main_standalone_seld.py \
  --config-file configs/experiments_standalone_seld/foa/preliminary_conformer/conformer_starss23_cse.yaml
```

Warm-start standalone SELD from a NAS-SELD checkpoint:

```bash
.venv/bin/python main_standalone_seld.py \
  --config-file configs/experiments_standalone_seld/foa/preliminary_conformer/conformer_starss23_cse.yaml \
  --pretrained-ckpt runs/at2seld_nas/finetune/exp_ft_mixed_focal/checkpoints/exp_ft_mixed_focal/30_0.4702.ckpt
```

Both training entry points write checkpoints, TensorBoard event files, state files, and raw metrics under `runs/`.

FOA augmentation: `augmentation.foa_channel_order` (`legacy_wxyz`, default = published behaviour; `acn` = patterns applied to the actual ACN channels W,Y,Z,X).

## Analysis and Diagnostics

All analysis scripts are executable and resolve paths relative to the repository root. The scripts consume existing datasets and/or `runs/` outputs and write derived artifacts to `results/` and `figures/`.

### Dataset checks and figures

Run dataset smoke tests and dataset statistics:

```bash
.venv/bin/python analysis/datasets/test_seld_datasets.py --dataset all --analysis-only
```

Run class-balance analysis:

```bash
.venv/bin/python analysis/datasets/test_seld_class_balance.py --dataset STARSS23
.venv/bin/python analysis/datasets/test_seld_class_balance.py --dataset TAU-NIGENS2021
```

Regenerate class-balance figures from CSV assets:

```bash
.venv/bin/python analysis/datasets/plot_seld_class_balance.py \
  --csv-dir results/datasets \
  --out figures/datasets
```

Generate BalancedSTARSS23 diagnostics:

```bash
.venv/bin/python analysis/datasets/test_balanced_seld_dataset.py
.venv/bin/python analysis/datasets/plot_balanced_seld_dataset.py
```

> `analysis/**/test_*.py` are data-dependent *scripts* (they need datasets, checkpoints or an accelerator); `pytest` collects only `tests/` (`pytest.ini`).

### SED threshold calibration

Run validation-to-test class-specific threshold calibration:

```bash
.venv/bin/python analysis/diagnostics/class_specific_threshold_calibration.py \
  --preset nas_expa \
  --root data/STARSS23 \
  --dataset balanced_starss23 \
  --calibration-split val \
  --evaluation-split test \
  --device auto
```

Aggregate calibration outputs:

```bash
.venv/bin/python analysis/diagnostics/summarize_class_threshold_calibration.py \
  --results-root results/diagnostics/class_threshold_calibration
```

### Activation and DOA diagnostics

Run threshold-sweep activation diagnostics:

```bash
.venv/bin/python analysis/diagnostics/activation_test.py \
  --dataset starss23 \
  --split test \
  --root data/STARSS23 \
  --models nas_seld_x1 standalone_conformer standalone_resnet \   # plus --nas-config/--nas-checkpoint, --conformer-*, --resnet-*
  --pool-size 128 \
  --sample-policy random \
  --device auto
```

Run DOA-oracle diagnostics:

```bash
.venv/bin/python analysis/diagnostics/doa_oracle_test.py \
  --dataset starss23 \
  --split test \
  --root data/STARSS23 \
  --models nas_seld_x1 standalone_conformer standalone_resnet \   # plus --nas-config/--nas-checkpoint, --conformer-*, --resnet-*
  --align-standalone \    # tPIT-align the slots of the standalone models (they are trained with tPITLoss); see below
  --pool-size 256 \
  --device auto
```

> **Alignment.** NAS-SELD slots are always aligned with the checkpoint's `tPITLoss` before the oracle statistics are computed. The standalone Conformer/ResNet baselines are trained with the same `tPITLoss`, but **without `--align-standalone` their slots are left in the raw network order** (the historical behaviour, kept so that the stored logs stay reproducible): such statistics are not comparable with the NAS-SELD ones (the matched-loss standalone Conformer: 10.8 % raw vs 28.5 % aligned oracle accuracy at 20 deg on the same checkpoint and windows).

### Cross-dataset generalization matrix

Run the complete model-by-dataset matrix:

```bash
.venv/bin/python analysis/maintenance/run_dataset_generalization_matrix.py \
  --manifest analysis/diagnostics/dataset_generalization_manifest.json \
  --device auto \
  --reuse-cache
```

Aggregate completed matrix runs:

```bash
.venv/bin/python analysis/diagnostics/summarize_dataset_generalization.py \
  --input-root results/diagnostics/dataset_generalization \
  --figures-dir figures/diagnostics/dataset_generalization/aggregate
```

Run or rebuild the coverage-aware DOA-oracle matrix:

```bash
.venv/bin/python analysis/diagnostics/dataset_generalization_doa_oracle.py \
  --manifest analysis/diagnostics/dataset_generalization_manifest.json \
  --device auto \
  --reuse-cache

.venv/bin/python analysis/diagnostics/dataset_generalization_doa_oracle.py \
  --aggregate-only
```

### Profiling

Profile NAS configuration complexity:

```bash
.venv/bin/python analysis/profiling/test_seld_nas_configs.py \
  --configs-dir configs/experiments_nas-seld/foa/shallow_spec
```

Profile standalone configurations:

```bash
.venv/bin/python analysis/profiling/test_seld_standalone_configs.py \
  --configs-dir configs/experiments_standalone_seld/foa/preliminary_conformer
```

Profile spatial modules and regenerate profiling figures:

```bash
.venv/bin/python analysis/profiling/test_spatial_modules.py
.venv/bin/python analysis/profiling/plot_spatial_modules.py \
  --csv results/profiling/spatial_modules_profile_20260407_114849.csv
```

Parameter accounting by instantiation (CPU, no data): `analysis/profiling/param_accounting.py` (see Quick Start); re-profile of the Stage 1-3 architectures with the executed configuration: `analysis/profiling/reprofile_effective_configs.py`. Note that the NAS-table latencies were measured on CPU with 30-s inputs and that the "GFLOPs" columns of the stored profiles are multiply-accumulate counts (GMACs).

---

## Output Conventions

| Output area | Contents |
| --- | --- |
| `runs/` | Checkpoints, TensorBoard logs, raw training metrics, experiment state. |
| `results/datasets/` | Dataset statistics, balance tables, event distributions, JSON summaries. |
| `results/experiments/` | NAS/fine-tuning summaries, TensorBoard curves, per-class experiment tables, `run_configs.csv` (executed-configuration ledger), `ablation_family.csv`. |
| `results/diagnostics/` | Threshold sweeps, DOA diagnostics, cross-dataset matrices, NPZ/cache artifacts; `*_phase2/` folders hold the additional analyses (each with a `provenance.json`). |
| `results/profiling/` | Model/config/module profiling CSV files. |
| `figures/datasets/` | Dataset and augmentation figures. |
| `figures/experiments/` | NAS and fine-tuning plots. |
| `figures/diagnostics/` | Calibration, generalization, and DOA-oracle diagnostic heatmaps. |
| `figures/profiling/` | Spatial-module profiling figures. |

---

## Configuration Map

| Config path | Purpose |
| --- | --- |
| `configs/datasets/` | Dataset-specific settings. |
| `configs/experiments_nas-seld/foa/shallow_spec` | Stage-1 spectral shallow NAS search. |
| `configs/experiments_nas-seld/foa/shallow_gab` | Stage-1 raw-waveform variants. |
| `configs/experiments_nas-seld/foa/deep_spec` | Stage-2 depth allocation experiments. |
| `configs/experiments_nas-seld/foa/deep_reg` | Stage-3 regularization and cross-branch interaction. |
| `configs/experiments_nas-seld/foa/finetune` | Dataset/loss/architecture fine-tuning experiments. |
| `configs/experiments_nas-seld/foa/temporal_opt` | Temporal-module optimization experiments. |
| `configs/experiments_standalone_seld/foa/preliminary_conformer` | Standalone Conformer baselines. |
| `configs/experiments_standalone_seld/foa/resnet_standalone` | Standalone ResNet baselines. |
| `configs/experiments_standalone_seld/foa/spec_augment` | SpecAugment studies. |
| `configs/loss_experiments/` | Controlled SED and DOA loss-design experiments. |
| `configs/test_spatial_modules/` | Spatial-module profiling/test configurations. |

---

## Reproducibility and Repository Policy

* **What was executed** is recorded in the stored `runs/**/checkpoints/<exp>/version_0/hparams.yaml`; the YAML files under `configs/` are templates that were edited after many runs (some differ from the executed values: loss weights, batch size, dropout, number of tracks, learning-rate schedule, augmentation and dataset of individual runs). [`results/experiments/run_configs.csv`](results/experiments/run_configs.csv) exports the executed configuration, best epoch and validation/test metrics of all 69 runs (`analysis/experiments/export_run_configs.py --check` verifies it against `runs/`).
* **Checkpoints and logs** (`runs/`, ≈ 17 GB) are distributed as an archive (`runs/README.md`).
* **Seeds**: every run uses seed 42 and no configuration was repeated; sampling uncertainty of the test set is quantified by a recording-level bootstrap (`analysis/diagnostics/bootstrap_oracle_recording.py`), not by seed variance.
* **Re-runs**: inference/evaluation re-runs are supported (see Quick Start); training requires the cluster configuration and is not repeated here.

## Current Boundaries

Honest, evidence-based limitations of the repository:

* **Single seed, no controls trained without the semantic prior.** All AT2SELD runs start from the pretrained E-PANNs and are not frozen; no random-initialised, frozen or no-bridge control of equal capacity exists. What exists: 25 standalone Conformer/ResNet baselines (including matched-loss controls), the bridge arms of Stage 3, and inference-time probes of the trained ExpA.
* **Data of the headline model.** The stored hparams of ExpA show native STARSS23 data **without** augmentation (the template YAML is named after BalancedSTARSS23); the reference models and ExpB/ExpC/posw use BalancedSTARSS23 with augmentation.
* **Augmentation channel order.** The 16-pattern augmentation treated the channels of the ACN files as W,X,Y,Z; runs with augmentation (listed in `run_configs.csv`) are affected, ExpA is not. A default-off fix exists (`foa_channel_order: acn`).
* **Metric.** The reported SELD score is the repository's frame-level, gated metric, not the DCASE evaluation; DCASE-style variants (re-implementation) are provided as additional rows. Post-hoc (test-optimised) thresholds are labelled as such.
* **Validation pools.** The cross-dataset diagnostics use a STARSS23 validation pool of which most clips were training clips (seed 2 vs 42); native TAU2019 validation rotates over training folds.
* **Training is not re-runnable on a laptop** (cluster configuration); the training-time test metrics were computed on CUDA, the diagnostics on MPS.

## Troubleshooting

| Symptom | Likely cause | Action |
| --- | --- | --- |
| `FileNotFoundError` for dataset paths | Dataset root not present under `data/` or custom path not supplied. | Check `data/README.md`, run `data/setup_data.sh`, or pass explicit root arguments. |
| Missing checkpoint errors | The requested config expects a checkpoint under `runs/`. | Verify the run completed or update the checkpoint path in the preset/command. |
| CUDA/MPS memory pressure | Full-clip/window size or batch size too large. | Reduce `--batch-size`, use `--window-s`, or use CPU for diagnostics. |
| Matrix job interrupted | Long cross-dataset inference run stopped mid-way. | Re-run with `--reuse-cache`; completed pairs are skipped or restored. |
| W&B login / network error at start-up | The runner creates a W&B logger by default. | Use `--wandb offline` or `--wandb disabled`. |
| `pytest` collects nothing / collects analysis scripts | Wrong working directory or an old pytest configuration. | Run `pytest` from the repository root (`pytest.ini` restricts collection to `tests/`). |

## Funding

Research activities were mainly funded under the Ministerial Decree (DM) 118/2023, Mission 4, Component 1, Investment 4.1 of the National Recovery and Resilience Plan (PNRR) – “PNRR Research” – CUP: E11I23000100001.

<p align="center">
  <img src="assets/next_generation_eu_logo.png" alt="NextGenerationEU" width="200">
  &nbsp;&nbsp;&nbsp;
  <img src="assets/mur_blue_logo.jpg" alt="Ministry of University and Research (MUR)" width="190">
  &nbsp;&nbsp;&nbsp;
  <img src="assets/italiadomani_logo.png" alt="Italia Domani - National Recovery and Resilience Plan" width="170">
</p>

## Citation

### How to cite

If you find this work useful or use part of it, please cite the technical report below; [`CITATION.cff`](CITATION.cff) holds the machine-readable metadata (GitHub shows it under "Cite this repository").

```bibtex
@techreport{giacomelli_at2seld_2026,
  author      = {Giacomelli, Stefano and Damiano, Stefano and Rinaldi, Claudia and Graziosi, Fabio and van Waterschoot, Toon},
  title       = {{From General-Purpose Audio Tagging to Spatially Grounded Sound Event Localization and Detection}},
  institution = {KU Leuven -- University of L'Aquila},
  type        = {Technical Report},
  number      = {arXiv:2606.27751},
  year        = {2026},
  month       = jun,
  doi         = {10.48550/arXiv.2606.27751},
  url         = {https://arxiv.org/abs/2606.27751},
  note        = {Technical report (KU Leuven -- UnivAQ)}
}
```

### Related publications

- **AT2SELD technical report** (this repository): S. Giacomelli, S. Damiano, C. Rinaldi, F. Graziosi, and T. van Waterschoot, "From General-Purpose Audio Tagging to Spatially Grounded Sound Event Localization and Detection," Technical Report, arXiv:2606.27751, 2026. DOI: [10.48550/arXiv.2606.27751](https://doi.org/10.48550/arXiv.2606.27751).
- **GP-AT backbone (E-PANNs lineage).** S. Giacomelli, M. Giordano, C. Rinaldi, and F. Graziosi, "From General-Purpose Audio Tagging to Real-Time Emergency Vehicle Siren Detection," IEEE Transactions on Audio, Speech and Language Processing, vol. 34, pp. 3428-3444, 2026. DOI: [10.1109/TASLPRO.2026.3703169](https://doi.org/10.1109/TASLPRO.2026.3703169).
- **Dataset curation.** S. Giacomelli, M. Giordano, C. Rinaldi, and F. Graziosi, "AudioSet-Tools: A Python Framework for Taxonomy-Aware AudioSet Curation and Reproducible Audio Research," EURASIP Journal on Audio, Speech, and Music Processing, vol. 2026, no. 1, art. no. 2, 2026. DOI: [10.1186/s13636-025-00436-z](https://doi.org/10.1186/s13636-025-00436-z).

```bibtex
@article{giacomelli_gpat_ev_2026,
  author  = {Giacomelli, Stefano and Giordano, Marco and Rinaldi, Claudia and Graziosi, Fabio},
  title   = {{From General-Purpose Audio Tagging to Real-Time Emergency Vehicle Siren Detection}},
  journal = {IEEE Transactions on Audio, Speech and Language Processing},
  volume  = {34},
  pages   = {3428--3444},
  year    = {2026},
  doi     = {10.1109/TASLPRO.2026.3703169}
}

@article{giacomelli_audioset_tools_2026,
  author    = {Giacomelli, Stefano and Giordano, Marco and Rinaldi, Claudia and Graziosi, Fabio},
  title     = {{AudioSet-Tools: A Python Framework for Taxonomy-Aware AudioSet Curation and Reproducible Audio Research}},
  journal   = {EURASIP Journal on Audio, Speech, and Music Processing},
  volume    = {2026},
  number    = {1},
  eid       = {2},
  year      = {2026},
  publisher = {Springer Nature},
  doi       = {10.1186/s13636-025-00436-z}
}

```

## License

The code and the documentation of this repository are licensed under the **Creative Commons Attribution 4.0 International**
licence (CC BY 4.0; [`LICENSE`](LICENSE), <https://creativecommons.org/licenses/by/4.0/>). Third-party components keep their own licences, which this licence does not replace: the E-PANNs backbone in [`semantic_backbone/epanns/`](semantic_backbone/epanns/LICENSE) (MIT) and the dataset licences of STARSS23, TAU-NIGENS2020/2021 and TAU2019 (see `data/README.md`).

## Corresponding Author and Contacts

**Stefano Giacomelli, Ph.D. candidate**
Ph.D. in ICT candidate @ DISIM Dpt — University of L'Aquila, Italy

| Channel | Link |
| --- | --- |
| Email | [stefano.giacomelli@graduate.univaq.it](mailto:stefano.giacomelli@graduate.univaq.it) |
| GitHub | [StefanoGiacomelli](https://github.com/StefanoGiacomelli) |
| ORCID | [0009-0009-0438-1748](https://orcid.org/0009-0009-0438-1748) |
| Google Scholar | [Stefano Giacomelli](https://scholar.google.com/citations?hl=&user=l-n0hl4AAAAJ) |
