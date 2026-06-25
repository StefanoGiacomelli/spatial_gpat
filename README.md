# Spatial GP-AT / AT2SELD

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-SELD%20Models-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-Training-792EE5?logo=lightning&logoColor=white)](https://lightning.ai/)
[![TorchAudio](https://img.shields.io/badge/TorchAudio-Multichannel%20Audio-2C5BB4)](https://pytorch.org/audio/stable/)
[![Research](https://img.shields.io/badge/Research-AT2SELD%20%7C%20GP--AT%20%7C%20SELD-0A7C66)](#research-context)
[![Status](https://img.shields.io/badge/Status-Research%20Code-orange)](#reproducibility-and-repository-policy)

This repository contains the research code for **AT2SELD**, a semantic-to-spatial framework that extends general-purpose audio tagging backbones toward **Sound Event Localization and Detection (SELD)**. The central idea is to reuse high-level semantic representations learned by large-scale audio tagging models as acoustic priors, and to connect them with trainable multichannel spatial processing, temporal modeling, and track-wise SED/DOA output heads.

The project is not a single-model demo. It is a complete experimental workspace for studying how semantic audio representations, spatial front-ends, temporal modules, loss design, calibration, dataset imbalance, and cross-dataset label-space mismatch interact in localization-aware acoustic scene analysis.

## Highlights

| Area | What this repository provides |
| --- | --- |
| Semantic-to-spatial SELD | AT2SELD/NAS-SELD models combining E-PANNs-style semantic priors with spatial and temporal SELD branches. |
| Standalone SELD baselines | Conformer and ResNet-based SELD systems without the GP-AT semantic branch, used as ablations and references. |
| Track-wise output representation | Multi-track SED logits and Cartesian DOA vectors for same-class overlap and permutation-aware training. |
| Informed NAS workflow | Staged architecture search over spatial inputs, early/late modules, smoothers, depth allocation, and cross-branch interaction. |
| Diagnostic tooling | Threshold calibration, DOA-oracle analysis, activity-loss diagnostics, cross-dataset projection, profiling, and dataset statistics. |
| Reproducible artifacts | Training outputs are isolated in `runs/`; derived CSV/JSON/NPZ artifacts are in `results/`; figures are in `figures/`. |

## Research Context

General-purpose audio tagging models pretrained on large audio corpora provide robust semantic evidence, but they do not directly estimate source direction, source identity, or track continuity. SELD requires exactly those missing dimensions: framewise sound event activity, spatial direction of active sources, and assignment of concurrent events to output tracks.

AT2SELD addresses this transition by treating semantic audio tagging as one component of a broader localization-aware architecture. The framework combines:

- a pretrained semantic backbone used as an acoustic prior;
- multichannel spatial front-ends for FOA audio;
- configurable early and late spatial modules;
- temporal smoothers such as BiGRU, TrackTransformer, Conformer/TCN-style modules, and related variants;
- track-wise SED and DOA heads;
- permutation-aware training and diagnostic evaluation for multi-source scenes.

The repository includes experiments on **STARSS23**, **TAU-NIGENS2020**, **TAU-NIGENS2021**, **TAU2019**, and **BalancedSTARSS23**. It also contains dataset-projection diagnostics for evaluating how models trained in one label space behave when tested on another.

## Main Experimental Findings

The current artifacts document a staged experimental campaign. SELD score follows the standard convention used in the codebase: **lower is better**. `F_20` and localization recall (`LR_CD`) are higher-is-better.

### NAS Stage Summary

| Stage | Best configuration | Test SELD score | F_20 | LR_CD | LE_CD | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Shallow screening | `exp_spe_rn_tt_bg` | 0.6868 | 0.1342 | 0.0788 | 13.74 deg | Spectral input, ResNet early module, TrackTransformer late module, BiGRU smoother. |
| Depth allocation | `exp_e3_l1_s1` | 0.6271 | 0.2289 | 0.1221 | 13.23 deg | Deeper early spatial processing improved over shallow screening. |
| Cross-branch interaction | `exp_r3_t1_cs01` | 0.6242 | 0.2264 | 0.1422 | 12.49 deg | Late cross-stitch interaction was the best regularized variant. |

### Fine-Tuning and Dataset Effects

Balanced multi-dataset training improves validation behavior but exposes difficult operating-point and dataset-transfer effects. The mixed focal-loss setting (`exp_ft_mixed_focal`) improves STARSS23 test SELD score over the mixed BCE setting (`0.6659` vs. `0.6985`) and increases STARSS23 `F_20` (`0.1746` vs. `0.1112`). Native TAU2019 fine-tuning is much easier in-domain, reaching a single-domain SELD score of `0.1217` and `F_20 = 0.8667`.

The cross-dataset matrix shows that semantic transfer is not symmetric across datasets. Native-domain results are strongest when the train/test label spaces and annotation regimes match, while cross-domain performance depends heavily on class coverage and projection quality.

| Training dataset | Test STARSS23 | Test TAU2019 | Test TAU-NIGENS2020 | Test TAU-NIGENS2021 |
| --- | ---: | ---: | ---: | ---: |
| STARSS23 | 0.6263 | 0.7157 | 0.7448 | 0.7414 |
| TAU2019 | 0.5046 | 0.1415 | 0.6385 | 0.6825 |
| TAU-NIGENS2020 | 0.6296 | 0.5508 | 0.5405 | 0.6033 |
| TAU-NIGENS2021 | 0.5864 | 0.4273 | 0.4845 | 0.5051 |

### Calibration and DOA Diagnostics

Threshold calibration is a major factor in SED/SELD behavior. Fixed `0.5` thresholds are often too conservative; validation-selected global or class-specific thresholds improve SED F1 for most model families. For example, NAS-SELD ExpA improves SED F1 from `0.375` at the fixed threshold to `0.435` with a validation-selected global threshold and `0.441` with class-specific thresholds.

The DOA-oracle diagnostics separate localization quality from activity-detection errors by using ground-truth activity gates. Native-domain median angular error ranges from about `5.96 deg` on TAU2019 to about `23.97 deg` on STARSS23, showing that localization difficulty depends strongly on dataset geometry, annotation regime, class space, and learned operating point.

## Repository Structure

| Path | Role |
| --- | --- |
| `main_nas-seld.py` | Resumable NAS-SELD/AT2SELD training entry point. |
| `main_standalone_seld.py` | Resumable standalone SELD training entry point for Conformer/ResNet-style baselines. |
| `configs/` | YAML configurations for datasets, NAS experiments, standalone SELD, loss experiments, and profiling tests. |
| `datasets/` | Dataset implementations, target construction, augmentation, projection wrappers, and Lightning DataModules. |
| `semantic_backbone/` | E-PANNs-style semantic audio backbone and split/freeze wrappers. |
| `spatial_modules/` | Candidate spatial/temporal modules used by NAS-SELD and standalone SELD systems. |
| `experiments_nas_seld/` | NAS-SELD model, Lightning module, loss functions, metrics, and training logic. |
| `experiments_standalone_seld/` | Standalone SELD model definitions, checkpoint filtering, SpecAugment, and training logic. |
| `analysis/datasets/` | Dataset validation, class balance, distribution analysis, and dataset figures. |
| `analysis/experiments/` | Experiment-result parsers and plotting scripts for NAS/fine-tuning phases. |
| `analysis/diagnostics/` | Threshold calibration, DOA-oracle analysis, cross-dataset matrices, and aggregate summaries. |
| `analysis/profiling/` | NAS config profiling, standalone profiling, spatial-module profiling, and minimum-window checks. |
| `analysis/maintenance/` | Helpers for resuming or completing experiment and diagnostic matrices. |
| `results/` | Machine-readable generated artifacts: CSV, JSON, NPZ, and caches. |
| `figures/` | Generated figures grouped by dataset, experiment, diagnostic, and profiling domain. |
| `runs/` | Training checkpoints, TensorBoard logs, metrics, and experiment state. |
| `data/` | Local dataset roots and data setup notes. |

## Installation

Create an isolated environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Core dependencies include PyTorch, TorchAudio, PyTorch Lightning, NumPy, SciPy, pandas, scikit-learn, soundfile, torchlibrosa, torchmetrics, torchinfo, TensorBoard, matplotlib, seaborn, and DCASE/SELD evaluation utilities.

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

Dataset code supports full-clip and windowed modes. STARSS23 validation splitting is controlled by seed and fraction parameters in the relevant scripts/configs. TAU2019 uses fold-aware handling where applicable. BalancedSTARSS23 builds a 13-class STARSS23-anchored training space by combining STARSS23 with projected TAU-NIGENS2021 and TAU2019 material.

## Training Workflows

### NAS-SELD / AT2SELD

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

Run a short smoke/debug job:

```bash
.venv/bin/python main_nas-seld.py \
  --config-file configs/experiments_nas-seld/foa/shallow_spec/exp_spe_rn_tt_bg.yaml \
  --epochs 1 \
  --limit-batches 5
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

### Experiment summaries and plots

Aggregate NAS phase results:

```bash
.venv/bin/python analysis/experiments/analyze_shallow_nas_foa.py
.venv/bin/python analysis/experiments/analyze_deep_nas_foa.py
.venv/bin/python analysis/experiments/analyze_reg_nas_foa.py
.venv/bin/python analysis/experiments/analyze_finetune_nas_foa.py
```

Regenerate experiment figures:

```bash
.venv/bin/python analysis/experiments/plot_deep_nas_foa.py \
  --results-dir results/experiments \
  --out figures/experiments

.venv/bin/python analysis/experiments/plot_reg_nas_foa.py \
  --results-dir results/experiments \
  --out figures/experiments

.venv/bin/python analysis/experiments/plot_finetune_nas_foa.py
```

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
  --models nas conformer resnet \
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
  --models nas conformer resnet \
  --pool-size 256 \
  --device auto
```

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

Profile NAS-SELD branch ownership and incremental spatial cost:

```bash
.venv/bin/python analysis/diagnostics/profile_nas_seld_branches.py \
  --manifest analysis/diagnostics/dataset_generalization_manifest.json \
  --device auto
```

## Output Conventions

| Output area | Contents |
| --- | --- |
| `runs/` | Checkpoints, TensorBoard logs, raw training metrics, experiment state. |
| `results/datasets/` | Dataset statistics, balance tables, event distributions, JSON summaries. |
| `results/experiments/` | NAS/fine-tuning summaries, TensorBoard curves, per-class experiment tables. |
| `results/diagnostics/` | Threshold sweeps, DOA diagnostics, cross-dataset matrices, NPZ/cache artifacts. |
| `results/profiling/` | Model/config/module profiling CSV files. |
| `figures/datasets/` | Dataset and augmentation figures. |
| `figures/experiments/` | NAS and fine-tuning plots. |
| `figures/diagnostics/` | Calibration, generalization, and DOA-oracle diagnostic heatmaps. |
| `figures/profiling/` | Spatial-module profiling figures. |

## Configuration Map

| Config path | Purpose |
| --- | --- |
| `configs/datasets/` | Dataset-specific settings. |
| `configs/experiments_nas-seld/foa/shallow_spec` | Stage-1 spectral shallow NAS search. |
| `configs/experiments_nas-seld/foa/shallow_gab` | Stage-1 geometry-aware baseline variants. |
| `configs/experiments_nas-seld/foa/deep_spec` | Stage-2 depth allocation experiments. |
| `configs/experiments_nas-seld/foa/deep_reg` | Stage-3 regularization and cross-branch interaction. |
| `configs/experiments_nas-seld/foa/finetune` | Dataset/loss/architecture fine-tuning experiments. |
| `configs/experiments_nas-seld/foa/temporal_opt` | Temporal-module optimization experiments. |
| `configs/experiments_standalone_seld/foa/preliminary_conformer` | Standalone Conformer baselines. |
| `configs/experiments_standalone_seld/foa/resnet_standalone` | Standalone ResNet baselines. |
| `configs/experiments_standalone_seld/foa/spec_augment` | Standalone augmentation studies. |
| `configs/loss_experiments/` | Controlled loss-design experiments. |
| `configs/test_spatial_modules/` | Spatial-module profiling/test configurations. |

## Reproducibility and Repository Policy

The repository separates source code, training outputs, and derived analysis artifacts. Training scripts write to `runs/`; analysis scripts write to `results/` and `figures/`. Existing `runs/` content is treated as an input archive for downstream analysis. Cross-dataset diagnostics use cache files so interrupted matrices can be resumed without recomputing completed model/dataset pairs.

Recommended workflow:

```bash
source .venv/bin/activate

# 1. Train or inspect experiment status
.venv/bin/python main_nas-seld.py --list

# 2. Run selected analysis scripts
.venv/bin/python analysis/experiments/analyze_deep_nas_foa.py
.venv/bin/python analysis/diagnostics/summarize_dataset_generalization.py --allow-incomplete

# 3. Regenerate figures from machine-readable assets
.venv/bin/python analysis/experiments/plot_deep_nas_foa.py
```

## Troubleshooting

| Symptom | Likely cause | Action |
| --- | --- | --- |
| `FileNotFoundError` for dataset paths | Dataset root not present under `data/` or custom path not supplied. | Check `data/README.md`, run `data/setup_data.sh`, or pass explicit root arguments. |
| Missing checkpoint errors | The requested config expects a checkpoint under `runs/`. | Verify the run completed or update the checkpoint path in the preset/command. |
| Slow first plotting command | Matplotlib/font cache initialization. | Re-run the command; subsequent calls are usually faster. |
| CUDA/MPS memory pressure | Full-clip/window size or batch size too large. | Reduce `--batch-size`, use `--window-s`, or use CPU for diagnostics. |
| Matrix job interrupted | Long cross-dataset inference run stopped mid-way. | Re-run with `--reuse-cache`; completed pairs are skipped or restored. |

## Funding and Acknowledgements

This work was developed in the context of doctoral research at the University of L'Aquila, with collaboration and scientific supervision across audio machine learning, acoustic scene analysis, and spatial audio research. The research activity was funded by the Italian Ministry of University and Research under Ministerial Decree No. 118/2023, within the Italian National Recovery and Resilience Plan (NRRP/PNRR), Mission 4, Component 1, Investment 4.1, "PNRR Research", CUP E11I2300010000.

## Citation

The formal citation will be updated after the arXiv release.

```bibtex
@misc{giacomelli2026at2seld,
  title         = {From General-Purpose Audio Tagging to Spatially Grounded Sound Event Localization and Detection},
  author        = {Giacomelli, Stefano and Damiano, Stefano and Rinaldi, Claudia and Graziosi, Fabio and van Waterschoot, Toon},
  year          = {2026},
  archivePrefix = {arXiv},
  eprint        = {TODO},
  primaryClass  = {cs.SD},
  url           = {TODO}
}
```

## Author and Contact

**Stefano Giacomelli, Ph.D. candidate**  
DISIM Department — University of L'Aquila  
Researcher | Developer | Electroacoustic Musician

| Channel | Link |
| --- | --- |
| GitHub | [StefanoGiacomelli](https://github.com/StefanoGiacomelli) |
| Website | [stefanogiacomelli.github.io](https://stefanogiacomelli.github.io/) |
| Google Scholar | [Stefano Giacomelli](https://scholar.google.com/citations?hl=&user=l-n0hl4AAAAJ) |
| ORCID | [0009-0009-0438-1748](https://orcid.org/0009-0009-0438-1748) |
| LinkedIn | [stefano-giacomelli-811654135](https://www.linkedin.com/in/stefano-giacomelli-811654135/) |
| Email | [stefano.giacomelli@graduate.univaq.it](mailto:stefano.giacomelli@graduate.univaq.it) |
