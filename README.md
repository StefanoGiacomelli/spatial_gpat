# Spatial GP-AT / AT2SELD

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-AT2SELD-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-Training-792EE5?logo=lightning&logoColor=white)](https://lightning.ai/)
[![Research](https://img.shields.io/badge/Research-GP--AT%20%7C%20SED%20%7C%20DOA%20%7C%20SELD-0A7C66)](#research-context)
[![Status](https://img.shields.io/badge/Status-Research%20Code-orange)](#reproducibility-and-repository-policy)

This repository contains the research code for **AT2SELD**, a semantic-to-spatial framework that extends General Purpose Audio Tagging (GP-AT) backbones toward **Sound Event Localization and Detection (SELD)**. The main idea is to reuse high-level semantic representations learned by large-scale Audio Tagging (AT) models as acoustic priors, and to connect them with trainable multichannel spatial processing, temporal modeling, and track-wise SED/DOA output regressors.

## Highlights

| Area | What this repository provides |
| --- | --- |
| Semantic-to-spatial SELD | AT2SELD models combining E-PANNs semantic priors with a spatio-temporal SELD processing branch. |
| Standalone SELD baselines | Conformer and ResNet-based SELD systems without the GP-AT semantic branch, used as ablations and references. |
| Track-wise representation | Multi-ACCDOA inputs, SED logits and Cartesian DOA vectors outputs and permutation-Invariant Training (tPIT). |
| Informed NAS workflow | Staged architecture search over early/late modules, smoothers, depth allocation, and cross-branch interaction. |
| Diagnostics | activity threshold calibration, DOA-oracle analysis, activity-loss diagnostics, cross-dataset projection, architectural profiling, and dataset statistics. |

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

ToDo

### NAS Stage Summary

ToDo

### Fine-Tuning and Dataset Effects

ToDo

### Calibration and DOA Diagnostics

ToDo

---

## Repository Structure

| Path | Role |
| --- | --- |
| `main_nas-seld.py` | Resumable AT2SELD training entry point. |
| `main_standalone_seld.py` | Resumable standalone SELD training entry point for Conformer/ResNet baselines. |
| `configs/` | YAML configurations for datasets, NAS experiments, standalone SELD, loss experiments, and profiling tests. |
| `datasets/` | Dataset implementations, target construction, augmentation, projection wrappers, and Lightning DataModules. |
| `semantic_backbone/` | E-PANNs semantic audio backbone and architecture split/freeze wrappers. |
| `spatial_modules/` | Candidate spatial/temporal modules used by NAS. |
| `experiments_nas_seld/` | AT2SELD model, Lightning module, loss functions, metrics, and training logic. |
| `experiments_standalone_seld/` | Standalone SELD model definitions, checkpoint filtering, SpecAugment, and training logic. |
| `analysis/datasets/` | Dataset validation, class balance, distribution analysis, and related figures. |
| `analysis/experiments/` | Experiment parsers and plotting scripts for NAS/fine-tuning phases. |
| `analysis/diagnostics/` | Threshold calibration, DOA-oracle analysis, cross-dataset matrices, and aggregate summaries. |
| `analysis/profiling/` | NAS config profiling, standalone profiling, spatial-module profiling, and minimum-window checks. |
| `analysis/maintenance/` | Helpers for resuming or completing experiment and diagnostic matrices. |
| `results/` | Generated artifacts: CSV, JSON, NPZ, and caches. |
| `figures/` | Generated figures grouped by dataset, experiment, diagnostic, and profiling domain. |
| `runs/` | Training checkpoints, TensorBoard logs, metrics, and experiment state. |
| `data/` | Local dataset roots and data setup notes. |

## Installation

Create a Python environment and install the dependencies:

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

Run a short smoke test:

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

---

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

## Troubleshooting

| Symptom | Likely cause | Action |
| --- | --- | --- |
| `FileNotFoundError` for dataset paths | Dataset root not present under `data/` or custom path not supplied. | Check `data/README.md`, run `data/setup_data.sh`, or pass explicit root arguments. |
| Missing checkpoint errors | The requested config expects a checkpoint under `runs/`. | Verify the run completed or update the checkpoint path in the preset/command. |
| CUDA/MPS memory pressure | Full-clip/window size or batch size too large. | Reduce `--batch-size`, use `--window-s`, or use CPU for diagnostics. |
| Matrix job interrupted | Long cross-dataset inference run stopped mid-way. | Re-run with `--reuse-cache`; completed pairs are skipped or restored. |

## Funding and Acknowledgements

This work was developed in the context of doctoral research at the University of L'Aquila, with collaboration and scientific supervision across audio machine learning, acoustic scene analysis, and spatial audio research. The research activity was funded by the Italian Ministry of University and Research under Ministerial Decree No. 118/2023, within the Italian National Recovery and Resilience Plan (NRRP/PNRR), Mission 4, Component 1, Investment 4.1, "PNRR Research", CUP E11I2300010000.

## Citation

If you find this work useful or use part of it, please cite:

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

## Corresponding Author and Contacts

**Stefano Giacomelli, Ph.D. candidate**  
Ph.D. in ICT candidate @ DISIM Dpt — University of L'Aquila, Italy

| Channel | Link |
| --- | --- |
| Google Scholar | [Stefano Giacomelli](https://scholar.google.com/citations?hl=&user=l-n0hl4AAAAJ) |
| ORCID | [0009-0009-0438-1748](https://orcid.org/0009-0009-0438-1748) |
| Email | [stefano.giacomelli@graduate.univaq.it](mailto:stefano.giacomelli@graduate.univaq.it) |
