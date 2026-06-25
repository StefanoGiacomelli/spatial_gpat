# Spatial GP-AT for Sound Event Localization and Detection

This repository contains research code for semantic-to-spatial sound event
localization and detection (SELD).  The project studies AT2SELD models that
combine an Audio Tagging (AT) semantic prior, derived from E-PANNs backbones,
with trainable spatial modules and SELD heads for joint Sound Event
Detection (SED) and Direction-of-Arrival (DOA) estimation.

The codebase supports two related experimental families:

- **NAS-SELD (AT2SELD)** models, where spatial processing branches are selected from a
  configurable Neural Architecture Search (NAS) space and trained for SELD.
- **Standalone SELD** ablation baselines, including Conformer and ResNet-based models
  without the GP-AT backbone.

The experimental workflow covers native-domain training, cross-dataset
generalization, class-threshold calibration, DOA-oracle diagnostics, model
profiling, and dataset-level analysis on STARSS23, TAU-NIGENS2020,
TAU-NIGENS2021, and TAU2019.

## Repository Layout

```text
.
├── analysis/                    # Re-runnable analysis, diagnostics, profiling scripts
│   ├── datasets/                # Dataset statistics, balance checks, dataset figures
│   ├── diagnostics/             # SED/DOA diagnostics and cross-dataset summaries
│   ├── experiments/             # Experiment result parsers and plotting scripts
│   ├── maintenance/             # Resume/completion helpers for experiment matrices
│   ├── profiling/               # Architecture and spatial-module profiling scripts
│   └── paths.py                 # Shared filesystem conventions for analysis code
├── configs/                     # YAML experiment and dataset configurations
├── data/                        # Local datasets; see data/README.md
├── datasets/                    # Dataset implementations and data modules
├── experiments_nas_seld/        # NAS-SELD Lightning module, model, and losses
├── experiments_standalone_seld/ # Standalone SELD models and training module
├── figures/                     # Generated visual assets, grouped by analysis domain
├── results/                     # Generated CSV, JSON, NPZ, and cache artifacts
├── runs/                        # Training checkpoints, metrics, state, TensorBoard logs
├── semantic_backbone/           # Semantic audio-tagging backbone integration
├── spatial_modules/             # Candidate spatial processing modules
├── utils/                       # Shared metric and utility code
├── main_nas-seld.py             # NAS-SELD training entry point
├── main_standalone_seld.py      # Standalone SELD training entry point
└── requirements.txt             # Python dependencies
```

Derived artifacts produced after training belong in `results/` or `figures/`.

## Installation

Create and activate a Python environment, then install the project
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The code expects PyTorch, PyTorch Lightning, torchaudio, pandas, NumPy,
matplotlib, seaborn, and DCASE/SELD-related utilities listed in
`requirements.txt`.

## Data

Dataset roots are expected under `data/`:

```text
data/
├── STARSS23/
├── TAU-NIGENS2020/
├── TAU-NIGENS2021/
└── TAU2019/
```

The dataset-specific setup notes live in `data/README.md`, and the helper script
`data/setup_data.sh` is the data bootstrap entry point.

## Training

Run NAS-SELD experiments through:

```bash
.venv/bin/python main_nas-seld.py \
  --config-dir configs/experiments_nas-seld/foa/shallow_spec \
  --epochs 20 \
  --batch-size 4
```

Run standalone SELD experiments through:

```bash
.venv/bin/python main_standalone_seld.py \
  --config-dir configs/experiments_standalone_seld/foa/preliminary_conformer
```

Both entry points are resumable.  Checkpoints, TensorBoard logs, experiment
state, and raw per-run metrics are written under `runs/`.

## Analysis Workflow

All analysis scripts live under `analysis/` and are executable.  They are grouped
by purpose:

- `analysis/datasets/`: dataset statistics, balance checks, augmentation
  figures, and dataset-distribution analyses.
- `analysis/experiments/`: parsers and figure generators for NAS and
  fine-tuning experiment results.
- `analysis/diagnostics/`: SED threshold diagnostics, DOA-oracle analysis,
  cross-dataset generalization matrices, and aggregate summaries.
- `analysis/profiling/`: model, NAS-branch, and spatial-module profiling.
- `analysis/maintenance/`: helpers for completing or resuming diagnostic
  matrices.

Generated numeric artifacts use `results/`:

```text
results/
├── datasets/
├── diagnostics/
├── experiments/
├── profiling/
└── cache/
```

Generated visual artifacts use `figures/`:

```text
figures/
├── datasets/
├── diagnostics/
├── experiments/
└── profiling/
```

Analysis outputs are restricted to machine-readable artifacts
such as CSV, JSON, NPZ, and cache files, plus visual figures in `figures/`.

## Reproducibility Notes

- Configuration is YAML-driven and stored under `configs/`.
- Training runs are resumable through JSON state files inside `runs/`.
- Cross-dataset diagnostics cache intermediate inference results to avoid
  recomputing completed model/dataset pairs.
- Analysis code resolves paths relative to the repository root, so scripts can
  be launched from the root with `.venv/bin/python analysis/.../script.py`.
