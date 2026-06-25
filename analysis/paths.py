"""Shared filesystem conventions for analysis scripts.

The training entry points keep writing checkpoints, TensorBoard logs, state
files, and raw metrics under ``runs/``.  Analysis scripts should write derived
assets only under ``results/`` and ``figures/`` so that code folders remain
free of generated files.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = REPO_ROOT / "analysis"
CONFIGS_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"

DATASET_RESULTS_DIR = RESULTS_DIR / "datasets"
EXPERIMENT_RESULTS_DIR = RESULTS_DIR / "experiments"
DIAGNOSTIC_RESULTS_DIR = RESULTS_DIR / "diagnostics"
PROFILING_RESULTS_DIR = RESULTS_DIR / "profiling"
CACHE_RESULTS_DIR = RESULTS_DIR / "cache"

DATASET_FIGURES_DIR = FIGURES_DIR / "datasets"
EXPERIMENT_FIGURES_DIR = FIGURES_DIR / "experiments"
DIAGNOSTIC_FIGURES_DIR = FIGURES_DIR / "diagnostics"
PROFILING_FIGURES_DIR = FIGURES_DIR / "profiling"


def ensure_dir(path: Path) -> Path:
    """Create ``path`` if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def project_path(path: str | Path) -> Path:
    """Resolve a user path relative to the repository root."""
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value
