"""
experiments_at2sed/omap.py — Ontology Mean Average Precision (oMAP)
====================================================================
Provenance
----------
Ablated from:
  ontology-aware-audio-tagging (Apache 2.0 licence)
  Original functions:
    ``ontology_mean_average_precision()``,  ``build_weight()``,
    ``_binary_clf_curve()``,               ``precision_recall_curve()``,
    ``_average_precision()``

Adaptations
-----------
  * All state encapsulated in ``OntologyMAP`` class (no global state,
    no hardcoded cache path).
  * Distance matrix passed as argument instead of loaded from
    ``get_audioset_graph_distance()`` (uses our 456×456 submatrix).
  * Cache path is configurable (default: alongside metrics.json).
  * ``update()`` / ``compute()`` / ``reset()`` API for use in Lightning
    ``validation_step`` / ``on_validation_epoch_end``.

Performance notes
-----------------
  * ``_build_weight()`` uses Numba JIT and is the bottleneck on first call
    (~30–60 s).  Results are pickled to ``cache_path`` for subsequent epochs.
  * ``compute()`` iterates over ~15 distance thresholds × 456 classes; for
    ~100 k val samples expect 30–90 s per oMAP computation.
  * Use ``omap_every_n`` in the YAML to limit how often oMAP is computed.

References
----------
H. Liu et al., "Ontology-aware Learning and Evaluation for Audio Tagging,"
arXiv 2206.03491, 2022.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import sklearn
import sklearn.utils
import sklearn.utils.multiclass
import sklearn.utils.extmath
import sklearn.metrics._ranking
import torch
from numba import jit
from tqdm import tqdm

_DEFAULT_DIST = (
    Path(__file__).resolve().parent.parent
    / 'configs' / 'datasets' / 'audioset_strong_graph_distance.npy'
)


# ══════════════════════════════════════════════════════════════════════════════
# Low-level functions (ablated, minimal changes from original)
# ══════════════════════════════════════════════════════════════════════════════

def _precision_recall_curve(
    y_true, probas_pred, *, pos_label=None, tps_weight=None, fps_weight=None
):
    fps, tps, thresholds = _binary_clf_curve(
        y_true,
        probas_pred,
        pos_label=pos_label,
        tps_weight=tps_weight,
        fps_weight=fps_weight,
    )

    ps = tps + fps
    precision = np.divide(tps, ps, where=(ps != 0))

    if tps[-1] == 0:
        warnings.warn(
            "No positive class found in y_true, "
            "recall is set to one for all thresholds."
        )
        recall = np.ones_like(tps)
    else:
        recall = tps / tps[-1]

    sl = slice(None, None, -1)
    return np.hstack((precision[sl], 1)), np.hstack((recall[sl], 0)), thresholds[sl]


def _binary_clf_curve(
    y_true, y_score, pos_label=None, tps_weight=None, fps_weight=None
):
    y_type = sklearn.utils.multiclass.type_of_target(y_true)
    if not (y_type == "binary" or (y_type == "multiclass" and pos_label is not None)):
        raise ValueError("{0} format is not supported".format(y_type))

    sklearn.utils.check_consistent_length(y_true, y_score, fps_weight)
    sklearn.utils.check_consistent_length(y_true, y_score, tps_weight)
    y_true  = sklearn.utils.column_or_1d(y_true)
    y_score = sklearn.utils.column_or_1d(y_score)
    sklearn.utils.assert_all_finite(y_true)
    sklearn.utils.assert_all_finite(y_score)

    if fps_weight is not None:
        fps_weight = sklearn.utils.column_or_1d(fps_weight)
        fps_weight = sklearn.utils.validation._check_sample_weight(fps_weight, y_true)

    if tps_weight is not None:
        tps_weight = sklearn.utils.column_or_1d(tps_weight)
        tps_weight = sklearn.utils.validation._check_sample_weight(tps_weight, y_true)

    pos_label = sklearn.metrics._ranking._check_pos_label_consistency(pos_label, y_true)

    if tps_weight is None:
        y_true = y_true == pos_label
    else:
        y_true = tps_weight
        assert np.max(y_true) <= 1.0

    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true  = y_true[desc_score_indices]

    if fps_weight is not None:
        weight = fps_weight[desc_score_indices]
    else:
        weight = 1.0

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    tps = sklearn.utils.extmath.stable_cumsum(y_true)[threshold_idxs]

    if fps_weight is not None:
        fps = sklearn.utils.extmath.stable_cumsum((1 - y_true) * weight)[threshold_idxs]
    else:
        fps = 1 + threshold_idxs - tps

    return fps, tps, y_score[threshold_idxs]


def _average_precision(y_true, pred_scores, tps_weight=None, fps_weight=None):
    precisions, recalls, _ = _precision_recall_curve(
        y_true, pred_scores, tps_weight=tps_weight, fps_weight=fps_weight
    )
    precisions = np.array(precisions)
    recalls    = np.array(recalls)
    return float(np.sum((recalls[:-1] - recalls[1:]) * precisions[:-1]))


def _mask_weight(weight: np.ndarray, threshold: float) -> np.ndarray:
    w = weight.copy()
    w[weight <= threshold] = 0.0
    diag = np.eye(w.shape[0], dtype=bool)
    off_diag = w[~diag]
    if off_diag.mean() > 1e-9:
        w = w / off_diag.mean()
    return w


@jit(nopython=True)
def _build_ontology_fps_weight_min(
    target: np.ndarray,
    weight: np.ndarray,
    class_idx: int,
) -> list:
    """Numba-JIT: per-sample minimum graph weight from positive labels to class_idx."""
    ret = []
    for i in range(target.shape[0]):
        positive_indices = np.where(target[i] == 1)[0]
        if len(positive_indices) == 0:
            ret.append(0.0)
            continue
        min_dist = np.min(weight[positive_indices][:, class_idx])
        ret.append(min_dist)
    return ret


def _build_weight(
    target: np.ndarray,
    weight: np.ndarray,
    cache_path: Optional[Path] = None,
    refresh: bool = False,
) -> Dict:
    """
    Build per-class, per-threshold FPS weights.

    Results are cached to ``cache_path`` (pickle).  Pass ``refresh=True``
    to force recomputation even if the cache exists.
    """
    ret: Dict = {}

    if cache_path is not None and not refresh and cache_path.exists():
        with open(cache_path, 'rb') as fh:
            return pickle.load(fh)

    max_val = int(np.max(weight))
    print('[oMAP] Building ontology weight cache — this may take 1–2 minutes …')
    for threshold in tqdm(np.linspace(0, max_val, max_val + 1)):
        ret[threshold] = {}
        masked = _mask_weight(weight, threshold)
        for i in range(target.shape[1]):
            ret[threshold][i] = _build_ontology_fps_weight_min(target, masked, i)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as fh:
            pickle.dump(ret, fh)
        print(f'[oMAP] Weight cache saved → {cache_path}')

    return ret


def _compute_omap(
    target: np.ndarray,
    clipwise_output: np.ndarray,
    graph_distance: np.ndarray,
    cache_path: Optional[Path] = None,
) -> Tuple[float, Dict[float, float], Dict[float, np.ndarray]]:
    """
    Compute oMAP at all ontology distance thresholds.

    Returns
    -------
    omap_average : float
        Scalar mean oMAP.
    omap_per_threshold : dict[threshold → float]
        Mean AP per threshold.
    omap_per_threshold_per_class : dict[threshold → ndarray(C)]
        AP per class at each threshold.
    """
    ontology_weight = _build_weight(target, graph_distance, cache_path=cache_path)
    omap_details: Dict[float, np.ndarray] = {}
    max_val = int(np.max(graph_distance))

    while True:
        try:
            for threshold in tqdm(
                np.linspace(0, max_val, max_val + 1),
                desc='[oMAP] Computing',
                leave=False,
            ):
                fps_ap = []
                for i in range(target.shape[1]):
                    fps_weight = ontology_weight[threshold][i]
                    fps_ap.append(
                        _average_precision(
                            target[:, i],
                            clipwise_output[:, i],
                            tps_weight=None,
                            fps_weight=fps_weight,
                        )
                    )
                omap_details[threshold] = np.array(fps_ap)
            break
        except Exception:
            # Cache may be stale — rebuild once
            ontology_weight = _build_weight(
                target, graph_distance, cache_path=cache_path, refresh=True
            )

    omap_per_threshold = {k: float(np.mean(v)) for k, v in omap_details.items()}
    omap_average       = float(np.mean(list(omap_per_threshold.values())))

    return omap_average, omap_per_threshold, omap_details


# ══════════════════════════════════════════════════════════════════════════════
# Public accumulator class
# ══════════════════════════════════════════════════════════════════════════════

class OntologyMAP:
    """
    Accumulator for Ontology Mean Average Precision (oMAP).

    Designed for use inside a Lightning ``validation_step`` /
    ``on_validation_epoch_end`` loop:

    .. code-block:: python

        omap = OntologyMAP(cache_path=exp_dir / 'omap_cache.pkl')

        # Inside validation_step:
        probs = torch.sigmoid(logits)
        omap.update(probs, targets)

        # Inside on_validation_epoch_end (every N epochs):
        val = omap.compute()
        omap.reset()

    Args:
        dist_path:  Path to the ``(456, 456)`` float32 graph distance matrix.
                    Defaults to ``configs/datasets/audioset_strong_graph_distance.npy``.
        cache_path: Path for caching the per-class weight arrays (pickle).
                    Highly recommended — first build takes 1–2 minutes.
                    ``None`` disables caching (recomputes every time).
    """

    def __init__(
        self,
        dist_path: Optional[Union[str, Path]] = None,
        cache_path: Optional[Union[str, Path]] = None,
    ) -> None:
        p = Path(dist_path) if dist_path else _DEFAULT_DIST
        if not p.exists():
            raise FileNotFoundError(
                f'Distance matrix not found: {p}\n'
                f'Run:  .venv/bin/python experiments_at2sed/build_strong_ontology.py'
            )
        self._dist       = np.load(str(p)).astype(np.float32)
        self._cache_path = Path(cache_path) if cache_path else None
        self._probs:   List[np.ndarray] = []
        self._targets: List[np.ndarray] = []

    # ── Accumulation API ──────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear accumulated predictions."""
        self._probs.clear()
        self._targets.clear()

    @torch.no_grad()
    def update(
        self,
        probs:   torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """
        Accumulate predictions for one batch.

        Args:
            probs   : ``(B, C)`` sigmoid probabilities (float32).
            targets : ``(B, C)`` float32 multi-hot labels ∈ {0, 1}.
        """
        self._probs.append(probs.detach().cpu().float().numpy())
        self._targets.append(targets.detach().cpu().float().numpy())

    def compute(self) -> float:
        """
        Compute scalar oMAP from all accumulated predictions.

        Returns:
            Scalar float oMAP in [0, 1].

        Note:
            This is computationally expensive the first time
            (cache is built with Numba JIT).  Subsequent calls
            reuse the cache.
        """
        if not self._probs:
            return 0.0

        all_probs   = np.concatenate(self._probs,   axis=0)   # (N, C)
        all_targets = np.concatenate(self._targets, axis=0)   # (N, C)
        all_targets_int = (all_targets > 0.5).astype(np.float32)

        omap_avg, _, _ = _compute_omap(
            all_targets_int, all_probs, self._dist, cache_path=self._cache_path
        )
        return omap_avg
