"""
NASLightningModule — PyTorch Lightning Training Wrapper for NASModel
=====================================================================
Wraps ``NASModel`` into a ``pl.LightningModule`` with:

  * ``training_step``              — tPIT loss (spatial) + BCE loss (semantic)
  * ``validation_step``            — tPIT + BCE losses; SELD + semantic metrics
  * ``on_validation_epoch_end``    — flush and log SELD + semantic metrics; JSON log;
                                     DOA oracle metrics; sed_eval SED metrics; CSV output
  * ``test_step``                  — accumulate SELD + semantic metrics
  * ``on_test_epoch_end``          — flush and log all metrics + confusion matrix to JSON
  * ``configure_optimizers``       — Adam + ReduceLROnPlateau scheduler
  * ``set_metrics_path(path)``     — inject JSON log path after construction

Loss design
-----------
.. math::

    L = \\lambda_s \\cdot L_{\\text{tPIT}}(\\text{act\\_logits}, \\text{doa\\_raw}, T_s)
      + \\lambda_e \\cdot L_{\\text{BCE}}(P_e, \\text{presence}(T_s))

Batch convention
----------------
.. code-block::

    batch = (x, tgt_spatial)
    x           : (B, n_mics, T_samples)  raw waveform at original SR
    tgt_spatial : (B, T_mel,  N, C, 3)   Multi-ACCDOA at mel resolution

TensorBoard logging
-------------------
Training (per step + per epoch):
  ``train/loss``, ``train/loss_tpit``, ``train/loss_bce``
  ``train/loss_doa_active``, ``train/loss_doa_dir``, ``train/loss_doa_norm``,
  ``train/loss_doa_inactive``, ``train/loss_doa_smooth``,
  ``train/pred_norm_active_median``

Validation (per epoch):
  ``val/loss_tpit``, ``val/loss_sed``, ``val/loss_doa``, ``val/loss_bce``,
  ``val/LE_CD``, ``val/LR_CD``, ``val/ER_20``, ``val/F_20``, ``val/SELD_score``
  ``val/doa_oracle_median_AE``, ``val/doa_oracle_acc_20deg``,
  ``val/doa_oracle_pred_norm_active_med``, ``val/doa_oracle_zero_norm_lt_0.10``
  ``val/sed_eval_F1``, ``val/sed_eval_ER`` (when sed_eval is installed)

References
----------
W. Falcon et al., "PyTorch Lightning," GitHub 2019.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import lightning as L

from .nas_model import NASModel
from .loss import (
    tPITLoss,
    SemanticClassificationMetrics,
    SemanticPresenceLoss,
    SELDMetrics,
    TARGET_THRESH,
)
from utils.doa_metrics import (
    accumulate_active_doa,
    align_doa_to_target,
    compute_doa_oracle_metrics,
    compute_doa_oracle_per_class,
)

# ── sed_eval: optional dependency ─────────────────────────────────────────────
try:
    import sed_eval as _sed_eval_lib
    _SED_EVAL_AVAILABLE = True
except ImportError:
    _sed_eval_lib = None
    _SED_EVAL_AVAILABLE = False

# Dataset → number of classes mapping
_N_CLASSES: Dict[str, int] = {
    'STARSS23':          13,
    'TAU-NIGENS2021':    12,
    'TAU2019':           11,
    'BalancedSTARSS23':  13,
}

_ORIG_SR: Dict[str, int] = {
    'STARSS23':          24000,
    'TAU-NIGENS2021':    24000,
    'TAU2019':           24000,
    'BalancedSTARSS23':  24000,
}

_STARSS23_CLASS_NAMES = [
    'Female speech', 'Male speech', 'Clapping', 'Telephone', 'Laughter',
    'Domestic sounds', 'Walk, footsteps', 'Door open/close', 'Music',
    'Musical instrument', 'Water tap', 'Bell', 'Knock',
]


# ══════════════════════════════════════════════════════════════════════════════
# NASLightningModule
# ══════════════════════════════════════════════════════════════════════════════

class NASLightningModule(L.LightningModule):
    """PyTorch Lightning wrapper for NASModel."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters({'cfg': cfg})

        train_cfg    = cfg.get('training', {})
        self.lambda_s = float(train_cfg.get('lambda_spatial',  0.9))
        self.lambda_e = float(train_cfg.get('lambda_semantic', 0.1))
        angle_thresh  = float(train_cfg.get('angle_thresh',   20.0))

        N            = cfg['heads']['n_tracks']
        dataset_name = cfg.get('semantic', {}).get('dataset', 'STARSS23')
        n_classes    = _N_CLASSES.get(dataset_name, 13)
        self.orig_sr = _ORIG_SR.get(dataset_name, 24000)
        self.n_classes = n_classes

        self.model = NASModel(cfg)

        # ── DOA loss config ───────────────────────────────────────────────
        doa_loss_cfg = train_cfg.get('doa_loss', None)
        beta_sed     = float(train_cfg.get('beta_sed', 0.5))
        focal_gamma  = float(train_cfg.get('focal_gamma', 0.0))
        self.tpit_loss = tPITLoss(
            N=N, beta=beta_sed,
            focal_gamma=focal_gamma, doa_loss_cfg=doa_loss_cfg,
        )
        self.sem_loss    = SemanticPresenceLoss()
        self.metrics     = SELDMetrics(angle_thresh=angle_thresh)
        self.sem_metrics = SemanticClassificationMetrics(n_classes)

        self.metrics_starss  = SELDMetrics(angle_thresh=angle_thresh)
        self.metrics_tau     = SELDMetrics(angle_thresh=angle_thresh)
        self.metrics_tau2019 = SELDMetrics(angle_thresh=angle_thresh)
        self._mixed_test = (dataset_name == 'BalancedSTARSS23')

        self._perclass_metrics = bool(train_cfg.get('perclass_metrics', False))

        # ── Metrics config ────────────────────────────────────────────────
        metrics_cfg     = cfg.get('metrics', {})
        self.act_thresh = float(metrics_cfg.get('act_thresh', 0.5))

        # ── sed_eval availability ─────────────────────────────────────────
        self._sed_eval_available = _SED_EVAL_AVAILABLE
        if not _SED_EVAL_AVAILABLE:
            print('[NASLightningModule] sed_eval not installed — SED-only metrics disabled')

        # ── Test accumulators for DOA oracle and sed_eval ─────────────────
        # Only used in test_step / on_test_epoch_end (not during validation).
        # Keyed by dataloader_idx (0=STARSS, 1=TAU, 2=TAU2019).
        self._test_doa_pred:   Dict[int, List[np.ndarray]] = {0: [], 1: [], 2: []}
        self._test_doa_tgt:    Dict[int, List[np.ndarray]] = {0: [], 1: [], 2: []}
        self._test_doa_cls:    Dict[int, List[np.ndarray]] = {0: [], 1: [], 2: []}
        self._test_sed_logits: Dict[int, List[np.ndarray]] = {0: [], 1: [], 2: []}
        self._test_sed_tgt:    Dict[int, List[np.ndarray]] = {0: [], 1: [], 2: []}

        # ── Loss component accumulator for epoch-level CSV ────────────────
        self._loss_comps_epoch: List[dict] = []
        self._debug_loss_components = bool(train_cfg.get('debug_loss_components', False))

        # ── JSON metrics log ──────────────────────────────────────────────
        self._metrics_path: Optional[Path] = None
        self._epoch_metrics_log: list      = []

        # ── Print configuration banner ────────────────────────────────────
        self._print_banner(cfg, doa_loss_cfg, beta_sed, focal_gamma)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def set_metrics_path(self, path: Path) -> None:
        """Inject JSON metrics log path after construction."""
        self._metrics_path = Path(path)

    def _print_banner(self, cfg: dict, doa_loss_cfg: Optional[dict],
                      beta_sed: float, focal_gamma: float) -> None:
        doa_cfg = doa_loss_cfg or {}
        mode    = doa_cfg.get('mode', 'legacy_mse_all')
        legacy  = (mode == 'legacy_mse_all')
        train_cfg = cfg.get('training', {})
        dataset_name = cfg.get('semantic', {}).get('dataset', 'STARSS23')
        print('\n' + '=' * 80)
        print('SELD TRAINING WITH CONFIGURABLE DOA LOSS')
        print('=' * 80)
        print(f'  Model family:          NAS-SELD')
        print(f'  Dataset:               {dataset_name}')
        print(f'  Augmentation:          {cfg.get("augmentation", {}).get("enabled", False)}')
        print()
        print('  Loss configuration:')
        print(f'    beta_sed:            {beta_sed}')
        print(f'    focal_gamma:         {focal_gamma}')
        print(f'    lambda_spatial:      {train_cfg.get("lambda_spatial", 0.9)}')
        print(f'    lambda_semantic:     {train_cfg.get("lambda_semantic", 0.1)}')
        print(f'    doa_loss.mode:       {mode}')
        print(f'    mse_weight:          {doa_cfg.get("mse_weight", 1.0)}')
        print(f'    direction_weight:    {doa_cfg.get("direction_weight", 1.0)}')
        print(f'    norm_weight:         {doa_cfg.get("norm_weight", 0.0)}')
        print(f'    inactive_weight:     {doa_cfg.get("inactive_weight", 0.0)}')
        print(f'    smooth_weight:       {doa_cfg.get("smooth_weight", 0.0)}')
        print(f'    legacy mode:         {legacy}')
        print()
        print('  Metrics:')
        print(f'    internal SELD:       val + test')
        print(f'    sed_eval SED:        test-only{"" if self._sed_eval_available else " (sed_eval not installed — will be skipped)"}')
        print(f'    DOA oracle:          test-only')
        print(f'    act_thresh:          {self.act_thresh}')
        print()
        print('  Retro-compatibility:')
        print(f'    old config detected: {"yes" if doa_loss_cfg is None else "no"}')
        print(f'    using legacy DOA:    {legacy}')
        print('=' * 80 + '\n')

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        x, tgt_spatial = batch
        act_logits, doa_tanh, out_e = self.model(x, orig_sr=self.orig_sr)

        T = min(act_logits.shape[1], tgt_spatial.shape[1])
        act_logits  = act_logits[:, :T]
        doa_tanh    = doa_tanh[:, :T]
        tgt_spatial = tgt_spatial[:, :T]

        loss_tpit, _, _ = self.tpit_loss(act_logits, doa_tanh, tgt_spatial)
        loss_bce        = self.sem_loss(out_e, tgt_spatial)
        loss = self.lambda_s * loss_tpit + self.lambda_e * loss_bce

        self.log('train/loss',      loss,      on_step=True,  on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.log('train/loss_tpit', loss_tpit, on_step=False, on_epoch=True,
                 sync_dist=True)
        self.log('train/loss_bce',  loss_bce,  on_step=False, on_epoch=True,
                 sync_dist=True)

        # ── Log DOA loss components ───────────────────────────────────────
        comps = self.tpit_loss.last_components
        self.log('train/loss_doa_active',   comps.get('loss_doa_mse_active', 0.0),
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log('train/loss_doa_dir',      comps.get('loss_doa_direction',  0.0),
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log('train/loss_doa_norm',     comps.get('loss_doa_norm',       0.0),
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log('train/loss_doa_inactive', comps.get('loss_doa_inactive',   0.0),
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log('train/loss_doa_smooth',   comps.get('loss_doa_smooth',     0.0),
                 on_step=False, on_epoch=True, sync_dist=True)
        v = comps.get('pred_norm_active_median', float('nan'))
        if not math.isnan(v):
            self.log('train/pred_norm_active_median', v,
                     on_step=False, on_epoch=True, sync_dist=True)

        # ── Accumulate loss components for epoch CSV ──────────────────────
        if self._debug_loss_components:
            row = {
                'epoch': self.current_epoch, 'split': 'train',
                'step':  self.global_step,
                **{k: float(val) for k, val in comps.items()},
            }
            self._loss_comps_epoch.append(row)

        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        x, tgt_spatial = batch
        act_logits, doa_tanh, out_e = self.model(x, orig_sr=self.orig_sr)

        T = min(act_logits.shape[1], tgt_spatial.shape[1])
        act_logits  = act_logits[:, :T]
        doa_tanh    = doa_tanh[:, :T]
        tgt_spatial = tgt_spatial[:, :T]

        # Single tPITLoss call — access last_best_perm immediately after
        loss_tpit, loss_sed, loss_doa = self.tpit_loss(act_logits, doa_tanh, tgt_spatial)
        loss_bce = self.sem_loss(out_e, tgt_spatial)

        self.log('val/loss_tpit', loss_tpit, on_epoch=True, sync_dist=True)
        self.log('val/loss_sed',  loss_sed,  on_epoch=True, sync_dist=True)
        self.log('val/loss_doa',  loss_doa,  on_epoch=True, sync_dist=True)
        self.log('val/loss_bce',  loss_bce,  on_epoch=True, sync_dist=True)

        self.metrics.update(act_logits, doa_tanh, tgt_spatial)
        self.sem_metrics.update(out_e, tgt_spatial)

    def on_validation_epoch_end(self) -> None:
        # ── SELD metrics — compute perclass BEFORE reset ──────────────────
        results      = self.metrics.compute()
        perclass_rows = self.metrics.compute_perclass()
        self.metrics.reset()

        for key, val in results.items():
            self.log(f'val/{key}', val, on_epoch=True, sync_dist=True,
                     prog_bar=(key == 'SELD_score'))
        self.log('val_seld_score', float(results['SELD_score']),
                 on_epoch=True, sync_dist=True)

        # ── Semantic metrics ──────────────────────────────────────────────
        sem = self.sem_metrics.compute(include_confusion_matrix=False)
        self.sem_metrics.reset()
        for key, val in sem.items():
            self.log(f'val/{key}', float(val), on_epoch=True, sync_dist=True)

        # ── JSON log (SELD-only — DOA oracle and sed_eval are test-only) ──
        if self._metrics_path is not None:
            epoch_dict: dict = {'epoch': self.current_epoch}
            epoch_dict.update({f'val/{k}': float(v) for k, v in results.items()})
            epoch_dict.update({f'val/{k}': float(v) for k, v in sem.items()})
            self._epoch_metrics_log.append(epoch_dict)
            self._flush_metrics_log()
            self._flush_csv_metrics(
                epoch=self.current_epoch, split='val',
                seld_results=results, doa_results={}, sed_results={},
            )
            if self._perclass_metrics:
                self._flush_perclass_metrics(
                    split='val', epoch=self.current_epoch, rows=perclass_rows
                )
            if self._loss_comps_epoch:
                self._flush_loss_components_csv()

        # ── Terminal summary ──────────────────────────────────────────────
        self._print_val_summary('val', results)

        # ── LR ────────────────────────────────────────────────────────────
        opt = self.optimizers()
        if opt is not None:
            self.log('train/lr', opt.param_groups[0]['lr'], on_epoch=True, sync_dist=True)

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        x, tgt_spatial = batch
        act_logits, doa_tanh, out_e = self.model(x, orig_sr=self.orig_sr)
        T = min(act_logits.shape[1], tgt_spatial.shape[1])
        act_logits  = act_logits[:, :T]
        doa_tanh    = doa_tanh[:, :T]
        tgt_spatial = tgt_spatial[:, :T]

        # Run loss with no_grad to populate last_best_perm for DOA oracle alignment
        with torch.no_grad():
            self.tpit_loss(act_logits, doa_tanh, tgt_spatial)

        # Accumulate DOA oracle data (using PIT-aligned predictions)
        B, T2, N, C, _ = doa_tanh.shape
        pred_aligned = align_doa_to_target(
            doa_tanh, self.tpit_loss.last_best_perm, B, T2, N, C
        )
        p, t, cls = accumulate_active_doa(
            pred_aligned, tgt_spatial,
            active_eps=self.tpit_loss.doa_active_eps,
        )
        self._test_doa_pred[dataloader_idx].append(p)
        self._test_doa_tgt[dataloader_idx].append(t)
        self._test_doa_cls[dataloader_idx].append(cls)

        if self._sed_eval_available:
            self._test_sed_logits[dataloader_idx].append(act_logits.detach().cpu().numpy())
            self._test_sed_tgt[dataloader_idx].append(tgt_spatial.detach().cpu().numpy())

        if self._mixed_test:
            if dataloader_idx == 0:
                self.metrics_starss.update(act_logits, doa_tanh, tgt_spatial)
            elif dataloader_idx == 1:
                self.metrics_tau.update(act_logits, doa_tanh, tgt_spatial)
            else:
                self.metrics_tau2019.update(act_logits, doa_tanh, tgt_spatial)
        else:
            self.metrics.update(act_logits, doa_tanh, tgt_spatial)
            self.sem_metrics.update(out_e, tgt_spatial)

    def on_test_epoch_end(self) -> None:
        _SRC = {0: 'test_starss', 1: 'test_tau', 2: 'test_tau2019'}

        if self._mixed_test:
            # ── Per-source DOA oracle and sed_eval ────────────────────────
            doa_per_src:    Dict[int, dict] = {}
            sed_per_src:    Dict[int, dict] = {}
            doa_pc_per_src: Dict[int, list] = {}
            for idx, src_label in _SRC.items():
                if self._test_doa_pred[idx]:
                    pred_all = np.concatenate(self._test_doa_pred[idx], axis=0)
                    tgt_all  = np.concatenate(self._test_doa_tgt[idx],  axis=0)
                    cls_all  = np.concatenate(self._test_doa_cls[idx],  axis=0)
                    doa_m  = compute_doa_oracle_metrics(pred_all, tgt_all)
                    doa_pc = compute_doa_oracle_per_class(
                        pred_all, tgt_all, cls_all, self.n_classes, _STARSS23_CLASS_NAMES
                    )
                    doa_per_src[idx]    = doa_m
                    doa_pc_per_src[idx] = doa_pc
                    for k, v in doa_m.items():
                        if not math.isnan(float(v)):
                            self.log(f'{src_label}/doa_oracle_{k}', float(v),
                                     on_epoch=True, sync_dist=True)
                self._test_doa_pred[idx].clear()
                self._test_doa_tgt[idx].clear()
                self._test_doa_cls[idx].clear()
                if self._sed_eval_available and self._test_sed_logits[idx]:
                    sed_m = self._compute_sed_eval_metrics(
                        self._test_sed_logits[idx], self._test_sed_tgt[idx]
                    )
                    sed_per_src[idx] = sed_m
                    for k, v in sed_m.items():
                        self.log(f'{src_label}/{k}', float(v), on_epoch=True, sync_dist=True)
                self._test_sed_logits[idx].clear()
                self._test_sed_tgt[idx].clear()

            # ── Per-source SELD metrics ───────────────────────────────────
            starss    = self.metrics_starss.compute()
            starss_pc = self.metrics_starss.compute_perclass()
            self.metrics_starss.reset()
            tau       = self.metrics_tau.compute()
            self.metrics_tau.reset()
            tau2019   = self.metrics_tau2019.compute()
            self.metrics_tau2019.reset()
            for key, val in starss.items():
                self.log(f'test_starss/{key}', float(val), on_epoch=True, sync_dist=True)
            for key, val in tau.items():
                self.log(f'test_tau/{key}', float(val), on_epoch=True, sync_dist=True)
            for key, val in tau2019.items():
                self.log(f'test_tau2019/{key}', float(val), on_epoch=True, sync_dist=True)

            if self._metrics_path is not None:
                test_dict: dict = {'epoch': 'test'}
                test_dict.update({f'test_starss/{k}':  float(v) for k, v in starss.items()})
                test_dict.update({f'test_tau/{k}':     float(v) for k, v in tau.items()})
                test_dict.update({f'test_tau2019/{k}': float(v) for k, v in tau2019.items()})
                for idx, src_label in _SRC.items():
                    for k, v in doa_per_src.get(idx, {}).items():
                        test_dict[f'{src_label}/doa_oracle_{k}'] = float(v)
                    for k, v in sed_per_src.get(idx, {}).items():
                        test_dict[f'{src_label}/{k}'] = float(v)
                self._epoch_metrics_log.append(test_dict)
                self._flush_metrics_log()
                seld_by_src = {0: starss, 1: tau, 2: tau2019}
                for idx, src_label in _SRC.items():
                    self._flush_csv_metrics(
                        epoch='test', split=src_label,
                        seld_results=seld_by_src[idx],
                        doa_results=doa_per_src.get(idx, {}),
                        sed_results=sed_per_src.get(idx, {}),
                    )
                if doa_pc_per_src.get(0):
                    self._flush_doa_per_class_csv('test', 'test_starss', doa_pc_per_src[0])
                if self._perclass_metrics:
                    self._flush_perclass_metrics('test', 'test', starss_pc)
            self._print_val_summary('test_starss', starss,
                                    doa_per_src.get(0, {}), sed_per_src.get(0, {}))
            return

        # ── Non-mixed test (single dataset) ──────────────────────────────
        doa_results: dict = {}
        doa_pc_rows: list = []
        if self._test_doa_pred[0]:
            pred_all = np.concatenate(self._test_doa_pred[0], axis=0)
            tgt_all  = np.concatenate(self._test_doa_tgt[0],  axis=0)
            cls_all  = np.concatenate(self._test_doa_cls[0],  axis=0)
            doa_results = compute_doa_oracle_metrics(pred_all, tgt_all)
            doa_pc_rows = compute_doa_oracle_per_class(
                pred_all, tgt_all, cls_all, self.n_classes, _STARSS23_CLASS_NAMES
            )
            for k, v in doa_results.items():
                if not math.isnan(float(v)):
                    self.log(f'test/doa_oracle_{k}', float(v), on_epoch=True, sync_dist=True)
        self._test_doa_pred[0].clear()
        self._test_doa_tgt[0].clear()
        self._test_doa_cls[0].clear()

        sed_results: dict = {}
        if self._sed_eval_available and self._test_sed_logits[0]:
            sed_results = self._compute_sed_eval_metrics(
                self._test_sed_logits[0], self._test_sed_tgt[0]
            )
            for k, v in sed_results.items():
                self.log(f'test/{k}', float(v), on_epoch=True, sync_dist=True)
        self._test_sed_logits[0].clear()
        self._test_sed_tgt[0].clear()

        seld = self.metrics.compute()
        seld_pc = self.metrics.compute_perclass()
        self.metrics.reset()
        sem  = self.sem_metrics.compute(include_confusion_matrix=True)
        self.sem_metrics.reset()
        conf_mat = sem.pop('sem_confusion_matrix', None)

        for key, val in seld.items():
            self.log(f'test/{key}', float(val), on_epoch=True, sync_dist=True)
        for key, val in sem.items():
            self.log(f'test/{key}', float(val), on_epoch=True, sync_dist=True)

        if self._metrics_path is not None:
            test_dict2: dict = {'epoch': 'test'}
            test_dict2.update({f'test/{k}': v for k, v in seld.items()})
            test_dict2.update({f'test/{k}': v for k, v in sem.items()})
            test_dict2.update({f'test/doa_{k}': v for k, v in doa_results.items()})
            test_dict2.update({f'test/{k}': v for k, v in sed_results.items()})
            self._epoch_metrics_log.append(test_dict2)
            self._flush_metrics_log()
            self._flush_csv_metrics(
                epoch='test', split='test',
                seld_results=seld, doa_results=doa_results, sed_results=sed_results,
            )
            if doa_pc_rows:
                self._flush_doa_per_class_csv('test', 'test', doa_pc_rows)
            if self._perclass_metrics:
                self._flush_perclass_metrics('test', 'test', seld_pc)
            if conf_mat is not None:
                cm_path = self._metrics_path.with_name(
                    self._metrics_path.stem + '_confusion_matrix.json'
                )
                with open(cm_path, 'w') as fh:
                    json.dump({'sem_confusion_matrix': conf_mat}, fh, indent=2)

        self._print_val_summary('test', seld, doa_results, sed_results)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        train_cfg = self.hparams['cfg'].get('training', {})
        lr        = float(train_cfg.get('lr', 5e-4))
        factor    = float(train_cfg.get('lr_reduce_factor', 0.75))
        patience  = int(train_cfg.get('lr_patience', 20))
        lr_min    = float(train_cfg.get('lr_min', 1e-5))

        params    = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=factor, patience=patience, min_lr=lr_min,
        )
        return {
            'optimizer':    optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor':   'val_seld_score',
                'interval':  'epoch',
                'frequency': 1,
            },
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_sed_eval_metrics(
        self,
        logits_parts: List[np.ndarray],
        tgt_parts:    List[np.ndarray],
    ) -> dict:
        """Compute sed_eval frame-based SED metrics from accumulated batch data."""
        try:
            import sed_eval
            # Convert accumulated batches to binary activity matrices
            # logits_parts: list of (B, T, N, C)  — raw SED logits
            # tgt_parts:    list of (B, T, N, C, 3) — Multi-ACCDOA targets
            all_pred_frames = []
            all_tgt_frames  = []
            for logits_b, tgt_b in zip(logits_parts, tgt_parts):
                B, T, N, C = logits_b.shape
                # pred activity: sigmoid(logit) > act_thresh → (B, T, N, C) → any over N → (B, T, C)
                pred_act = (1.0 / (1.0 + np.exp(-logits_b)) > self.act_thresh)
                pred_cls = pred_act.any(axis=2)   # (B, T, C) bool
                # target activity: norm(target_doa) > TARGET_THRESH → (B, T, N, C) → any → (B, T, C)
                tgt_norm = np.linalg.norm(tgt_b, axis=-1)          # (B, T, N, C)
                tgt_act  = (tgt_norm > TARGET_THRESH).any(axis=2)  # (B, T, C) bool
                # Flatten B*T rows
                all_pred_frames.append(pred_cls.reshape(B * T, C))
                all_tgt_frames.append(tgt_act.reshape(B * T, C))

            pred_mat = np.concatenate(all_pred_frames, axis=0).astype(float)  # (Frames, C)
            tgt_mat  = np.concatenate(all_tgt_frames,  axis=0).astype(float)  # (Frames, C)

            C = pred_mat.shape[1]
            metrics_inst = sed_eval.sound_event.SegmentBasedMetrics(
                event_label_list=[str(c) for c in range(C)],
                time_resolution=1.0,
            )
            # Feed frame by frame (each frame = 1-second segment for sed_eval API)
            for i in range(len(pred_mat)):
                ref_events = [
                    {'event_label': str(c), 'event_onset': float(i), 'event_offset': float(i + 1)}
                    for c in range(C) if tgt_mat[i, c] > 0
                ]
                est_events = [
                    {'event_label': str(c), 'event_onset': float(i), 'event_offset': float(i + 1)}
                    for c in range(C) if pred_mat[i, c] > 0
                ]
                metrics_inst.evaluate(
                    reference_event_list=ref_events,
                    estimated_event_list=est_events,
                )
            out  = metrics_inst.results_overall_metrics()
            f1   = out.get('f_measure', {}).get('f_measure',  float('nan'))
            er   = out.get('error_rate', {}).get('error_rate', float('nan'))
            prec = out.get('f_measure', {}).get('precision',  float('nan'))
            rec  = out.get('f_measure', {}).get('recall',     float('nan'))
            return {
                'sed_eval_F1':        float(f1)   if f1   is not None else float('nan'),
                'sed_eval_ER':        float(er)   if er   is not None else float('nan'),
                'sed_eval_precision': float(prec) if prec is not None else float('nan'),
                'sed_eval_recall':    float(rec)  if rec  is not None else float('nan'),
            }
        except Exception as exc:
            print(f'[NASLightningModule] sed_eval computation failed: {exc}')
            return {}

    def _print_val_summary(
        self,
        split: str,
        seld_results: dict,
        doa_results:  Optional[dict] = None,
        sed_results:  Optional[dict] = None,
    ) -> None:
        print(f'\n{split} metrics (epoch {self.current_epoch}):')
        print(f'  SELD_score: {seld_results.get("SELD_score", float("nan")):.4f}')
        print(f'  F_20:       {seld_results.get("F_20",       float("nan")):.4f}')
        print(f'  ER_20:      {seld_results.get("ER_20",      float("nan")):.4f}')
        print(f'  LE_CD:      {seld_results.get("LE_CD",      float("nan")):.2f}°')
        print(f'  LR_CD:      {seld_results.get("LR_CD",      float("nan")):.4f}')
        if sed_results:
            print(f'  sed_eval_F1:        {sed_results.get("sed_eval_F1",        float("nan")):.4f}')
            print(f'  sed_eval_ER:        {sed_results.get("sed_eval_ER",        float("nan")):.4f}')
            print(f'  sed_eval_precision: {sed_results.get("sed_eval_precision", float("nan")):.4f}')
            print(f'  sed_eval_recall:    {sed_results.get("sed_eval_recall",    float("nan")):.4f}')
        if doa_results:
            print(f'  doa_oracle_median_AE:            {doa_results.get("angular_error_median",   float("nan")):.2f}°')
            print(f'  doa_oracle_acc_20deg:            {doa_results.get("acc_20deg",              float("nan")):.4f}')
            print(f'  doa_oracle_pred_norm_active_med: {doa_results.get("pred_norm_active_median",float("nan")):.4f}')
            print(f'  doa_oracle_zero_norm_lt_0.10:    {doa_results.get("zero_norm_active_lt_0.10",float("nan")):.4f}')

    def _flush_metrics_log(self) -> None:
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._metrics_path.with_suffix('.tmp')
        with open(tmp, 'w') as fh:
            json.dump(self._epoch_metrics_log, fh, indent=2, default=float)
        tmp.replace(self._metrics_path)

    def _flush_csv_metrics(
        self,
        epoch,
        split: str,
        seld_results: dict,
        doa_results:  dict,
        sed_results:  dict,
    ) -> None:
        """Append one row to metrics_summary.csv."""
        if self._metrics_path is None:
            return
        csv_path = self._metrics_path.with_name('metrics_summary.csv')
        cfg       = self.hparams['cfg']
        train_cfg = cfg.get('training', {})
        doa_cfg   = train_cfg.get('doa_loss', {}) or {}
        row = {
            'epoch':              epoch,
            'split':              split,
            'model_name':         cfg.get('experiment', {}).get('name', ''),
            'dataset':            cfg.get('semantic', {}).get('dataset', ''),
            'loss_mode':          doa_cfg.get('mode', 'legacy_mse_all'),
            'beta_sed':           train_cfg.get('beta_sed', ''),
            'SELD_score':         seld_results.get('SELD_score', ''),
            'ER_20':              seld_results.get('ER_20', ''),
            'F_20':               seld_results.get('F_20', ''),
            'LE_CD':              seld_results.get('LE_CD', ''),
            'LR_CD':              seld_results.get('LR_CD', ''),
            'sed_eval_ER':        sed_results.get('sed_eval_ER', ''),
            'sed_eval_F1':        sed_results.get('sed_eval_F1', ''),
            'sed_eval_precision': sed_results.get('sed_eval_precision', ''),
            'sed_eval_recall':    sed_results.get('sed_eval_recall', ''),
            'doa_oracle_median_AE':            doa_results.get('angular_error_median', ''),
            'doa_oracle_acc_20deg':             doa_results.get('acc_20deg', ''),
            'doa_oracle_pred_norm_active_median': doa_results.get('pred_norm_active_median', ''),
            'doa_oracle_zero_norm_active_lt_0.10': doa_results.get('zero_norm_active_lt_0.10', ''),
        }
        write_header = not csv_path.exists()
        with open(csv_path, 'a', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _flush_doa_per_class_csv(self, epoch, split: str, rows: list) -> None:
        if self._metrics_path is None:
            return
        csv_path = self._metrics_path.with_name('doa_oracle_per_class.csv')
        write_header = not csv_path.exists()
        with open(csv_path, 'a', newline='') as fh:
            fieldnames = ['epoch', 'split'] + list(rows[0].keys())
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for r in rows:
                writer.writerow({'epoch': epoch, 'split': split, **r})

    def _flush_loss_components_csv(self) -> None:
        if self._metrics_path is None or not self._loss_comps_epoch:
            return
        csv_path = self._metrics_path.with_name('loss_components.csv')
        write_header = not csv_path.exists()
        with open(csv_path, 'a', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(self._loss_comps_epoch[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(self._loss_comps_epoch)
        self._loss_comps_epoch.clear()

    def _flush_perclass_metrics(self, split: str, epoch, rows: list) -> None:
        pc_path = self._metrics_path.with_name('perclass_sed_metrics.json')
        existing: list = []
        if pc_path.exists():
            with open(pc_path) as fh:
                try:
                    existing = json.load(fh)
                except json.JSONDecodeError:
                    pass
        for row in rows:
            existing.append({'epoch': epoch, 'split': split, **row})
        tmp = pc_path.with_suffix('.tmp')
        with open(tmp, 'w') as fh:
            json.dump(existing, fh, indent=2, default=float)
        tmp.replace(pc_path)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def trainable_params(self) -> List[torch.nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]
