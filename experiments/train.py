"""
NASLightningModule — PyTorch Lightning Training Wrapper for NASModel
=====================================================================
Wraps ``NASModel`` into a ``pl.LightningModule`` with:

  * ``training_step``              — tPIT loss (spatial) + BCE loss (semantic)
  * ``validation_step``            — tPIT + BCE losses; SELD + semantic metrics
  * ``on_validation_epoch_end``    — flush and log SELD + semantic metrics; JSON log
  * ``test_step``                  — accumulate SELD + semantic metrics
  * ``on_test_epoch_end``          — flush and log all metrics + confusion matrix to JSON
  * ``configure_optimizers``       — Adam + CosineAnnealingLR scheduler
  * ``set_metrics_path(path)``     — inject JSON log path after construction

Loss design
-----------
.. math::

    L = \\lambda_s \\cdot L_{\\text{tPIT}}(\\text{act\\_logits}, \\text{doa\\_raw}, T_s)
      + \\lambda_e \\cdot L_{\\text{BCE}}(P_e, \\text{presence}(T_s))

where ``act_logits`` :math:`\\in \\mathbb{R}^{B \\times T \\times N \\times C}` are the
raw SED logits, ``doa_tanh`` :math:`\\in \\mathbb{R}^{B \\times T \\times N \\times C \\times 3}`
are the tanh-bounded DOA vectors, and :math:`P_e \\in \\mathbb{R}^{B \\times T_{\\text{sem}}
\\times C}` is the semantic class-presence logit tensor.  The BCE target is derived
from the spatial target by nearest-neighbour downsampling followed by a per-class
binary activity collapse (``any`` over tracks).

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

Validation (per epoch):
  ``val/loss_tpit``, ``val/loss_sed``, ``val/loss_doa``, ``val/loss_bce``,
  ``val/LE_CD``, ``val/LR_CD``, ``val/ER_20``, ``val/F_20``, ``val/SELD_score``

References
----------
W. Falcon et al., "PyTorch Lightning," GitHub 2019.
https://lightning.ai/docs/pytorch/stable/

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import lightning as L

from .nas_model import NASModel
from .loss import (
    tPITLoss,
    SemanticClassificationMetrics,
    SemanticPresenceLoss,
    SELDMetrics,
)

# Dataset → number of classes mapping (mirrors main_experiments_1.py DATASET_N_CLASSES)
_N_CLASSES: Dict[str, int] = {
    'STARSS23':       13,
    'TAU-NIGENS2021': 12,
}

# Dataset → native (original) sample rate; both current datasets are 24 kHz.
# NASModel.forward resamples to cfg['input']['sr'] only when orig_sr is passed.
_ORIG_SR: Dict[str, int] = {
    'STARSS23':       24000,
    'TAU-NIGENS2021': 24000,
}


# ══════════════════════════════════════════════════════════════════════════════
# NASLightningModule
# ══════════════════════════════════════════════════════════════════════════════

class NASLightningModule(L.LightningModule):
    """
    PyTorch Lightning wrapper for NASModel.

    Args:
        cfg (dict): merged config dict (nas_base.yaml + nas_train.yaml).

    Logged metrics (TensorBoard)
    ----------------------------
    Training (per step + per epoch):
      train/loss        — weighted total loss
      train/loss_tpit   — spatial tPIT loss (β·L_SED + (1-β)·L_DOA)
      train/loss_bce    — semantic head BCE loss (auxiliary)

    Validation (per epoch):
      val/loss_tpit     — spatial tPIT loss
      val/loss_sed      — SED component of tPIT (β-weighted BCE)
      val/loss_doa      — DOA component of tPIT ((1-β)-weighted MSE)
      val/loss_bce      — semantic head BCE loss
      val/LE_CD         — mean angular error for TP events (°)
      val/LR_CD         — localisation recall
      val/ER_20         — location-dependent error rate   (20° tolerance)
      val/F_20          — location-dependent F-score      (20° tolerance)
      val/SELD_score    — combined SELD score (lower = better)
      val/sem_accuracy  — multilabel accuracy (macro)
      val/sem_f1        — multilabel F1 (macro)
      val/sem_auroc     — multilabel AUROC (macro)

    Test (once per run, after trainer.test()):
      test/SELD_score, test/LE_CD, …  (same spatial set)
      test/sem_accuracy, …            (same semantic set)
      confusion matrix → JSON only (not TensorBoard)
    """

    def __init__(self, cfg: dict):
        """Initialize NASLightningModule with model, losses, and metrics."""
        super().__init__()
        self.save_hyperparameters({'cfg': cfg})

        train_cfg    = cfg.get('training', {})
        self.lambda_s = float(train_cfg.get('lambda_spatial',  0.9))
        self.lambda_e = float(train_cfg.get('lambda_semantic', 0.1))
        angle_thresh  = float(train_cfg.get('angle_thresh',   20.0))

        N            = cfg['heads']['n_tracks']
        dataset_name = cfg.get('semantic', {}).get('dataset', 'STARSS23')
        n_classes    = _N_CLASSES.get(dataset_name, 13)
        self.orig_sr = _ORIG_SR.get(dataset_name, 24000)   # passed to model.forward

        self.model = NASModel(cfg)

        # ── Per-class BCE pos_weight (class-imbalance correction) ────────────
        # Computed from training CSV annotations: pos_weight[c] = neg[c]/pos[c].
        # Supported for both STARSS23 and TAU-NIGENS2021.
        # Graceful fallback to None (uniform weighting) if path is absent.
        pos_weight = None
        root_path  = cfg.get('data', {}).get('root_path', None)
        if root_path:
            try:
                if dataset_name == 'STARSS23':
                    from datasets.starss23 import compute_pos_weights
                elif dataset_name == 'TAU-NIGENS2021':
                    from datasets.tau_nigens import compute_pos_weights
                else:
                    compute_pos_weights = None
                if compute_pos_weights is not None:
                    pos_weight = compute_pos_weights(Path(root_path), n_classes=n_classes)
                    # Cap pos_weight to avoid extreme gradient magnitudes for very rare classes.
                    # Configured via training.pos_weight_max in configs/datasets/{NAME}.yaml.
                    pw_cap = float(train_cfg.get('pos_weight_max', float('inf')))
                    if pw_cap < float('inf'):
                        pos_weight = pos_weight.clamp(max=pw_cap)
                    pw_min = pos_weight.min().item()
                    pw_max = pos_weight.max().item()
                    print(f'[NASLightningModule] pos_weight computed  '
                          f'min={pw_min:.2f}  max={pw_max:.1f}  cap={pw_cap}'
                          f'  (classes={n_classes})')
            except Exception as exc:
                print(f'[NASLightningModule] pos_weight unavailable: {exc}')

        beta_sed       = float(train_cfg.get('beta_sed', 0.5))
        focal_gamma    = float(train_cfg.get('focal_gamma', 0.0))
        self.tpit_loss = tPITLoss(N=N, beta=beta_sed, pos_weight=pos_weight,
                                  focal_gamma=focal_gamma)
        self.sem_loss  = SemanticPresenceLoss(pos_weight=pos_weight)
        self.metrics     = SELDMetrics(angle_thresh=angle_thresh)
        self.sem_metrics = SemanticClassificationMetrics(n_classes)

        self._perclass_metrics = bool(train_cfg.get('perclass_metrics', False))

        # JSON metrics log — path injected via set_metrics_path() by training script
        self._metrics_path: Optional[Path] = None
        self._epoch_metrics_log: list      = []

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def set_metrics_path(self, path: Path) -> None:
        """Inject JSON metrics log path after construction.

        Called by ``main_experiments_1.py`` so that per-epoch and test metrics
        are persisted to ``{metrics_dir}/{exp_name}.json``.

        Args:
            path : destination file path for the JSON log.
        """
        self._metrics_path = Path(path)

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Compute weighted ADPIT + BCE loss for one training batch.

        Parameters
        ----------
        batch : tuple of (x, tgt_spatial)
            x           : (B, n_mics, T_samples)
            tgt_spatial : (B, T_mel, N, C, 3)
        batch_idx : int
            Index of the current batch (unused; required by Lightning).

        Returns
        -------
        torch.Tensor
            Scalar total loss :math:`L = \\lambda_s L_{\\text{ADPIT}} + \\lambda_e L_{\\text{BCE}}`.

        Logged keys
        -----------
        ``train/loss`` (step + epoch), ``train/loss_tpit`` (epoch),
        ``train/loss_bce`` (epoch).
        """
        x, tgt_spatial = batch                           # tgt: (B, T_mel, N, C, 3)
        act_logits, doa_tanh, out_e = self.model(x, orig_sr=self.orig_sr)
        # act_logits: (B, T_mel, N, C)     raw SED logits
        # doa_tanh  : (B, T_mel, N, C, 3)  tanh-bounded DOA ∈ [-1,1]³
        # out_e     : (B, T_sem, C)        class-presence logits

        # T_mel alignment: resampling rounding may produce ±1 frame difference
        # between padded audio (full-clip mode) and pre-computed target T.
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
        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Accumulate validation losses and both metric accumulators.

        One forward pass per batch; updates SELDMetrics (spatial head) and
        SemanticClassificationMetrics (semantic head) simultaneously.

        Parameters
        ----------
        batch : tuple of (x, tgt_spatial)
        batch_idx : int
        """
        x, tgt_spatial = batch
        act_logits, doa_tanh, out_e = self.model(x, orig_sr=self.orig_sr)

        # T_mel alignment (see training_step)
        T = min(act_logits.shape[1], tgt_spatial.shape[1])
        act_logits  = act_logits[:, :T]
        doa_tanh    = doa_tanh[:, :T]
        tgt_spatial = tgt_spatial[:, :T]

        loss_tpit, loss_sed, loss_doa = self.tpit_loss(act_logits, doa_tanh, tgt_spatial)
        loss_bce = self.sem_loss(out_e, tgt_spatial)

        self.log('val/loss_tpit', loss_tpit, on_epoch=True, sync_dist=True)
        self.log('val/loss_sed',  loss_sed,  on_epoch=True, sync_dist=True)
        self.log('val/loss_doa',  loss_doa,  on_epoch=True, sync_dist=True)
        self.log('val/loss_bce',  loss_bce,  on_epoch=True, sync_dist=True)

        self.metrics.update(act_logits, doa_tanh, tgt_spatial)   # SELD metrics
        self.sem_metrics.update(out_e, tgt_spatial)               # semantic metrics

    def on_validation_epoch_end(self) -> None:
        """Flush both metric accumulators, log all val metrics, update JSON log."""
        # ── SELD metrics ─────────────────────────────────────────────────────
        results = self.metrics.compute()
        self.metrics.reset()
        for key, val in results.items():
            self.log(f'val/{key}', val, on_epoch=True, sync_dist=True,
                     prog_bar=(key == 'SELD_score'))
        # Alias without slash — safe for ModelCheckpoint filename and EarlyStopping
        self.log('val_seld_score', float(results['SELD_score']),
                 on_epoch=True, sync_dist=True)

        # ── Semantic metrics ─────────────────────────────────────────────────
        sem = self.sem_metrics.compute(include_confusion_matrix=False)
        self.sem_metrics.reset()
        for key, val in sem.items():
            self.log(f'val/{key}', float(val), on_epoch=True, sync_dist=True)

        # ── JSON log ─────────────────────────────────────────────────────────
        if self._metrics_path is not None:
            epoch_dict: dict = {'epoch': self.current_epoch}
            epoch_dict.update({f'val/{k}': float(v) for k, v in results.items()})
            epoch_dict.update({f'val/{k}': float(v) for k, v in sem.items()})
            self._epoch_metrics_log.append(epoch_dict)
            self._flush_metrics_log()

            if self._perclass_metrics:
                self._flush_perclass_metrics(
                    split='val', epoch=self.current_epoch,
                    rows=self.metrics.compute_perclass(),
                )

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Accumulate SELD and semantic metrics on the test split.

        Parameters
        ----------
        batch : tuple of (x, tgt_spatial)
        batch_idx : int
        """
        x, tgt_spatial = batch
        act_logits, doa_tanh, out_e = self.model(x, orig_sr=self.orig_sr)
        # T_mel alignment (see training_step)
        T = min(act_logits.shape[1], tgt_spatial.shape[1])
        act_logits  = act_logits[:, :T]
        doa_tanh    = doa_tanh[:, :T]
        tgt_spatial = tgt_spatial[:, :T]
        self.metrics.update(act_logits, doa_tanh, tgt_spatial)
        self.sem_metrics.update(out_e, tgt_spatial)

    def on_test_epoch_end(self) -> None:
        """Flush all test metrics; log scalars to TensorBoard, full results to JSON."""
        seld = self.metrics.compute()
        self.metrics.reset()
        sem  = self.sem_metrics.compute(include_confusion_matrix=True)
        self.sem_metrics.reset()

        # Pop confusion matrix before logging / JSON (saved to a separate file)
        conf_mat = sem.pop('sem_confusion_matrix', None)

        for key, val in seld.items():
            self.log(f'test/{key}', float(val), on_epoch=True, sync_dist=True)
        for key, val in sem.items():
            self.log(f'test/{key}', float(val), on_epoch=True, sync_dist=True)

        if self._metrics_path is not None:
            test_dict: dict = {'epoch': 'test'}
            test_dict.update({f'test/{k}': v for k, v in seld.items()})
            test_dict.update({f'test/{k}': v for k, v in sem.items()})
            self._epoch_metrics_log.append(test_dict)
            self._flush_metrics_log()

            if self._perclass_metrics:
                self._flush_perclass_metrics(
                    split='test', epoch='test',
                    rows=self.metrics.compute_perclass(),
                )

            if conf_mat is not None:
                cm_path = self._metrics_path.with_name(
                    self._metrics_path.stem + '_confusion_matrix.json'
                )
                with open(cm_path, 'w') as fh:
                    json.dump({'sem_confusion_matrix': conf_mat}, fh, indent=2)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        """Return Adam optimizer with CosineAnnealingLR scheduler.

        Learning rate is read from ``cfg['training']['lr']``.
        The scheduler decays LR from ``lr`` to ``lr * 0.1`` over ``max_epochs``
        following a cosine curve.

        Returns
        -------
        dict with 'optimizer' and 'lr_scheduler' keys (Lightning format).
        """
        lr        = float(self.hparams['cfg'].get('training', {}).get('lr', 5e-4))
        params    = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max   = self.trainer.max_epochs,
            eta_min = lr * 0.1,
        )
        return {
            'optimizer':    optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _flush_metrics_log(self) -> None:
        """Atomically write ``_epoch_metrics_log`` to ``_metrics_path`` as JSON."""
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._metrics_path.with_suffix('.tmp')
        with open(tmp, 'w') as fh:
            json.dump(self._epoch_metrics_log, fh, indent=2, default=float)
        tmp.replace(self._metrics_path)

    def _flush_perclass_metrics(self, split: str, epoch, rows: list) -> None:
        """Append per-class SED metrics to ``perclass_sed_metrics.json``.

        File format: list of dicts, each with ``epoch``, ``split``, and
        per-class keys (``class_idx``, ``tp``, ``fp``, ``fn``,
        ``precision``, ``recall``, ``f1``).
        """
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
        """Return all parameters with ``requires_grad=True``.

        Returns
        -------
        list of torch.nn.Parameter
        """
        return [p for p in self.model.parameters() if p.requires_grad]
