"""
StandaloneSELDModule — PyTorch Lightning Training Wrapper
=========================================================
Wraps ``StandaloneSELDModel`` with:

  * ``training_step``           — single tPITLoss (SED from sed_branch, DOA from doa_branch)
  * ``validation_step``         — tPIT losses + SELD metrics accumulation
  * ``on_validation_epoch_end`` — flush metrics, JSON log, LR log
  * ``test_step``               — two-dataloader test (STARSS23 idx=0, TAU idx=1)
  * ``on_test_epoch_end``       — flush both metric accumulators, JSON log
  * ``configure_optimizers``    — Adam + ReduceLROnPlateau (mode='min', val_seld_score)

Loss design
-----------
Single ``tPITLoss`` instance: ``act_logits`` from the SED branch and ``doa_tanh``
from the DOA branch are passed together.  The optimal permutation is found from the
combined β·BCE + (1-β)·MSE cost, then SED gradient flows to ``sed_branch`` only
and DOA gradient flows to ``doa_branch`` only.

Batch convention
----------------
.. code-block::

    batch = (x, tgt_spatial)
    x           : (B, n_mics, T_samples)  raw waveform at dataset SR
    tgt_spatial : (B, T_mel, N, C, 3)    Multi-ACCDOA at mel resolution

Test dataloaders
----------------
``test_step`` accepts ``dataloader_idx``:
  * 0 → STARSS23 test set   (metrics_starss)
  * 1 → TAU-NIGENS test set (metrics_tau)
                             TAU targets are zero-padded C: 12 → 13 in test_step.

TensorBoard logging
-------------------
Training (per step + per epoch):
  ``train/loss``, ``train/loss_tpit``, ``train/loss_sed``, ``train/loss_doa``, ``train/lr``

Validation (per epoch):
  ``val/loss_tpit``, ``val/loss_sed``, ``val/loss_doa``,
  ``val/LE_CD``, ``val/LR_CD``, ``val/ER_20``, ``val/F_20``, ``val/SELD_score``,
  ``val_seld_score`` (alias for ModelCheckpoint / EarlyStopping / scheduler)

Test (once per run):
  ``test_starss/{metric}``, ``test_tau/{metric}``

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import lightning as L

from .model import StandaloneSELDModel
from experiments_nas_seld.loss import tPITLoss, SELDMetrics


_ORIG_SR = {
    'STARSS23':          24000,
    'TAU-NIGENS2021':    24000,
    'TAU2019':           24000,   # TAU2019Dataset resamples to model_sr internally
    'BalancedSTARSS23':  24000,
}


class StandaloneSELDModule(L.LightningModule):
    """
    PyTorch Lightning wrapper for StandaloneSELDModel.

    Args:
        cfg (dict): merged YAML config.

    Logged metrics (val)
    --------------------
    val/loss_tpit, val/loss_sed, val/loss_doa,
    val/LE_CD, val/LR_CD, val/ER_20, val/F_20, val/SELD_score,
    val_seld_score (alias for monitor).

    Logged metrics (test)
    ---------------------
    test_starss/{metric}, test_tau/{metric}.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters({'cfg': cfg})

        train_cfg    = cfg.get('training', {})
        angle_thresh = float(train_cfg.get('angle_thresh', 20.0))
        N            = cfg.get('heads', {}).get('n_tracks', 2)
        n_classes    = cfg.get('heads', {}).get('n_classes', 13)
        dataset_mode = train_cfg.get('dataset_mode', 'STARSS23')

        self.orig_sr = _ORIG_SR.get(dataset_mode, 24000)
        self.model   = StandaloneSELDModel(cfg)

        # ── Per-class pos_weight for SED head ────────────────────────────────
        pos_weight = None
        try:
            if dataset_mode == 'STARSS23':
                root = cfg.get('data', {}).get('root_path') or cfg.get('data', {}).get('starss_root')
                if root:
                    from datasets.starss23 import compute_pos_weights
                    pos_weight = compute_pos_weights(Path(root), n_classes=n_classes)
            elif dataset_mode == 'TAU-NIGENS2021':
                root = cfg.get('data', {}).get('root_path') or cfg.get('data', {}).get('tau_root')
                if root:
                    from datasets.tau_nigens import compute_pos_weights
                    pos_weight = compute_pos_weights(Path(root), n_classes=12)
                    # zero-pad to 13 classes (Knock class absent in TAU)
                    pos_weight = F.pad(pos_weight, (0, 1), value=1.0)
            elif dataset_mode == 'TAU2019':
                root = cfg.get('data', {}).get('root_path')
                if root:
                    try:
                        from datasets.tau2019 import compute_pos_weights as _cpw_tau2019
                        pos_weight = _cpw_tau2019(Path(root), n_classes=n_classes)
                    except (ImportError, AttributeError):
                        pass  # compute_pos_weights not available for TAU2019 — skip
            elif dataset_mode == 'BalancedSTARSS23':
                root = cfg.get('data', {}).get('starss_root')
                if root:
                    from datasets.starss23 import compute_pos_weights
                    pos_weight = compute_pos_weights(Path(root), n_classes=n_classes)
            if pos_weight is not None:
                pw_cap = float(train_cfg.get('pos_weight_max', float('inf')))
                if pw_cap < float('inf'):
                    pos_weight = pos_weight.clamp(max=pw_cap)
                print(f'[StandaloneSELDModule] pos_weight  '
                      f'min={pos_weight.min():.2f}  max={pos_weight.max():.1f}  '
                      f'cap={pw_cap}')
        except (ImportError, AttributeError):
            pass   # compute_pos_weights not implemented for this dataset — skip silently
        except Exception as exc:
            print(f'[StandaloneSELDModule] pos_weight error: {exc}')

        beta_sed    = float(train_cfg.get('beta_sed', 0.6))
        focal_gamma = float(train_cfg.get('focal_gamma', 0.0))
        self.tpit_loss = tPITLoss(
            N           = N,
            beta        = beta_sed,
            pos_weight  = pos_weight,
            focal_gamma = focal_gamma,
        )

        self.metrics_val      = SELDMetrics(angle_thresh=angle_thresh)
        self.metrics_starss   = SELDMetrics(angle_thresh=angle_thresh)
        self.metrics_tau      = SELDMetrics(angle_thresh=angle_thresh)
        self.metrics_tau2019  = SELDMetrics(angle_thresh=angle_thresh)

        # JSON metrics log
        self._metrics_path: Optional[Path] = None
        self._epoch_metrics_log: list      = []

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def set_metrics_path(self, path: Path) -> None:
        self._metrics_path = Path(path)

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        x, tgt = batch
        act_logits, doa_tanh = self.model(x, orig_sr=self.orig_sr)

        T = min(act_logits.shape[1], tgt.shape[1])
        act_logits = act_logits[:, :T]
        doa_tanh   = doa_tanh[:, :T]
        tgt        = tgt[:, :T]

        stride = int(self.hparams['cfg'].get('training', {}).get('gt_stride', 1))
        if stride > 1:
            act_logits = act_logits[:, ::stride]
            doa_tanh   = doa_tanh[:, ::stride]
            tgt        = tgt[:, ::stride]

        loss, loss_sed, loss_doa = self.tpit_loss(act_logits, doa_tanh, tgt)

        self.log('train/loss',     loss,     on_step=True,  on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.log('train/loss_tpit', loss,    on_step=False, on_epoch=True, sync_dist=True)
        self.log('train/loss_sed', loss_sed, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train/loss_doa', loss_doa, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        x, tgt = batch
        act_logits, doa_tanh = self.model(x, orig_sr=self.orig_sr)

        T = min(act_logits.shape[1], tgt.shape[1])
        act_logits = act_logits[:, :T]
        doa_tanh   = doa_tanh[:, :T]
        tgt        = tgt[:, :T]

        loss, loss_sed, loss_doa = self.tpit_loss(act_logits, doa_tanh, tgt)

        self.log('val/loss_tpit', loss,     on_epoch=True, sync_dist=True)
        self.log('val/loss_sed',  loss_sed, on_epoch=True, sync_dist=True)
        self.log('val/loss_doa',  loss_doa, on_epoch=True, sync_dist=True)
        self.log('val_loss_tpit', loss,     on_epoch=True, sync_dist=True)

        self.metrics_val.update(act_logits, doa_tanh, tgt)

    def on_validation_epoch_end(self) -> None:
        results = self.metrics_val.compute()
        self.metrics_val.reset()

        for key, val in results.items():
            self.log(f'val/{key}', val, on_epoch=True, sync_dist=True,
                     prog_bar=(key == 'SELD_score'))
        self.log('val_seld_score', float(results['SELD_score']),
                 on_epoch=True, sync_dist=True)

        # LR log
        opt = self.optimizers()
        if opt is not None:
            self.log('train/lr', opt.param_groups[0]['lr'],
                     on_epoch=True, sync_dist=True)

        if self._metrics_path is not None:
            epoch_dict: dict = {'epoch': self.current_epoch}
            epoch_dict.update({f'val/{k}': float(v) for k, v in results.items()})
            self._epoch_metrics_log.append(epoch_dict)
            self._flush_metrics_log()

    # ------------------------------------------------------------------
    # Test — 1 or 3 dataloaders (single dataset or BalancedSTARSS23)
    # ------------------------------------------------------------------

    # Maps dataset_mode → log_prefix (None = multi-loader, prefix from dataloader_idx)
    _TEST_PREFIX = {
        'STARSS23':          'test_starss',
        'TAU-NIGENS2021':    'test_tau',
        'TAU2019':           'test_tau2019',
        'BalancedSTARSS23':  None,   # 3 DataLoaders: idx=0/1/2 → starss/tau/tau2019
    }

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        x, tgt = batch
        dataset_mode = self.hparams['cfg']['training'].get('dataset_mode', 'STARSS23')

        act_logits, doa_tanh = self.model(x, orig_sr=self.orig_sr)

        # Pad class dim when target has fewer classes than model output
        if tgt.shape[3] < act_logits.shape[3]:
            pad = act_logits.shape[3] - tgt.shape[3]
            tgt = F.pad(tgt, (0, 0, 0, pad))

        T = min(act_logits.shape[1], tgt.shape[1])
        act_logits = act_logits[:, :T]
        doa_tanh   = doa_tanh[:, :T]
        tgt        = tgt[:, :T]

        if dataset_mode == 'BalancedSTARSS23':
            # Three DataLoaders: idx=0 → STARSS23, idx=1 → TAU-NIGENS, idx=2 → TAU2019
            if dataloader_idx == 0:
                self.metrics_starss.update(act_logits, doa_tanh, tgt)
            elif dataloader_idx == 1:
                self.metrics_tau.update(act_logits, doa_tanh, tgt)
            else:
                self.metrics_tau2019.update(act_logits, doa_tanh, tgt)
        else:
            # Single DataLoader — always accumulate into metrics_starss
            self.metrics_starss.update(act_logits, doa_tanh, tgt)

    def on_test_epoch_end(self) -> None:
        dataset_mode = self.hparams['cfg']['training'].get('dataset_mode', 'STARSS23')
        prefix = self._TEST_PREFIX.get(dataset_mode, 'test_starss')

        if dataset_mode == 'BalancedSTARSS23':
            starss   = self.metrics_starss.compute();   self.metrics_starss.reset()
            tau      = self.metrics_tau.compute();      self.metrics_tau.reset()
            tau2019  = self.metrics_tau2019.compute();  self.metrics_tau2019.reset()

            for key, val in starss.items():
                self.log(f'test_starss/{key}',   float(val), on_epoch=True, sync_dist=True)
            for key, val in tau.items():
                self.log(f'test_tau/{key}',      float(val), on_epoch=True, sync_dist=True)
            for key, val in tau2019.items():
                self.log(f'test_tau2019/{key}',  float(val), on_epoch=True, sync_dist=True)

            if self._metrics_path is not None:
                test_dict: dict = {'epoch': 'test'}
                test_dict.update({f'test_starss/{k}':  float(v) for k, v in starss.items()})
                test_dict.update({f'test_tau/{k}':     float(v) for k, v in tau.items()})
                test_dict.update({f'test_tau2019/{k}': float(v) for k, v in tau2019.items()})
                self._epoch_metrics_log.append(test_dict)
                self._flush_metrics_log()
        else:
            results = self.metrics_starss.compute()
            self.metrics_starss.reset()

            for key, val in results.items():
                self.log(f'{prefix}/{key}', float(val), on_epoch=True, sync_dist=True)

            if self._metrics_path is not None:
                test_dict = {'epoch': 'test'}
                test_dict.update({f'{prefix}/{k}': float(v) for k, v in results.items()})
                self._epoch_metrics_log.append(test_dict)
                self._flush_metrics_log()

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        """AdamW + cosine (default) or ReduceLROnPlateau, selected via training.scheduler."""
        train_cfg     = self.hparams['cfg'].get('training', {})
        lr            = float(train_cfg.get('lr', 1e-3))
        lr_min        = float(train_cfg.get('lr_min', 1e-6))
        scheduler_type = train_cfg.get('scheduler', 'cosine')

        params    = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)

        if scheduler_type == 'reduce_on_plateau':
            factor      = float(train_cfg.get('lr_reduce_factor', 0.75))
            lr_patience = int(  train_cfg.get('lr_patience',       20))
            scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode     = 'min',
                factor   = factor,
                patience = lr_patience,
                min_lr   = lr_min,
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
        else:
            # cosine (default) — linear warmup then cosine annealing
            warmup_epochs = int(train_cfg.get('lr_warmup_epochs', 5))
            total_epochs  = self.trainer.max_epochs
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor = 1e-3,
                end_factor   = 1.0,
                total_iters  = max(warmup_epochs, 1),
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max   = max(total_epochs - warmup_epochs, 1),
                eta_min = lr_min,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers = [warmup, cosine],
                milestones = [warmup_epochs],
            )
            return {
                'optimizer':    optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval':  'epoch',
                    'frequency': 1,
                },
            }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _flush_metrics_log(self) -> None:
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._metrics_path.with_suffix('.tmp')
        with open(tmp, 'w') as fh:
            json.dump(self._epoch_metrics_log, fh, indent=2, default=float)
        tmp.replace(self._metrics_path)

    def trainable_params(self):
        return [p for p in self.model.parameters() if p.requires_grad]
