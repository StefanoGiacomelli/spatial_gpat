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
    'STARSS23':       24000,
    'TAU-NIGENS2021': 24000,
    'MixedSELD':      24000,
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
            elif dataset_mode == 'MixedSELD':
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
        except Exception as exc:
            print(f'[StandaloneSELDModule] pos_weight unavailable: {exc}')

        beta_sed    = float(train_cfg.get('beta_sed', 0.6))
        focal_gamma = float(train_cfg.get('focal_gamma', 0.0))
        self.tpit_loss = tPITLoss(
            N           = N,
            beta        = beta_sed,
            pos_weight  = pos_weight,
            focal_gamma = focal_gamma,
        )

        self.metrics_val    = SELDMetrics(angle_thresh=angle_thresh)
        self.metrics_starss = SELDMetrics(angle_thresh=angle_thresh)
        self.metrics_tau    = SELDMetrics(angle_thresh=angle_thresh)

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
    # Test — two dataloaders (STARSS23 idx=0, TAU-NIGENS idx=1)
    # ------------------------------------------------------------------

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        x, tgt = batch

        # TAU-NIGENS targets have C=12; pad to 13 so model output matches
        if tgt.shape[3] == 12:
            tgt = F.pad(tgt, (0, 0, 0, 1))   # (B, T, N, 12, 3) → (B, T, N, 13, 3)

        act_logits, doa_tanh = self.model(x, orig_sr=self.orig_sr)

        T = min(act_logits.shape[1], tgt.shape[1])
        act_logits = act_logits[:, :T]
        doa_tanh   = doa_tanh[:, :T]
        tgt        = tgt[:, :T]

        if dataloader_idx == 0:
            self.metrics_starss.update(act_logits, doa_tanh, tgt)
        else:
            self.metrics_tau.update(act_logits, doa_tanh, tgt)

    def on_test_epoch_end(self) -> None:
        starss = self.metrics_starss.compute()
        self.metrics_starss.reset()
        tau    = self.metrics_tau.compute()
        self.metrics_tau.reset()

        for key, val in starss.items():
            self.log(f'test_starss/{key}', float(val), on_epoch=True, sync_dist=True)
        for key, val in tau.items():
            self.log(f'test_tau/{key}', float(val), on_epoch=True, sync_dist=True)

        if self._metrics_path is not None:
            test_dict: dict = {'epoch': 'test'}
            test_dict.update({f'test_starss/{k}': float(v) for k, v in starss.items()})
            test_dict.update({f'test_tau/{k}': float(v) for k, v in tau.items()})
            self._epoch_metrics_log.append(test_dict)
            self._flush_metrics_log()

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        """Adam + linear warmup + cosine annealing."""
        train_cfg     = self.hparams['cfg'].get('training', {})
        lr            = float(train_cfg.get('lr', 1e-3))
        lr_min        = float(train_cfg.get('lr_min', 1e-6))
        warmup_epochs = int(  train_cfg.get('lr_warmup_epochs', 5))
        total_epochs  = self.trainer.max_epochs

        params    = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)

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
