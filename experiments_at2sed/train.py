"""
AT2SEDLightningModule — PyTorch Lightning Training Wrapper for AT2SEDModel
==========================================================================
Wraps ``AT2SEDModel`` for multi-label SED fine-tuning on AudioSet-Strong.

  * ``training_step``           — BCEWithLogitsLoss (with pos_weight)
  * ``validation_step``         — loss + metric accumulation
  * ``on_validation_epoch_end`` — flush metrics, TensorBoard log, JSON append
  * ``test_step``               — metric accumulation on test split
  * ``on_test_epoch_end``       — flush metrics, JSON append
  * ``configure_optimizers``    — Adam with fixed learning rate (no scheduler)
  * ``set_metrics_path(path)``  — inject JSON log path after construction
  * ``set_pos_weight(pw)``      — inject class-imbalance weights after DataModule setup

Batch convention::

    batch = (x, y)
    x : (B, T_samples)  mono waveform at 32 kHz
    y : (B, C)          float32 multi-hot label tensor

TensorBoard logging
-------------------
Training (per step + per epoch):
  ``train/loss``

Validation (per epoch):
  ``val/loss``, ``val/map_macro``, ``val/auroc_macro``, ``val/f1_macro``,
  ``val/f1_micro``, ``val/precision_macro``, ``val/recall_macro``

Test (once, after ``trainer.test()``):
  ``test/map_macro``, ``test/auroc_macro``, …  (same metric set)

Checkpoint monitor: ``val/map_macro``  (higher is better).

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import lightning as L

from .model import AT2SEDModel
from .loss  import SEDMultiLabelLoss, SEDMultiLabelMetrics


class AT2SEDLightningModule(L.LightningModule):
    """
    Lightning wrapper for AudioSet-Strong SED fine-tuning.

    Args:
        cfg (dict): merged config dict (experiment YAML + dataset YAML + CLI).

    Logged metrics (TensorBoard)
    ----------------------------
    Training (per step + per epoch):
      train/loss         — BCE loss

    Validation (per epoch):
      val/loss           — BCE loss
      val/map_macro      — mean Average Precision, macro (PRIMARY — used for checkpointing)
      val/auroc_macro    — AUROC, macro
      val/f1_macro       — F1, macro
      val/f1_micro       — F1, micro
      val/precision_macro
      val/recall_macro

    Test (once per run):
      test/{metric}      — same metric set as validation
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters({'cfg': cfg})

        self.model    = AT2SEDModel(cfg)
        self.loss_fn  = SEDMultiLabelLoss(pos_weight=None)   # weights set later
        self.metrics  = SEDMultiLabelMetrics(
            n_cls     = self.model.n_cls,
            threshold = float(cfg.get('training', {}).get('inference_threshold', 0.5)),
        )

        # JSON metrics log — path injected via set_metrics_path() after construction
        self._metrics_path: Optional[Path] = None
        self._epoch_metrics_log: list      = []

    # ── Public helpers ────────────────────────────────────────────────────────

    def set_metrics_path(self, path: Path) -> None:
        """
        Inject the JSON log path after construction.

        Called by ``main_sed_only.py`` so that per-epoch and test metrics are
        persisted to ``{run_dir}/metrics/{exp_name}.json``.
        """
        self._metrics_path = Path(path)

    def set_pos_weight(self, pos_weight: torch.Tensor) -> None:
        """
        Inject class-imbalance weights computed from the training dataset.

        Called by ``main_sed_only.py`` after ``DataModule.setup('fit')``
        has computed the per-class statistics.

        Args:
            pos_weight : ``(n_cls,)`` float32 Tensor.
        """
        self.loss_fn.set_pos_weight(pos_weight)
        pw_min = pos_weight.min().item()
        pw_max = pos_weight.max().item()
        print(
            f'[AT2SEDLightningModule] pos_weight injected  '
            f'min={pw_min:.2f}  max={pw_max:.1f}'
        )

    # ── Training ──────────────────────────────────────────────────────────────

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Compute BCE loss for one training batch.

        Parameters
        ----------
        batch : (x, y)
            x : ``(B, T_samples)``
            y : ``(B, C)`` float32 multi-hot labels
        batch_idx : int (unused, required by Lightning)

        Returns
        -------
        torch.Tensor — scalar loss

        Logged keys
        -----------
        ``train/loss`` (per step + per epoch)
        """
        x, y     = batch
        logits   = self.model(x)          # (B, n_cls)
        loss     = self.loss_fn(logits, y)

        self.log('train/loss', loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    # ── Validation ────────────────────────────────────────────────────────────

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """
        Accumulate validation loss and metrics for one batch.

        Logged keys (on_epoch=True): ``val/loss``.
        Metric accumulation flushed in ``on_validation_epoch_end``.
        """
        x, y   = batch
        logits = self.model(x)
        loss   = self.loss_fn(logits, y)

        self.log('val/loss', loss, on_epoch=True, sync_dist=True)
        self.metrics.update(logits, y)

    def on_validation_epoch_end(self) -> None:
        """
        Flush metric accumulators, log all val metrics, append to JSON log.
        """
        results = self.metrics.compute()   # no confusion matrix on val (too large)
        self.metrics.reset()

        for key, val in results.items():
            self.log(f'val/{key}', float(val), on_epoch=True, sync_dist=True,
                     prog_bar=(key == 'map_macro'))

        # Alias without slash for ModelCheckpoint monitor
        self.log('val_map_macro', float(results['map_macro']),
                 on_epoch=True, sync_dist=True)

        # ── JSON log ─────────────────────────────────────────────────────────
        if self._metrics_path is not None:
            epoch_dict: dict = {'epoch': self.current_epoch}
            epoch_dict.update({f'val/{k}': float(v) for k, v in results.items()})
            self._epoch_metrics_log.append(epoch_dict)
            self._flush_metrics_log()

    # ── Test ─────────────────────────────────────────────────────────────────

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Accumulate test metrics for one batch."""
        x, y   = batch
        logits = self.model(x)
        self.metrics.update(logits, y)

    def on_test_epoch_end(self) -> None:
        """Flush test metrics, log to TensorBoard and append to JSON log.

        The confusion matrix (``n_cls × 2 × 2``) is saved as a separate
        ``confusion_matrix.npy`` file alongside ``metrics.json``; it is not
        logged to TensorBoard (too large).
        """
        import numpy as np
        results = self.metrics.compute(include_cm=True)
        self.metrics.reset()

        cm = results.pop('confusion_matrix', None)   # remove before TensorBoard log

        for key, val in results.items():
            self.log(f'test/{key}', float(val), on_epoch=True, sync_dist=True)

        if self._metrics_path is not None:
            test_dict: dict = {'epoch': 'test'}
            test_dict.update({f'test/{k}': float(v) for k, v in results.items()})
            self._epoch_metrics_log.append(test_dict)
            self._flush_metrics_log()

            # Save confusion matrix as .npy alongside metrics.json
            if cm is not None:
                cm_path = self._metrics_path.with_name('confusion_matrix.npy')
                np.save(str(cm_path), cm)
                print(f'[AT2SEDLightningModule] Confusion matrix saved → {cm_path.name}')

    # ── Optimizer ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        """
        Adam optimizer + CosineAnnealingLR scheduler.

        LR anneals from ``cfg['training']['lr']`` (default 1e-3) down to
        ``cfg['training']['lr_min']`` (default 1e-5) over the full training
        budget (``cfg['training']['epochs']``).

        Returns
        -------
        dict with ``optimizer`` and ``lr_scheduler`` keys (Lightning format).
        """
        train_cfg = self.hparams['cfg'].get('training', {})
        lr        = float(train_cfg.get('lr',     1e-3))
        lr_min    = float(train_cfg.get('lr_min', 1e-5))
        t_max     = int(train_cfg.get('epochs',   30))

        params    = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=lr_min,
        )
        return {
            'optimizer':    optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval':  'epoch',
                'frequency': 1,
                'name':      'lr/cosine',
            },
        }

    # ── Trainable params helper ───────────────────────────────────────────────

    def trainable_params(self) -> List[torch.nn.Parameter]:
        """Return all parameters with ``requires_grad=True``."""
        return [p for p in self.model.parameters() if p.requires_grad]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _flush_metrics_log(self) -> None:
        """Atomically write ``_epoch_metrics_log`` to ``_metrics_path`` as JSON."""
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._metrics_path.with_suffix('.tmp')
        with open(tmp, 'w') as fh:
            json.dump(self._epoch_metrics_log, fh, indent=2, default=float)
        tmp.replace(self._metrics_path)
