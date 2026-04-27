"""
SpatialLightningModule — PyTorch Lightning wrapper for SpatialSRPModel
=======================================================================
Trains the standalone spatial SELD model (NeuralSRPEncoder + DOA/activity/
distance heads) on STARSS23 MIC format audio.

Checkpoint monitor : ``val_lf`` (Localization F-score %, mode=max).

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

import json
from pathlib import Path
from typing import Optional

import torch
import lightning as L

from spatial_modules.neural_srp_modules import SpatialSRPModel
from experiments_spatial.loss    import SpatialPITLoss, SpatialHNetLoss
from experiments_spatial.metrics import SpatialMetrics


def _infer_mode(metric: str) -> str:
    """Infer checkpoint/early-stopping mode from metric name."""
    return 'min' if any(k in metric for k in ('loss', 'error', '_deg', 'mae')) else 'max'


class SpatialLightningModule(L.LightningModule):
    """
    Args:
        cfg : merged config dict (loaded from experiment YAML + CLI overrides).
              Expected top-level sections: ``encoder``, ``heads``, ``training``.

    Config keys used
    ----------------
    ``encoder.*``            : kwargs forwarded to ``NeuralSRPEncoder``
                               (``mic_positions`` key is consumed here, not forwarded)
    ``encoder.mic_positions``: ``(n_mics, 3)`` list-of-lists in metres —
                               registered as a non-trainable buffer.
    ``heads.n_tracks``       : number of simultaneous source tracks.
    ``training.loss_type``   : ``'pit'`` (default) or ``'hnet'``.
    ``training.hnet_ckpt``   : path to HNetGRU checkpoint (required for ``hnet``).
    ``training.lambda_act``  : activity loss weight.
    ``training.lambda_doa``  : DOA loss weight.
    ``training.lambda_dist`` : distance loss weight.
    ``training.lr``          : Adam learning rate.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        t_cfg = cfg.get('training', {})

        # ── Encoder config (pop mic_positions before forwarding) ─────────────
        encoder_cfg   = dict(cfg['encoder'])
        mic_pos_list  = encoder_cfg.pop('mic_positions', None)
        n_tracks      = int(cfg['heads']['n_tracks'])

        self.model = SpatialSRPModel(encoder_cfg, n_tracks=n_tracks)

        # ── Fixed mic-position buffer ────────────────────────────────────────
        if mic_pos_list is not None:
            mic_pos = torch.tensor(mic_pos_list, dtype=torch.float32)  # (n_mics, 3)
            self.register_buffer('mic_pos', mic_pos)
        else:
            self.mic_pos: Optional[torch.Tensor] = None

        # ── Loss ────────────────────────────────────────────────────────────
        loss_type   = str(t_cfg.get('loss_type', 'pit')).lower()
        lambda_act  = float(t_cfg.get('lambda_act',  1.0))
        lambda_doa  = float(t_cfg.get('lambda_doa',  1.0))
        lambda_dist = float(t_cfg.get('lambda_dist', 0.5))

        if loss_type == 'hnet':
            self.loss_fn = SpatialHNetLoss(
                hnet_ckpt   = str(t_cfg['hnet_ckpt']),
                lambda_act  = lambda_act,
                lambda_doa  = lambda_doa,
                lambda_dist = lambda_dist,
                dmota_wt    = float(t_cfg.get('dMOTA_wt', 1.0)),
                dmotp_wt    = float(t_cfg.get('dMOTP_wt', 1.0)),
                ids_wt      = float(t_cfg.get('IDS_wt',   1.0)),
                n_tracks    = n_tracks,
            )
        else:
            self.loss_fn = SpatialPITLoss(
                lambda_act  = lambda_act,
                lambda_doa  = lambda_doa,
                lambda_dist = lambda_dist,
            )

        self.lr = float(t_cfg.get('lr', 1e-4))

        # ── Metrics ─────────────────────────────────────────────────────────
        self._val_metrics  = SpatialMetrics()
        self._test_metrics = SpatialMetrics()

        # ── Metrics JSON log ────────────────────────────────────────────────
        self._metrics_path:      Optional[Path] = None
        self._epoch_metrics_log: list           = []

    # ── Helpers ──────────────────────────────────────────────────────────────

    def set_metrics_path(self, path: Path) -> None:
        self._metrics_path = Path(path)

    def _expand_mic_pos(self, B: int) -> torch.Tensor:
        if self.mic_pos is None:
            raise RuntimeError(
                'mic_pos not configured — add encoder.mic_positions to the YAML.'
            )
        return self.mic_pos.unsqueeze(0).expand(B, -1, -1)   # (B, n_mics, 3)

    def _forward_step(self, batch):
        wav, _accdoa, doa_t, act_t, dist_t = batch
        B = wav.shape[0]
        pred = self.model(wav, self._expand_mic_pos(B))

        # Temporal alignment: encoder may produce ±1 frame vs. target
        T = min(pred['doa'].shape[1], doa_t.shape[1])
        pred_trunc = {
            'doa':      pred['doa'][:, :T],
            'activity': pred['activity'][:, :T],
            'distance': pred['distance'][:, :T],
        }
        targets = {
            'doa':      doa_t[:, :T],
            'activity': act_t[:, :T],
            'distance': dist_t[:, :T],
        }
        return pred_trunc, targets

    # ── Lightning hooks ───────────────────────────────────────────────────────

    def training_step(self, batch, _batch_idx):
        pred, targets = self._forward_step(batch)
        loss, parts   = self.loss_fn(pred, targets)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        for k, v in parts.items():
            if k != 'loss':
                self.log(f'train_{k}', v, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, _batch_idx):
        pred, targets = self._forward_step(batch)
        loss, _       = self.loss_fn(pred, targets)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)

        self._val_metrics.update(
            pred['doa'], pred['activity'], pred['distance'],
            targets['doa'], targets['activity'], targets['distance'],
        )

    def on_validation_epoch_end(self) -> None:
        results = self._val_metrics.get_results(reset=True)
        for k, v in results.items():
            self.log(k, v,
                     prog_bar=(k in ('val_lf', 'val_le_deg')),
                     sync_dist=True)

        entry = {'epoch': self.current_epoch, **results}
        self._epoch_metrics_log.append(entry)
        self._flush_metrics()

        opt = self.optimizers()
        if opt is not None:
            self.log('train/lr', opt.param_groups[0]['lr'], on_epoch=True, sync_dist=True)

    def test_step(self, batch, _batch_idx):
        pred, targets = self._forward_step(batch)
        loss, _       = self.loss_fn(pred, targets)
        self.log('test_loss', loss, sync_dist=True)

        self._test_metrics.update(
            pred['doa'], pred['activity'], pred['distance'],
            targets['doa'], targets['activity'], targets['distance'],
        )

    def on_test_epoch_end(self) -> None:
        results = self._test_metrics.get_results(reset=True)
        test_results = {k.replace('val_', 'test_'): v for k, v in results.items()}
        for k, v in test_results.items():
            self.log(k, v, sync_dist=True)
        for k, v in test_results.items():
            print(f'  {k}: {v:.4f}')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    # ── Private ───────────────────────────────────────────────────────────────

    def _flush_metrics(self) -> None:
        if self._metrics_path is None:
            return
        tmp = self._metrics_path.with_suffix('.tmp')
        with open(tmp, 'w') as fh:
            json.dump(self._epoch_metrics_log, fh, indent=2, default=float)
        tmp.replace(self._metrics_path)
