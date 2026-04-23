"""
SpatialDataModule — Lightning DataModule for the standalone spatial SELD model
===============================================================================
Wraps ``STARSS23WithDistanceDataset`` (MIC format, with distance annotations)
for training, validation, and evaluation of ``SpatialSRPModel``.

Each batch is a 5-tuple:
    wav        : ``(B, n_mics, T_samples)``              float32
    accdoa     : ``(B, T_mel, N_tracks, C_classes, 3)``  Multi-ACCDOA (unused by spatial model)
    doa_target : ``(B, T_mel, N_tracks, 3)``             class-agnostic DOA targets
    act_target : ``(B, T_mel, N_tracks)``                binary activity
    dist_target: ``(B, T_mel, N_tracks)``                distance in cm

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader

from datasets.starss23 import STARSS23WithDistanceDataset


class SpatialDataModule(L.LightningDataModule):
    """
    Args:
        cfg : merged config dict.

    Config keys used (under ``data.`` and ``training.``)
    -----------------------------------------------------
    ``data.root``        : path to STARSS23 dataset root.
    ``data.sr``          : model sample rate (must equal STARSS23.ORIG_SR = 24000).
    ``data.hop_length``  : hop length at ``data.sr`` (must equal encoder hop_size).
    ``data.window_s``    : window duration in seconds (None = full clip).
    ``data.hop_s``       : window stride in seconds (None = non-overlapping).
    ``data.num_workers`` : DataLoader worker processes (default 4).
    ``data.val_fraction``: fraction of dev-train reserved for val (default 0.2).
    ``data.val_seed``    : RNG seed for reproducible train/val split (default 42).
    ``heads.n_tracks``   : number of source tracks.
    ``training.batch_size``: batch size (default 4).
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        d_cfg = cfg['data']

        self.root         = Path(d_cfg['root'])
        self.model_sr     = int(d_cfg['sr'])
        self.hop_length   = int(d_cfg['hop_length'])
        self.n_tracks     = int(cfg['heads']['n_tracks'])
        self.window_s     = d_cfg.get('window_s')
        self.hop_s        = d_cfg.get('hop_s')
        self.val_fraction = float(d_cfg.get('val_fraction', 0.2))
        self.val_seed     = int(d_cfg.get('val_seed', 42))
        self.batch_size   = int(cfg.get('training', {}).get('batch_size', 4))
        self.num_workers  = int(d_cfg.get('num_workers', 4))

        self.train_ds = None
        self.val_ds   = None
        self.test_ds  = None

    def setup(self, stage: str = None) -> None:
        common = dict(
            root         = self.root,
            model_sr     = self.model_sr,
            hop_length   = self.hop_length,
            n_tracks     = self.n_tracks,
            window_s     = self.window_s,
            hop_s        = self.hop_s,
            audio_format = 'mic',    # NeuralSRP needs raw MIC signals
            augment      = None,     # FOA-specific augmentations don't apply here
            val_fraction = self.val_fraction,
            val_seed     = self.val_seed,
        )
        if stage in (None, 'fit'):
            self.train_ds = STARSS23WithDistanceDataset(split='train', **common)
            self.val_ds   = STARSS23WithDistanceDataset(split='val',   **common)
        if stage in (None, 'test'):
            self.test_ds  = STARSS23WithDistanceDataset(split='test',  **common)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size  = self.batch_size,
            shuffle     = True,
            num_workers = self.num_workers,
            pin_memory  = True,
            drop_last   = True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size  = self.batch_size,
            shuffle     = False,
            num_workers = self.num_workers,
            pin_memory  = True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size  = self.batch_size,
            shuffle     = False,
            num_workers = self.num_workers,
        )
