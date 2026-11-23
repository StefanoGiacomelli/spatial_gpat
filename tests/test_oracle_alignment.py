"""Data-free regression test of the slot alignment in ``analysis/diagnostics/doa_oracle_test.py``.

The standalone Conformer/ResNet baselines are trained with ``tPITLoss`` (their slot order is arbitrary with
respect to the target slots).  ``doa_oracle_test.py`` historically aligned the DOA slots only for NAS-SELD models,
so the stored oracle logs of the standalone baselines are raw-slot statistics.  ``--align-standalone`` passes the
checkpoint's ``tPITLoss`` to ``run_inference``; this test uses a synthetic "perfect but permuted" model to check that

* without a ``tpit_loss`` (historical behaviour) the permuted slots give a poor oracle accuracy;
* with a ``tpit_loss`` the oracle accuracy is 1.0 (the permutation is undone).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from analysis.diagnostics.doa_oracle_test import run_inference
from experiments_nas_seld.loss import tPITLoss
from utils.doa_metrics import angular_error_deg

B, T, N, C = 1, 20, 3, 2


def unit(az_deg: float, el_deg: float = 0.0) -> np.ndarray:
    a, e = math.radians(az_deg), math.radians(el_deg)
    return np.array([math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e)], dtype=np.float32)


def make_target() -> torch.Tensor:
    t = np.zeros((B, T, N, C, 3), np.float32)
    t[0, 0:10, 0, 0] = unit(0.0)          # class 0, slot 0
    t[0, 5:15, 1, 0] = unit(90.0)         # class 0, slot 1 (overlaps the first source for frames 5-9)
    t[0, 10:20, 0, 1] = unit(200.0)       # class 1, slot 0
    return torch.from_numpy(t)


class PermutedPerfectModel(torch.nn.Module):
    """Returns the ground truth with the track slots permuted: prediction slot j holds target slot ``perm[j]``."""

    def __init__(self, target: torch.Tensor, perm=(1, 2, 0)):
        super().__init__()
        self.target, self.perm = target, list(perm)

    def forward(self, x, orig_sr=None):
        tgt = self.target[:, :, self.perm]                                   # (B, T, N, C, 3)
        active = (tgt.norm(dim=-1) > 0.5).float()                            # (B, T, N, C)
        return 12.0 * active - 6.0, tgt


def oracle_accuracy(tpit_loss):
    target = make_target()
    loader = [(torch.zeros(B, 4, 100), target)]
    doa_list, tgt_list, _, _ = run_inference(PermutedPerfectModel(target), loader, torch.device("cpu"), 24000, 320,
                                             False, "dummy", tpit_loss)
    pred, tgt = np.concatenate(doa_list), np.concatenate(tgt_list)
    active = np.linalg.norm(tgt, axis=-1) > 0.5
    err = angular_error_deg(pred[active], tgt[active])
    return float((err <= 20.0).mean()), int(active.sum())


def test_raw_slot_order_penalises_a_permuted_model():
    acc, n_active = oracle_accuracy(None)
    assert n_active == 30
    assert acc < 0.5


def test_tpit_alignment_undoes_the_slot_permutation():
    tpit = tPITLoss(N=N, beta=0.5, doa_loss_cfg={"mode": "active_mse"}).eval()
    acc, n_active = oracle_accuracy(tpit)
    assert n_active == 30
    assert acc == 1.0
