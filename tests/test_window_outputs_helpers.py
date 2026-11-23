"""Helpers of the window cache: window offsets and target reconstruction (no model, no audio)."""
from __future__ import annotations

import numpy as np
import pytest

from analysis.diagnostics.window_outputs import window_start_frame, window_target
from conftest import ROOT, requires_data
from test_dataset_protocols import ToyDataset


def make_toy():
    ds = ToyDataset(split="test", orig_sr=24000, model_sr=32000, hop_length=320, n_tracks=3,
                    window_s=10.0, hop_s=2.5)
    ds._events = [[(5, 1, 0, 30.0, 10.0), (60, 2, 3, -90.0, 0.0)]]
    ds._items = [(0, k * 60000, False) for k in range(4)]
    return ds


def test_window_start_frames_are_multiples_of_250():
    ds = make_toy()
    assert [window_start_frame(ds, i) for i in range(4)] == [0, 250, 500, 750]


def test_window_target_matches_build_target_and_places_events():
    ds = make_toy()
    t = window_target(ds, 1).numpy()                 # window of frames 250..1250 -> csv frames 25..125
    assert t.shape == (1001, 3, 4, 3)
    # event of csv frame 60 (class 2) appears 350 model frames after the window start (+/- the bin artefact)
    active_frames = np.nonzero(np.linalg.norm(t[:, 0, 2], axis=-1) > 0.5)[0]
    assert active_frames.min() in (349, 350) and active_frames.max() in (358, 359, 360)
    assert np.linalg.norm(t[:, :, 1], axis=-1).max() == 0       # class-1 event (frame 5) lies before the window


@requires_data("STARSS23", "foa_dev", "dev-test-sony")
def test_window_target_equals_dataset_item():
    from datasets.starss23 import STARSS23Dataset
    ds = STARSS23Dataset(root=ROOT / "data" / "STARSS23", split="test", model_sr=32000, hop_length=320,
                         n_tracks=3, window_s=10.0, hop_s=2.5, val_fraction=0.2, val_seed=42)
    for i in (0, 123, 2500):
        assert (ds[i][1] == window_target(ds, i)).all()
