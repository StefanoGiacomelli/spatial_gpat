"""Audio-label consistency of the 16-pattern FOA augmentation (data-free).

Synthetic single-source ACN plane waves (the channel order of every corpus used
here: W, Y, Z, X, SN3D) are augmented with each of the 15 patterns; the DOA
implied by the augmented audio (time-domain intensity vector) must equal the
augmented label.  Findings documented in ``docs/CONVENTIONS.md``:

* with the legacy channel order (default, published behaviour) only pattern 11
  (full inversion) is consistent on ACN data -- the other 14 disagree by 90-180 deg;
* ``foa_channel_order='acn'`` makes all 15 patterns consistent;
* the default code path is bit-identical to the earlier (legacy) implementation.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from datasets.utils import SELDAugment

T_AUDIO, T_LAB, N_TR, N_CLS = 4800, 5, 3, 2


def label_xyz(az_deg: float, el_deg: float) -> np.ndarray:
    a, e = math.radians(az_deg), math.radians(el_deg)
    return np.array([math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e)])


def acn_plane_wave(az_deg: float, el_deg: float, rng) -> torch.Tensor:
    """SN3D/ACN (W, Y, Z, X) plane wave from DCASE-frame direction (az, el)."""
    a, e = math.radians(az_deg), math.radians(el_deg)
    x_d, y_d, z_d = math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e)
    s = torch.from_numpy(rng.normal(size=T_AUDIO).astype(np.float32))
    return torch.stack([s, y_d * s, z_d * s, x_d * s])


def audio_direction_acn(wav: torch.Tensor) -> np.ndarray:
    """DCASE-frame (X, Y, Z) direction implied by ACN-ordered audio (intensity vector)."""
    w = wav[0].double()
    iv = np.array([float((w * wav[3].double()).sum()),    # X = ch3
                   float((w * wav[1].double()).sum()),    # Y = ch1
                   float((w * wav[2].double()).sum())])   # Z = ch2
    return iv / np.linalg.norm(iv)


def label_direction_dcase(target: torch.Tensor) -> np.ndarray:
    """Convert the (x_l, y_l, z_l) label of the active slot to the DCASE frame."""
    v = target[0, 0, 0].double().numpy()
    return np.array([v[1], v[0], v[2]]) / np.linalg.norm(v)


def angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(np.dot(u, v), -1.0, 1.0))))


def sample_pair(rng):
    az, el = rng.uniform(-180, 180), math.degrees(math.asin(rng.uniform(-1, 1)))
    wav = acn_plane_wave(az, el, rng)
    tgt = torch.zeros(T_LAB, N_TR, N_CLS, 3)
    tgt[:, 0, 0] = torch.tensor(label_xyz(az, el), dtype=torch.float32)
    return wav, tgt


def max_disagreement(aug: SELDAugment, pattern: int, n: int = 12) -> float:
    rng = np.random.default_rng(100 + pattern)
    worst = 0.0
    for _ in range(n):
        wav, tgt = sample_pair(rng)
        w2, t2 = aug.apply_pattern(wav.clone(), tgt.clone(), pattern)
        worst = max(worst, angle_deg(audio_direction_acn(w2), label_direction_dcase(t2)))
    return worst


def test_identity_is_consistent():
    rng = np.random.default_rng(0)
    wav, tgt = sample_pair(rng)
    assert angle_deg(audio_direction_acn(wav), label_direction_dcase(tgt)) < 1e-3


def test_acn_order_makes_all_15_patterns_consistent():
    aug = SELDAugment(enabled=True, p_16patterns=1.0, p_gain=0.0, p_noise=0.0,
                      foa_channel_order="acn")
    for k in range(15):
        assert max_disagreement(aug, k) < 1e-3, f"pattern {k + 1}"


def test_legacy_order_is_consistent_only_for_full_inversion():
    aug = SELDAugment(enabled=True, p_16patterns=1.0, p_gain=0.0, p_noise=0.0)  # default
    assert aug.foa_channel_order == "legacy_wxyz"
    consistent = [k + 1 for k in range(15) if max_disagreement(aug, k) < 1e-3]
    assert consistent == [11]
    for k in range(15):
        if k != 10:
            assert max_disagreement(aug, k) > 60.0, f"pattern {k + 1} unexpectedly consistent"


def _reference_legacy(aug: SELDAugment, wav, target, idx):
    """The earlier (legacy) implementation (copied verbatim) used as bit-exactness oracle."""
    pat = aug._PATTERNS[idx]
    ch1_s, ch1_f, ch2_s, ch2_f, ch3_s, cx_s, cx_f, cy_s, cy_f, cz_s = pat
    x_orig, y_orig, z_orig = wav[1].clone(), wav[2].clone(), wav[3].clone()
    wav[1] = ch1_s * (x_orig if ch1_f == 1 else y_orig)
    wav[2] = ch2_s * (x_orig if ch2_f == 1 else y_orig)
    wav[3] = ch3_s * z_orig
    cx_orig, cy_orig, cz_orig = target[..., 0].clone(), target[..., 1].clone(), target[..., 2].clone()
    target[..., 0] = cx_s * (cx_orig if cx_f == 0 else cy_orig)
    target[..., 1] = cy_s * (cx_orig if cy_f == 0 else cy_orig)
    target[..., 2] = cz_s * cz_orig
    return wav, target


def test_default_is_bit_identical_to_legacy_implementation():
    aug = SELDAugment(enabled=True, p_16patterns=1.0, p_gain=0.0, p_noise=0.0)
    rng = np.random.default_rng(5)
    for k in range(15):
        wav, tgt = sample_pair(rng)
        a_w, a_t = aug.apply_pattern(wav.clone(), tgt.clone(), k)
        r_w, r_t = _reference_legacy(aug, wav.clone(), tgt.clone(), k)
        assert torch.equal(a_w, r_w) and torch.equal(a_t, r_t)
    # same random stream through __call__ (pattern draw + clone semantics)
    wav, tgt = sample_pair(rng)
    torch.manual_seed(1234)
    c_w, c_t = aug(wav, tgt)
    torch.manual_seed(1234)
    torch.multinomial(torch.tensor([1.0, 0.0, 0.0, 0.0]), 1)   # technique draw of __call__
    idx = int(torch.randint(15, (1,)).item())
    r_w, r_t = _reference_legacy(aug, wav.clone(), tgt.clone(), idx)
    assert torch.equal(c_w, r_w) and torch.equal(c_t, r_t)


def test_invalid_order_and_flag_plumbing():
    with pytest.raises(ValueError):
        SELDAugment(foa_channel_order="wxyz")
    assert SELDAugment(**{"enabled": True, "foa_channel_order": "acn", "unknown_key": 1}
                       ).foa_channel_order == "acn"


def test_pattern_table_targets_of_rows_4_7_12_15():
    """A tabulation of the patterns lists rows 4, 7, 12, 15 as (x,-y,z), (-x,y,z), (x,-y,-z), (-x,y,-z);
    the code produces (-x,y,z), (x,-y,z), (-x,y,-z), (x,-y,-z) (docs/CONVENTIONS.md)."""
    def target_signs(k):
        _, _, _, _, _, cx_s, cx_f, cy_s, cy_f, cz_s = SELDAugment._PATTERNS[k]
        name = lambda s, f: ("" if s > 0 else "-") + ("x" if f == 0 else "y")
        return (name(cx_s, cx_f), name(cy_s, cy_f), ("" if cz_s > 0 else "-") + "z")
    assert target_signs(3) == ("-x", "y", "z")
    assert target_signs(6) == ("x", "-y", "z")
    assert target_signs(11) == ("-x", "y", "-z")
    assert target_signs(14) == ("x", "-y", "-z")
    assert target_signs(10) == ("-x", "-y", "-z")   # pattern 11: full inversion
