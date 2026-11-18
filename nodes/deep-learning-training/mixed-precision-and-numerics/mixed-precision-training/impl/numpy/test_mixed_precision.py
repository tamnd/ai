# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/mixed-precision-training/ (ai://EPEFQ9).
"""Tests of the NumPy reference: the format limits against numpy.finfo and the tables of the sources, BF16 rounding
and truncation, underflow and lost updates in FP16, the constant loss scale, the dynamic loss scaler and its fixture
shared with the PyTorch tests, FP32 master weights, and a small regression trained in FP16 with and without them."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mixed_precision import (
    FORMATS,
    DynamicLossScaler,
    MixedPrecisionSGD,
    format_limits,
    largest_safe_scale,
    round_to_bf16,
    truncate_to_bf16,
    update_is_lost,
)

# The shared fixtures. The PyTorch tests check LIMITS against torch.finfo, BF16_CASES against tensor.to(bfloat16),
# and SCALER against torch.amp.GradScaler.
LIMITS = {
    "fp32": (2.0**-23, 2.0**-126, 2.0**-149, 3.4028234663852886e38),
    "fp16": (2.0**-10, 2.0**-14, 2.0**-24, 65504.0),
    "bf16": (2.0**-7, 2.0**-126, 2.0**-133, 3.3895313892515355e38),
    "e5m2": (2.0**-2, 2.0**-14, 2.0**-16, 57344.0),
}
E4M3_FN = (2.0**-3, 2.0**-6, 2.0**-9, 448.0)
BF16_CASES = [
    (1.0 + 2.0**-8, 1.0),
    (1.0 + 3 * 2.0**-8, 1.0 + 2.0**-6),
    (1.0 + 2.0**-8 + 2.0**-20, 1.0 + 2.0**-7),
    (-(1.0 + 2.0**-8 + 2.0**-20), -(1.0 + 2.0**-7)),
    (3.140625, 3.140625),
    (2.0**-133, 2.0**-133),
    (2.0**-134, 0.0),
    (3.3895313892515355e38, 3.3895313892515355e38),
    (3.4028234663852886e38, np.inf),
    (np.inf, np.inf),
    (-np.inf, -np.inf),
]
SCALER = {
    "init_scale": 4.0,
    "growth_interval": 3,
    "overflow": [
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        True,
        False,
    ],
    "scales": [4.0, 4.0, 4.0, 8.0, 8.0, 4.0, 4.0, 4.0, 8.0, 4.0, 2.0],
    "final": 2.0,
}


@pytest.mark.parametrize("name", sorted(LIMITS))
def test_format_limits(name):
    got = format_limits(*FORMATS[name])
    assert (got["eps"], got["min_normal"], got["min_subnormal"], got["max"]) == LIMITS[name]


@pytest.mark.parametrize("dt,name", [(np.float16, "fp16"), (np.float32, "fp32")])
def test_format_limits_match_finfo(dt, name):
    fi, got = np.finfo(dt), format_limits(*FORMATS[name])
    assert got["eps"] == fi.eps and got["min_normal"] == fi.smallest_normal
    assert got["min_subnormal"] == fi.smallest_subnormal and got["max"] == fi.max


def test_e4m3_reclaims_the_top_exponent():
    got = format_limits(4, 3, ieee=False)
    assert (got["eps"], got["min_normal"], got["min_subnormal"], got["max"]) == E4M3_FN
    assert format_limits(4, 3)["max"] == 240.0


def test_format_limits_rejects_bad_widths():
    with pytest.raises(ValueError):
        format_limits(1, 3)
    with pytest.raises(ValueError):
        format_limits(5, 0)
    # The smallest widths allowed: 2 exponent bits and 1 mantissa bit, bias 1.
    assert format_limits(2, 1) == {"eps": 0.5, "min_normal": 1.0, "min_subnormal": 0.5, "max": 3.0}


def test_formats_table():
    """The field widths of FORMATS give the limits of the derivation number-formats."""
    assert format_limits(*FORMATS["e4m3"], ieee=False)["max"] == 448.0
    assert format_limits(*FORMATS["e5m2"])["max"] == 57344.0
    assert format_limits(*FORMATS["fp16"])["max"] == 65504.0
    assert format_limits(*FORMATS["bf16"])["eps"] == 2.0**-7
    assert format_limits(*FORMATS["fp32"])["min_subnormal"] == 2.0**-149


def test_tables_of_the_sources():
    # Kalamkar et al., Table 1, and Micikevicius et al. (2022), Table 1.
    assert f"{LIMITS['fp16'][3]:.2e}" == "6.55e+04" and f"{LIMITS['fp16'][1]:.2e}" == "6.10e-05"
    assert f"{LIMITS['fp16'][2]:.2e}" == "5.96e-08" and f"{LIMITS['fp32'][2]:.2e}" == "1.40e-45"
    assert LIMITS["e5m2"][3] == 1.75 * 2**15 and E4M3_FN[3] == 1.75 * 2**8


@pytest.mark.parametrize("x,want", BF16_CASES)
def test_round_to_bf16_cases(x, want):
    with np.errstate(over="ignore"):
        got = round_to_bf16(np.float32(x))
    assert got == np.float32(want) and np.signbit(got) == np.signbit(np.float32(want))


def test_round_to_bf16_keeps_nan_and_signed_zero():
    assert np.isnan(round_to_bf16(np.float32(np.nan)))
    z = round_to_bf16(np.float32(-0.0))
    assert z == 0 and np.signbit(z)


@settings(max_examples=300, deadline=None)
@given(
    st.floats(
        width=32,
        allow_nan=False,
        allow_infinity=False,
        min_value=-(2.0**127),
        max_value=2.0**127,
    )
)
def test_round_to_bf16_is_nearest(x):
    x = np.float32(x)
    r, t = round_to_bf16(x), truncate_to_bf16(x)
    assert r.view(np.uint32) & 0xFFFF == 0 and t.view(np.uint32) & 0xFFFF == 0
    assert abs(t) <= abs(x)
    # The neighbours of x in BF16 are t and the next value away from zero; r is the nearer, ties to even.
    up = (t.view(np.uint32) + np.uint32(0x10000)).view(np.float32) if t != x else t
    d_t, d_up = abs(float(x) - float(t)), abs(float(x) - float(up))
    assert float(r) in (float(t), float(up))
    assert abs(float(x) - float(r)) == min(d_t, d_up)
    if d_t == d_up and t != up:
        assert (r.view(np.uint32) >> 16) & 1 == 0


def test_truncation_rounds_toward_zero():
    x = np.float32(1.0 + 2.0**-7 - 2.0**-20)
    assert truncate_to_bf16(x) == 1.0 and round_to_bf16(x) == 1.0 + 2.0**-7
    assert truncate_to_bf16(np.float32(-x)) == -1.0


def test_fp16_underflow_threshold():
    t = 2.0**-25
    assert np.float16(t) == 0 and np.float16(np.nextafter(t, 1)) == 2.0**-24
    assert np.float16(2.0**-24) == 2.0**-24 and np.float16(2.0**-26) == 0


def test_update_is_lost_edges():
    assert update_is_lost(1.0, 2.0**-11)  # a tie, rounded to the even 1
    assert not update_is_lost(1.0, 2.0**-11 + 2.0**-21)
    assert update_is_lost(1.0, 2.0**-12) and not update_is_lost(1.0, 2.0**-10)
    assert not update_is_lost(np.float32(1.0), np.float32(2.0**-12), dtype=np.float32)
    assert update_is_lost(np.float32(1.0), np.float32(2.0**-25), dtype=np.float32)
    # 1 + 2^-24 is a tie that rounds back to 1, while 1 - 2^-24 is a float32 number.
    assert update_is_lost(np.float32(1.0), np.float32(2.0**-24), dtype=np.float32)
    with pytest.raises(ValueError):
        update_is_lost(1.0, 1.0, dtype=np.float64)


fp16_normal = st.integers(0x0400, 0x7BFF).map(lambda b: float(np.uint16(b).view(np.float16)))


@settings(max_examples=400, deadline=None)
@given(fp16_normal, st.floats(9.0, 14.0), st.booleans())
def test_update_is_lost_bounds(w, log_ratio, negative):
    u = float(np.float16(w / 2.0**log_ratio * (-1 if negative else 1)))
    if u == 0:
        return
    r = w / abs(u)
    if r > 4096:
        assert update_is_lost(w, u)
    if r < 2048:
        assert not update_is_lost(w, u)


def test_largest_safe_scale():
    assert largest_safe_scale(1.0) == 2.0**15
    assert largest_safe_scale(3.0) == 2.0**14 and largest_safe_scale(1e-3) == 2.0**25
    for g in (1e-3, 0.5, 3.0, 700.0):
        s = largest_safe_scale(g)
        assert np.isfinite(np.float16(s * g)) and s * g * 2 > 65504.0
    with pytest.raises(ValueError):
        largest_safe_scale(0.0)


def test_scaler_fixture():
    s = DynamicLossScaler(init_scale=SCALER["init_scale"], growth_interval=SCALER["growth_interval"])
    used = []
    for bad in SCALER["overflow"]:
        used.append(float(s.scale))
        s.update(bad)
    assert used == SCALER["scales"] and s.scale == SCALER["final"]


def test_scaler_defaults_are_those_of_grad_scaler():
    s = DynamicLossScaler()
    assert (s.scale, s.growth_factor, s.backoff_factor, s.growth_interval) == (
        65536.0,
        2.0,
        0.5,
        2000,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"init_scale": 0.0},
        {"init_scale": np.inf},
        {"growth_factor": 1.0},
        {"backoff_factor": 1.0},
        {"backoff_factor": 0.0},
        {"growth_interval": 0},
        {"growth_interval": 1.5},
    ],
)
def test_scaler_rejects_bad_settings(kwargs):
    with pytest.raises(ValueError):
        DynamicLossScaler(**kwargs)


def test_scaler_does_not_grow_to_inf():
    s = DynamicLossScaler(init_scale=2.0**127, growth_interval=1)
    s.update(False)
    assert s.scale == 2.0**127 and s.growth_tracker == 0
    s.update(True)
    assert s.scale == 2.0**126


@settings(max_examples=200, deadline=None)
@given(st.lists(st.booleans(), max_size=60), st.integers(1, 5), st.integers(-3, 20))
def test_scaler_counts_doublings_and_backoffs(flags, n, k):
    s = DynamicLossScaler(init_scale=2.0**k, growth_interval=n)
    for f in flags:
        s.update(f)
    runs = "".join("x" if f else "." for f in flags).split("x")
    doublings = sum(len(r) // n for r in runs)
    assert s.scale == 2.0 ** (k + doublings - sum(flags))
    assert s.growth_tracker == len(runs[-1]) % n


def test_unscale_is_exact_and_finds_overflow():
    s = DynamicLossScaler(init_scale=1024.0)
    g16 = [
        np.array([1.0, -3.0, 2.0**-24], dtype=np.float16),
        np.array([65504.0], dtype=np.float16),
    ]
    out, bad = s.unscale(g16)
    assert not bad and out[0].dtype == np.float32
    np.testing.assert_array_equal(out[0], np.array([1.0, -3.0, 2.0**-24]) / 1024)
    for poison in (np.inf, -np.inf, np.nan):
        _, bad = s.unscale([np.array([1.0, poison], dtype=np.float16)])
        assert bad
    assert s.scale_loss(np.float32(0.5)) == 512.0


def test_skipped_step_leaves_master_weights():
    opt = MixedPrecisionSGD([np.array([1.0, 2.0])], lr=0.1, scaler=DynamicLossScaler(init_scale=8.0))
    before = opt.master[0].copy()
    assert opt.step([np.array([np.inf, 1.0], dtype=np.float16)])
    np.testing.assert_array_equal(opt.master[0], before)
    assert opt.scaler.scale == 4.0
    assert not opt.step([np.array([4.0, 8.0], dtype=np.float16)])
    np.testing.assert_allclose(opt.master[0], [0.9, 1.8], rtol=1e-6)


def test_master_weights_keep_small_updates():
    # 1000 updates of 2^-12 to a weight of 1, a ratio of 4096: lost in FP16, kept in the FP32 master copy.
    w16 = np.float16(1.0)
    opt = MixedPrecisionSGD([np.array([1.0])], lr=1.0, scaler=DynamicLossScaler(init_scale=1.0))
    for _ in range(1000):
        w16 = np.float16(np.float64(w16) + 2.0**-12)
        opt.step([np.array([-(2.0**-12)], dtype=np.float16)])
    assert w16 == 1.0 and opt.master[0][0] == np.float32(1.244140625)
    assert opt.half_params()[0][0] == np.float16(1.244140625)


# A linear regression whose loss carries a small weight C, as a loss term of a larger objective might: the
# gradients are C times those of the mean squared error, small enough that FP16 rounds them to zero.
C = 1e-6
LR = 0.1 / (2 * C)


def _fp16_grad(x16, w16, y16, scale):
    """The gradient of scale * C * mean squared error of a linear model, with every operation in float16."""
    r = x16 @ w16 - y16
    up = (np.float16(2.0 * C * scale / len(y16)) * r).astype(np.float16)
    return (x16.T @ up).astype(np.float16)


def _train(mode, steps=300):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(256, 4))
    w_true = np.array([3.0, -1.0, 2.0, 0.5])
    y = x @ w_true
    x16, y16 = x.astype(np.float16), y.astype(np.float16)
    if mode == "fp16":
        w = np.zeros(4, dtype=np.float16)
        for _ in range(steps):
            w = (w - np.float16(LR) * _fp16_grad(x16, w, y16, 1.0)).astype(np.float16)
        return np.abs(w.astype(np.float64) - w_true).max() / np.abs(w_true).max(), 0
    opt = MixedPrecisionSGD(
        [np.zeros(4)],
        lr=LR,
        scaler=DynamicLossScaler(init_scale=2.0**40, growth_interval=50),
    )
    skipped = 0
    for _ in range(steps):
        (w16,) = opt.half_params()
        with np.errstate(over="ignore", invalid="ignore"):
            g = _fp16_grad(x16, w16, y16, float(opt.scaler.scale))
        skipped += opt.step([g])
    return np.abs(opt.master[0].astype(np.float64) - w_true).max() / np.abs(w_true).max(), skipped


def test_fp16_without_loss_scaling_does_not_learn_small_gradients():
    err, _ = _train("fp16")
    assert err > 0.5


def test_mixed_precision_with_dynamic_scaling_learns():
    err, skipped = _train("mixed")
    assert err < 0.002
    # Starting at 2^40 the scaler overflows and halves for a run of steps before training starts.
    assert 5 <= skipped <= 15
