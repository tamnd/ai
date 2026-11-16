# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/cosine-learning-rate-schedule/ (ai://NEW-cosine-learning-rate-schedule).
"""Tests of the NumPy reference: the worked example, the ends, mean and slope of the cosine, the warmup joints, the
cooldown shapes, the restart epochs against a loop, the SGDR rates, the noise-floor numbers and the error cases."""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cosine_learning_rate_schedule import (
    cooldown_factor,
    cosine,
    linear_decay,
    quadratic_variance,
    rates,
    restart_epoch,
    schedule,
    sgdr,
    sgdr_position,
    stationary_variance,
    step_decay,
    wsd,
)

# The worked example: peak 0.1, floor 0, 100 steps. The PyTorch tests check these numbers too.
STEPS = [0, 25, 50, 75, 100]
COSINE = [0.1, 0.08535533905932738, 0.05, 0.014644660940672627, 0.0]
LINEAR = [0.1, 0.075, 0.05, 0.025, 0.0]
STEP = [0.1, 0.1, 0.01, 0.001, 0.001]
WSD = {0: 0.1, 50: 0.1, 80: 0.1, 90: 0.05, 95: 0.025, 100: 0.0}


def test_worked_example():
    for t, c, lin, s in zip(STEPS, COSINE, LINEAR, STEP, strict=True):
        assert cosine(t, 100, 0.1) == pytest.approx(c, abs=1e-15)
        assert linear_decay(t, 100, 0.1) == pytest.approx(lin, abs=1e-15)
        assert step_decay(t, 0.1, [50, 75]) == pytest.approx(s, rel=1e-12)
    for t, r in WSD.items():
        assert wsd(t, 100, 0.1, 20) == pytest.approx(r, abs=1e-15)


def test_warmup_then_cosine():
    got = [schedule(t, "cosine", 110, 0.1, warmup=10) for t in (0, 5, 9, 10, 60, 110)]
    np.testing.assert_allclose(got, [0.0, 0.05, 0.09, 0.1, 0.05, 0.0], atol=1e-15)
    # Past the end the rate stays at the floor.
    assert schedule(130, "cosine", 110, 0.1, final=0.01, warmup=10) == pytest.approx(0.01, abs=1e-15)


def test_schedule_kinds():
    assert schedule(7, "constant", 100, 0.3) == 0.3
    assert schedule(7, "constant", 100, 0.3, warmup=10) == pytest.approx(0.21)
    assert schedule(60, "step", 100, 0.1, warmup=10, milestones=[50, 75]) == pytest.approx(0.01)
    assert schedule(80, "step", 100, 0.1, warmup=10, milestones=[50, 75], gamma=0.5) == pytest.approx(0.025)
    assert schedule(55, "linear", 100, 0.1, final=0.01, warmup=10) == pytest.approx(0.055)
    assert schedule(90, "wsd", 110, 0.1, warmup=10, decay_steps=20) == pytest.approx(0.1)
    assert schedule(100, "wsd", 110, 0.1, warmup=10, decay_steps=20) == pytest.approx(0.05)
    assert schedule(100, "wsd", 110, 0.1, warmup=10, decay_steps=20, shape="1-sqrt") == pytest.approx(
        0.1 * (1 - math.sqrt(0.5))
    )
    for t in range(0, 111):
        assert schedule(t, "cosine", 110, 0.1, warmup=0) == pytest.approx(cosine(t, 110, 0.1), abs=1e-15)


def test_cosine_reaches_floor():
    for final in (0.0, 0.01):
        assert cosine(0, 50, 0.1, final) == 0.1
        assert cosine(50, 50, 0.1, final) == pytest.approx(final, abs=1e-15)
        assert cosine(75, 50, 0.1, final) == pytest.approx(final, abs=1e-15)
        assert linear_decay(80, 50, 0.1, final) == pytest.approx(final, abs=1e-15)
        assert wsd(80, 50, 0.1, 10, final) == pytest.approx(final, abs=1e-15)


def test_mean_rate():
    lr = np.array([cosine(t, 100, 0.1) for t in range(101)])
    assert lr.mean() == pytest.approx(0.05, abs=1e-15)
    assert lr[:-1].mean() == pytest.approx(0.05 + 0.1 / 200, abs=1e-15)
    assert rates(cosine, 100, total=100, peak=0.1).sum() == pytest.approx(5.05, abs=1e-12)
    assert rates(linear_decay, 100, total=100, peak=0.1).sum() == pytest.approx(5.05, abs=1e-12)


@settings(max_examples=60, deadline=None)
@given(st.integers(2, 400), st.floats(1e-4, 1.0), st.floats(0.0, 0.99))
def test_mean_rate_any(total, peak, frac):
    final = frac * peak
    lr = np.array([cosine(t, total, peak, final) for t in range(total + 1)])
    assert lr.mean() == pytest.approx((peak + final) / 2, rel=1e-9)
    assert lr[:-1].mean() == pytest.approx((peak + final) / 2 + (peak - final) / (2 * total), rel=1e-9)


def test_slope():
    total, peak = 1000, 1.0
    d = np.diff(rates(cosine, total + 1, total=total, peak=peak))
    assert abs(d[0]) < 1e-5 and abs(d[-1]) < 1e-5
    assert int(np.argmax(-d)) in (total // 2 - 1, total // 2)
    assert -d.min() == pytest.approx(math.pi / 2 * peak / total, rel=1e-5)
    lin = rates(linear_decay, total + 1, total=total, peak=peak)
    cos = rates(cosine, total + 1, total=total, peak=peak)
    assert np.all(cos[1:500] > lin[1:500]) and np.all(cos[501:1000] < lin[501:1000])


@settings(max_examples=60, deadline=None)
@given(st.integers(1, 300), st.floats(0.0, 1.0), st.floats(1e-3, 1.0), st.floats(0.0, 0.9))
def test_monotone_and_bounded(total, a, peak, frac):
    final = frac * peak
    t1 = a * total
    for fn in (cosine, linear_decay):
        r0, r1 = (
            fn(t1, total, peak, final),
            fn(min(t1 + 0.5, total * 1.5), total, peak, final),
        )
        assert final - 1e-12 <= r1 <= r0 + 1e-12 <= peak + 2e-12
    w = wsd(t1, total, peak, max(1, total // 5), final)
    assert final - 1e-12 <= w <= peak


def test_step_decay():
    assert step_decay(31999, 0.1, [32000, 48000]) == 0.1
    assert step_decay(32000, 0.1, [32000, 48000]) == pytest.approx(0.01)
    assert step_decay(48000, 0.1, [32000, 48000]) == pytest.approx(0.001)
    assert step_decay(160, 0.1, [60, 120, 160], gamma=0.2) == pytest.approx(0.1 * 0.2**3)
    assert step_decay(5, 0.1, []) == 0.1


def test_cooldown_shapes():
    for shape in ("linear", "1-sqrt", "cosine"):
        assert cooldown_factor(0.0, shape) == 1.0
        assert cooldown_factor(1.0, shape) == pytest.approx(0.0, abs=1e-15)
    assert cooldown_factor(0.25, "linear") == 0.75
    assert cooldown_factor(0.25, "1-sqrt") == 0.5
    assert cooldown_factor(0.25, "cosine") == pytest.approx(0.5 * (1 + math.cos(math.pi / 4)))
    # (1-sqrt) drops faster than linear early on.
    assert cooldown_factor(0.04, "1-sqrt") == pytest.approx(0.8)
    assert wsd(90, 100, 0.1, 20, shape="cosine") == pytest.approx(0.05)
    assert wsd(85, 100, 0.1, 20, final=0.02, shape="1-sqrt") == pytest.approx(0.02 + 0.08 * 0.5)


def test_restart_epochs():
    assert [restart_epoch(k, 10, 2) for k in range(6)] == [0, 10, 30, 70, 150, 310]
    assert [restart_epoch(k, 1, 2) for k in range(5)] == [0, 1, 3, 7, 15]
    assert [restart_epoch(k, 5, 1) for k in range(4)] == [0, 5, 10, 15]
    assert restart_epoch(3, 2.0, 1.5) == pytest.approx(2.0 * (1.5**3 - 1) / 0.5)
    assert isinstance(restart_epoch(4, 10, 2), int)


@settings(max_examples=80, deadline=None)
@given(st.integers(1, 20), st.integers(1, 4), st.integers(0, 8))
def test_restart_epochs_match_loop(t0, t_mult, k):
    start, length = 0, t0
    for _ in range(k):
        start += length
        length *= t_mult
    assert restart_epoch(k, t0, t_mult) == start
    assert sgdr_position(start, t0, t_mult) == (k, 0, length)
    if start > 0:
        i, t_cur, t_i = sgdr_position(start - 0.5, t0, t_mult)
        assert (i, t_cur, t_i) == (k - 1, length // t_mult - 0.5, length // t_mult)


def test_sgdr():
    assert sgdr(0, 10, 0.05, t_mult=2) == 0.05
    assert sgdr(5, 10, 0.05, t_mult=2) == pytest.approx(0.025)
    assert sgdr(20, 10, 0.05, t_mult=2) == pytest.approx(0.025)
    assert sgdr(29.9, 10, 0.05, t_mult=2) == pytest.approx(0.025 * (1 + math.cos(math.pi * 19.9 / 20)))
    assert sgdr(29.9, 10, 0.05, t_mult=2) < 1e-5
    assert sgdr(30, 10, 0.05, t_mult=2) == 0.05
    assert sgdr(50, 10, 0.05, t_mult=2) == pytest.approx(0.025)
    assert sgdr(15, 10, 0.05, final=0.01) == pytest.approx(0.03)
    assert sgdr(37.5, 10, 0.1, t_mult=1) == pytest.approx(cosine(7.5, 10, 0.1))


def test_noise_floor():
    assert stationary_variance(0.1, 1.0, 1.0) == pytest.approx(0.1 / 1.9)
    assert stationary_variance(0.01, 1.0, 1.0) == pytest.approx(0.005025125628140704)
    assert stationary_variance(0.1, 2.0, 3.0) == pytest.approx(0.3 / (2 * 1.8))
    n = 1000
    const = quadratic_variance([0.1] * n, 1.0, 1.0, 1.0)
    assert const == pytest.approx(stationary_variance(0.1, 1.0, 1.0), rel=1e-9)
    cos = quadratic_variance(rates(cosine, n, total=n, peak=0.1), 1.0, 1.0, 1.0)
    lin = quadratic_variance(rates(linear_decay, n, total=n, peak=0.1), 1.0, 1.0, 1.0)
    stp = quadratic_variance(rates(step_decay, n, base=0.1, milestones=[500, 750]), 1.0, 1.0, 1.0)
    w = quadratic_variance(rates(wsd, n, total=n, peak=0.1, decay_steps=200), 1.0, 1.0, 1.0)
    assert round(cos, 4) == 0.0036 and round(lin, 4) == 0.0044 and round(stp, 4) == 0.0034 and round(w, 4) == 0.0100
    assert max(cos, lin, stp, w) < const
    # One step of the recursion by hand.
    assert quadratic_variance([0.5], 2.0, 3.0, 4.0) == pytest.approx(0.0 * 4.0 + 0.25 * 3.0)
    assert quadratic_variance([0.1, 0.2], 1.0, 1.0, 2.0) == pytest.approx(0.64 * (0.81 * 2.0 + 0.01) + 0.04)


@pytest.mark.parametrize(
    "call",
    [
        lambda: cosine(-1, 10, 0.1),
        lambda: cosine(1, 0, 0.1),
        lambda: linear_decay(1, 0, 0.1),
        lambda: linear_decay(-0.5, 10, 0.1),
        lambda: step_decay(-1, 0.1, [5]),
        lambda: wsd(1, 10, 0.1, 0),
        lambda: wsd(1, 10, 0.1, 11),
        lambda: wsd(-1, 10, 0.1, 2),
        lambda: cooldown_factor(1.5),
        lambda: cooldown_factor(-0.1),
        lambda: cooldown_factor(0.5, "exp"),
        lambda: schedule(1, "cosine", 10, 0.1, warmup=10),
        lambda: schedule(1, "cosine", 10, 0.1, warmup=-1),
        lambda: schedule(5, "poly", 10, 0.1),
        lambda: schedule(-1, "cosine", 10, 0.1),
        lambda: restart_epoch(-1, 10, 2),
        lambda: restart_epoch(1, 0, 2),
        lambda: restart_epoch(1, 10, 0.5),
        lambda: sgdr_position(-1, 10),
        lambda: sgdr_position(1, 0),
        lambda: sgdr_position(1, 10, 0),
        lambda: stationary_variance(2.0, 1.0, 1.0),
        lambda: stationary_variance(0.0, 1.0, 1.0),
    ],
)
def test_errors(call):
    with pytest.raises(ValueError):
        call()


def test_edges_accepted():
    assert cosine(0, 1, 0.1) == 0.1
    assert wsd(10, 10, 0.1, 10) == pytest.approx(0.0, abs=1e-15)
    assert wsd(0, 10, 0.1, 10) == 0.1
    assert schedule(0, "cosine", 10, 0.1, warmup=0) == 0.1
    assert restart_epoch(0, 10, 2) == 0
    assert sgdr_position(0, 10) == (0, 0, 10)
    assert sgdr_position(10, 10) == (1, 0, 10)
    assert stationary_variance(1.999, 1.0, 1.0) > 0
