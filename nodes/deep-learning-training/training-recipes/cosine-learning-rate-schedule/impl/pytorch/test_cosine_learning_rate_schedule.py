# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/cosine-learning-rate-schedule/ (ai://NEW-cosine-learning-rate-schedule).
"""Tests of the PyTorch implementation: the worked example, agreement with CosineAnnealingLR and
CosineAnnealingWarmRestarts (including fractional epochs and the periodic behaviour past T_max), the LambdaLR wrapper
with warmup, the restart positions against a loop, and the error cases."""

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

from cosine_learning_rate_schedule import (
    build_scheduler,
    cosine,
    linear_decay,
    lr_factor,
    sgdr,
    sgdr_position,
    step_decay,
    wsd,
)

torch.set_default_dtype(torch.float64)

# The worked example of the NumPy tests: peak 0.1, floor 0, 100 steps.
STEPS = [0, 25, 50, 75, 100]
COSINE = [0.1, 0.08535533905932738, 0.05, 0.014644660940672627, 0.0]
LINEAR = [0.1, 0.075, 0.05, 0.025, 0.0]
STEP = [0.1, 0.1, 0.01, 0.001, 0.001]
WSD = {0: 0.1, 50: 0.1, 80: 0.1, 90: 0.05, 95: 0.025, 100: 0.0}


def _optimizer(lr):
    return torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=lr)


def _trace(sched, opt, n):
    out = []
    for _ in range(n):
        out.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return out


def test_worked_example():
    t = torch.tensor(STEPS, dtype=torch.float64)
    torch.testing.assert_close(cosine(t, 100, 0.1), torch.tensor(COSINE), atol=1e-15, rtol=0)
    torch.testing.assert_close(linear_decay(t, 100, 0.1), torch.tensor(LINEAR), atol=1e-15, rtol=0)
    torch.testing.assert_close(step_decay(t, 0.1, [50, 75]), torch.tensor(STEP), atol=1e-15, rtol=1e-12)
    w = wsd(torch.tensor(list(WSD), dtype=torch.float64), 100, 0.1, 20)
    torch.testing.assert_close(w, torch.tensor(list(WSD.values())), atol=1e-15, rtol=0)


@pytest.mark.parametrize("eta_min", [0.0, 0.01])
@pytest.mark.parametrize("t_max", [1, 7, 100])
def test_cosine_annealing_lr(t_max, eta_min):
    opt = _optimizer(0.1)
    got = _trace(CosineAnnealingLR(opt, T_max=t_max, eta_min=eta_min), opt, t_max + 1)
    want = cosine(torch.arange(t_max + 1), t_max, 0.1, eta_min)
    torch.testing.assert_close(torch.tensor(got), want, atol=1e-12, rtol=0)


def test_cosine_annealing_lr_past_t_max():
    opt = _optimizer(0.1)
    got = _trace(CosineAnnealingLR(opt, T_max=10), opt, 21)
    # PyTorch keeps going round the cosine: back up after T_max, at the peak at 2 T_max.
    assert got[10] == pytest.approx(0.0, abs=1e-15)
    assert got[15] == pytest.approx(0.05, abs=1e-12)
    assert got[20] == pytest.approx(0.1, abs=1e-12)
    # The page's cosine stays at the floor.
    assert float(cosine(20, 10, 0.1)) == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("t0,t_mult,eta_min", [(10, 1, 0.0), (10, 2, 0.0), (3, 2, 0.005), (1, 3, 0.0)])
def test_cosine_annealing_warm_restarts(t0, t_mult, eta_min):
    opt = _optimizer(0.05)
    got = _trace(
        CosineAnnealingWarmRestarts(opt, T_0=t0, T_mult=t_mult, eta_min=eta_min),
        opt,
        160,
    )
    want = sgdr(torch.arange(160), t0, 0.05, eta_min, t_mult)
    torch.testing.assert_close(torch.tensor(got), want, atol=1e-12, rtol=0)


def test_warm_restarts_fractional_epochs():
    opt = _optimizer(0.05)
    sched = CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)
    times = [i / 7 for i in range(7 * 75)]
    got = []
    for e in times:
        sched.step(e)  # pyright: ignore[reportArgumentType]
        got.append(opt.param_groups[0]["lr"])
    torch.testing.assert_close(
        torch.tensor(got),
        sgdr(torch.tensor(times), 10, 0.05, t_mult=2),
        atol=1e-12,
        rtol=0,
    )


def test_sgdr_worked_example():
    assert float(sgdr(20, 10, 0.05, t_mult=2)) == pytest.approx(0.025)
    assert float(sgdr(29.9, 10, 0.05, t_mult=2)) < 1e-5
    assert float(sgdr(30, 10, 0.05, t_mult=2)) == 0.05
    starts = torch.tensor([0.0, 10, 30, 70, 150, 310])
    t_cur, t_i = sgdr_position(starts, 10, 2)
    torch.testing.assert_close(t_cur, torch.zeros(6), atol=0, rtol=0)
    torch.testing.assert_close(t_i, torch.tensor([10.0, 20, 40, 80, 160, 320]), atol=0, rtol=0)


@settings(max_examples=80, deadline=None)
@given(st.integers(1, 20), st.integers(1, 4), st.floats(0.0, 2000.0))
def test_sgdr_position_matches_loop(t0, t_mult, t):
    start, length = 0, t0
    while t >= start + length:
        start += length
        length *= t_mult
    t_cur, t_i = sgdr_position(t, t0, t_mult)
    assert float(t_i) == length
    assert float(t_cur) == pytest.approx(t - start, abs=1e-9)


def test_lambda_lr_with_warmup():
    opt = _optimizer(0.1)
    got = _trace(build_scheduler(opt, "cosine", 110, warmup=10), opt, 111)
    for t, r in zip((0, 5, 9, 10, 60, 110), (0.0, 0.05, 0.09, 0.1, 0.05, 0.0), strict=True):
        assert got[t] == pytest.approx(r, abs=1e-15)
    # With no warmup the wrapper is CosineAnnealingLR with eta_min = final_ratio * base.
    opt2 = _optimizer(0.1)
    a = _trace(build_scheduler(opt2, "cosine", 50, final_ratio=0.1), opt2, 51)
    opt3 = _optimizer(0.1)
    b = _trace(CosineAnnealingLR(opt3, T_max=50, eta_min=0.01), opt3, 51)
    torch.testing.assert_close(torch.tensor(a), torch.tensor(b), atol=1e-12, rtol=0)


def test_lambda_lr_kinds():
    f = lr_factor("step", 100, warmup=10, milestones=[50, 75], gamma=0.5)
    assert [f(t) for t in (5, 49, 50, 75)] == pytest.approx([0.5, 1.0, 0.5, 0.25])
    f = lr_factor("linear", 100, warmup=10, final_ratio=0.1)
    assert f(55) == pytest.approx(0.55)
    f = lr_factor("wsd", 110, warmup=10, decay_steps=20)
    assert [f(t) for t in (90, 100, 110)] == pytest.approx([1.0, 0.5, 0.0])
    f = lr_factor("wsd", 110, warmup=10, decay_steps=20, shape="1-sqrt")
    assert f(95) == pytest.approx(0.5)
    assert lr_factor("constant", 100, warmup=4)(2) == 0.5
    assert lr_factor("constant", 100)(70) == 1.0


def test_cooldown_shapes():
    t = torch.tensor([80.0, 85.0, 90.0, 100.0])
    torch.testing.assert_close(
        wsd(t, 100, 1.0, 20, shape="1-sqrt"),
        torch.tensor([1.0, 0.5, 1 - math.sqrt(0.5), 0.0]),
    )
    torch.testing.assert_close(wsd(t, 100, 1.0, 20, shape="cosine")[2], torch.tensor(0.5))
    torch.testing.assert_close(wsd(t, 100, 1.0, 20, final=0.2)[1], torch.tensor(0.8))


@pytest.mark.parametrize(
    "call",
    [
        lambda: cosine(-1.0, 10, 0.1),
        lambda: cosine(1.0, 0, 0.1),
        lambda: linear_decay(1.0, 0, 0.1),
        lambda: step_decay(-1.0, 0.1, [5]),
        lambda: wsd(1.0, 10, 0.1, 0),
        lambda: wsd(1.0, 10, 0.1, 11),
        lambda: wsd(1.0, 10, 0.1, 2, shape="exp"),
        lambda: sgdr_position(-1.0, 10),
        lambda: sgdr_position(1.0, 0),
        lambda: sgdr_position(1.0, 10, 0.5),
        lambda: lr_factor("cosine", 10, warmup=10),
        lambda: lr_factor("cosine", 10, warmup=-1),
        lambda: lr_factor("poly", 10),
    ],
)
def test_errors(call):
    with pytest.raises(ValueError):
        call()
