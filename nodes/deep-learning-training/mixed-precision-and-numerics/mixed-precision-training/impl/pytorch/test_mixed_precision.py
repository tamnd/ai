# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/mixed-precision-training/ (ai://EPEFQ9).
"""Tests of the PyTorch code on CPU: the shared fixtures against torch.finfo and the bfloat16 conversion, the dynamic
loss scaler against torch.amp.GradScaler and its update kernel, training with float32 master weights against
GradScaler step for step, FP16 and BF16 underflow and lost updates, a regression in float16 and bfloat16, and two
behaviours of PyTorch that the page warns about."""

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import nn

from mixed_precision import DynamicLossScaler, MasterWeights, round_to_bf16_bits

torch.set_default_dtype(torch.float64)


def grad_of(p):
    """The gradient of a parameter, which must have one."""
    assert p.grad is not None
    return p.grad


# The shared fixtures of the NumPy tests.
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
    (3.4028234663852886e38, math.inf),
    (math.inf, math.inf),
    (-math.inf, -math.inf),
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
DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "e5m2": torch.float8_e5m2,
}


@pytest.mark.parametrize("name", sorted(LIMITS))
def test_limits_match_finfo(name):
    fi = torch.finfo(DTYPES[name])
    eps, min_normal, min_sub, top = LIMITS[name]
    assert (fi.eps, fi.smallest_normal, fi.max) == (eps, min_normal, top)
    if name != "e5m2":
        # The smallest subnormal is representable and half of it is not.
        t = torch.tensor([min_sub, min_sub / 2], dtype=torch.float64).to(DTYPES[name]).to(torch.float64)
        assert t.tolist() == [min_sub, 0.0]


def test_e4m3_limits():
    fi = torch.finfo(torch.float8_e4m3fn)
    assert (fi.eps, fi.smallest_normal, fi.max) == (E4M3_FN[0], E4M3_FN[1], E4M3_FN[3])
    t = torch.tensor([E4M3_FN[2], E4M3_FN[2] / 2]).to(torch.float8_e4m3fn).to(torch.float64)
    assert t.tolist() == [E4M3_FN[2], 0.0]


@pytest.mark.parametrize("x,want", BF16_CASES)
def test_bf16_cases(x, want):
    t = torch.tensor([x], dtype=torch.float32)
    assert t.to(torch.bfloat16).to(torch.float64).item() == want
    assert round_to_bf16_bits(t).to(torch.float64).item() == want


@settings(max_examples=300, deadline=None)
@given(st.integers(0, 2**32 - 1))
def test_bit_rounding_matches_torch(bits):
    signed = bits - 2**32 if bits >= 2**31 else bits
    x = torch.tensor([signed], dtype=torch.int32).view(torch.float32)
    ours, theirs = round_to_bf16_bits(x), x.to(torch.bfloat16).to(torch.float32)
    if torch.isnan(x).item():
        assert torch.isnan(ours).item() and torch.isnan(theirs).item()
    else:
        assert torch.equal(ours.view(torch.int32), theirs.view(torch.int32))


def test_round_to_bf16_bits_wants_float32():
    with pytest.raises(ValueError):
        round_to_bf16_bits(torch.tensor([1.0], dtype=torch.float64))


def test_scaler_fixture():
    s = DynamicLossScaler(init_scale=SCALER["init_scale"], growth_interval=SCALER["growth_interval"])
    used = []
    for bad in SCALER["overflow"]:
        used.append(s.get_scale())
        s.update(bad)
    assert used == SCALER["scales"] and s.get_scale() == SCALER["final"]


def _grad_scaler_run(init_scale, interval, flags):
    """Run torch.amp.GradScaler on CPU through the flags, poisoning a gradient with inf on the flagged steps.
    Return the scales used, the growth tracker after each step, and whether each step changed the weights."""
    w = nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float32))
    opt = torch.optim.SGD([w], lr=0.01)
    gs = torch.amp.GradScaler(
        "cpu",
        init_scale=init_scale,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=interval,
    )
    scales, trackers, moved = [], [], []
    for bad in flags:
        opt.zero_grad()
        scales.append(gs.get_scale())
        gs.scale((w * w).sum()).backward()
        if bad:
            grad_of(w)[0] = math.inf
        before = w.detach().clone()
        gs.step(opt)
        gs.update()
        trackers.append(int(gs._get_growth_tracker()))
        moved.append(not torch.equal(before, w.detach()))
    return scales, trackers, moved, gs.get_scale()


def test_scaler_fixture_against_grad_scaler():
    scales, _, moved, final = _grad_scaler_run(SCALER["init_scale"], SCALER["growth_interval"], SCALER["overflow"])
    assert scales == SCALER["scales"] and final == SCALER["final"]
    assert moved == [not f for f in SCALER["overflow"]]


@settings(max_examples=40, deadline=None)
@given(st.lists(st.booleans(), max_size=25), st.integers(1, 4), st.integers(-4, 24))
def test_scaler_matches_grad_scaler(flags, interval, k):
    ours = DynamicLossScaler(init_scale=2.0**k, growth_interval=interval)
    scales, trackers = [], []
    for bad in flags:
        scales.append(ours.get_scale())
        ours.update(bad)
        trackers.append(ours.growth_tracker)
    g_scales, g_trackers, _, g_final = _grad_scaler_run(2.0**k, interval, flags)
    assert scales == g_scales and trackers == g_trackers and ours.get_scale() == g_final


@pytest.mark.parametrize(
    "scale,tracker,found,interval",
    [
        (8.0, 0, 0.0, 2),
        (8.0, 1, 0.0, 2),
        (8.0, 1, 1.0, 2),
        (3e38, 0, 0.0, 1),
        (2.0**127, 0, 0.0, 1),
        (1.0, 5, 1.0, 9),
    ],
)
def test_update_matches_the_kernel(scale, tracker, found, interval):
    s_t = torch.tensor([scale], dtype=torch.float32)
    tr_t = torch.tensor([tracker], dtype=torch.int32)
    torch._amp_update_scale_(  # pyright: ignore[reportPrivateImportUsage]
        s_t, tr_t, torch.tensor([found], dtype=torch.float32), 2.0, 0.5, interval
    )
    ours = DynamicLossScaler(init_scale=scale, growth_interval=interval)
    ours.growth_tracker = tracker
    ours.update(bool(found))
    assert ours.get_scale() == s_t.item() and ours.growth_tracker == tr_t.item()


def test_scaler_does_not_grow_to_inf():
    s = DynamicLossScaler(init_scale=2.0**127, growth_interval=1)
    s.update(False)
    assert s.get_scale() == 2.0**127 and s.growth_tracker == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"init_scale": -1.0},
        {"growth_factor": 0.5},
        {"backoff_factor": 2.0},
        {"growth_interval": 0},
    ],
)
def test_scaler_rejects_bad_settings(kwargs):
    with pytest.raises(ValueError):
        DynamicLossScaler(**kwargs)


def test_scaler_defaults_are_those_of_grad_scaler():
    ours, theirs = DynamicLossScaler(), torch.amp.GradScaler("cpu")
    assert ours.get_scale() == theirs.get_scale() == 2.0**16
    assert ours.growth_interval == theirs.get_growth_interval() == 2000
    assert ours.growth_factor == theirs.get_growth_factor() == 2.0
    assert ours.backoff_factor == theirs.get_backoff_factor() == 0.5


@pytest.mark.parametrize(
    "kwargs", [{"init_scale": 0.0}, {"growth_factor": 1.0}, {"backoff_factor": 0.0}, {"backoff_factor": 1.0}]
)
def test_scaler_rejects_edge_settings(kwargs):
    with pytest.raises(ValueError):
        DynamicLossScaler(**kwargs)


def test_scaler_accepts_settings_near_the_edges():
    s = DynamicLossScaler(
        init_scale=2.0**-149, growth_factor=1.0 + 2.0**-20, backoff_factor=2.0**-20, growth_interval=1
    )
    assert s.get_scale() == 2.0**-149 and s.growth_interval == 1
    s = DynamicLossScaler(backoff_factor=1.0 - 2.0**-20)
    assert s.backoff_factor == 1.0 - 2.0**-20


def test_unscale_wants_float32_and_finds_overflow():
    s = DynamicLossScaler(init_scale=1024.0)
    with pytest.raises(ValueError):
        s.unscale_([torch.ones(2, dtype=torch.float16)])
    g = torch.tensor([1024.0, 2.0**-14], dtype=torch.float32)
    assert not s.unscale_([g, None])
    assert g.tolist() == [1.0, 2.0**-24]
    for poison in (math.inf, -math.inf, math.nan):
        assert s.unscale_([torch.tensor([1.0, poison], dtype=torch.float32)])


def test_grad_scaler_refuses_fp16_gradients():
    # GradScaler expects float32 master parameters; unscaling float16 gradients raises.
    w = nn.Parameter(torch.ones(2, dtype=torch.float16))
    opt = torch.optim.SGD([w], lr=0.1)
    gs = torch.amp.GradScaler("cpu")
    gs.scale((w.float() * 3).sum()).backward()
    with pytest.raises(ValueError, match="FP16"):
        gs.unscale_(opt)


def _net(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 1)).to(torch.float32)


def test_master_weights_match_grad_scaler_step_for_step():
    # In float32 the hand-written loop and GradScaler do the same arithmetic, so the weights agree bit for bit,
    # including the steps skipped because of an injected overflow.
    x = torch.randn(16, 3, dtype=torch.float32, generator=torch.Generator().manual_seed(1))
    y = torch.randn(16, 1, dtype=torch.float32, generator=torch.Generator().manual_seed(2))
    poison = {3, 4, 9}
    ref = _net(0)
    ref_opt = torch.optim.SGD(ref.parameters(), lr=0.05, momentum=0.9)
    gs = torch.amp.GradScaler("cpu", init_scale=2.0**10, growth_interval=2)
    mw = MasterWeights(
        _net(0),
        dtype=torch.float32,
        scaler=DynamicLossScaler(init_scale=2.0**10, growth_interval=2),
    )
    opt = torch.optim.SGD(mw.master_params(), lr=0.05, momentum=0.9)
    for t in range(12):
        ref_opt.zero_grad()
        gs.scale(((ref(x) - y) ** 2).mean()).backward()
        mw.backward(((mw.model(x) - y) ** 2).mean())
        if t in poison:
            grad_of(next(ref.parameters()))[0, 0] = math.nan
            mw.params[0].grad[0, 0] = math.nan
        gs.step(ref_opt)
        gs.update()
        skipped = mw.step(opt)
        assert skipped == (t in poison)
        assert gs.get_scale() == mw.scaler.get_scale()
        for a, b in zip(ref.parameters(), mw.model.parameters(), strict=True):
            assert torch.equal(a, b)


def test_skipped_step_leaves_weights():
    mw = MasterWeights(_net(3), dtype=torch.float16, scaler=DynamicLossScaler(init_scale=2.0**20))
    opt = torch.optim.SGD(mw.master_params(), lr=0.1)
    before = [p.detach().clone() for p in mw.model.parameters()]
    x = torch.full((4, 3), 100.0, dtype=torch.float16)
    mw.backward(mw.model(x).sum())  # 2^20 times this sum overflows float16
    assert mw.step(opt)
    assert all(torch.equal(a, b) for a, b in zip(before, mw.model.parameters(), strict=True))
    assert mw.scaler.get_scale() == 2.0**19
    assert all(p.dtype == torch.float16 for p in mw.model.parameters())
    assert all(m.dtype == torch.float32 for m in mw.master_params())


def test_master_weights_rejects_other_dtypes():
    with pytest.raises(ValueError):
        MasterWeights(_net(0), dtype=torch.float64)


def test_underflow_fp16_and_bf16():
    t = torch.tensor([2.0**-25, 2.0**-24, 2.0**-40], dtype=torch.float32)
    assert t.to(torch.float16).to(torch.float64).tolist() == [0.0, 2.0**-24, 0.0]
    assert t.to(torch.bfloat16).to(torch.float64).tolist() == [
        2.0**-25,
        2.0**-24,
        2.0**-40,
    ]


@pytest.mark.parametrize("dtype,m", [(torch.float16, 10), (torch.bfloat16, 7)])
def test_updates_lost_above_the_ratio(dtype, m):
    w = torch.ones((), dtype=dtype)
    small, big = (
        torch.tensor(2.0 ** -(m + 2), dtype=dtype),
        torch.tensor(2.0 ** -(m + 1) * 1.5, dtype=dtype),
    )
    assert (w + small).item() == 1.0  # ratio 2^(m+2): lost
    assert (w + big).item() > 1.0  # ratio below 2^(m+1): kept
    acc = w.clone()
    for _ in range(100):
        acc = acc + small
    assert acc.item() == 1.0
    acc32 = torch.ones((), dtype=torch.float32)
    for _ in range(100):
        acc32 = acc32 + small.to(torch.float32)
    assert acc32.item() == 1.0 + 100 * 2.0 ** -(m + 2)


# A linear regression whose loss carries a small weight C, as in the NumPy tests.
C = 1e-6
LR = 0.1 / (2 * C)


def _regression(dtype, scaler):
    g = torch.Generator().manual_seed(7)
    x = torch.randn(256, 4, generator=g, dtype=torch.float32)
    w_true = torch.tensor([[3.0], [-1.0], [2.0], [0.5]], dtype=torch.float32)
    y = x @ w_true
    model = nn.Linear(4, 1, bias=False).to(torch.float32)
    with torch.no_grad():
        model.weight.zero_()
    mw = MasterWeights(model, dtype=dtype, scaler=scaler)
    opt = torch.optim.SGD(mw.master_params(), lr=LR)
    xs, ys = x.to(dtype), y.to(dtype)
    skipped = 0
    for _ in range(300):
        mw.backward(C * ((mw.model(xs) - ys) ** 2).mean())
        skipped += mw.step(opt)
    err = (mw.master[0].detach().reshape(-1, 1) - w_true).abs().max().item() / 3.0
    return err, skipped


def _no_scaling():
    return DynamicLossScaler(init_scale=1.0, growth_interval=10**9)


def test_fp16_without_scaling_loses_the_gradients():
    err, _ = _regression(torch.float16, _no_scaling())
    assert err > 0.5


def test_fp16_with_dynamic_scaling_learns():
    # A start of 2^40 is far too large: the first steps overflow and back off, then about one step in 51 is skipped.
    err, skipped = _regression(torch.float16, DynamicLossScaler(init_scale=2.0**40, growth_interval=50))
    assert err < 0.002 and 15 <= skipped <= 45


def test_bf16_learns_without_scaling():
    # BF16 has the exponent range of FP32, so the small gradients survive with no loss scale; its 8-bit
    # significand limits how close the model gets.
    err, skipped = _regression(torch.bfloat16, _no_scaling())
    assert err < 0.02 and skipped == 0


def test_autocast_on_cpu_keeps_float32_weights():
    lin = nn.Linear(4, 2).to(torch.float32)
    x = torch.randn(3, 4, dtype=torch.float32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = lin(x)
    assert out.dtype == torch.bfloat16 and lin.weight.dtype == torch.float32


def test_torch_e4m3_cast_saturates():
    # Micikevicius et al. (2022) say infinities should become NaN when converted to E4M3. The CPU cast of this
    # PyTorch saturates them to 448, so an inf check after the cast does not see the overflow. E5M2 keeps inf.
    t = torch.tensor([1e5, math.inf], dtype=torch.float32)
    assert t.to(torch.float8_e4m3fn).to(torch.float32).tolist() == [448.0, 448.0]
    assert torch.isinf(t.to(torch.float8_e5m2).to(torch.float32)).all()


def test_clip_after_unscaling():
    # Clipping must see the true gradients: clipped after GradScaler.unscale_ they equal the clipped float32 gradients,
    # clipped before it the threshold applies to S times the gradients and cuts them far too much.
    def grads(order):
        torch.manual_seed(0)
        lin = nn.Linear(3, 1).to(torch.float32)
        x = torch.randn(8, 3, dtype=torch.float32)
        opt = torch.optim.SGD(lin.parameters(), lr=0.1)
        gs = torch.amp.GradScaler("cpu", init_scale=2.0**12)
        loss = (lin(x) ** 2).mean()
        if order == "plain":
            loss.backward()
            nn.utils.clip_grad_norm_(lin.parameters(), 0.5)
        else:
            gs.scale(loss).backward()
            if order == "unscale-then-clip":
                gs.unscale_(opt)
                nn.utils.clip_grad_norm_(lin.parameters(), 0.5)
            else:
                nn.utils.clip_grad_norm_(lin.parameters(), 0.5)
                gs.unscale_(opt)
        return torch.cat([grad_of(p).reshape(-1) for p in lin.parameters()])

    plain = grads("plain")
    assert torch.allclose(grads("unscale-then-clip"), plain, rtol=1e-6, atol=0)
    wrong = grads("clip-then-unscale")
    assert wrong.norm() < plain.norm() / 1000


def test_bit_rounding_keeps_the_sign():
    x = torch.tensor([-0.0, -1.0, -(1.0 + 2.0**-8), -(2.0**-133), -3.0e38, -math.inf], dtype=torch.float32)
    ours, theirs = round_to_bf16_bits(x), x.to(torch.bfloat16).to(torch.float32)
    assert torch.equal(ours.view(torch.int32), theirs.view(torch.int32))
    assert bool((torch.signbit(ours)).all())
