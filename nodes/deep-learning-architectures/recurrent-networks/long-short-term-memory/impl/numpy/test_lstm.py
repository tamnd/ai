# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/long-short-term-memory/ (ai://YVW8D3).
"""Tests of the NumPy reference: the worked example, the shape contract, the shared fixture and the properties."""

import json
import pathlib

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lstm import backward, cell, config, flops, forward, init, param_count, sigmoid

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def cfg_of(c):
    return config(c["config"]["input_size"], c["config"]["hidden_size"])


# The worked example of the page: one cell, one input, no recurrent weights, no bias.
EX = {"W": np.array([[0.0, 1.0, 1.0, 0.0]]), "U": np.zeros((1, 4)), "b": np.zeros(4)}


def test_worked_example():
    h, c, g = cell(EX, np.array([[1.0]]), np.zeros((1, 1)), np.array([[1.0]]))
    np.testing.assert_allclose(
        [g["i"][0, 0], g["f"][0, 0], g["g"][0, 0], g["o"][0, 0]],
        [0.5, 0.7310585786, 0.7615941560, 0.5],
        rtol=0,
        atol=1e-10,
    )
    np.testing.assert_allclose(c[0, 0], 1.1118556566, rtol=0, atol=1e-10)
    np.testing.assert_allclose(h[0, 0], 0.4023586789, rtol=0, atol=1e-10)


def test_worked_example_jacobian():
    # With U = 0 the cell state is the only path from c_0 to c_T, and dc_T/dc_0 is the product of the f_t.
    x = np.ones((1, 3, 1))
    _, cache = forward(EX, x, np.zeros((1, 1)), np.array([[1.0]]))
    _, _, _, dc0 = backward(EX, cache, np.zeros((1, 3, 1)), None, np.ones((1, 1)))
    np.testing.assert_allclose(dc0[0, 0], 0.7310585786**3, rtol=0, atol=1e-10)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_shapes(c):
    cfg, k, blocks = cfg_of(c), c["config"], c["blocks"]
    rng = np.random.default_rng(0)
    params = init(rng, cfg)
    h, cache = forward(params, rng.normal(size=(k["B"], k["T"], k["I"])))
    assert list(h.shape) == blocks["layer"]["outputs"]["h"]
    assert list(cache["c"].shape) == blocks["layer"]["outputs"]["c"]
    for name, shape in blocks["cell"]["internals"].items():
        assert list(cache[name].shape) == shape, name
    assert list(cache["h_T"].shape) == blocks["layer"]["inputs"]["h0"]


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_param_count_and_flops(c):
    cfg, k = cfg_of(c), c["config"]
    assert param_count(cfg) == c["model"]["params"]
    assert flops(cfg, k["B"], k["T"]) == c["model"]["flops_forward"]
    assert sum(a.size for a in init(np.random.default_rng(0), cfg).values()) == c["model"]["params"]


def test_shared_fixture():
    params = dict(np.load(FIXTURES / "params-small.npz"))
    inp = np.load(FIXTURES / "input-small.npz")
    want = np.load(FIXTURES / "expected-small.npz")
    h, cache = forward(params, inp["x"], inp["h0"], inp["c0"])
    for name in ("h", "c", "h_T", "c_T", "i", "f", "g", "o"):
        np.testing.assert_allclose(cache[name], want[name], rtol=1e-12, atol=1e-14, err_msg=name)
    dx, dp, dh0, dc0 = backward(params, cache, inp["r"], inp["r_h"], inp["r_c"])
    got = {
        "dx": dx,
        "dh0": dh0,
        "dc0": dc0,
        "dW": dp["W"],
        "dU": dp["U"],
        "db": dp["b"],
    }
    for name, v in got.items():
        np.testing.assert_allclose(v, want[name], rtol=1e-12, atol=1e-14, err_msg=name)


def small(seed=1, n_in=3, H=4, B=2, T=4, forget_bias=0.0):
    rng = np.random.default_rng(seed)
    params = init(rng, config(n_in, H), forget_bias)
    return (
        params,
        rng.normal(size=(B, T, n_in)),
        rng.normal(0, 0.5, (B, H)),
        rng.normal(0, 0.5, (B, H)),
        rng,
    )


def loss_of(params, x, h0, c0, r, r_h, r_c):
    h, cache = forward(params, x, h0, c0)
    return np.sum(h * r) + np.sum(cache["h_T"] * r_h) + np.sum(cache["c_T"] * r_c)


@pytest.mark.parametrize("T", [1, 4])
def test_gradient(T):
    """Central finite differences in float64 on every parameter, input and initial state entry."""
    params, x, h0, c0, rng = small(T=T, forget_bias=0.5)
    r, r_h, r_c = (
        rng.normal(size=(2, T, 4)),
        rng.normal(size=(2, 4)),
        rng.normal(size=(2, 4)),
    )
    _, cache = forward(params, x, h0, c0)
    dx, dp, dh0, dc0 = backward(params, cache, r, r_h, r_c)

    def check(arr, grad):
        num = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            old = arr[idx]
            eps = 1e-6 * max(1.0, abs(old))
            arr[idx] = old + eps
            up = loss_of(params, x, h0, c0, r, r_h, r_c)
            arr[idx] = old - eps
            down = loss_of(params, x, h0, c0, r, r_h, r_c)
            arr[idx] = old
            num[idx] = (up - down) / (2 * eps)
        np.testing.assert_allclose(grad, num, rtol=1e-6, atol=1e-8)

    for name in ("W", "U", "b"):
        check(params[name], dp[name])
    check(x, dx)
    check(h0, dh0)
    check(c0, dc0)


@settings(max_examples=30, deadline=None)
@given(st.integers(0, 2**31 - 1), st.floats(-5.0, 5.0))
def test_gates_in_unit_interval(seed, scale):
    params, x, h0, c0, _ = small(seed)
    _, cache = forward(params, scale * x, h0, c0)
    for name in ("i", "f", "o"):
        assert np.all((cache[name] > 0.0) & (cache[name] < 1.0)), name
    assert np.all(np.abs(cache["g"]) <= 1.0)
    assert np.all(np.abs(cache["h"]) < 1.0)  # |h| = o |tanh(c)| < 1


def test_sigmoid_is_stable():
    z = np.array([-800.0, -50.0, 0.0, 50.0, 800.0])
    s = sigmoid(z)
    assert np.all(np.isfinite(s))
    np.testing.assert_allclose(s, [0.0, np.exp(-50.0) / (1 + np.exp(-50.0)), 0.5, 1.0, 1.0], rtol=1e-15, atol=0)
    np.testing.assert_allclose(sigmoid(-z), 1.0 - s, atol=1e-15)


def test_copy_when_forget_is_one_and_input_is_zero():
    # A forget bias of +800 and an input bias of -800 give f = 1 and i = 0 exactly in float64,
    # so the cell keeps c_0 bit for bit over the whole sequence, whatever the inputs.
    params, x, h0, c0, _ = small(T=50)
    H = 4
    params["b"][:H] = -800.0
    params["b"][H : 2 * H] = 800.0
    _, cache = forward(params, 3.0 * x, h0, c0)
    assert np.all(cache["i"] == 0.0) and np.all(cache["f"] == 1.0)
    for t in range(50):
        assert np.array_equal(cache["c"][:, t], c0)
    np.testing.assert_allclose(cache["h"][:, -1], cache["o"][:, -1] * np.tanh(c0), rtol=1e-15)


def test_cell_jacobian_without_recurrence_is_diag_f():
    # With U = 0, h_{t-1} does not reach c_t, and dc_t/dc_{t-1} is the diagonal matrix of f_t.
    params, x, h0, c0, _ = small(T=1)
    params["U"][...] = 0.0
    _, cache = forward(params, x, h0, c0)
    J = np.zeros((2, 4, 4))
    for b in range(2):
        for k in range(4):
            dcT = np.zeros((2, 4))
            dcT[b, k] = 1.0
            _, _, _, dc0 = backward(params, cache, np.zeros((2, 1, 4)), None, dcT)
            J[b, k] = dc0[b]
    for b in range(2):
        np.testing.assert_allclose(J[b], np.diag(cache["f"][b, 0]), atol=1e-15)


def test_zero_forget_gate_erases_the_past():
    params, x, h0, c0, _ = small(T=3)
    params["b"][4:8] = -800.0  # f = 0
    _, a = forward(params, x, h0, c0)
    _, b = forward(params, x, h0, c0 + 5.0)
    np.testing.assert_array_equal(a["c"][:, 0], b["c"][:, 0])


def test_default_state_is_zero():
    params, x, _, _, _ = small()
    h1, _ = forward(params, x)
    h2, _ = forward(params, x, np.zeros((2, 4)), np.zeros((2, 4)))
    np.testing.assert_array_equal(h1, h2)


def test_init_forget_bias():
    p0 = init(np.random.default_rng(3), config(2, 5))
    p1 = init(np.random.default_rng(3), config(2, 5), forget_bias=1.0)
    np.testing.assert_allclose(p1["b"] - p0["b"], np.r_[np.zeros(5), np.ones(5), np.zeros(10)])
    assert np.all(np.abs(p0["W"]) <= 5**-0.5) and np.all(np.abs(p0["U"]) <= 5**-0.5)


def test_config_rejects_bad_sizes():
    with pytest.raises(ValueError):
        config(0, 4)
    with pytest.raises(ValueError):
        config(3, 0)


def flop_case(c):
    """The forward call of a config of shapes.json, for aik flopcount."""
    cfg, k = cfg_of(c), c["config"]
    rng = np.random.default_rng(0)
    params = init(rng, cfg)
    x = rng.normal(size=(k["B"], k["T"], k["I"]))  # (B, T, I)
    return forward, (params, x)


def test_default_config_is_the_example():
    assert config() == cfg_of(SHAPES[0])
