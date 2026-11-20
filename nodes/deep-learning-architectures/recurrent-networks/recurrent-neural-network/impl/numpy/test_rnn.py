# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/recurrent-neural-network/ (ai://FKJKTR).
"""Tests of the NumPy reference: the worked example, the shape contract, the shared fixture and the properties."""

import json
import pathlib

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rnn import (
    NONLINEARITIES,
    backward,
    cell,
    config,
    flops,
    forward,
    init,
    irnn_init,
    one_bias_param_count,
    param_count,
    step_jacobian,
)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def cfg_of(c):
    return config(c["config"]["input_size"], c["config"]["hidden_size"])


# The worked example of the page: one unit, input weight 1, recurrent weight 0.9, no bias, one impulse at t = 1.
SCALAR = {"W": np.array([[1.0]]), "U": np.array([[0.9]]), "b_i": np.zeros(1), "b_h": np.zeros(1)}


def impulse(T=10):
    x = np.zeros((1, T, 1))
    x[0, 0, 0] = 1.0
    return x


def test_worked_example_tanh():
    h, cache = forward(SCALAR, impulse(), np.zeros((1, 1)), "tanh")
    np.testing.assert_allclose(h[0, 0, 0], np.tanh(1.0), rtol=1e-15)
    np.testing.assert_allclose(h[0, 1, 0], 0.5950411899, rtol=0, atol=1e-10)
    np.testing.assert_allclose(h[0, 9, 0], 0.1892358120, rtol=0, atol=1e-10)
    # dh_10/dx_1 is the product of the ten local slopes 1 - h_t^2 and nine factors of the recurrent weight 0.9.
    dx, _, _ = backward(SCALAR, cache, np.zeros((1, 10, 1)), np.ones((1, 1)))
    hs = h[0, :, 0]
    np.testing.assert_allclose(dx[0, 0, 0], 0.9**9 * np.prod(1.0 - hs**2), rtol=1e-13)
    np.testing.assert_allclose(dx[0, 0, 0], 0.0419487706, rtol=0, atol=1e-10)


def test_worked_example_relu():
    # With relu the state is 0.9^(t-1) and the gradient is 0.9^9 = 0.387: only the recurrent weight shrinks it.
    h, cache = forward(SCALAR, impulse(), np.zeros((1, 1)), "relu")
    np.testing.assert_allclose(h[0, :, 0], 0.9 ** np.arange(10), rtol=1e-14)
    dx, _, _ = backward(SCALAR, cache, np.zeros((1, 10, 1)), np.ones((1, 1)))
    np.testing.assert_allclose(dx[0, 0, 0], 0.9**9, rtol=1e-14)
    np.testing.assert_allclose(dx[0, 0, 0], 0.387420489, rtol=0, atol=1e-12)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_shapes(c):
    cfg, k, blocks = cfg_of(c), c["config"], c["blocks"]
    rng = np.random.default_rng(0)
    params = init(rng, cfg)
    for nl in NONLINEARITIES:
        h, cache = forward(params, rng.normal(size=(k["B"], k["T"], k["I"])), nonlinearity=nl)
        assert list(h.shape) == blocks["layer"]["outputs"]["h"]
        assert list(cache["h_T"].shape) == blocks["layer"]["outputs"]["h_T"]
        assert list(cache["a"].shape) == blocks["step"]["internals"]["a"]


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_param_count_and_flops(c):
    cfg, k, m = cfg_of(c), c["config"], c["model"]
    assert param_count(cfg) == m["params"]
    assert one_bias_param_count(cfg) == m["one_bias_params"]
    assert flops(cfg, k["B"], k["T"]) == m["flops_forward"]
    assert sum(a.size for a in init(np.random.default_rng(0), cfg).values()) == m["params"]
    # In the torch form the GRU has exactly three times and the LSTM four times the parameters and the FLOPs.
    assert m["gru_params"] == 3 * m["params"] and m["lstm_params"] == 4 * m["params"]
    assert m["gru_flops"] == 3 * m["flops_forward"] and m["lstm_flops"] == 4 * m["flops_forward"]


def test_le_pixel_mnist_size():
    # 100 units reading one pixel per step: 100 + 10,000 + 200 parameters, and 784 steps per image.
    cfg = config(1, 100)
    assert param_count(cfg) == 10_300
    assert flops(cfg, 1, 784) == 2 * 784 * 100 * 101


def fixture():
    params = dict(np.load(FIXTURES / "params-small.npz"))
    inp = np.load(FIXTURES / "input-small.npz")
    want = np.load(FIXTURES / "expected-small.npz")
    return params, inp, want


@pytest.mark.parametrize("nl", NONLINEARITIES)
def test_shared_fixture(nl):
    params, inp, want = fixture()
    sfx = "" if nl == "tanh" else "_relu"
    h, cache = forward(params, inp["x"], inp["h0"], nl)
    for name in ("h", "h_T", "a"):
        np.testing.assert_allclose(cache[name], want[name + sfx], rtol=1e-12, atol=1e-14, err_msg=name)
    dx, dp, dh0 = backward(params, cache, inp["r"], inp["r_h"])
    got = {"dx": dx, "dh0": dh0} | {"d" + k: v for k, v in dp.items()}
    for name, v in got.items():
        np.testing.assert_allclose(v, want[name + sfx], rtol=1e-12, atol=1e-14, err_msg=name)


def small(seed=1, n_in=3, H=4, B=2, T=4):
    rng = np.random.default_rng(seed)
    params = init(rng, config(n_in, H))
    return params, rng.normal(size=(B, T, n_in)), rng.normal(0, 0.5, (B, H)), rng


def loss_of(params, x, h0, r, r_h, nl):
    h, cache = forward(params, x, h0, nl)
    return np.sum(h * r) + np.sum(cache["h_T"] * r_h)


@pytest.mark.parametrize("nl", NONLINEARITIES)
@pytest.mark.parametrize("T", [1, 5])
def test_gradient(T, nl):
    """Central finite differences in float64 on every parameter, input and initial state entry."""
    params, x, h0, rng = small(T=T)
    r, r_h = rng.normal(size=(2, T, 4)), rng.normal(size=(2, 4))
    _, cache = forward(params, x, h0, nl)
    if nl == "relu":  # finite differences need every pre-activation away from the kink of relu
        assert np.min(np.abs(cache["a"])) > 1e-4
    dx, dp, dh0 = backward(params, cache, r, r_h)

    def check(arr, grad):
        num = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            old = arr[idx]
            eps = 1e-6 * max(1.0, abs(old))
            arr[idx] = old + eps
            up = loss_of(params, x, h0, r, r_h, nl)
            arr[idx] = old - eps
            down = loss_of(params, x, h0, r, r_h, nl)
            arr[idx] = old
            num[idx] = (up - down) / (2 * eps)
        np.testing.assert_allclose(grad, num, rtol=1e-6, atol=1e-8)

    for name in params:
        check(params[name], dp[name])
    check(x, dx)
    check(h0, dh0)


@pytest.mark.parametrize("nl", NONLINEARITIES)
def test_shared_weight_gradient_is_the_sum_over_steps(nl):
    # Unroll the layer into T copies with their own weights: the gradient of the shared U is the sum of theirs.
    params, x, h0, rng = small(T=6)
    r = rng.normal(size=(2, 6, 4))
    _, cache = forward(params, x, h0, nl)
    _, dp, _ = backward(params, cache, r)
    total = np.zeros_like(params["U"])
    for s in range(6):
        eps, num = 1e-6, np.zeros_like(params["U"])
        for idx in np.ndindex(params["U"].shape):

            def run(delta):
                h_t, out = h0, 0.0
                for t in range(6):
                    p = dict(params)
                    if t == s:  # perturb the copy of U used at step s only  # noqa: B023
                        p["U"] = params["U"].copy()
                        p["U"][idx] += delta  # noqa: B023
                    h_t, _ = cell(p, x[:, t], h_t, nl)
                    out += np.sum(h_t * r[:, t])
                return out

            num[idx] = (run(eps) - run(-eps)) / (2 * eps)
        total += num
    np.testing.assert_allclose(dp["U"], total, rtol=1e-6, atol=1e-8)


def test_step_jacobian_matches_backward():
    params, x, h0, _ = small(T=1)
    for nl in NONLINEARITIES:
        _, cache = forward(params, x, h0, nl)
        for b in range(2):
            J = np.zeros((4, 4))
            for k in range(4):
                dhT = np.zeros((2, 4))
                dhT[b, k] = 1.0
                _, _, dh0 = backward(params, cache, np.zeros((2, 1, 4)), dhT)
                J[k] = dh0[b]
            np.testing.assert_allclose(J, step_jacobian(params, cache["a"][b, 0], cache["h"][b, 0], nl), atol=1e-15)


@settings(max_examples=30, deadline=None)
@given(st.integers(0, 2**31 - 1), st.floats(0.1, 3.0), st.integers(1, 12))
def test_gradient_bound(seed, scale, T):
    # Pascanu et al.: ||dh_T/dh_0|| <= (gamma ||U||_2)^T, with gamma = 1 the bound of the slope of tanh.
    params, x, h0, _ = small(seed, T=T)
    params["U"] = scale * params["U"]
    _, cache = forward(params, x, h0, "tanh")
    for b in range(2):
        J = np.eye(4)
        for t in range(T):
            J = step_jacobian(params, cache["a"][b, t], cache["h"][b, t], "tanh") @ J
        assert np.linalg.norm(J, 2) <= np.linalg.norm(params["U"], 2) ** T * (1 + 1e-12) + 1e-300


def test_vanishing_and_exploding():
    # A recurrent matrix of spectral norm 0.5 makes the gradient over 50 steps at most 0.5^50; with relu and an
    # orthogonal U scaled by 1.5 on a state that stays positive, the gradient grows as 1.5^50.
    rng = np.random.default_rng(3)
    Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    params = {"W": np.zeros((1, 4)), "U": 0.5 * Q, "b_i": np.zeros(4), "b_h": np.zeros(4)}
    x, h0 = np.zeros((1, 50, 1)), rng.normal(0, 0.5, (1, 4))
    _, cache = forward(params, x, h0, "tanh")
    _, _, dh0 = backward(params, cache, np.zeros((1, 50, 4)), np.ones((1, 4)))
    assert np.linalg.norm(dh0) <= 0.5**50 * 2.0 + 1e-300
    params = {"W": np.zeros((1, 4)), "U": 1.5 * np.eye(4), "b_i": np.zeros(4), "b_h": np.zeros(4)}
    _, cache = forward(params, x, np.abs(h0) + 0.1, "relu")
    _, _, dh0 = backward(params, cache, np.zeros((1, 50, 4)), np.ones((1, 4)))
    np.testing.assert_allclose(dh0, 1.5**50, rtol=1e-12)


def test_irnn_keeps_its_state_without_input():
    # Le et al.: with U = I, zero biases and no input, a relu network stays in the same state, and the error
    # derivatives stay constant when they are backpropagated; U = 0.9 I forgets as 0.9^t.
    cfg = config(2, 5)
    params = irnn_init(np.random.default_rng(4), cfg)
    assert np.array_equal(params["U"], np.eye(5)) and np.std(params["W"]) < 0.005
    h0 = np.abs(np.random.default_rng(5).normal(size=(3, 5)))
    _, cache = forward(params, np.zeros((3, 200, 2)), h0, "relu")
    assert np.array_equal(cache["h_T"], h0)
    dhT = np.random.default_rng(6).normal(size=(3, 5))
    _, _, dh0 = backward(params, cache, np.zeros((3, 200, 5)), dhT)
    assert np.array_equal(dh0, dhT)
    params = irnn_init(np.random.default_rng(4), cfg, scale=0.9)
    _, cache = forward(params, np.zeros((3, 20, 2)), h0, "relu")
    np.testing.assert_allclose(cache["h_T"], 0.9**20 * h0, rtol=1e-13)


@settings(max_examples=30, deadline=None)
@given(st.integers(0, 2**31 - 1), st.floats(-20.0, 20.0))
def test_tanh_state_is_bounded_and_relu_state_is_nonnegative(seed, scale):
    params, x, h0, _ = small(seed, T=6)
    _, cache = forward(params, scale * x, h0, "tanh")
    assert np.all(np.abs(cache["h"]) <= 1.0)
    _, cache = forward(params, scale * x, h0, "relu")
    assert np.all(cache["h"] >= 0.0)


def test_two_biases_only_enter_as_a_sum():
    params, x, h0, _ = small(T=5)
    moved = dict(params, b_i=params["b_i"] + 0.7, b_h=params["b_h"] - 0.7)
    for nl in NONLINEARITIES:
        np.testing.assert_allclose(forward(params, x, h0, nl)[0], forward(moved, x, h0, nl)[0], rtol=1e-14, atol=1e-15)


def test_permuting_units_is_a_symmetry():
    # Renaming the hidden units with a permutation P maps W to W P, U to P^T U P and the biases to b P.
    params, x, h0, _ = small(T=5)
    P = np.eye(4)[[2, 0, 3, 1]]
    perm = {"W": params["W"] @ P, "U": P.T @ params["U"] @ P, "b_i": params["b_i"] @ P, "b_h": params["b_h"] @ P}
    for nl in NONLINEARITIES:
        h, _ = forward(params, x, h0, nl)
        hp, _ = forward(perm, x, h0 @ P, nl)
        np.testing.assert_allclose(hp, h @ P, rtol=1e-14, atol=1e-15)


def test_default_state_is_zero():
    params, x, _, _ = small()
    h1, _ = forward(params, x)
    h2, _ = forward(params, x, np.zeros((2, 4)))
    np.testing.assert_array_equal(h1, h2)


def test_bad_config_is_an_error():
    with pytest.raises(ValueError):
        config(3, 4, "sigmoid")
    with pytest.raises(ValueError):
        config(0, 4)
    params, x, _, _ = small()
    with pytest.raises(ValueError):
        forward(params, x, nonlinearity="gelu")


def flop_case(c):
    """The forward call of a config of shapes.json, for aik flopcount."""
    cfg, k = cfg_of(c), c["config"]
    rng = np.random.default_rng(0)
    x = rng.normal(size=(k["B"], k["T"], k["input_size"]))  # (B, T, I)
    return forward, (init(rng, cfg), x)
