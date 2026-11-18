# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/gated-recurrent-unit/ (ai://XAGPFN).
"""Tests of the NumPy reference: the worked example, the shape contract, the shared fixture and the properties."""

import json
import pathlib

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from gru import (
    backward,
    cell,
    config,
    flops,
    forward,
    init,
    lstm_param_count,
    param_count,
    sigmoid,
    tanh_param_count,
)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]
RESETS = ("after", "before")


def cfg_of(c):
    return config(c["config"]["input_size"], c["config"]["hidden_size"])


# The worked example of the page: one unit, update bias 2, candidate weight 1 on the input, nothing else.
LEAKY = {
    "W": np.array([[0.0, 0.0, 1.0]]),
    "U": np.zeros((1, 3)),
    "b_i": np.array([0.0, 2.0, 0.0]),
    "b_h": np.zeros(3),
}


@pytest.mark.parametrize("reset", RESETS)
def test_worked_example_leaky_integrator(reset):
    z, n = sigmoid(np.array(2.0)), np.tanh(1.0)
    h, cache = forward(LEAKY, np.ones((1, 10, 1)), np.zeros((1, 1)), reset)
    np.testing.assert_allclose(cache["z"], z, rtol=0, atol=1e-15)
    np.testing.assert_allclose(h[0, 0, 0], 0.0907842488, rtol=0, atol=1e-10)
    np.testing.assert_allclose(h[0, 7, 0], 0.4857076084, rtol=0, atol=1e-10)
    np.testing.assert_allclose(h[0, :, 0], n * (1.0 - z ** np.arange(1, 11)), rtol=1e-14)
    np.testing.assert_allclose(1.0 / (1.0 - z), 1.0 + np.exp(2.0), rtol=1e-14)  # the time constant, 8.39 steps
    # The gradient of h_10 with respect to h_0 is z^10 = 0.281: the only path is the direct one of eq-update.
    _, _, dh0 = backward(LEAKY, cache, np.zeros((1, 10, 1)), np.ones((1, 1)))
    np.testing.assert_allclose(dh0[0, 0], z**10, rtol=1e-14)
    np.testing.assert_allclose(dh0[0, 0], 0.2810338623, rtol=0, atol=1e-10)


def test_worked_example_two_resets():
    # One unit, x = h = 1: a = (1, 0, 1), h U_n = 1 and b_hn = 1, so r = sigmoid(1) and z = 1/2.
    p = {
        "W": np.array([[1.0, 0.0, 1.0]]),
        "U": np.array([[0.0, 0.0, 1.0]]),
        "b_i": np.zeros(3),
        "b_h": np.array([0.0, 0.0, 1.0]),
    }
    one = np.ones((1, 1))
    h_after, s_after = cell(p, one, one, "after")  # n = tanh(1 + r (1 + 1))
    h_before, s_before = cell(p, one, one, "before")  # n = tanh(1 + r + 1)
    np.testing.assert_allclose(s_after["r"][0, 0], 0.7310585786, rtol=0, atol=1e-10)
    np.testing.assert_allclose(s_after["n"][0, 0], np.tanh(1.0 + 2.0 * s_after["r"][0, 0]), rtol=1e-15)
    np.testing.assert_allclose(s_before["n"][0, 0], np.tanh(2.0 + s_before["r"][0, 0]), rtol=1e-15)
    np.testing.assert_allclose(h_after[0, 0], 0.9927841575, rtol=0, atol=1e-10)
    np.testing.assert_allclose(h_before[0, 0], 0.9957733803, rtol=0, atol=1e-10)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_shapes(c):
    cfg, k, blocks = cfg_of(c), c["config"], c["blocks"]
    rng = np.random.default_rng(0)
    params = init(rng, cfg)
    for reset in RESETS:
        h, cache = forward(params, rng.normal(size=(k["B"], k["T"], k["I"])), reset=reset)
        assert list(h.shape) == blocks["layer"]["outputs"]["h"]
        assert list(cache["h_T"].shape) == blocks["layer"]["outputs"]["h_T"]
        for name in ("a", "m"):
            assert list(cache[name].shape) == blocks["gates"]["outputs"][name], name
        for name, shape in blocks["cell"]["internals"].items():
            assert list(cache[name].shape) == shape, name


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_param_count_and_flops(c):
    cfg, k, m = cfg_of(c), c["config"], c["model"]
    assert param_count(cfg) == m["params"]
    assert flops(cfg, k["B"], k["T"]) == m["flops_forward"]
    assert sum(a.size for a in init(np.random.default_rng(0), cfg).values()) == m["params"]
    assert lstm_param_count(cfg) == m["lstm_params"]
    assert lstm_param_count(cfg, peepholes=True) == m["lstm_peephole_params"]
    assert tanh_param_count(cfg) == m["tanh_params"]


def test_chung_table_1_speech_sizes():
    # The speech models of Table 1 of Chung et al. read 20 samples per step; counting the recurrent layer alone with
    # one bias per gate gives their counts, and the LSTM needs its three peephole vectors to reach 169.1 thousand.
    gru, lstm, tanh = config(20, 227), config(20, 195), config(20, 400)
    assert param_count(gru) - 3 * 227 == 168_888  # 168.9e3
    assert lstm_param_count(lstm, peepholes=True) == 169_065  # 169.1e3
    assert lstm_param_count(lstm) == 168_480
    assert tanh_param_count(tanh) == 168_400  # 168.4e3


def fixture():
    params = dict(np.load(FIXTURES / "params-small.npz"))
    inp = np.load(FIXTURES / "input-small.npz")
    want = np.load(FIXTURES / "expected-small.npz")
    return params, inp, want


@pytest.mark.parametrize("reset", RESETS)
def test_shared_fixture(reset):
    params, inp, want = fixture()
    sfx = "" if reset == "after" else "_before"
    h, cache = forward(params, inp["x"], inp["h0"], reset)
    for name in ("h", "h_T", "r", "z", "n"):
        np.testing.assert_allclose(cache[name], want[name + sfx], rtol=1e-12, atol=1e-14, err_msg=name)
    dx, dp, dh0 = backward(params, cache, inp["r"], inp["r_h"])
    got = {"dx": dx, "dh0": dh0} | {"d" + k: v for k, v in dp.items()}
    for name, v in got.items():
        np.testing.assert_allclose(v, want[name + sfx], rtol=1e-12, atol=1e-14, err_msg=name)


def small(seed=1, n_in=3, H=4, B=2, T=4):
    rng = np.random.default_rng(seed)
    params = init(rng, config(n_in, H))
    return params, rng.normal(size=(B, T, n_in)), rng.normal(0, 0.5, (B, H)), rng


def loss_of(params, x, h0, r, r_h, reset):
    h, cache = forward(params, x, h0, reset)
    return np.sum(h * r) + np.sum(cache["h_T"] * r_h)


@pytest.mark.parametrize("reset", RESETS)
@pytest.mark.parametrize("T", [1, 4])
def test_gradient(T, reset):
    """Central finite differences in float64 on every parameter, input and initial state entry."""
    params, x, h0, rng = small(T=T)
    r, r_h = rng.normal(size=(2, T, 4)), rng.normal(size=(2, 4))
    _, cache = forward(params, x, h0, reset)
    dx, dp, dh0 = backward(params, cache, r, r_h)

    def check(arr, grad):
        num = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            old = arr[idx]
            eps = 1e-6 * max(1.0, abs(old))
            arr[idx] = old + eps
            up = loss_of(params, x, h0, r, r_h, reset)
            arr[idx] = old - eps
            down = loss_of(params, x, h0, r, r_h, reset)
            arr[idx] = old
            num[idx] = (up - down) / (2 * eps)
        np.testing.assert_allclose(grad, num, rtol=1e-6, atol=1e-8)

    for name in params:
        check(params[name], dp[name])
    check(x, dx)
    check(h0, dh0)


@settings(max_examples=30, deadline=None)
@given(st.integers(0, 2**31 - 1), st.floats(-5.0, 5.0), st.sampled_from(RESETS))
def test_gates_in_unit_interval_and_state_in_the_box(seed, scale, reset):
    # h_t is a convex combination of h_{t-1} and n_t, and |n_t| <= 1, so a state that starts in [-1, 1] stays there.
    params, x, h0, _ = small(seed, T=6)
    h0 = np.clip(h0, -1.0, 1.0)
    _, cache = forward(params, scale * x, h0, reset)
    for name in ("r", "z"):
        assert np.all((cache[name] >= 0.0) & (cache[name] <= 1.0)), name
    assert np.all(np.abs(cache["n"]) <= 1.0)
    assert np.all(np.abs(cache["h"]) <= 1.0)
    prev = np.concatenate([h0[:, None], cache["h"][:, :-1]], axis=1)  # (B, T, H)
    lo, hi = np.minimum(prev, cache["n"]), np.maximum(prev, cache["n"])
    assert np.all((cache["h"] >= lo - 1e-15) & (cache["h"] <= hi + 1e-15))


def test_sigmoid_is_stable():
    z = np.array([-800.0, -50.0, 0.0, 50.0, 800.0])
    s = sigmoid(z)
    assert np.all(np.isfinite(s))
    np.testing.assert_allclose(s, [0.0, np.exp(-50.0) / (1 + np.exp(-50.0)), 0.5, 1.0, 1.0], rtol=1e-15, atol=0)
    np.testing.assert_allclose(sigmoid(-z), 1.0 - s, atol=1e-15)


@pytest.mark.parametrize("reset", RESETS)
def test_copy_when_update_is_one(reset):
    # An update bias of +800 gives z = 1 exactly in float64: the state is h_0 bit for bit, whatever the inputs.
    params, x, h0, _ = small(T=50)
    params["b_i"][4:8] = 800.0
    _, cache = forward(params, 3.0 * x, h0, reset)
    assert np.all(cache["z"] == 1.0)
    for t in range(50):
        assert np.array_equal(cache["h"][:, t], h0)


@pytest.mark.parametrize("reset", RESETS)
def test_update_zero_and_reset_one_is_a_tanh_rnn(reset):
    # z = 0 and r = 1 turn the GRU into h_t = tanh(x_t W_n + b_in + h_{t-1} U_n + b_hn), the plain recurrent network.
    params, x, h0, _ = small(T=5)
    params["b_i"][:4] = 800.0  # r = 1
    params["b_i"][4:8] = -800.0  # z = 0
    h, _ = forward(params, x, h0, reset)
    W_n, U_n, b = params["W"][:, 8:], params["U"][:, 8:], params["b_i"][8:] + params["b_h"][8:]
    h_t = h0
    for t in range(5):
        h_t = np.tanh(x[:, t] @ W_n + h_t @ U_n + b)
        np.testing.assert_allclose(h[:, t], h_t, rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("reset", RESETS)
def test_reset_zero_forgets_the_state_in_the_candidate(reset):
    # With r = 0 the candidate does not depend on h_{t-1}: n = tanh(x W_n + b_in + c), with c = 0 after and b_hn before.
    params, x, h0, _ = small(T=1)
    params["b_i"][:4] = -800.0
    _, a = forward(params, x, h0, reset)
    _, b = forward(params, x, h0 + 5.0, reset)
    np.testing.assert_array_equal(a["n"], b["n"])
    c = 0.0 if reset == "after" else params["b_h"][8:]
    np.testing.assert_allclose(a["n"][:, 0], np.tanh(x[:, 0] @ params["W"][:, 8:] + params["b_i"][8:] + c), rtol=1e-14)


@pytest.mark.parametrize("reset", RESETS)
def test_jacobian_without_recurrent_weights_is_diag_z(reset):
    # With U = 0 the gates and the candidate do not see h_{t-1}, and dh_t/dh_{t-1} is the diagonal matrix of z_t.
    params, x, h0, _ = small(T=1)
    params["U"][...] = 0.0
    _, cache = forward(params, x, h0, reset)
    for b in range(2):
        J = np.zeros((4, 4))
        for k in range(4):
            dhT = np.zeros((2, 4))
            dhT[b, k] = 1.0
            _, _, dh0 = backward(params, cache, np.zeros((2, 1, 4)), dhT)
            J[k] = dh0[b]
        np.testing.assert_allclose(J, np.diag(cache["z"][b, 0]), atol=1e-15)


def test_the_two_resets_agree_when_U_n_is_diagonal_and_b_hn_is_zero():
    params, x, h0, _ = small(T=6)
    params["U"][:, 8:] = np.diag(np.diag(params["U"][:, 8:]))
    params["b_h"][8:] = 0.0
    h_a, _ = forward(params, x, h0, "after")
    h_b, _ = forward(params, x, h0, "before")
    np.testing.assert_allclose(h_a, h_b, rtol=1e-14, atol=1e-15)
    params["U"][0, 9] = 0.3  # one off-diagonal entry is enough to tell them apart
    h_a, _ = forward(params, x, h0, "after")
    h_b, _ = forward(params, x, h0, "before")
    assert np.max(np.abs(h_a - h_b)) > 1e-3


@pytest.mark.parametrize("reset", RESETS)
def test_the_flipped_convention_is_the_same_family(reset):
    # Chung et al. write h_t = (1 - z) h_{t-1} + z n. Negating the weights and the biases of the update gate maps
    # z to 1 - z, so the two conventions describe the same functions with different parameters.
    params, x, h0, _ = small(T=5)
    h, cache = forward(params, x, h0, reset)
    flipped = {k: v.copy() for k, v in params.items()}
    for name in flipped:
        if name in ("W", "U"):
            flipped[name][:, 4:8] *= -1.0
        else:
            flipped[name][4:8] *= -1.0
    h_t = h0
    for t in range(5):
        _, step = cell(flipped, x[:, t], h_t, reset)  # the gates of the flipped parameters at the same state
        np.testing.assert_allclose(step["z"], 1.0 - cache["z"][:, t], rtol=1e-13, atol=1e-15)
        h_t = (1.0 - step["z"]) * h_t + step["z"] * step["n"]  # the update as Chung et al. write it
        np.testing.assert_allclose(h_t, h[:, t], rtol=1e-13, atol=1e-15)


def test_default_state_is_zero():
    params, x, _, _ = small()
    h1, _ = forward(params, x)
    h2, _ = forward(params, x, np.zeros((2, 4)))
    np.testing.assert_array_equal(h1, h2)


def test_unknown_reset_is_an_error():
    params, x, _, _ = small()
    with pytest.raises(ValueError):
        forward(params, x, reset="never")
