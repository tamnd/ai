# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/multilayer-perceptron/ (ai://YA59M7).
"""Tests of the NumPy reference: the XOR example, the gradient, the linear collapse, the shapes, He initialization and training."""

import json
import math
import pathlib

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from multilayer_perceptron import (
    ACTIVATIONS,
    backward,
    config,
    flops,
    forward,
    init,
    mse_loss,
    param_count,
    sgd_step,
    train,
    widths,
    xor_params,
)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]

# The four points of XOR and the table of the worked example: z = W_0 x + b_0, h = relu(z), y = W_1 h + b_1.
X_XOR = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
T_XOR = np.array([[0.0], [1.0], [1.0], [0.0]])
Z_XOR = np.array([[0.0, -1.0], [1.0, 0.0], [1.0, 0.0], [2.0, 1.0]])
H_XOR = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 1.0]])


def cfg_of(c, activation="tanh"):
    k = c["config"]
    return config(k["d_in"], k["d_hidden"], k["d_out"], k["n_hidden"], activation)


def random_params(rng, cfg):
    params = init(rng, cfg, "glorot")
    params["b"] = [rng.normal(0.0, 0.5, b.shape) for b in params["b"]]
    return params


def test_worked_example():
    y, cache = forward(xor_params(), X_XOR, config(2, 2, 1, 1, "relu"))
    np.testing.assert_array_equal(cache["z"][0], Z_XOR)
    np.testing.assert_array_equal(cache["h"][1], H_XOR)
    np.testing.assert_array_equal(y, T_XOR)


def test_linear_model_cannot_fit_xor():
    # The least-squares affine fit to XOR predicts 1/2 at every point; no affine map separates the classes.
    A = np.hstack([X_XOR, np.ones((4, 1))])
    coef, *_ = np.linalg.lstsq(A, T_XOR, rcond=None)
    np.testing.assert_allclose(A @ coef, np.full((4, 1), 0.5), atol=1e-12)
    np.testing.assert_allclose(coef.ravel(), [0.0, 0.0, 0.5], atol=1e-12)


@pytest.mark.parametrize("activation", ["relu", "tanh", "sigmoid", "gelu", "identity"])
def test_gradient_finite_differences(activation):
    rng = np.random.default_rng(1)
    cfg = config(3, 5, 2, 2, activation)
    params = random_params(rng, cfg)
    x, t = rng.normal(size=(4, 3)), rng.normal(size=(4, 2))

    def loss_of(p, xx):
        return mse_loss(forward(p, xx, cfg)[0], t)[0]

    y, cache = forward(params, x, cfg)
    dx, grads = backward(params, cache, mse_loss(y, t)[1])
    eps = 1e-6
    for key in ("W", "b"):
        for l, p in enumerate(params[key]):  # noqa: E741
            for idx in np.ndindex(p.shape):
                plus = {k: [a.copy() for a in v] for k, v in params.items()}
                minus = {k: [a.copy() for a in v] for k, v in params.items()}
                plus[key][l][idx] += eps
                minus[key][l][idx] -= eps
                numeric = (loss_of(plus, x) - loss_of(minus, x)) / (2 * eps)
                assert grads[key][l][idx] == pytest.approx(numeric, rel=1e-5, abs=1e-8)
    for idx in np.ndindex(x.shape):
        xp, xm = x.copy(), x.copy()
        xp[idx] += eps
        xm[idx] -= eps
        assert dx[idx] == pytest.approx((loss_of(params, xp) - loss_of(params, xm)) / (2 * eps), rel=1e-5, abs=1e-8)


def test_linear_collapse():
    # With the identity activation, three layers are one affine map: W = W_2 W_1 W_0, b = W_2 (W_1 b_0 + b_1) + b_2.
    rng = np.random.default_rng(2)
    cfg = config(3, 6, 2, 2, "identity")
    p = random_params(rng, cfg)
    x = rng.normal(size=(7, 3))
    W = p["W"][2] @ p["W"][1] @ p["W"][0]
    b = p["W"][2] @ (p["W"][1] @ p["b"][0] + p["b"][1]) + p["b"][2]
    np.testing.assert_allclose(forward(p, x, cfg)[0], x @ W.T + b, rtol=1e-12, atol=1e-12)
    assert np.linalg.matrix_rank(W) <= 2


def test_nonlinearity_breaks_collapse():
    rng = np.random.default_rng(2)
    cfg = config(3, 6, 2, 2, "relu")
    p = random_params(rng, cfg)
    x = rng.normal(size=(40, 3))
    y = forward(p, x, cfg)[0]
    A = np.hstack([x, np.ones((40, 1))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    assert np.abs(A @ coef - y).max() > 1e-3


@pytest.mark.parametrize("c", SHAPES, ids=[c["name"] for c in SHAPES])
def test_shape_contract(c):
    cfg, k = cfg_of(c), c["config"]
    params = init(np.random.default_rng(0), cfg)
    y, cache = forward(params, np.zeros((k["B"], k["d_in"])), cfg)
    assert list(y.shape) == c["blocks"]["output"]["outputs"]["y"]
    assert list(cache["z"][0].shape) == c["blocks"]["first"]["internals"]["z"]
    assert all(list(z.shape) == c["blocks"]["hidden"]["internals"]["z"] for z in cache["z"][1:])
    assert param_count(cfg) == c["model"]["params"] == sum(a.size for v in params.values() for a in v)
    assert flops(cfg, k["B"]) == c["model"]["flops_forward"]


@settings(max_examples=25, deadline=None)
@given(st.integers(1, 6), st.integers(1, 5), st.integers(1, 3), st.integers(1, 3), st.sampled_from(sorted(ACTIVATIONS)))
def test_batch_independence(B, d_hidden, d_out, n_hidden, activation):
    rng = np.random.default_rng(B * 100 + d_hidden)
    cfg = config(3, d_hidden, d_out, n_hidden, activation)
    p = random_params(rng, cfg)
    x = rng.normal(size=(B, 3))
    y = forward(p, x, cfg)[0]
    assert y.shape == (B, d_out)
    rows = np.vstack([forward(p, x[i : i + 1], cfg)[0] for i in range(B)])
    np.testing.assert_allclose(y, rows, rtol=1e-12, atol=1e-14)


def test_activations():
    z = np.array([-2.0, -0.5, 0.0, 1.0, 3.0])
    act = {k: f(z) for k, (f, _) in ACTIVATIONS.items()}
    np.testing.assert_array_equal(act["relu"], [0.0, 0.0, 0.0, 1.0, 3.0])
    np.testing.assert_allclose(act["sigmoid"], 1.0 / (1.0 + np.exp(-z)), rtol=1e-14)
    np.testing.assert_allclose(act["tanh"], np.tanh(z), rtol=1e-15)
    gelu = [x * 0.5 * (1.0 + math.erf(x / math.sqrt(2.0))) for x in z]
    np.testing.assert_allclose(act["gelu"], gelu, rtol=1e-14)
    assert act["gelu"][3] == pytest.approx(0.8413447460685429, rel=1e-14)
    np.testing.assert_array_equal(act["identity"], z)
    np.testing.assert_array_equal(ACTIVATIONS["relu"][1](z), [0.0, 0.0, 0.0, 1.0, 1.0])


def test_he_variance_preserved():
    # eq-he keeps the variance of the pre-activations of a deep ReLU network about constant from layer to layer.
    rng = np.random.default_rng(3)
    cfg = config(256, 256, 256, 10, "relu")
    x = rng.normal(size=(2000, 256))
    _, cache = forward(init(rng, cfg, "he"), x, cfg)
    v = np.array([z.var() for z in cache["z"]])
    assert v[0] == pytest.approx(2.0, rel=0.1)  # Var[z_1] = n Var[w] E[x^2] = 2 for a standard Gaussian input
    assert np.all(np.abs(v[1:] / v[:-1] - 1.0) < 0.15)  # each layer keeps the variance within 15%
    assert 0.5 < v[-1] / v[0] < 2.0  # and the drift over 9 layers stays far from the 2^-9 of test_half_variance_shrinks


def test_half_variance_shrinks():
    # With variance 1 / n_in instead of 2 / n_in, each ReLU layer halves the variance: the factor of 2 matters.
    rng = np.random.default_rng(3)
    cfg = config(256, 256, 256, 10, "relu")
    params = init(rng, cfg, "he")
    params["W"] = [w / math.sqrt(2.0) for w in params["W"]]
    _, cache = forward(params, rng.normal(size=(2000, 256)), cfg)
    ratio = cache["z"][-1].var() / cache["z"][0].var()
    assert ratio == pytest.approx(2.0**-9, rel=0.3)


def test_init_variances():
    rng = np.random.default_rng(4)
    cfg = config(300, 500, 100, 1)
    he, gl = init(rng, cfg, "he"), init(rng, cfg, "glorot")
    assert he["W"][0].var() == pytest.approx(2.0 / 300, rel=0.03)
    assert gl["W"][0].var() == pytest.approx(2.0 / 800, rel=0.03)
    assert gl["W"][1].var() == pytest.approx(2.0 / 600, rel=0.03)
    assert np.abs(gl["W"][0]).max() <= math.sqrt(6.0 / 800)
    assert he["W"][1].shape == (100, 500) and not np.any(he["b"][0])


def test_training_reduces_loss():
    cfg = config(2, 4, 1, 1, "tanh")
    params = init(np.random.default_rng(0), cfg, "glorot")
    trained, losses = train(params, X_XOR, T_XOR, cfg, lr=0.5, steps=500)
    assert losses[-1] < 1e-3 * losses[0]
    assert losses[100] < 0.01 * losses[0]
    np.testing.assert_allclose(forward(trained, X_XOR, cfg)[0], T_XOR, atol=0.05)


def test_sgd_step():
    p = xor_params()
    g = {"W": [np.ones((2, 2)), np.ones((1, 2))], "b": [np.ones(2), np.ones(1)]}
    q = sgd_step(p, g, 0.25)
    np.testing.assert_array_equal(q["W"][1], [[0.75, -2.25]])
    np.testing.assert_array_equal(q["b"][0], [-0.25, -1.25])
    np.testing.assert_array_equal(p["W"][1], [[1.0, -2.0]])  # the old parameters are unchanged


def test_shared_fixture():
    raw = dict(np.load(FIXTURES / "params-small.npz"))
    params = {k: [raw[f"{k}/{i}"] for i in range(3)] for k in ("W", "b")}
    inp, want = np.load(FIXTURES / "input-small.npz"), np.load(FIXTURES / "expected-small.npz")
    cfg = cfg_of(SHAPES[0])
    y, cache = forward(params, inp["x"], cfg)
    np.testing.assert_allclose(y, want["y"], rtol=1e-12)
    loss, dy = mse_loss(y, inp["t"])
    assert loss == pytest.approx(float(want["loss"]), rel=1e-12)
    dx, grads = backward(params, cache, dy)
    np.testing.assert_allclose(dx, want["dx"], rtol=1e-12)
    for k in ("W", "b"):
        for i in range(3):
            np.testing.assert_allclose(grads[k][i], want[f"d{k}/{i}"], rtol=1e-12)


def test_config_errors():
    assert widths(config(3, 4, 2, 2)) == [3, 4, 4, 2]
    with pytest.raises(ValueError):
        config(0, 4, 2, 1)
    with pytest.raises(ValueError):
        config(3, 4, 2, 0)
    with pytest.raises(ValueError):
        config(3, 4, 2, 1, "softplus")
    with pytest.raises(ValueError):
        init(np.random.default_rng(0), config(), "lecun")


def flop_case(c):
    """The forward call of a config of shapes.json, for aik flopcount."""
    cfg, k = cfg_of(c), c["config"]
    rng = np.random.default_rng(0)
    return forward, (init(rng, cfg), rng.normal(size=(k["B"], k["d_in"])), cfg)
