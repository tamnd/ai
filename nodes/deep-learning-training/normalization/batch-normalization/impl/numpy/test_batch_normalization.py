# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/batch-normalization/ (ai://NEW-batch-normalization).
"""Tests of the NumPy reference: the worked example, the gradient, the running statistics, evaluation mode, the
convolutional case and agreement with torch.nn.BatchNorm1d and torch.nn.BatchNorm2d when PyTorch is installed."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from batch_normalization import backward, config, fold, forward, init, init_state, update_state

# The worked example of the page: one feature, eps = 0, gamma = 2, beta = 1.
WX = np.array([[2.0], [4.0], [5.0], [6.0], [8.0]])
WY = [[-2.0], [0.0], [1.0], [2.0], [4.0]]
WG = np.array([[1.0], [0.0], [0.0], [0.0], [0.0]])
WDX = [[0.35], [-0.35], [-0.2], [-0.05], [0.25]]

# The shared fixture. The PyTorch and JAX tests hold the same numbers.
X = np.array([[0.5, -1.5, 2.0], [1.0, 3.0, -2.5], [-1.25, 0.5, 1.5], [2.0, -0.75, 0.25]])
DY = np.array([[1.0, 2.0, -1.0], [-2.0, 1.0, 0.0], [0.25, -0.5, 2.0], [1.5, 0.5, -3.0]])
GAMMA = np.array([1.5, 0.5, 2.0])
BETA = np.array([0.1, -0.2, 0.3])
X4 = ((np.arange(48.0) * 7) % 11 - 5).reshape(4, 3, 2, 2) / 2  # (N, C, H, W)
DY4 = ((np.arange(48.0) * 5) % 13 - 6).reshape(4, 3, 2, 2) / 3


def params():
    return {"gamma": GAMMA.copy(), "beta": BETA.copy()}


def numeric_grad(fun, a, h=1e-6):
    g = np.zeros_like(a)
    for idx in np.ndindex(*a.shape):
        e = np.zeros_like(a)
        e[idx] = h
        g[idx] = (fun(a + e) - fun(a - e)) / (2 * h)
    return g


def test_worked_example():
    cfg = config(1, eps=0.0, momentum=0.1)
    p = {"gamma": np.array([2.0]), "beta": np.array([1.0])}
    y, cache = forward(p, WX, cfg)
    np.testing.assert_allclose(y, WY, atol=1e-12)
    dx, grads = backward(p, cache, WG)
    np.testing.assert_allclose(dx, WDX, atol=1e-12)
    np.testing.assert_allclose(grads["gamma"], [-1.5], atol=1e-12)
    np.testing.assert_allclose(grads["beta"], [1.0], atol=1e-12)
    state = update_state(init_state(cfg), cache, cfg)
    np.testing.assert_allclose(state["mean"], [0.5], atol=1e-12)
    np.testing.assert_allclose(state["var"], [1.4], atol=1e-12)


@pytest.mark.parametrize(("x", "dy"), [(X, DY), (X4, DY4)])
def test_gradient_finite_differences(x, dy):
    cfg = config(3, eps=1e-3)
    p = params()

    def loss_x(a):
        return float(np.sum(dy * forward(p, a, cfg)[0]))

    def loss_gamma(g):
        return float(np.sum(dy * forward({"gamma": g, "beta": p["beta"]}, x, cfg)[0]))

    _, cache = forward(p, x, cfg)
    dx, grads = backward(p, cache, dy)
    np.testing.assert_allclose(dx, numeric_grad(loss_x, x), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(grads["gamma"], numeric_grad(loss_gamma, p["gamma"]), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(grads["beta"], dy.sum(axis=(0,) + tuple(range(2, dy.ndim))), rtol=1e-12)


def test_statistics_treated_as_constants_are_wrong():
    cfg = config(3)
    _, cache = forward(params(), X, cfg)
    dx, _ = backward(params(), cache, DY)
    naive = GAMMA * cache["inv"] * DY
    assert np.abs(dx - naive).max() > 0.1


@settings(max_examples=50, deadline=None)
@given(st.lists(st.floats(-10, 10), min_size=3, max_size=12), st.integers(0, 10**6))
def test_input_gradient_sums_to_zero_and_is_orthogonal(values, seed):
    x = np.array(values)[:, None]
    if x.std() < 1e-3:
        return
    cfg = config(1, eps=0.0)
    p = {"gamma": np.array([1.7]), "beta": np.array([0.3])}
    _, cache = forward(p, x, cfg)
    dy = np.random.default_rng(seed).normal(size=x.shape)
    dx, _ = backward(p, cache, dy)
    scale = np.abs(dx).sum() + np.abs(dy).sum()
    assert abs(dx.sum()) <= 1e-9 * scale
    assert abs(np.sum(dx * cache["xhat"])) <= 1e-9 * scale * max(1.0, np.abs(cache["xhat"]).max())


def test_eval_uses_running_stats():
    cfg = config(3, momentum=0.3)
    p, state = params(), init_state(cfg)
    for k in range(3):
        _, cache = forward(p, X + k, cfg)
        state = update_state(state, cache, cfg)
    y, cache = forward(p, X, cfg, state=state, training=False)
    expected = GAMMA * (X - state["mean"]) / np.sqrt(state["var"] + cfg["eps"]) + BETA
    np.testing.assert_allclose(y, expected, rtol=1e-12)
    # Each example is treated on its own: one row alone gives the same output as in the batch.
    y1, _ = forward(p, X[1:2], cfg, state=state, training=False)
    np.testing.assert_allclose(y1, y[1:2], rtol=1e-12)
    dx, _ = backward(p, cache, DY)
    np.testing.assert_allclose(dx, GAMMA / np.sqrt(state["var"] + cfg["eps"]) * DY, rtol=1e-12)
    assert update_state(state, cache, cfg) is state


def test_momentum_convention():
    alpha = 0.1
    cfg = config(3, momentum=alpha)
    state = init_state(cfg)
    rho = 1 - alpha  # the Keras or Flax momentum
    keras_mean, keras_var = np.zeros(3), np.ones(3)
    for k in range(4):
        x = X * (k + 1) - k
        _, cache = forward(params(), x, cfg)
        state = update_state(state, cache, cfg)
        keras_mean = rho * keras_mean + (1 - rho) * x.mean(axis=0)
        keras_var = rho * keras_var + (1 - rho) * x.var(axis=0, ddof=1)
    np.testing.assert_allclose(state["mean"], keras_mean, rtol=1e-12)
    np.testing.assert_allclose(state["var"], keras_var, rtol=1e-12)


def test_conv_per_channel():
    cfg = config(3, eps=0.0)
    y, cache = forward({"gamma": np.ones(3), "beta": np.zeros(3)}, X4, cfg)
    np.testing.assert_allclose(y.mean(axis=(0, 2, 3)), 0.0, atol=1e-12)
    np.testing.assert_allclose(y.var(axis=(0, 2, 3)), 1.0, rtol=1e-12)
    assert cache["count"] == 4 * 2 * 2
    per_pixel = (X4 - X4.mean(axis=0)) / X4.std(axis=0)
    assert np.abs(per_pixel - y).max() > 0.1


def test_single_value_rejected():
    cfg = config(3)
    with pytest.raises(ValueError, match="more than 1 value"):
        forward(params(), X[:1], cfg)
    y, _ = forward(params(), X4[:1], cfg)  # one feature map still has 4 values per channel
    assert y.shape == X4[:1].shape


def test_scale_invariance():
    rng = np.random.default_rng(20261114)
    u, w = rng.normal(size=(8, 5)), rng.normal(size=(3, 5))
    cfg = config(3, eps=0.0)
    y1, _ = forward(params(), u @ w.T, cfg)
    y2, _ = forward(params(), u @ (4.0 * w).T, cfg)
    np.testing.assert_allclose(y1, y2, rtol=1e-12, atol=1e-12)


def test_fold():
    rng = np.random.default_rng(20261114)
    cfg = config(3)
    state = {"mean": rng.normal(size=3), "var": rng.uniform(0.5, 2.0, size=3), "count": 5}
    w, b, z = rng.normal(size=(3, 4)), rng.normal(size=3), rng.normal(size=(6, 4))
    y, _ = forward(params(), z @ w.T + b, cfg, state=state, training=False)
    w2, b2 = fold(params(), state, cfg, w, b)
    np.testing.assert_allclose(z @ w2.T + b2, y, rtol=1e-12)


def test_running_stats_match_torch():
    torch = pytest.importorskip("torch")
    for x, cls in [(X, torch.nn.BatchNorm1d), (X4, torch.nn.BatchNorm2d)]:
        for momentum in [0.1, None]:
            cfg = config(3, momentum=momentum)
            ref = cls(3, momentum=momentum).double()
            state = init_state(cfg)
            for k in range(3):
                xk = x * (k + 1) + k
                y, cache = forward({"gamma": np.ones(3), "beta": np.zeros(3)}, xk, cfg)
                state = update_state(state, cache, cfg)
                yr = ref(torch.tensor(xk)).detach().numpy()
                np.testing.assert_allclose(y, yr, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(state["mean"], ref.running_mean.numpy(), rtol=1e-12)
            np.testing.assert_allclose(state["var"], ref.running_var.numpy(), rtol=1e-12)
            ref.eval()
            ye, _ = forward({"gamma": np.ones(3), "beta": np.zeros(3)}, x, cfg, state=state, training=False)
            np.testing.assert_allclose(ye, ref(torch.tensor(x)).detach().numpy(), rtol=1e-10, atol=1e-12)


def test_config_errors():
    with pytest.raises(ValueError):
        config(0)
    with pytest.raises(ValueError):
        config(3, momentum=1.5)
    with pytest.raises(ValueError):
        forward(params(), np.zeros((4, 2)), config(3))
    with pytest.raises(ValueError):
        forward(params(), X, config(3), training=False)
