# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/batch-normalization/ (ai://TD3H76).
"""Tests of the JAX reference: the worked example, jax.grad against eq-bn-backward and finite differences, the running
statistics, evaluation mode and the convolutional case."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from batch_normalization import apply, config, init, init_state

jax.config.update("jax_enable_x64", True)

# The worked example of the page.
WX = [[2.0], [4.0], [5.0], [6.0], [8.0]]
WY = [[-2.0], [0.0], [1.0], [2.0], [4.0]]
WG = [[1.0], [0.0], [0.0], [0.0], [0.0]]
WDX = [[0.35], [-0.35], [-0.2], [-0.05], [0.25]]

# The shared fixture of the NumPy tests.
X = [[0.5, -1.5, 2.0], [1.0, 3.0, -2.5], [-1.25, 0.5, 1.5], [2.0, -0.75, 0.25]]
DY = [[1.0, 2.0, -1.0], [-2.0, 1.0, 0.0], [0.25, -0.5, 2.0], [1.5, 0.5, -3.0]]
GAMMA = [1.5, 0.5, 2.0]
BETA = [0.1, -0.2, 0.3]


def a(v):
    return jnp.asarray(v, dtype=jnp.float64)


def x4():
    return ((jnp.arange(48.0) * 7) % 11 - 5).reshape(4, 3, 2, 2) / 2


def params():
    return {"gamma": a(GAMMA), "beta": a(BETA)}


def test_worked_example():
    cfg = config(1, eps=0.0)
    p = {"gamma": a([2.0]), "beta": a([1.0])}
    y, state = apply(p, init_state(cfg), a(WX), cfg)
    np.testing.assert_allclose(y, WY, atol=1e-12)
    dx = jax.grad(lambda x: jnp.sum(apply(p, init_state(cfg), x, cfg)[0] * a(WG)))(a(WX))
    np.testing.assert_allclose(dx, WDX, atol=1e-12)
    dp = jax.grad(lambda q: jnp.sum(apply(q, init_state(cfg), a(WX), cfg)[0] * a(WG)))(p)
    np.testing.assert_allclose(dp["gamma"], [-1.5], atol=1e-12)
    np.testing.assert_allclose(dp["beta"], [1.0], atol=1e-12)
    np.testing.assert_allclose(state["mean"], [0.5], atol=1e-12)
    np.testing.assert_allclose(state["var"], [1.4], atol=1e-12)


def test_backward_formula():
    cfg = config(3)
    x, g = a(X), a(DY)
    dx = jax.grad(lambda z: jnp.sum(apply(params(), init_state(cfg), z, cfg)[0] * g))(x)
    m = 4
    s = jnp.sqrt(jnp.var(x, axis=0) + cfg["eps"])
    xhat = (x - x.mean(0)) / s
    expected = a(GAMMA) / (m * s) * (m * g - g.sum(0) - xhat * (g * xhat).sum(0))
    np.testing.assert_allclose(dx, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dx.sum(0), 0.0, atol=1e-12)


@pytest.mark.parametrize("make", [lambda: a(X), x4])
def test_gradient_finite_differences(make):
    cfg = config(3, eps=1e-3)

    def loss(x, gamma, beta):
        y, _ = apply({"gamma": gamma, "beta": beta}, init_state(cfg), x, cfg)
        return jnp.sum(y * jnp.sin(x))

    check_grads(loss, (make(), a(GAMMA), a(BETA)), order=2, modes=["rev"], atol=1e-6, rtol=1e-6)


def test_running_stats_closed_form():
    cfg = config(3, momentum=0.3)
    state = init_state(cfg)
    mean, var = np.zeros(3), np.ones(3)
    for k in range(3):
        x = np.array(X) * (k + 1) + k
        _, state = apply(params(), state, a(x), cfg)
        mean = 0.7 * mean + 0.3 * x.mean(0)
        var = 0.7 * var + 0.3 * x.var(0, ddof=1)
    np.testing.assert_allclose(state["mean"], mean, rtol=1e-12)
    np.testing.assert_allclose(state["var"], var, rtol=1e-12)


def test_momentum_convention():
    alpha, rho = 0.01, 0.99  # PyTorch alpha and the Flax default rho
    cfg = config(3, momentum=alpha)
    _, state = apply(params(), init_state(cfg), a(X), cfg)
    x = np.array(X)
    np.testing.assert_allclose(state["mean"], rho * 0.0 + (1 - rho) * x.mean(0), rtol=1e-12)
    np.testing.assert_allclose(state["var"], rho * 1.0 + (1 - rho) * x.var(0, ddof=1), rtol=1e-12)


def test_eval_uses_running_stats():
    cfg = config(3)
    state = {"mean": a([0.3, -0.1, 0.7]), "var": a([1.2, 0.4, 2.5])}
    y, new_state = apply(params(), state, a(X), cfg, training=False)
    expected = a(GAMMA) * (a(X) - state["mean"]) / jnp.sqrt(state["var"] + cfg["eps"]) + a(BETA)
    np.testing.assert_allclose(y, expected, rtol=1e-12)
    assert new_state is state
    y1, _ = apply(params(), state, a(X)[1:2], cfg, training=False)
    np.testing.assert_allclose(y1, y[1:2], rtol=1e-12)


def test_conv_per_channel():
    cfg = config(3, eps=0.0)
    y, _ = apply({"gamma": jnp.ones(3), "beta": jnp.zeros(3)}, init_state(cfg), x4(), cfg)
    np.testing.assert_allclose(y.mean(axis=(0, 2, 3)), 0.0, atol=1e-12)
    np.testing.assert_allclose(y.var(axis=(0, 2, 3)), 1.0, rtol=1e-12)


def test_single_value_rejected():
    cfg = config(3)
    with pytest.raises(ValueError, match="more than 1 value"):
        apply(params(), init_state(cfg), a(X)[:1], cfg)
    y, _ = apply(params(), init_state(cfg), x4()[:1], cfg)
    assert y.shape == (1, 3, 2, 2)


def test_jit():
    cfg = config(3)
    fn = jax.jit(lambda p, s, x: apply(p, s, x, cfg))
    y, s = fn(params(), init_state(cfg), a(X))
    y2, s2 = apply(params(), init_state(cfg), a(X), cfg)
    np.testing.assert_allclose(y, y2, rtol=1e-12)
    np.testing.assert_allclose(s["var"], s2["var"], rtol=1e-12)
