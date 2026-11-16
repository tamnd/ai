# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/layer-normalization/ (ai://NEW-layer-normalization).
"""Tests of the JAX reference: the worked example, jax.grad against eq-ln-backward and eq-rms-backward, finite
differences, the Jacobian, the invariances, independence from the batch, and the two placements."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from layer_normalization import config, init, layer_norm, post_ln, pre_ln, rms_config, rms_norm

jax.config.update("jax_enable_x64", True)

# The worked example of the page.
WA = [3.0, -4.0, 7.0, -1.0, 5.0]
WGAIN = [2.0, 1.0, 2.0, 1.0, 4.0]
WBIAS = [0.0, 1.0, 0.0, 1.0, 0.0]
WY = [0.5, -0.5, 2.5, 0.25, 3.0]
WDY = [0.0, 1.0, 0.0, 0.0, 0.0]
WDX = [-0.625 / 20, 1.75 / 20, 0.875 / 20, -2.125 / 20, 0.125 / 20]

# The shared fixture of the NumPy tests.
X = [[0.5, -1.5, 2.0, 1.0], [1.0, 3.0, -2.5, 0.0], [-1.25, 0.5, 1.5, 4.0]]
DY = [[1.0, 2.0, -1.0, 0.5], [-2.0, 1.0, 0.0, 3.0], [0.25, -0.5, 2.0, -1.0]]
GAIN = [1.5, 0.5, 2.0, -1.0]
BIAS = [0.1, -0.2, 0.3, 0.0]


def a(v):
    return jnp.asarray(v, dtype=jnp.float64)


def x3():
    return ((jnp.arange(24.0) * 7) % 11 - 5).reshape(2, 3, 4) / 2


def test_worked_example():
    cfg = config(5, eps=0.0)
    p = {"gain": a(WGAIN), "bias": a(WBIAS)}
    np.testing.assert_allclose(layer_norm(p, a(WA), cfg), WY, atol=1e-12)
    dx, dp = jax.grad(lambda x, q: jnp.sum(layer_norm(q, x, cfg) * a(WDY)), argnums=(0, 1))(a(WA), p)
    np.testing.assert_allclose(dx, WDX, atol=1e-12)
    np.testing.assert_allclose(dp["gain"], [0.0, -1.5, 0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(dp["bias"], WDY, atol=1e-12)
    j = jax.jacobian(lambda x: layer_norm({}, x, cfg))(a(WA))
    np.testing.assert_allclose(jnp.linalg.norm(j, 2), 0.25, rtol=1e-12)
    centered = a(WA) - a(WA).mean()
    np.testing.assert_allclose(rms_norm({}, centered, rms_config(5, eps=0.0)), layer_norm({}, a(WA), cfg), atol=1e-12)


def test_backward_formulas():
    x, g, h = a(X), a(DY), 4
    cfg = config(4)
    p = {"gain": a(GAIN), "bias": a(BIAS)}
    dx = jax.grad(lambda z: jnp.sum(layer_norm(p, z, cfg) * g))(x)
    s = jnp.sqrt(jnp.var(x, axis=-1, keepdims=True) + cfg["eps"])
    xhat = (x - x.mean(-1, keepdims=True)) / s
    u = g * a(GAIN)
    expected = (h * u - u.sum(-1, keepdims=True) - xhat * (u * xhat).sum(-1, keepdims=True)) / (h * s)
    np.testing.assert_allclose(dx, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dx.sum(-1), 0.0, atol=1e-12)
    rcfg = rms_config(4, eps=1e-3)
    dr = jax.grad(lambda z: jnp.sum(rms_norm({"gain": a(GAIN)}, z, rcfg) * g))(x)
    r = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + 1e-3)
    zhat = x / r
    np.testing.assert_allclose(dr, (u - zhat * (u * zhat).sum(-1, keepdims=True) / h) / r, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("rms", [False, True])
def test_gradient_finite_differences(rms):
    cfg = rms_config((3, 4), eps=1e-3) if rms else config((3, 4), eps=1e-3)
    norm = rms_norm if rms else layer_norm
    key = jax.random.PRNGKey(0)
    gain = jax.random.normal(key, (3, 4), dtype=jnp.float64)

    def loss(x, g):
        return jnp.sum(norm({"gain": g}, x, cfg) * jnp.sin(x))

    check_grads(loss, (x3(), gain), order=2, modes=["rev"], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("eps", [0.0, 0.5])
def test_jacobian_spectral_norm(eps):
    for h in [3, 16]:
        x = jax.random.normal(jax.random.PRNGKey(h), (h,), dtype=jnp.float64) * 3 + 2
        j = jax.jacobian(lambda z: layer_norm({}, z, config(h, eps=eps)))(x)
        np.testing.assert_allclose(jnp.linalg.norm(j, 2), 1 / jnp.sqrt(jnp.var(x) + eps), rtol=1e-10)
        np.testing.assert_allclose(j.sum(0), 0.0, atol=1e-12)


def test_invariances():
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(20261115), 3)
    w = jax.random.normal(k1, (6, 5), dtype=jnp.float64)
    x = jax.random.normal(k2, (7, 5), dtype=jnp.float64)
    gamma = jax.random.normal(k3, (5,), dtype=jnp.float64)
    ln, rms = config(6, eps=0.0), rms_config(6, eps=0.0)
    base, base_rms = layer_norm({}, x @ w.T, ln), rms_norm({}, x @ w.T, rms)
    w2 = 2.5 * w + jnp.outer(jnp.ones(6), gamma)
    np.testing.assert_allclose(layer_norm({}, x @ w2.T, ln), base, atol=1e-12)
    np.testing.assert_allclose(layer_norm({}, (4.0 * x) @ w.T, ln), base, atol=1e-12)
    np.testing.assert_allclose(rms_norm({}, x @ (2.5 * w).T, rms), base_rms, atol=1e-12)
    assert jnp.abs(rms_norm({}, x @ w2.T, rms) - base_rms).max() > 1e-3
    assert jnp.abs(layer_norm({}, x @ w.at[0].multiply(3.0).T, ln) - base).max() > 1e-3


def test_batch_independence_and_no_state():
    cfg = config(4)
    p = {"gain": a(GAIN), "bias": a(BIAS)}
    y = layer_norm(p, a(X), cfg)
    for i in range(3):
        np.testing.assert_allclose(layer_norm(p, a(X)[i], cfg), y[i], rtol=1e-14)
    np.testing.assert_allclose(jax.vmap(lambda r: layer_norm(p, r, cfg))(a(X)), y, rtol=1e-14)
    assert set(init(None, cfg)) == {"gain", "bias"} and init(None, cfg)["gain"].shape == (4,)


def test_placements():
    norm = lambda v: layer_norm({}, v, config(4, eps=0.0))  # noqa: E731
    np.testing.assert_allclose(pre_ln(a(X), lambda v: 0.0 * v, norm), a(X), rtol=1e-14)
    np.testing.assert_allclose(post_ln(a(X), lambda v: 0.0 * v, norm), norm(a(X)), rtol=1e-14)
    y = post_ln(a(X), lambda v: 3.0 * v + 1.0, norm)
    np.testing.assert_allclose(jnp.linalg.norm(y, axis=-1), 2.0, rtol=1e-12)


def test_rms_default_eps_is_machine_epsilon():
    x = a(X) * 1e-4
    cfg = rms_config(4)
    np.testing.assert_allclose(rms_norm({}, x, cfg), x / jnp.sqrt(jnp.mean(x ** 2, -1, keepdims=True)
                                                                + jnp.finfo(jnp.float64).eps), rtol=1e-14)


def test_config_errors():
    with pytest.raises(ValueError):
        config(0)
    with pytest.raises(ValueError):
        config(4, eps=-1.0)
    with pytest.raises(ValueError):
        config(4, eps=None)
    with pytest.raises(ValueError):
        layer_norm({}, a(X), config(5))
