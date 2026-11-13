# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/dropout/ (ai://Y8VNMB).
"""Tests of the JAX reference: the shared fixture, the gradient and the keys."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from dropout import apply, config, init  # noqa: E402

# The worked example of the page.
H = [2.0, 4.0, -1.0, 3.0]
EXAMPLES = [
    (0.5, [1, 0, 1, 1], [4.0, 0.0, -2.0, 6.0], [2.0, 0.0, 2.0, 2.0]),
    (0.2, [1, 1, 0, 1], [2.5, 5.0, 0.0, 3.75], [1.25, 1.25, 0.0, 1.25]),
]

# The shared fixture of the NumPy tests.
X = [[0.5, -1.5, 2.0, 0.25, -0.75], [1.0, 3.0, -2.5, 0.0, 4.0], [-1.25, 0.5, 1.5, -3.0, 2.25]]
M = [[1, 0, 1, 1, 0], [0, 1, 1, 0, 1], [1, 1, 0, 1, 1]]
DY = [[1.0, 2.0, -1.0, 0.5, 3.0], [-2.0, 1.0, 0.0, 4.0, -1.0], [0.25, -0.5, 2.0, 1.0, -3.0]]
P = 0.4


def a(v):
    return jnp.asarray(v, dtype=jnp.float64)


@pytest.mark.parametrize(("p", "mask", "y", "dx"), EXAMPLES)
def test_worked_example(p, mask, y, dx):
    cfg = config(p)
    params = init(None, cfg)
    out = apply(params, a(H), cfg, mask=a(mask))
    np.testing.assert_allclose(out, y, rtol=1e-12)
    grad = jax.grad(lambda h: apply(params, h, cfg, mask=a(mask)).sum())(a(H))
    np.testing.assert_allclose(grad, dx, rtol=1e-12)


def test_shared_fixture():
    cfg = config(P)
    y = apply({}, a(X), cfg, mask=a(M))
    np.testing.assert_allclose(y, np.array(X) * np.array(M) / 0.6, rtol=1e-10, atol=1e-12)
    dx = jax.grad(lambda x: (apply({}, x, cfg, mask=a(M)) * a(DY)).sum())(a(X))
    np.testing.assert_allclose(dx, np.array(DY) * np.array(M) / 0.6, rtol=1e-8, atol=1e-10)


def test_gradient():
    # Central differences of sum(y * DY) for a fixed mask.
    cfg = config(P)

    def f(x):
        return (apply({}, x, cfg, mask=a(M)) * a(DY)).sum()

    x = np.array(X)
    dx = jax.grad(f)(a(x))
    eps = 1e-6
    num = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        e = np.zeros_like(x)
        e[i] = eps
        num[i] = (f(a(x + e)) - f(a(x - e))) / (2 * eps)
    np.testing.assert_allclose(dx, num, rtol=1e-8, atol=1e-10)


def test_eval_is_identity():
    x = a(X)
    y = apply({}, x, config(0.5), key=jax.random.key(0), training=False)
    np.testing.assert_array_equal(y, x)


def test_masks_differ():
    # Split the key for each step. The same key gives the same mask, which is the pitfall of the page.
    cfg = config(0.5)
    x = jnp.ones(64)
    key = jax.random.key(1)
    k1, k2 = jax.random.split(key)
    assert not np.array_equal(apply({}, x, cfg, key=k1), apply({}, x, cfg, key=k2))
    np.testing.assert_array_equal(apply({}, x, cfg, key=k1), apply({}, x, cfg, key=k1))


def test_expectation():
    # The tolerance is 5 standard errors.
    p, n = 0.3, 200_000
    h = np.array(H)
    y = apply({}, jnp.broadcast_to(a(H), (n, 4)), config(p), key=jax.random.key(2))
    se = np.abs(h) * np.sqrt(p / (1 - p) / n)
    assert np.all(np.abs(np.asarray(y.mean(axis=0)) - h) <= 5 * se)


def test_jit():
    cfg = config(P)
    f = jax.jit(lambda x, k: apply({}, x, cfg, key=k))
    y = f(a(X), jax.random.key(3))
    assert y.shape == (3, 5)
    ok = np.isclose(y, 0.0) | np.isclose(y, np.array(X) / 0.6)
    assert np.all(ok)


def test_p_zero_is_identity():
    x = a(X)
    np.testing.assert_array_equal(apply({}, x, config(0.0), key=jax.random.key(4)), x)


@pytest.mark.parametrize("p", [-0.1, 1.0, 1.5])
def test_invalid_p(p):
    with pytest.raises(ValueError):
        config(p)


def test_training_needs_key_or_mask():
    with pytest.raises(ValueError):
        apply({}, a(X), config(0.5))
