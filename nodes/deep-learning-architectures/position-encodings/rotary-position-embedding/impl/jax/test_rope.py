# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/rotary-position-embedding/ (ai://0V8E5T).
"""Tests of the JAX reference: the worked example, the shared fixture, the complex form and the gradient."""

import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from rope import apply, config, frequencies, init  # noqa: E402

# The worked example of the page: d = 4, theta = (1, 0.01).
Q = jnp.array([[1.0, 2.0, 0.0, 1.0]])
K = jnp.array([[2.0, 1.0, 1.0, 0.0]])
Q_AT_1 = [-1.1426, 1.9221, -0.0100, 1.0000]
SCORES = [((0, 0), 4.0), ((5, 5), 4.0), ((1, 0), -0.3732), ((2, 1), -0.3732), ((7, 6), -0.3732),
          ((0, 1), 4.6956), ((3, 0), -4.4133), ((10, 7), -4.4133)]

# The shared fixture of the NumPy tests.
X = jnp.array([[0.5, -1.5, 2.0, 0.25, -0.75, 1.0], [1.0, 3.0, -2.5, 0.0, 4.0, -1.0], [-1.25, 0.5, 1.5, -3.0, 2.25, 0.5]])
POS = jnp.array([0.0, 3.0, 17.0])
BASE = 1000.0  # theta = (1, 0.1, 0.01) for d = 6


def score(cfg, q, k, m, n):
    return float(jnp.sum(apply({}, q, [m], cfg) * apply({}, k, [n], cfg)))


def test_worked_example():
    cfg = config(4)
    np.testing.assert_allclose(frequencies(cfg), [1.0, 0.01], rtol=1e-15)
    np.testing.assert_allclose(apply(init(None, cfg), Q, [1], cfg)[0], Q_AT_1, atol=5e-5)
    for (m, n), s in SCORES:
        assert round(score(cfg, Q, K, m, n), 4) == s


def test_shared_fixture():
    y = apply({}, X, POS, config(6, BASE))
    np.testing.assert_array_equal(y[0], X[0])
    for row, m in [(1, 3.0), (2, 17.0)]:
        for i, a in enumerate([m, m / 10, m / 100]):
            x1, x2 = float(X[row, 2 * i]), float(X[row, 2 * i + 1])
            expect = [x1 * math.cos(a) - x2 * math.sin(a), x1 * math.sin(a) + x2 * math.cos(a)]
            np.testing.assert_allclose(y[row, 2 * i : 2 * i + 2], expect, rtol=1e-12, atol=1e-12)


def test_complex_form_agrees():
    cfg = config(6, BASE)
    z = (X[:, 0::2] + 1j * X[:, 1::2]) * jnp.exp(1j * jnp.outer(POS, frequencies(cfg)))  # eq: eq-rope-complex
    y = jnp.stack([z.real, z.imag], axis=-1).reshape(3, 6)
    np.testing.assert_allclose(apply({}, X, POS, cfg), y, rtol=1e-12, atol=1e-12)


def test_relative_and_norm():
    cfg = config(6, BASE)
    q, k = jax.random.normal(jax.random.PRNGKey(0), (2, 1, 6))
    for m, n, shift in [(4, 1, 100), (0, 9, -37), (250, 250, 3)]:
        np.testing.assert_allclose(score(cfg, q, k, m, n), score(cfg, q, k, m + shift, n + shift), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(jnp.linalg.norm(apply({}, q, [m], cfg)), jnp.linalg.norm(q), rtol=1e-12)


def test_gradient_is_inverse_rotation():
    cfg = config(6, BASE)
    dy = jnp.arange(18.0).reshape(3, 6) / 7 - 1
    dx = jax.grad(lambda v: jnp.sum(apply({}, v, POS, cfg) * dy))(X)
    np.testing.assert_allclose(dx, apply({}, dy, -POS, cfg), rtol=1e-12, atol=1e-12)


def test_interpolation():
    pi, plain = config(6, BASE, scale=4.0), config(6, BASE)
    np.testing.assert_allclose(apply({}, X, POS, pi), apply({}, X, POS / 4, plain), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(("d", "base", "scale"), [(3, 10000.0, 1.0), (4, 1.0, 1.0), (4, 10.0, 0.0)])
def test_bad_config(d, base, scale):
    with pytest.raises(ValueError):
        config(d, base, scale)


def test_wrong_width():
    with pytest.raises(ValueError):
        apply({}, X[:, :4], POS, config(6, BASE))
