# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/lora/ (ai://NEW-lora).
"""Tests of the JAX reference: the worked example, the shared fixture, jax.grad against eq-lora-grads, the merge."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lora import apply, config, delta, init, merge, num_trainable  # noqa: E402

# The worked example of the page: d_out = 3, d_in = 4, r = 1, alpha = 1, so s = 1.
W0_EX = [[1.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, -1.0], [1.0, 1.0, 0.0, 0.0]]
A_EX = [[1.0, 0.0, -1.0, 2.0]]
B_EX = [[0.5], [0.0], [-1.0]]
X_EX = [[1.0, 2.0, 0.0, 1.0]]

# The shared fixture of the NumPy tests.
W0 = [[0.5, -1.0, 2.0, 0.0, 1.5], [1.0, 0.25, -0.5, 3.0, -2.0], [-1.5, 2.0, 0.0, 1.0, 0.5], [0.0, -0.75, 1.25, -1.0, 2.0]]
A = [[0.2, -0.4, 0.1, 0.3, -0.2], [-0.1, 0.5, 0.25, -0.3, 0.4]]
B = [[1.0, -0.5], [0.0, 2.0], [-1.5, 0.5], [0.75, 1.0]]
X = [[0.5, -1.5, 2.0, 0.25, -0.75], [1.0, 3.0, -2.5, 0.0, 4.0], [-1.25, 0.5, 1.5, -3.0, 2.25]]
DY = [[1.0, 2.0, -1.0, 0.5], [-2.0, 1.0, 0.0, 4.0], [0.25, -0.5, 2.0, 1.0]]


def arr(v):
    return jnp.asarray(v, dtype=jnp.float64)


def test_worked_example():
    cfg = config(4, 3, r=1, alpha=1.0)
    params = {"A": arr(A_EX), "B": arr(B_EX)}
    w0, x = arr(W0_EX), arr(X_EX)
    np.testing.assert_allclose(apply(params, w0, x, cfg), [[2.5, 1.0, 0.0]], rtol=1e-12)
    grads = jax.grad(lambda p: apply(p, w0, x, cfg).sum())(params)
    np.testing.assert_allclose(grads["B"], [[3.0], [3.0], [3.0]], rtol=1e-12)
    np.testing.assert_allclose(grads["A"], [[-0.5, -1.0, 0.0, -0.5]], rtol=1e-12)
    dx = jax.grad(lambda v: apply(params, w0, v, cfg).sum())(x)
    np.testing.assert_allclose(dx, [[1.5, 2.0, 2.5, -2.0]], rtol=1e-12)
    assert num_trainable(cfg) == 7


def test_shared_fixture_against_eq_lora_grads():
    cfg = config(5, 4, r=2, alpha=4.0)
    a, b, w0, x, dy = arr(A), arr(B), arr(W0), arr(X), arr(DY)
    params = {"A": a, "B": b}
    np.testing.assert_allclose(apply(params, w0, x, cfg), x @ (w0 + 2.0 * b @ a).T, rtol=1e-12)
    grads = jax.grad(lambda p: jnp.sum(apply(p, w0, x, cfg) * dy))(params)
    np.testing.assert_allclose(grads["B"], 2.0 * dy.T @ x @ a.T, rtol=1e-12)
    np.testing.assert_allclose(grads["A"], 2.0 * b.T @ dy.T @ x, rtol=1e-12)
    gw0 = jax.grad(lambda w: jnp.sum(apply(params, w, x, cfg) * dy))(w0)
    assert np.all(np.asarray(gw0) == 0.0)


def test_starts_at_the_pretrained_layer():
    cfg = config(5, 4, r=2)
    params = init(jax.random.PRNGKey(0), cfg, dtype=jnp.float64)
    b0 = arr([0.1, -0.2, 0.3, 0.0])
    np.testing.assert_allclose(apply(params, arr(W0), arr(X), cfg, b0=b0), arr(X) @ arr(W0).T + b0, rtol=1e-12)
    assert np.all(np.asarray(params["B"]) == 0.0)


def test_merge_and_rank():
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(1), 4)
    cfg = config(16, 12, r=3, alpha=6.0, rank_stabilized=True)
    params = {"A": jax.random.normal(k1, (3, 16)), "B": jax.random.normal(k2, (12, 3))}
    w0 = jax.random.normal(k3, (12, 16))
    x = jax.random.normal(k4, (5, 16))
    np.testing.assert_allclose(x @ merge(params, w0, cfg).T, apply(params, w0, x, cfg), rtol=1e-10, atol=1e-10)
    assert np.linalg.matrix_rank(np.asarray(delta(params, cfg))) == 3
    assert cfg["scale"] == pytest.approx(6.0 / 3**0.5)


@pytest.mark.parametrize("args", [(0, 4, 1), (4, 4, 0), (4, 4, 5)])
def test_bad_config(args):
    with pytest.raises(ValueError):
        config(*args)
