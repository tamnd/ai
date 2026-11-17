# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/residual-connection/ (ai://F5K1AF).
"""Tests of the JAX reference: the worked example of the NumPy tests, the identity at F = 0, the Jacobian, the gradient norm and the highway gate."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from residual import apply, branch, config, highway, init

jax.config.update("jax_enable_x64", True)  # before any array is made: the tests run in float64

# The worked example of the page, with the numbers of the NumPy tests.
P = [
    {"W1": [[1.0, 1.0], [1.0, -1.0]], "b1": [0.0, 0.0], "W2": [[0.5, 0.0], [0.0, -1.0]], "b2": [0.0, 0.0]},
    {"W1": [[0.0, 1.0], [1.0, 0.0]], "b1": [0.5, 0.0], "W2": [[1.0, 0.0], [0.0, 0.5]], "b2": [0.0, 0.25]},
]
EXPECTED = {
    True: ([2.5, -2.5], [1.25, 2.75], [[1.5, -0.75], [-2.0, 1.0]], [[1.5, 4.5], [1.0, 3.0]]),
    False: ([0.0, 0.5], [0.25, 0.25], [[0.5, -0.25], [0.0, 0.0]], [[0.5, 1.5], [0.0, 0.0]]),
}


def params():
    return [{k: jnp.asarray(v, dtype=jnp.float64) for k, v in p.items()} for p in P]


@pytest.mark.parametrize("residual", [True, False])
def test_worked_example(residual):
    cfg = config(2, 2, residual=residual)
    x = jnp.array([[2.0, -1.0]])
    want_y, want_dx, want_dw1, want_dw2 = EXPECTED[residual]
    np.testing.assert_allclose(apply(params(), x, cfg)[0], want_y, rtol=1e-12)
    dp, dx = jax.grad(lambda ps, v: apply(ps, v, cfg).sum(), argnums=(0, 1))(params(), x)
    np.testing.assert_allclose(dx[0], want_dx, rtol=1e-12)
    np.testing.assert_allclose(dp[0]["W1"], want_dw1, rtol=1e-12)
    np.testing.assert_allclose(dp[0]["W2"], want_dw2, rtol=1e-12)


def test_zero_branch_is_identity():
    cfg = config(5, 4, hidden=7, activation="tanh")
    ps = init(jax.random.PRNGKey(0), cfg, zero_last=True)
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 5))
    np.testing.assert_array_equal(apply(ps, x, cfg), x)
    np.testing.assert_array_equal(jax.jacfwd(lambda v: apply(ps, v, cfg))(x[0]), np.eye(5))
    with pytest.raises(ValueError):
        config(2, 1, activation="gelu")


def test_jacobian_is_product():
    cfg = config(4, 5, hidden=6, activation="tanh")
    ps = init(jax.random.PRNGKey(2), cfg, gain=0.7)
    x = jax.random.normal(jax.random.PRNGKey(3), (4,))
    prod, xi = np.eye(4), x
    for p in ps:
        j = jax.jacfwd(lambda v, p=p: branch(p, v, "tanh"))(xi)
        prod = (np.eye(4) + j) @ prod
        xi = xi + branch(p, xi, "tanh")
    np.testing.assert_allclose(jax.jacfwd(lambda v: apply(ps, v, cfg))(x), prod, rtol=1e-10, atol=1e-12)


def ratio(n, residual):
    cfg = config(16, n, activation="tanh", residual=residual)
    ps = init(jax.random.PRNGKey(1603), cfg, gain=0.1)
    x = jax.random.normal(jax.random.PRNGKey(20160316), (8, 16))
    dy = jax.random.normal(jax.random.PRNGKey(5), (8, 16))
    dx = jax.grad(lambda v: (apply(ps, v, cfg) * dy).sum())(x)
    return float(jnp.linalg.norm(dx) / jnp.linalg.norm(dy))


def test_gradient_norm_bounded_vs_plain():
    assert 0.8 < ratio(50, True) < 1.6
    assert ratio(50, False) < 1e-30


def test_highway_gates():
    cfg = config(3, 1, hidden=5)
    p = init(jax.random.PRNGKey(4), cfg)[0]
    x = jax.random.normal(jax.random.PRNGKey(5), (4, 3))
    f = branch(p, x, "relu")
    np.testing.assert_allclose(highway(p, x, "relu", t=1.0, c=1.0), apply([p], x, cfg), rtol=1e-12)
    np.testing.assert_allclose(highway(p, x, "relu", t=1.0), f, rtol=1e-12)
    np.testing.assert_array_equal(highway(p, x, "relu", t=0.0), x)
    gate = {"WT": jnp.zeros((3, 3)), "bT": jnp.zeros(3)}
    np.testing.assert_allclose(highway(p, x, "relu", gate=gate), 0.5 * f + 0.5 * x, rtol=1e-12)
