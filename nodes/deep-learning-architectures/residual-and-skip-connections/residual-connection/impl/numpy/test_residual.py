# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/residual-connection/ (ai://F5K1AF).
"""Tests of the NumPy reference: the worked example, the identity at F = 0, the Jacobian product, the gradient norm, the paths and the highway gate."""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from residual import act, backward, block_jacobian, branch, config, forward, highway, init, jacobian, path_terms

# The worked example of the page: two blocks of width 2. The PyTorch and JAX tests hold the same numbers.
P = [
    {
        "W1": np.array([[1.0, 1.0], [1.0, -1.0]]),
        "b1": np.zeros(2),
        "W2": np.array([[0.5, 0.0], [0.0, -1.0]]),
        "b2": np.zeros(2),
    },
    {
        "W1": np.array([[0.0, 1.0], [1.0, 0.0]]),
        "b1": np.array([0.5, 0.0]),
        "W2": np.array([[1.0, 0.0], [0.0, 0.5]]),
        "b2": np.array([0.0, 0.25]),
    },
]
X = np.array([[2.0, -1.0]])
G = np.array([[1.0, 1.0]])
EXPECTED = {
    True: ([2.5, -2.5], [1.25, 2.75], [[1.5, -0.75], [-2.0, 1.0]], [[1.5, 4.5], [1.0, 3.0]]),
    False: ([0.0, 0.5], [0.25, 0.25], [[0.5, -0.25], [0.0, 0.0]], [[0.5, 1.5], [0.0, 0.0]]),
}


def numeric_jacobian(fn, x, h=1e-6):
    return np.stack([(fn(x + h * e) - fn(x - h * e)) / (2 * h) for e in np.eye(x.size)], axis=1)


@pytest.mark.parametrize("residual", [True, False])
def test_worked_example(residual):
    cfg = config(2, 2, residual=residual)
    y, cache = forward(P, X, cfg)
    want_y, want_dx, want_dw1, want_dw2 = EXPECTED[residual]
    np.testing.assert_allclose(y[0], want_y, rtol=1e-12)
    dx, grads = backward(P, cache, G, cfg)
    np.testing.assert_allclose(dx[0], want_dx, rtol=1e-12)
    np.testing.assert_allclose(grads[0]["W1"], want_dw1, rtol=1e-12)
    np.testing.assert_allclose(grads[0]["W2"], want_dw2, rtol=1e-12)
    np.testing.assert_allclose(grads[1]["b2"], [1.0, 1.0], rtol=1e-12)


def test_activations():
    z = np.array([-1.5, 0.0, 2.0])
    np.testing.assert_array_equal(act(z, "relu"), [0.0, 0.0, 2.0])
    np.testing.assert_allclose(act(z, "tanh"), np.tanh(z))
    with pytest.raises(ValueError):
        config(2, 1, activation="gelu")


@pytest.mark.parametrize("activation", ["relu", "tanh"])
def test_zero_branch_is_identity(activation):
    # With the last layer of every branch at zero, F = 0: each block and the stack are the identity, and so is the Jacobian.
    rng = np.random.default_rng(0)
    cfg = config(5, 4, hidden=7, activation=activation)
    params = init(rng, cfg, zero_last=True)
    x = rng.normal(size=(3, 5))
    y, cache = forward(params, x, cfg)
    np.testing.assert_array_equal(y, x)
    dy = rng.normal(size=(3, 5))
    dx, grads = backward(params, cache, dy, cfg)
    np.testing.assert_array_equal(dx, dy)
    np.testing.assert_array_equal(grads[0]["W1"], np.zeros((7, 5)))
    np.testing.assert_array_equal(jacobian(params, x[0], cfg), np.eye(5))
    np.testing.assert_allclose(numeric_jacobian(lambda v: forward(params, v, cfg)[0][0], x[0]), np.eye(5), atol=1e-9)


@pytest.mark.parametrize(("activation", "residual"), [("tanh", True), ("tanh", False), ("relu", True)])
def test_jacobian_is_product(activation, residual):
    # The Jacobian of the stack is the ordered product of the (I + J_i), checked against central differences.
    rng = np.random.default_rng(1)
    cfg = config(4, 5, hidden=6, activation=activation, residual=residual)
    params = init(rng, cfg, gain=0.7)
    for p in params:
        p["b1"] = rng.normal(size=6)
        p["b2"] = rng.normal(size=4)
    x = rng.normal(size=4)
    prod, xi, eye = np.eye(4), x, np.eye(4)
    for p in params:
        j = block_jacobian(p, xi, activation)
        np.testing.assert_allclose(
            j, numeric_jacobian(lambda v, p=p: branch(p, v[None, :], activation)[0][0], xi), atol=1e-7
        )
        prod = ((eye + j) if residual else j) @ prod
        f = branch(p, xi[None, :], activation)[0][0]
        xi = xi + f if residual else f
    jac = jacobian(params, x, cfg)
    np.testing.assert_allclose(jac, prod, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(jac, numeric_jacobian(lambda v: forward(params, v, cfg)[0][0], x), atol=1e-6)
    np.testing.assert_allclose(forward(params, x, cfg)[0][0], xi, rtol=1e-12)


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), n=st.integers(1, 4), residual=st.booleans())
def test_backward_matches_finite_differences(seed, n, residual):
    rng = np.random.default_rng(seed)
    cfg = config(3, n, hidden=4, activation="tanh", residual=residual)
    params = init(rng, cfg)
    for p in params:
        p["b1"] = rng.normal(size=4)
    x = rng.normal(size=(2, 3))
    dy = rng.normal(size=(2, 3))
    y, cache = forward(params, x, cfg)
    dx, grads = backward(params, cache, dy, cfg)

    def loss(ps, xx):
        return float(np.sum(forward(ps, xx, cfg)[0] * dy))

    h = 1e-6
    num = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        e = np.zeros_like(x)
        e[idx] = h
        num[idx] = (loss(params, x + e) - loss(params, x - e)) / (2 * h)
    np.testing.assert_allclose(dx, num, atol=1e-6)
    for i in range(n):
        for name in ("W1", "b1", "W2", "b2"):
            num = np.zeros_like(params[i][name])
            for idx in np.ndindex(num.shape):
                saved = params[i][name][idx]
                params[i][name][idx] = saved + h
                up = loss(params, x)
                params[i][name][idx] = saved - h
                down = loss(params, x)
                params[i][name][idx] = saved
                num[idx] = (up - down) / (2 * h)
            np.testing.assert_allclose(grads[i][name], num, atol=1e-6)


def gradient_ratio(n, residual, gain=0.1, d=16):
    """||dx_0|| / ||dx_n|| for a batch of 8 through n tanh blocks drawn with a fixed seed."""
    cfg = config(d, n, activation="tanh", residual=residual)
    params = init(np.random.default_rng(1603), cfg, gain=gain)
    x = np.random.default_rng(20160316).normal(size=(8, d))
    y, cache = forward(params, x, cfg)
    dy = np.random.default_rng(5).normal(size=y.shape)
    dx, _ = backward(params, cache, dy, cfg)
    return np.linalg.norm(dx) / np.linalg.norm(dy)


def test_gradient_norm_bounded_vs_plain():
    # Small branches (gain 0.1), 50 blocks: with shortcuts the gradient at the input keeps its size; the plain
    # stack of the same blocks multiplies it by about 1e-45 (the numbers of the page come from this test).
    res, plain = gradient_ratio(50, True), gradient_ratio(50, False)
    assert 0.8 < res < 1.6
    assert plain < 1e-30
    res10, plain10 = gradient_ratio(10, True), gradient_ratio(10, False)
    assert 0.8 < res10 < 1.6
    assert plain < plain10 < 1e-5
    assert gradient_ratio(100, True) < 2.5


def test_gradient_within_derivation_bounds():
    # For one input, (1 - eps)^n <= ||g J|| / ||g|| <= (1 + eps)^n with eps the largest spectral norm of the J_i,
    # while the plain stack is at most the product of the norms.
    rng = np.random.default_rng(3)
    cfg = config(8, 20, activation="tanh")
    params = init(rng, cfg, gain=0.1)
    x = rng.normal(size=8)
    norms, xi = [], x
    for p in params:
        norms.append(np.linalg.norm(block_jacobian(p, xi, "tanh"), 2))
        xi = xi + branch(p, xi[None, :], "tanh")[0][0]
    eps = max(norms)
    assert eps < 1.0
    g = rng.normal(size=8)
    ratio = np.linalg.norm(g @ jacobian(params, x, cfg)) / np.linalg.norm(g)
    assert (1 - eps) ** 20 <= ratio <= (1 + eps) ** 20
    plain_norms, xi = [], x
    for p in params:
        plain_norms.append(np.linalg.norm(block_jacobian(p, xi, "tanh"), 2))
        xi = branch(p, xi[None, :], "tanh")[0][0]
    plain = config(8, 20, activation="tanh", residual=False)
    assert np.linalg.norm(g @ jacobian(params, x, plain)) / np.linalg.norm(g) <= math.prod(plain_norms) * (1 + 1e-9)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_path_count(n):
    # The product of (I + A_i) expands into 2^n paths, C(n, k) of them through k branches, and they sum to the product.
    rng = np.random.default_rng(n)
    mats = [rng.normal(scale=0.5, size=(3, 3)) for _ in range(n)]
    terms = path_terms(mats)
    assert len(terms) == 2**n
    lengths = [sum(k) for k in terms]
    assert [lengths.count(k) for k in range(n + 1)] == [math.comb(n, k) for k in range(n + 1)]
    full = np.eye(3)
    for a in mats:
        full = (np.eye(3) + a) @ full
    np.testing.assert_allclose(sum(terms.values()), full, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(terms[(0,) * n], np.eye(3))
    np.testing.assert_allclose(terms[(1,) * n], np.linalg.multi_dot([*mats[::-1], np.eye(3)]), rtol=1e-12)
    if n >= 2:
        np.testing.assert_allclose(terms[(1, 1) + (0,) * (n - 2)], mats[1] @ mats[0], rtol=1e-12)


def test_highway_gates():
    rng = np.random.default_rng(4)
    cfg = config(3, 1, hidden=5)
    p = init(rng, cfg)[0]
    x = rng.normal(size=(4, 3))
    f = branch(p, x, "relu")[0]
    # T = 1 and C = 1: the highway layer is the residual block.
    np.testing.assert_allclose(highway(p, x, "relu", t=1.0, c=1.0), forward([p], x, cfg)[0], rtol=1e-12)
    # The coupled gate C = 1 - T: T = 1 gives the plain layer, T = 0 the identity.
    np.testing.assert_allclose(highway(p, x, "relu", t=1.0), f, rtol=1e-12)
    np.testing.assert_array_equal(highway(p, x, "relu", t=0.0), x)
    # A learned gate with a very negative bias carries the input; with a very positive one it transforms it.
    gate = {"WT": rng.normal(size=(3, 3)), "bT": np.full(3, -40.0)}
    np.testing.assert_allclose(highway(p, x, "relu", gate=gate), x, atol=1e-12)
    gate["bT"] = np.full(3, 40.0)
    np.testing.assert_allclose(highway(p, x, "relu", gate=gate), f, atol=1e-12)
    gate = {"WT": np.zeros((3, 3)), "bT": np.zeros(3)}
    np.testing.assert_allclose(highway(p, x, "relu", gate=gate), 0.5 * f + 0.5 * x, rtol=1e-12)


def test_init_scales():
    rng = np.random.default_rng(6)
    cfg = config(64, 2, hidden=128)
    params = init(rng, cfg, gain=0.5)
    assert params[0]["W1"].shape == (128, 64) and params[0]["W2"].shape == (64, 128)
    assert abs(params[0]["W1"].std() - math.sqrt(2 / 64)) < 0.02
    assert abs(params[0]["W2"].std() - 0.5 / math.sqrt(128)) < 0.005
    np.testing.assert_array_equal(params[1]["b1"], np.zeros(128))
    assert config(4, 1)["hidden"] == 4


def test_learns_tiny_regression():
    # The learning smoke test: SGD on a residual stack fits 16 fixed points of a smooth map.
    rng = np.random.default_rng(7)
    cfg = config(4, 3, hidden=16, activation="tanh")
    params = init(rng, cfg, gain=0.5)
    x = rng.normal(size=(16, 4))
    target = x + 0.5 * np.sin(x)
    first = None
    for _ in range(300):
        y, cache = forward(params, x, cfg)
        err = y - target
        loss = float(np.mean(err**2))
        first = loss if first is None else first
        _, grads = backward(params, cache, 2 * err / err.size, cfg)
        for p, g in zip(params, grads, strict=True):
            for k in p:
                p[k] = p[k] - 0.5 * g[k]
    assert loss < 0.1 * first
