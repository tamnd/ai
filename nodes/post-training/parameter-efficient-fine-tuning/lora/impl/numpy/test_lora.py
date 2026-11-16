# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/lora/ (ai://NEW-lora).
"""Tests of the NumPy reference: the worked example, the start at W0, the merge, the gradients and the counts."""

import math

import numpy as np
import pytest

from lora import backward, config, delta, forward, init, merge, num_trainable, unmerge

# The worked example of the page: d_out = 3, d_in = 4, r = 1, alpha = 1, so s = 1.
W0_EX = np.array([[1.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, -1.0], [1.0, 1.0, 0.0, 0.0]])
A_EX = np.array([[1.0, 0.0, -1.0, 2.0]])
B_EX = np.array([[0.5], [0.0], [-1.0]])
X_EX = np.array([[1.0, 2.0, 0.0, 1.0]])

# The shared fixture. The PyTorch and JAX tests hold the same numbers.
W0 = np.array([[0.5, -1.0, 2.0, 0.0, 1.5], [1.0, 0.25, -0.5, 3.0, -2.0], [-1.5, 2.0, 0.0, 1.0, 0.5], [0.0, -0.75, 1.25, -1.0, 2.0]])
A = np.array([[0.2, -0.4, 0.1, 0.3, -0.2], [-0.1, 0.5, 0.25, -0.3, 0.4]])
B = np.array([[1.0, -0.5], [0.0, 2.0], [-1.5, 0.5], [0.75, 1.0]])
X = np.array([[0.5, -1.5, 2.0, 0.25, -0.75], [1.0, 3.0, -2.5, 0.0, 4.0], [-1.25, 0.5, 1.5, -3.0, 2.25]])
DY = np.array([[1.0, 2.0, -1.0, 0.5], [-2.0, 1.0, 0.0, 4.0], [0.25, -0.5, 2.0, 1.0]])
ALPHA = 4.0  # with r = 2, s = 2


def test_worked_example():
    cfg = config(4, 3, r=1, alpha=1.0)
    params = {"A": A_EX, "B": B_EX}
    y, cache = forward(params, W0_EX, X_EX, cfg)
    np.testing.assert_allclose(y, [[2.5, 1.0, 0.0]], rtol=1e-12)
    np.testing.assert_allclose(merge(params, W0_EX, cfg), [[1.5, 0.0, 1.5, 1.0], [0.0, 1.0, 0.0, -1.0], [0.0, 1.0, 1.0, -2.0]], rtol=1e-12)
    dx, grads = backward(params, W0_EX, cache, np.ones((1, 3)), cfg)
    np.testing.assert_allclose(grads["B"], [[3.0], [3.0], [3.0]], rtol=1e-12)
    np.testing.assert_allclose(grads["A"], [[-0.5, -1.0, 0.0, -0.5]], rtol=1e-12)
    np.testing.assert_allclose(dx, [[1.5, 2.0, 2.5, -2.0]], rtol=1e-12)
    assert num_trainable(cfg) == 7


def test_shared_fixture():
    cfg = config(5, 4, r=2, alpha=ALPHA)
    assert cfg["scale"] == 2.0
    params = {"A": A, "B": B}
    y, cache = forward(params, W0, X, cfg)
    np.testing.assert_allclose(y, X @ (W0 + 2.0 * B @ A).T, rtol=1e-12)
    dx, grads = backward(params, W0, cache, DY, cfg)
    np.testing.assert_allclose(grads["B"], 2.0 * DY.T @ X @ A.T, rtol=1e-12)
    np.testing.assert_allclose(grads["A"], 2.0 * B.T @ DY.T @ X, rtol=1e-12)
    np.testing.assert_allclose(dx, DY @ (W0 + 2.0 * B @ A), rtol=1e-12)


def test_starts_at_the_pretrained_layer():
    rng = np.random.default_rng(0)
    cfg = config(5, 4, r=3)
    params = init(rng, cfg)
    assert np.all(params["B"] == 0.0)
    assert params["A"].shape == (3, 5) and params["B"].shape == (4, 3)
    b0 = np.array([0.1, -0.2, 0.3, 0.0])
    y, _ = forward(params, W0, X, cfg, b0=b0)
    np.testing.assert_allclose(y, X @ W0.T + b0, rtol=1e-12)


def test_first_step_moves_only_b():
    # With B = 0 the gradient of A is zero, and the gradient of B is s dy^T (x A^T) (eq-lora-grads).
    rng = np.random.default_rng(1)
    cfg = config(5, 4, r=2, alpha=ALPHA)
    params = init(rng, cfg)
    _, cache = forward(params, W0, X, cfg)
    _, grads = backward(params, W0, cache, DY, cfg)
    assert np.all(grads["A"] == 0.0)
    assert np.linalg.norm(grads["B"]) > 0.0


def loss(params, w0, x, cfg, dy):
    y, _ = forward(params, w0, x, cfg)
    return float(np.sum(y * dy))


@pytest.mark.parametrize("rank_stabilized", [False, True])
def test_gradients_match_finite_differences(rank_stabilized):
    cfg = config(5, 4, r=2, alpha=ALPHA, rank_stabilized=rank_stabilized)
    params = {"A": A.copy(), "B": B.copy()}
    _, cache = forward(params, W0, X, cfg)
    dx, grads = backward(params, W0, cache, DY, cfg)
    eps = 1e-6
    for name in ("A", "B"):
        num = np.zeros_like(params[name])
        for idx in np.ndindex(num.shape):
            old = params[name][idx]
            params[name][idx] = old + eps
            up = loss(params, W0, X, cfg, DY)
            params[name][idx] = old - eps
            down = loss(params, W0, X, cfg, DY)
            params[name][idx] = old
            num[idx] = (up - down) / (2 * eps)
        np.testing.assert_allclose(grads[name], num, rtol=1e-6, atol=1e-8)
    numx = np.zeros_like(X)
    for idx in np.ndindex(X.shape):
        xp, xm = X.copy(), X.copy()
        xp[idx] += eps
        xm[idx] -= eps
        numx[idx] = (loss(params, W0, xp, cfg, DY) - loss(params, W0, xm, cfg, DY)) / (2 * eps)
    np.testing.assert_allclose(dx, numx, rtol=1e-6, atol=1e-8)


def test_merge_and_unmerge():
    rng = np.random.default_rng(2)
    cfg = config(64, 48, r=4, alpha=8.0)
    w0 = rng.normal(size=(48, 64))
    params = {"A": rng.normal(size=(4, 64)), "B": rng.normal(size=(48, 4))}
    x = rng.normal(size=(10, 64))
    y, _ = forward(params, w0, x, cfg)
    w = merge(params, w0, cfg)
    np.testing.assert_allclose(x @ w.T, y, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(unmerge(params, w, cfg), w0, rtol=1e-10, atol=1e-10)
    assert np.linalg.matrix_rank(delta(params, cfg)) == 4


def test_counts_and_scales():
    cfg = config(12288, 12288, r=4)
    assert num_trainable(cfg) == 4 * (12288 + 12288)
    assert config(10, 10, r=4, alpha=8.0)["scale"] == 2.0
    assert config(10, 10, r=4, alpha=8.0, rank_stabilized=True)["scale"] == 4.0
    assert config(10, 10, r=9, alpha=3.0, rank_stabilized=True)["scale"] == pytest.approx(3.0 / math.sqrt(9))
    assert config(10, 10, r=5)["scale"] == 1.0


@pytest.mark.parametrize("args", [(0, 4, 1), (4, 4, 0), (4, 4, 5), (3, 8, 4)])
def test_bad_config(args):
    with pytest.raises(ValueError):
        config(*args)


def test_bad_shapes():
    cfg = config(5, 4, r=2)
    params = {"A": A, "B": B}
    with pytest.raises(ValueError):
        forward(params, W0.T, X, cfg)
    with pytest.raises(ValueError):
        forward(params, W0, X[:, :4], cfg)
