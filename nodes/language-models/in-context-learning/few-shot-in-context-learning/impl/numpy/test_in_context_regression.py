# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/few-shot-in-context-learning/ (ai://JNEP9D).
"""Tests of the NumPy reference: the worked example of the page, the matrices of the construction, one linear
self-attention layer against one gradient step, stacked layers against several steps, the rescaling of the
weights, the averaging estimate, ridge against the posterior mean, the lam -> 0 limit, and input validation."""

import numpy as np
import pytest

from in_context_regression import (
    gd_predict,
    gd_step,
    linear_self_attention,
    lsa_predict,
    lsa_weights,
    ridge_predict,
)

# The worked example: d_x = d_y = 1, context (1, 2), (2, 4), query 3, eta = 0.5.
WX = np.array([[1.0], [2.0]])
WY = np.array([[2.0], [4.0]])
WQ = np.array([[3.0]])


def problem(seed, n=7, dx=4, dy=2, m=3):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(n, dx)),
        rng.normal(size=(n, dy)),
        rng.normal(size=(m, dx)),
        rng.normal(size=(dy, dx)),
    )


def test_worked_example_gradient_descent():
    assert np.allclose(gd_predict(WX, WY, WQ, 0.5), [[7.5]])
    assert np.allclose(gd_predict(WX, WY, WQ, 0.5, steps=2), [[5.625]])
    assert np.allclose(gd_predict(WX, WY, WQ, 0.5, w0=[[1.0]]), [[6.75]])
    assert np.allclose(gd_predict(WX, WY, WQ, 0.5, w0=[[1.0]], steps=0), [[3.0]])
    assert np.allclose(gd_predict(WX, WY, WQ, 0.5, steps=0), [[0.0]])


def test_worked_example_attention_and_ridge():
    assert np.allclose(lsa_predict(WX, WY, WQ, 0.5), [[7.5]])
    assert np.allclose(lsa_predict(WX, WY, WQ, 0.5, layers=2), [[5.625]])
    assert np.allclose(lsa_predict(WX, WY, WQ, 0.5, w0=[[1.0]]), [[6.75]])
    assert np.allclose(ridge_predict(WX, WY, WQ, 1.0), [[5.0]])
    assert np.allclose(ridge_predict(WX, WY, WQ, 0.0), [[6.0]])


def test_construction_matrices():
    wk, wq, wv, p = lsa_weights(2, 1, 0.6, 3, [[5.0, 7.0]])
    assert np.array_equal(wk, [[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    assert np.array_equal(wq, wk)
    assert np.array_equal(wv, [[0, 0, 0], [0, 0, 0], [5, 7, -1]])
    assert np.allclose(p, 0.2 * np.eye(3))
    wk, wq, wv, p = lsa_weights(1, 2, 1.0, 1)
    assert np.array_equal(wv, [[0, 0, 0], [0, -1, 0], [0, 0, -1]])
    assert np.array_equal(wk, [[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert np.allclose(p, np.eye(3))


def test_attention_layer_matches_definition():
    rng = np.random.default_rng(1)
    ctx, tok = rng.normal(size=(5, 3)), rng.normal(size=(2, 3))
    wk, wq, wv, p = (rng.normal(size=(3, 3)) for _ in range(4))
    want = np.array([e + p @ wv @ sum(c * (c @ wk.T @ wq @ e) for c in ctx) for e in tok])
    assert np.allclose(linear_self_attention(ctx, tok, wk, wq, wv, p), want)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_one_layer_is_one_gradient_step(seed):
    x, y, xq, w0 = problem(seed)
    eta = 0.3
    w1 = gd_step(x, y, w0, eta)
    wk, wq, wv, p = lsa_weights(4, 2, eta, len(x), w0)
    ctx = np.hstack([x, y])
    out = linear_self_attention(ctx, ctx, wk, wq, wv, p)
    assert np.allclose(out[:, :4], x)
    assert np.allclose(out[:, 4:], y - x @ (w1 - w0).T)
    assert np.allclose(lsa_predict(x, y, xq, eta, w0), xq @ w1.T)
    assert np.allclose(lsa_predict(x, y, xq, eta, w0), gd_predict(x, y, xq, eta, w0))


def test_gradient_step_is_the_gradient():
    x, y, _, w = problem(3)
    eta, h = 0.7, 1e-6

    def loss(v):
        return ((x @ v.T - y) ** 2).sum() / (2 * len(x))

    grad = np.zeros_like(w)
    for idx in np.ndindex(*w.shape):
        e = np.zeros_like(w)
        e[idx] = h
        grad[idx] = (loss(w + e) - loss(w - e)) / (2 * h)
    assert np.allclose(gd_step(x, y, w, eta), w - eta * grad, atol=1e-7)


@pytest.mark.parametrize("layers", [1, 2, 3, 5])
def test_stacked_layers_are_gradient_steps(layers):
    x, y, xq, w0 = problem(4)
    assert np.allclose(lsa_predict(x, y, xq, 0.2, w0, layers), gd_predict(x, y, xq, 0.2, w0, layers))
    assert np.allclose(lsa_predict(x, y, xq, 0.2, None, layers), gd_predict(x, y, xq, 0.2, None, layers))


def test_many_steps_reach_least_squares():
    x, y, xq, _ = problem(5, n=30, dx=3)
    assert np.allclose(lsa_predict(x, y, xq, 0.5, layers=400), ridge_predict(x, y, xq, 0.0), atol=1e-6)


@pytest.mark.parametrize("s", [-3.0, 0.5, 7.0])
def test_rescaling(s):
    x, y, xq, w0 = problem(6)
    wk, wq, wv, p = lsa_weights(4, 2, 0.4, len(x), w0)
    ctx = np.hstack([x, y])
    tok = np.hstack([xq, -xq @ w0.T])
    a = linear_self_attention(ctx, tok, wk, wq, wv, p)
    b = linear_self_attention(ctx, tok, np.eye(6), wk.T @ wq / s, s * p @ wv, np.eye(6))
    assert np.allclose(a, b)


def test_averaging_estimate():
    x, y, xq, _ = problem(7)
    assert np.allclose(lsa_predict(x, y, xq, 1.0), xq @ x.T @ y / len(x))


def test_ridge_is_posterior_mean():
    x, y, xq, _ = problem(8, n=5, dx=6, dy=1)
    tau2, sigma2 = 2.0, 0.5
    post = tau2 * x.T @ np.linalg.solve(tau2 * x @ x.T + sigma2 * np.eye(5), y)
    assert np.allclose(ridge_predict(x, y, xq, sigma2 / tau2), xq @ post)


def test_ridge_limit_and_least_squares():
    x, y, xq, _ = problem(9, n=3, dx=6)
    assert np.allclose(ridge_predict(x, y, xq, 1e-9), ridge_predict(x, y, xq, 0.0), atol=1e-5)
    w = np.linalg.pinv(x) @ y
    assert np.allclose(ridge_predict(x, y, xq, 0.0), xq @ w)
    x, y, xq, _ = problem(10, n=12, dx=3)
    ols = np.linalg.lstsq(x, y, rcond=None)[0]
    assert np.allclose(ridge_predict(x, y, xq, 0.0), xq @ ols)
    assert not np.allclose(ridge_predict(x, y, xq, 5.0), xq @ ols)


def test_integer_inputs():
    xi = np.array([[1], [2]])
    yi = np.array([[2], [4]])
    assert np.allclose(lsa_predict(xi, yi, [[3]], 0.5), [[7.5]])
    assert np.allclose(ridge_predict(xi, yi, [[3]], 1), [[5.0]])


def test_list_inputs():
    assert np.allclose(ridge_predict([[1.0], [2.0]], [[2.0], [4.0]], [[3.0]], 1.0), [[5.0]])
    assert np.allclose(gd_predict([[1.0], [2.0]], [[2.0], [4.0]], [[3.0]], 0.5), [[7.5]])
    out = linear_self_attention([[1.0, 2.0], [2.0, 4.0]], [[3.0, 0.0]], *lsa_weights(1, 1, 0.5, 2))
    assert np.allclose(out, [[3.0, -7.5]])


def test_validation():
    x, y, xq, w0 = problem(11)
    with pytest.raises(ValueError):
        lsa_weights(2, 1, 0.0, 3)
    with pytest.raises(ValueError):
        lsa_weights(2, 1, 0.1, 0)
    with pytest.raises(ValueError):
        lsa_weights(2, 1, 0.1, 3, np.zeros((2, 2)))
    with pytest.raises(ValueError):
        gd_predict(x, y, xq, 0.0)
    with pytest.raises(ValueError):
        gd_predict(x[:3], y, xq, 0.1)
    with pytest.raises(ValueError):
        gd_predict(x, y, xq[:, :2], 0.1)
    with pytest.raises(ValueError):
        gd_predict(x, y, xq, 0.1, steps=-1)
    with pytest.raises(ValueError):
        gd_predict(x, y, xq, 0.1, w0=w0.T)
    with pytest.raises(ValueError):
        lsa_predict(x, y, xq, 0.1, layers=0)
    with pytest.raises(ValueError):
        lsa_predict(x, y, xq, 0.0)
    with pytest.raises(ValueError):
        ridge_predict(x, y, xq, -1e-3)
    with pytest.raises(ValueError):
        ridge_predict(x[:, 0], y, xq, 1.0)
    with pytest.raises(ValueError):
        ridge_predict(x, y[:, 0], xq, 1.0)
    with pytest.raises(ValueError):
        ridge_predict(x, y, xq[0], 1.0)
    with pytest.raises(ValueError):
        ridge_predict(x[:3], y, xq, 1.0)
    with pytest.raises(ValueError):
        ridge_predict(x[:0], y[:0], xq, 1.0)
    with pytest.raises(ValueError):
        ridge_predict(x, y, xq[:, :2], 1.0)
