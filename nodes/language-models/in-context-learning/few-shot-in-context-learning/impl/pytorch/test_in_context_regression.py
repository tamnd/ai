# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/few-shot-in-context-learning/ (ai://JNEP9D).
"""Tests of the PyTorch version: the worked example of the page (the same numbers as the NumPy tests), the
matrices of the construction, the attention module against its definition, one layer against one gradient step
with the gradient from autograd, stacked layers against several steps, the rescaling of the weights, the
averaging estimate, ridge against the posterior mean, the lam -> 0 limit, and input validation."""

import pytest
import torch

from in_context_regression import (
    DTYPE,
    LinearSelfAttention,
    gd_predict,
    gd_step,
    lsa_predict,
    lsa_weights,
    ridge_predict,
)

WX = [[1.0], [2.0]]
WY = [[2.0], [4.0]]
WQ = [[3.0]]


def close(a, b, atol=1e-8):
    return torch.allclose(torch.as_tensor(a, dtype=DTYPE), torch.as_tensor(b, dtype=DTYPE), atol=atol)


def problem(seed, n=7, dx=4, dy=2, m=3):
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(*s, generator=g, dtype=DTYPE) for s in [(n, dx), (n, dy), (m, dx), (dy, dx)])


def test_worked_example():
    assert close(gd_predict(WX, WY, WQ, 0.5), [[7.5]])
    assert close(gd_predict(WX, WY, WQ, 0.5, steps=2), [[5.625]])
    assert close(gd_predict(WX, WY, WQ, 0.5, w0=[[1.0]]), [[6.75]])
    assert close(gd_predict(WX, WY, WQ, 0.5, w0=[[1.0]], steps=0), [[3.0]])
    assert close(gd_predict(WX, WY, WQ, 0.5, steps=0), [[0.0]])
    assert close(lsa_predict(WX, WY, WQ, 0.5), [[7.5]])
    assert close(lsa_predict(WX, WY, WQ, 0.5, layers=2), [[5.625]])
    assert close(lsa_predict(WX, WY, WQ, 0.5, w0=[[1.0]]), [[6.75]])
    assert close(ridge_predict(WX, WY, WQ, 1.0), [[5.0]])
    assert close(ridge_predict(WX, WY, WQ, 0.0), [[6.0]])
    assert close(ridge_predict([[1], [2]], [[2], [4]], [[3]], 1), [[5.0]])


def test_construction_matrices():
    wk, wq, wv, p = lsa_weights(2, 1, 0.6, 3, [[5.0, 7.0]])
    assert close(wk, [[1, 0, 0], [0, 1, 0], [0, 0, 0]], 0)
    assert close(wq, wk, 0)
    assert close(wv, [[0, 0, 0], [0, 0, 0], [5, 7, -1]], 0)
    assert close(p, 0.2 * torch.eye(3, dtype=DTYPE))
    wk, wq, wv, p = lsa_weights(1, 2, 1.0, 1)
    assert close(wv, [[0, 0, 0], [0, -1, 0], [0, 0, -1]], 0)
    assert close(wk, [[1, 0, 0], [0, 0, 0], [0, 0, 0]], 0)
    assert close(p, torch.eye(3, dtype=DTYPE))


def test_attention_module_matches_definition():
    g = torch.Generator().manual_seed(1)
    ctx, tok = torch.randn(5, 3, generator=g, dtype=DTYPE), torch.randn(2, 3, generator=g, dtype=DTYPE)
    wk, wq, wv, p = (torch.randn(3, 3, generator=g, dtype=DTYPE) for _ in range(4))
    want = torch.stack([e + p @ wv @ torch.stack([c * (c @ wk.T @ wq @ e) for c in ctx]).sum(0) for e in tok])
    with torch.no_grad():
        assert close(LinearSelfAttention(wk, wq, wv, p)(ctx, tok), want)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_one_layer_is_one_gradient_step(seed):
    x, y, xq, w0 = problem(seed)
    eta = 0.3
    w = w0.clone().requires_grad_(True)
    loss = ((x @ w.T - y) ** 2).sum() / (2 * len(x))
    loss.backward()
    assert w.grad is not None
    w1 = w0 - eta * w.grad
    assert close(gd_step(x, y, w0, eta), w1)
    layer = LinearSelfAttention(*lsa_weights(4, 2, eta, len(x), w0))
    ctx = torch.cat([x, y], dim=1)
    with torch.no_grad():
        out = layer(ctx, ctx)
    assert close(out[:, :4], x)
    assert close(out[:, 4:], y - x @ (w1 - w0).T)
    assert close(lsa_predict(x, y, xq, eta, w0), xq @ w1.T)


@pytest.mark.parametrize("layers", [1, 2, 3, 5])
def test_stacked_layers_are_gradient_steps(layers):
    x, y, xq, w0 = problem(4)
    assert close(lsa_predict(x, y, xq, 0.2, w0, layers), gd_predict(x, y, xq, 0.2, w0, layers))
    assert close(lsa_predict(x, y, xq, 0.2, None, layers), gd_predict(x, y, xq, 0.2, None, layers))


def test_many_steps_reach_least_squares():
    x, y, xq, _ = problem(5, n=30, dx=3)
    assert close(lsa_predict(x, y, xq, 0.5, layers=400), ridge_predict(x, y, xq, 0.0), atol=1e-6)


@pytest.mark.parametrize("s", [-3.0, 0.5, 7.0])
def test_rescaling(s):
    x, y, xq, w0 = problem(6)
    wk, wq, wv, p = lsa_weights(4, 2, 0.4, len(x), w0)
    ctx = torch.cat([x, y], dim=1)
    tok = torch.cat([xq, -xq @ w0.T], dim=1)
    eye = torch.eye(6, dtype=DTYPE)
    with torch.no_grad():
        a = LinearSelfAttention(wk, wq, wv, p)(ctx, tok)
        b = LinearSelfAttention(eye, wk.T @ wq / s, s * p @ wv, eye)(ctx, tok)
    assert close(a, b)


def test_averaging_estimate():
    x, y, xq, _ = problem(7)
    assert close(lsa_predict(x, y, xq, 1.0), xq @ x.T @ y / len(x))


def test_ridge_is_posterior_mean():
    x, y, xq, _ = problem(8, n=5, dx=6, dy=1)
    tau2, sigma2 = 2.0, 0.5
    post = tau2 * x.T @ torch.linalg.solve(tau2 * x @ x.T + sigma2 * torch.eye(5, dtype=DTYPE), y)
    assert close(ridge_predict(x, y, xq, sigma2 / tau2), xq @ post)


def test_ridge_limit_and_least_squares():
    x, y, xq, _ = problem(9, n=3, dx=6)
    assert close(ridge_predict(x, y, xq, 1e-9), ridge_predict(x, y, xq, 0.0), atol=1e-5)
    assert close(ridge_predict(x, y, xq, 0.0), xq @ torch.linalg.pinv(x) @ y)
    x, y, xq, _ = problem(10, n=12, dx=3)
    ols = torch.linalg.lstsq(x, y).solution
    assert close(ridge_predict(x, y, xq, 0.0), xq @ ols)
    assert not close(ridge_predict(x, y, xq, 5.0), xq @ ols)


def test_validation():
    x, y, xq, w0 = problem(11)
    bad = [
        lambda: lsa_weights(2, 1, 0.0, 3),
        lambda: lsa_weights(2, 1, 0.1, 0),
        lambda: lsa_weights(2, 1, 0.1, 3, torch.zeros(2, 2)),
        lambda: gd_predict(x, y, xq, 0.0),
        lambda: gd_predict(x, y, xq, 0.1, steps=-1),
        lambda: gd_predict(x, y, xq, 0.1, w0=w0.T),
        lambda: lsa_predict(x, y, xq, 0.1, layers=0),
        lambda: lsa_predict(x, y, xq, 0.0),
        lambda: ridge_predict(x, y, xq, -1e-3),
        lambda: ridge_predict(x[:, 0], y, xq, 1.0),
        lambda: ridge_predict(x, y[:, 0], xq, 1.0),
        lambda: ridge_predict(x, y, xq[0], 1.0),
        lambda: ridge_predict(x[:3], y, xq, 1.0),
        lambda: ridge_predict(x[:0], y[:0], xq, 1.0),
        lambda: ridge_predict(x, y, xq[:, :2], 1.0),
    ]
    for f in bad:
        with pytest.raises(ValueError):
            f()
