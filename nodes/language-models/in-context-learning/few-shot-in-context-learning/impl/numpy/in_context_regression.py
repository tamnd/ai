# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/few-shot-in-context-learning/ (ai://NEW-few-shot-in-context-learning).
"""In-context linear regression in NumPy: a linear self-attention layer with the weights of von Oswald et al.
(2023), which performs one gradient descent step on the in-context least-squares loss, the gradient descent
predictor it matches, and the ridge predictor that is Bayes-optimal under a Gaussian prior.

Shapes: the context inputs x are (n, d_x), the context outputs y are (n, d_y), the queries are (m, d_x), a linear
model W is (d_y, d_x) and predicts W x. Tokens are rows e = (x, y) of length d_x + d_y.
"""

import numpy as np


def _check_context(x, y, x_query):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_query = np.asarray(x_query, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or x_query.ndim != 2:
        raise ValueError("x, y and x_query must be 2-D arrays")
    if x.shape[0] != y.shape[0] or x.shape[0] == 0:
        raise ValueError("x and y must have the same, non-zero number of rows")
    if x_query.shape[1] != x.shape[1]:
        raise ValueError("x_query must have as many columns as x")
    return x, y, x_query


def _initial_weights(w0, dx, dy):
    if w0 is None:
        return np.zeros((dy, dx))
    w0 = np.asarray(w0, dtype=float)
    if w0.shape != (dy, dx):
        raise ValueError(f"w0 must have shape {(dy, dx)}")
    return w0


def lsa_weights(dx, dy, eta, n, w0=None):
    """Key, query, value and projection matrices (W_K, W_Q, W_V, P) of the construction of von Oswald et al.:
    W_K = W_Q keep the x block, W_V maps (x, y) to (0, W_0 x - y), and P = (eta / n) I."""
    if eta <= 0:
        raise ValueError("eta must be positive")
    if n < 1:
        raise ValueError("n must be at least 1")
    w0 = _initial_weights(w0, dx, dy)
    d = dx + dy
    wk = np.zeros((d, d))
    wk[:dx, :dx] = np.eye(dx)
    wv = np.zeros((d, d))
    wv[dx:, :dx] = w0
    wv[dx:, dx:] = -np.eye(dy)
    return wk, wk.copy(), wv, (eta / n) * np.eye(d)


def linear_self_attention(context, tokens, wk, wq, wv, p):
    """One linear self-attention layer without softmax: every row e of tokens becomes
    e + P W_V sum_i e_i e_i^T W_K^T W_Q e, with keys and values from the rows e_i of context."""
    context = np.asarray(context, dtype=float)
    tokens = np.asarray(tokens, dtype=float)
    scores = context @ wk.T @ wq @ tokens.T  # (n, m): e_i^T W_K^T W_Q e
    return tokens + (p @ wv @ context.T @ scores).T


def gd_step(x, y, w, eta):
    """One gradient descent step on L(W) = 1/(2n) sum_i ||W x_i - y_i||^2."""
    n = x.shape[0]
    return w - (eta / n) * (w @ x.T - y.T) @ x


def gd_predict(x, y, x_query, eta, w0=None, steps=1):
    """Predictions at x_query of the linear model after `steps` gradient descent steps from w0 (default 0)."""
    x, y, x_query = _check_context(x, y, x_query)
    if eta <= 0:
        raise ValueError("eta must be positive")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    w = _initial_weights(w0, x.shape[1], y.shape[1])
    for _ in range(steps):
        w = gd_step(x, y, w, eta)
    return x_query @ w.T


def lsa_predict(x, y, x_query, eta, w0=None, layers=1):
    """Predictions at x_query read off `layers` stacked linear self-attention layers with the construction:
    the query tokens start as (x_q, -W_0 x_q), and the prediction is minus their y block at the end."""
    x, y, x_query = _check_context(x, y, x_query)
    if layers < 1:
        raise ValueError("layers must be at least 1")
    dx, dy, n = x.shape[1], y.shape[1], x.shape[0]
    w0 = _initial_weights(w0, dx, dy)
    wk, wq, wv, p = lsa_weights(dx, dy, eta, n, w0)
    context = np.hstack([x, y])
    query = np.hstack([x_query, -x_query @ w0.T])
    for _ in range(layers):
        context, query = (
            linear_self_attention(context, context, wk, wq, wv, p),
            linear_self_attention(context, query, wk, wq, wv, p),
        )
    return -query[:, dx:]


def ridge_predict(x, y, x_query, lam):
    """Ridge regression fitted to the context, evaluated at x_query. lam = 0 gives minimum-norm least squares
    (the pseudo-inverse), which is also the lam -> 0 limit."""
    x, y, x_query = _check_context(x, y, x_query)
    if lam < 0:
        raise ValueError("lam must be non-negative")
    if lam == 0:
        w = np.linalg.pinv(x) @ y
    else:
        w = np.linalg.solve(x.T @ x + lam * np.eye(x.shape[1]), x.T @ y)
    return x_query @ w
