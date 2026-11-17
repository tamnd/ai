# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/weight-normalization/ (ai://NEW-weight-normalization).
"""Weight normalization, the NumPy reference.

A weight-normalized linear layer keeps a direction v of shape (out, in), a length g of shape (out,) and a bias b.
Row j of the weight is w_j = g_j v_j / ||v_j|| (eq-wn), and the layer computes y = x w^T + b (eq-neuron without the
nonlinearity). backward gives the gradients of eq-grad, and projection the matrix M_w of eq-proj.
data_dependent_init is eq-init, mean_only_forward and mean_only_backward are eq-mean-only and eq-mean-only-grad,
and spectral_norm and spectral_normalize are the power iteration of eq-power and the division of eq-sn.
"""

import numpy as np


def init(in_features, out_features, rng, scale=0.05):
    """Return the parameters: v from a normal of standard deviation `scale`, g = 1 and b = 0."""
    if in_features < 1 or out_features < 1:
        raise ValueError(f"the sizes must be positive, got in = {in_features} and out = {out_features}")
    v = rng.normal(0.0, scale, size=(out_features, in_features))
    return {"v": v, "g": np.ones(out_features), "b": np.zeros(out_features)}


def _row_norms(v):
    norms = np.sqrt(np.sum(v * v, axis=1))
    if np.any(norms == 0):
        raise ValueError("each row of v must be non-zero")
    return norms


def weight(v, g):
    """Return w with rows w_j = g_j v_j / ||v_j||, so that ||w_j|| = |g_j| for any non-zero v_j."""
    v = np.asarray(v, dtype=np.float64)
    # eq: eq-wn
    return (np.asarray(g, dtype=np.float64) / _row_norms(v))[:, None] * v


def forward(params, x):
    """Return (y, cache) with y = x w^T + b for a batch x of shape (N, in)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != params["v"].shape[1]:
        raise ValueError(f"x must have the shape (N, {params['v'].shape[1]}), got {x.shape}")
    w = weight(params["v"], params["g"])
    # eq: eq-neuron, the pre-activation w . x + b of each unit
    y = x @ w.T + params["b"]
    return y, {"x": x, "w": w}


def weight_grads(v, g, grad_w):
    """Return (grad_v, grad_g) from the gradient grad_w of the loss in w, row by row."""
    v = np.asarray(v, dtype=np.float64)
    norms = _row_norms(v)
    # eq: eq-grad, grad_g = grad_w . v / ||v|| and grad_v = (g / ||v||) grad_w - (g grad_g / ||v||^2) v
    grad_g = np.sum(grad_w * v, axis=1) / norms
    grad_v = (g / norms)[:, None] * grad_w - (g * grad_g / norms**2)[:, None] * v
    return grad_v, grad_g


def backward(params, cache, dy):
    """Return (dx, grads) for the output gradient dy of shape (N, out)."""
    dy = np.asarray(dy, dtype=np.float64)
    grad_w = dy.T @ cache["x"]
    grad_v, grad_g = weight_grads(params["v"], params["g"], grad_w)
    dx = dy @ cache["w"]
    return dx, {"v": grad_v, "g": grad_g, "b": dy.sum(axis=0)}


def projection(w):
    """Return M_w = I - w w^T / ||w||^2 for one weight vector w, the projection onto the complement of w."""
    w = np.asarray(w, dtype=np.float64)
    # eq: eq-proj
    return np.eye(w.size) - np.outer(w, w) / np.dot(w, w)


def data_dependent_init(params, x, eps=0.0):
    """Return new parameters with g = 1 / sigma[t] and b = -mu[t] / sigma[t] for t = v . x / ||v|| on the batch x.

    sigma is the biased standard deviation over the batch, so the pre-activations of x have mean 0 and variance 1
    when eps = 0. v is kept.
    """
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(params["v"], dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError(f"the batch must have at least two rows, got shape {x.shape}")
    # eq: eq-init, t = v . x / ||v||, then g <- 1 / sigma[t] and b <- -mu[t] / sigma[t]
    t = x @ v.T / _row_norms(v)
    mu = t.mean(axis=0)
    sigma = np.sqrt(((t - mu) ** 2).mean(axis=0) + eps)
    if np.any(sigma == 0):
        raise ValueError("a unit has a constant pre-activation on the batch; use a larger batch or eps > 0")
    return {"v": v.copy(), "g": 1.0 / sigma, "b": -mu / sigma}


def mean_only_forward(t, b):
    """Return t - mean(t) + b over the batch axis, mean-only batch normalization in training mode."""
    t = np.asarray(t, dtype=np.float64)
    # eq: eq-mean-only
    return t - t.mean(axis=0) + b


def mean_only_backward(dt_tilde):
    """Return (dt, db): the centred gradient of eq-mean-only-grad and the gradient of the bias."""
    dt_tilde = np.asarray(dt_tilde, dtype=np.float64)
    # eq: eq-mean-only-grad
    return dt_tilde - dt_tilde.mean(axis=0), dt_tilde.sum(axis=0)


def spectral_norm(w, u, n_iter=1):
    """Return (sigma, u, v) after n_iter rounds of the power iteration of eq-power, started from u.

    Each round sets v <- W^T u / ||W^T u|| and then u <- W v / ||W v||; the estimate is sigma = u^T W v.
    """
    w = np.asarray(w, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    if w.ndim != 2 or u.shape != (w.shape[0],):
        raise ValueError(f"W must be a matrix and u a vector of {w.shape[0]} entries")
    if n_iter < 1:
        raise ValueError(f"n_iter must be at least 1, got {n_iter}")
    for _ in range(n_iter):
        # eq: eq-power
        v = w.T @ u
        v = v / np.linalg.norm(v)
        u = w @ v
        u = u / np.linalg.norm(u)
    return float(u @ w @ v), u, v


def spectral_normalize(w, u, n_iter=1):
    """Return (W / sigma, u), with sigma from spectral_norm; the new u is kept for the next step."""
    sigma, u, _ = spectral_norm(w, u, n_iter)
    # eq: eq-sn
    return np.asarray(w, dtype=np.float64) / sigma, u
