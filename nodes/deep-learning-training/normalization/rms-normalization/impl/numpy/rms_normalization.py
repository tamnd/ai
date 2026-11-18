# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/rms-normalization/ (ai://3FZB17).
"""RMSNorm and pRMSNorm, the NumPy reference.

The statistic is taken over the last axis of the input, separately for each position of the leading axes: each
example, or each token of each sequence. forward is eq-rms, or eq-prms when p < 1; backward is eq-rms-grad. There is
no bias, no running state and no training mode: the same computation runs in training and at test time.
"""

import math

import numpy as np


def partial_count(n, p):
    """Return k = ceil(n p), the number of leading entries that pRMSNorm uses for its statistic."""
    n, p = int(n), float(p)
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    if not 0.0 < p <= 1.0:
        raise ValueError(f"p must be in (0, 1], got {p}")
    return max(1, math.ceil(round(n * p, 9)))  # round first, so that 10 * 0.3 gives 3 and not 4


def config(n, eps=1e-6, p=1.0, elementwise_affine=True):
    """Return the configuration of an RMSNorm layer over a last axis of size n."""
    if eps < 0:
        raise ValueError(f"eps must be non-negative, got {eps}")
    return {"n": int(n), "k": partial_count(n, p), "eps": float(eps), "affine": bool(elementwise_affine)}


def init(cfg):
    """Return the parameters: a gain of ones of size n, or nothing when the layer has no gain."""
    return {"gain": np.ones(cfg["n"])} if cfg["affine"] else {}


def forward(params, x, cfg):
    """Return (y, cache): x divided by the root mean square of its first k entries on the last axis, times the gain."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim < 1 or x.shape[-1] != cfg["n"]:
        raise ValueError(f"the last axis of x must have size {cfg['n']}, got shape {x.shape}")
    k = cfg["k"]
    # eq: eq-rms (k = n) and eq-prms (k = ceil(n p))
    r = np.sqrt(np.mean(x[..., :k] ** 2, axis=-1, keepdims=True) + cfg["eps"])
    z = x / r
    y = z * params["gain"] if "gain" in params else z
    return y, {"x": x, "r": r, "z": z, "k": k}


def backward(params, cache, dy):
    """Return (dx, grads) for the output gradient dy. The statistic depends on x, and the gradient flows through it."""
    dy = np.asarray(dy, dtype=np.float64)
    x, r, k = cache["x"], cache["r"], cache["k"]
    u = dy * params["gain"] if "gain" in params else dy
    # eq: eq-rms-grad, dx = (u - s * x (x . u) / (k r^2)) / r, with s the mask of the first k entries
    xu = np.sum(x * u, axis=-1, keepdims=True)
    dx = u / r
    dx[..., :k] -= x[..., :k] * xu / (k * r ** 3)
    grads = {}
    if "gain" in params:
        grads["gain"] = np.sum(dy * cache["z"], axis=tuple(range(dy.ndim - 1)))
    return dx, grads


def layer_norm(x, gain=None, eps=1e-6):
    """Layer normalization over the last axis, with no bias: RMSNorm of the centred input, eq-ln-as-rms."""
    x = np.asarray(x, dtype=np.float64)
    c = x - x.mean(axis=-1, keepdims=True)
    z = c / np.sqrt(np.mean(c ** 2, axis=-1, keepdims=True) + eps)
    return z if gain is None else z * gain
