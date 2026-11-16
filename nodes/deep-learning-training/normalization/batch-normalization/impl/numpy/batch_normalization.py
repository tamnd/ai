# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/batch-normalization/ (ai://NEW-batch-normalization).
"""Batch normalization, the NumPy reference.

The input has the shape (N, C) or (N, C, ...), as in PyTorch: the statistics of channel c are taken over the batch
axis and all axes after the channel axis. In training the layer normalizes with the biased batch variance
(eq-batch-stats, eq-bn-forward), and update_state updates the running averages with the unbiased one (eq-running).
In evaluation it uses the running averages (eq-inference). backward is eq-bn-backward, and fold is the folding of
derivations/inference-fold.md.
"""

import numpy as np


def config(num_features, eps=1e-5, momentum=0.1):
    """Return the configuration. momentum is the PyTorch convention: the weight of the new batch, or None for a
    cumulative average. A Keras or Flax momentum rho is momentum = 1 - rho here."""
    num_features = int(num_features)
    if num_features < 1:
        raise ValueError(f"num_features must be positive, got {num_features}")
    if eps < 0:
        raise ValueError(f"eps must be non-negative, got {eps}")
    if momentum is not None and not 0.0 <= momentum <= 1.0:
        raise ValueError(f"momentum must be in [0, 1] or None, got {momentum}")
    return {"num_features": num_features, "eps": float(eps), "momentum": None if momentum is None else float(momentum)}


def init(rng, cfg):
    """Return the parameters: gamma at 1 and beta at 0, the PyTorch, Keras and Flax defaults."""
    del rng
    c = cfg["num_features"]
    return {"gamma": np.ones(c), "beta": np.zeros(c)}


def init_state(cfg):
    """Return the running statistics: mean 0, variance 1 and no batch seen, the PyTorch defaults."""
    c = cfg["num_features"]
    return {"mean": np.zeros(c), "var": np.ones(c), "count": 0}


def _axes(x, cfg):
    if x.ndim < 2 or x.shape[1] != cfg["num_features"]:
        raise ValueError(f"x must have the shape (N, {cfg['num_features']}, ...), got {x.shape}")
    return (0,) + tuple(range(2, x.ndim))


def _bcast(v, x):
    return v.reshape((1, -1) + (1,) * (x.ndim - 2))


def forward(params, x, cfg, state=None, training=True):
    """Return (y, cache). In training, normalize with the batch statistics; in evaluation, with state."""
    x = np.asarray(x, dtype=np.float64)
    axes = _axes(x, cfg)
    count = x.size // x.shape[1]  # the values per channel: N, or N H W for a feature map
    if training:
        if count < 2:
            raise ValueError(f"training needs more than 1 value per channel, got {count}")
        mean = x.mean(axis=axes)
        var = ((x - _bcast(mean, x)) ** 2).mean(axis=axes)  # the biased variance of Algorithm 1
    else:
        if state is None:
            raise ValueError("evaluation needs the running statistics in state")
        mean, var = state["mean"], state["var"]
    inv = 1.0 / np.sqrt(var + cfg["eps"])
    xhat = (x - _bcast(mean, x)) * _bcast(inv, x)
    y = _bcast(params["gamma"], x) * xhat + _bcast(params["beta"], x)
    cache = {"xhat": xhat, "inv": inv, "axes": axes, "count": count, "training": training,
             "batch_mean": mean if training else None, "batch_var": var if training else None}
    return y, cache


def update_state(state, cache, cfg):
    """Return the running statistics after one training step, eq-running: the unbiased batch variance goes in."""
    if not cache["training"]:
        return state
    m = cache["count"]
    unbiased = cache["batch_var"] * m / (m - 1)
    count = state["count"] + 1
    alpha = 1.0 / count if cfg["momentum"] is None else cfg["momentum"]
    return {"mean": (1 - alpha) * state["mean"] + alpha * cache["batch_mean"],
            "var": (1 - alpha) * state["var"] + alpha * unbiased, "count": count}


def backward(params, cache, dy):
    """Return (dx, grads) for the output gradient dy, eq-bn-backward.

    In training the gradient flows through the batch mean and variance. In evaluation the layer is affine, and
    dx = gamma / sqrt(var + eps) dy.
    """
    dy = np.asarray(dy, dtype=np.float64)
    xhat, axes = cache["xhat"], cache["axes"]
    grads = {"gamma": np.sum(dy * xhat, axis=axes), "beta": np.sum(dy, axis=axes)}
    scale = _bcast(params["gamma"] * cache["inv"], dy)
    if not cache["training"]:
        return scale * dy, grads
    m = cache["count"]
    dx = scale / m * (m * dy - _bcast(grads["beta"], dy) - xhat * _bcast(grads["gamma"], dy))
    return dx, grads


def fold(params, state, cfg, w, b=None):
    """Fold the layer in evaluation mode into the linear layer x = z W^T + b before it.

    w has the shape (C, ...): a dense weight (C, k) or a convolution filter (C, k, kh, kw). Return (w', b').
    """
    a = params["gamma"] / np.sqrt(state["var"] + cfg["eps"])
    b = np.zeros(cfg["num_features"]) if b is None else np.asarray(b, dtype=np.float64)
    w2 = np.asarray(w, dtype=np.float64) * a.reshape((-1,) + (1,) * (np.ndim(w) - 1))
    return w2, a * (b - state["mean"]) + params["beta"]
