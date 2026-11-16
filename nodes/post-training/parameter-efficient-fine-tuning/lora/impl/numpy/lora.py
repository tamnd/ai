# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/lora/ (ai://NEW-lora).
"""A linear layer with a low-rank adapter (LoRA), the NumPy reference.

The layer keeps the pretrained weight W0 frozen, of shape (d_out, d_in), and trains B, of shape (d_out, r), and A, of
shape (r, d_in). For a batch X of shape (n, d_in) it gives Y = X W0^T + b0 + s (X A^T) B^T, eq-lora-forward of the page,
with s = alpha / r (eq-lora-scale), or s = alpha / sqrt(r) for the rank-stabilized variant. The backward pass is
eq-lora-grads, and merge gives the single matrix W0 + s B A of eq-lora-merge.
"""

import math

import numpy as np


def config(d_in, d_out, r=8, alpha=None, rank_stabilized=False):
    """Return the configuration of the layer. alpha defaults to r, so that s = 1, as Hu et al. (2021) do."""
    d_in, d_out, r = int(d_in), int(d_out), int(r)
    if min(d_in, d_out) < 1:
        raise ValueError(f"the sizes must be positive, got d_in={d_in} and d_out={d_out}")
    if not 1 <= r <= min(d_in, d_out):
        raise ValueError(f"r must be in [1, min(d_in, d_out)] = [1, {min(d_in, d_out)}], got {r}")
    alpha = float(r if alpha is None else alpha)
    scale = alpha / math.sqrt(r) if rank_stabilized else alpha / r
    return {"d_in": d_in, "d_out": d_out, "r": r, "alpha": alpha, "rank_stabilized": bool(rank_stabilized), "scale": scale}


def init(rng, cfg, std=None):
    """Return the trainable parameters: A Gaussian with standard deviation std (default 1/sqrt(d_in)), and B zero.

    With B = 0 the update s B A is zero, so the layer starts as the pretrained layer.
    """
    std = 1.0 / math.sqrt(cfg["d_in"]) if std is None else float(std)
    a = rng.normal(0.0, std, size=(cfg["r"], cfg["d_in"]))
    b = np.zeros((cfg["d_out"], cfg["r"]))
    return {"A": a, "B": b}


def num_trainable(cfg):
    """The number of trainable entries of one adapter: r (d_in + d_out), eq-lora-count."""
    return cfg["r"] * (cfg["d_in"] + cfg["d_out"])


def forward(params, w0, x, cfg, b0=None):
    """Return (y, cache) for a batch x of shape (n, d_in): y = x W0^T + b0 + s (x A^T) B^T."""
    x = np.asarray(x, dtype=np.float64)
    w0 = np.asarray(w0, dtype=np.float64)
    if w0.shape != (cfg["d_out"], cfg["d_in"]):
        raise ValueError(f"W0 has the shape {w0.shape}, and the configuration needs {(cfg['d_out'], cfg['d_in'])}")
    if x.ndim != 2 or x.shape[1] != cfg["d_in"]:
        raise ValueError(f"x must have the shape (n, {cfg['d_in']}), got {x.shape}")
    z = x @ params["A"].T  # (n, r): the input in the coordinates of the adapter
    y = x @ w0.T + cfg["scale"] * (z @ params["B"].T)
    if b0 is not None:
        y = y + b0
    return y, {"x": x, "z": z}


def backward(params, w0, cache, dy, cfg):
    """Return (dx, grads) for the output gradient dy of shape (n, d_out). W0 and b0 get no gradient.

    dB = s dy^T (x A^T), dA = s (dy B)^T x and dx = dy W0 + s (dy B) A, eq-lora-grads.
    """
    dy = np.asarray(dy, dtype=np.float64)
    s = cfg["scale"]
    g = dy @ params["B"]  # (n, r)
    grads = {"A": s * (g.T @ cache["x"]), "B": s * (dy.T @ cache["z"])}
    dx = dy @ np.asarray(w0, dtype=np.float64) + s * (g @ params["A"])
    return dx, grads


def delta(params, cfg):
    """The update s B A, of the shape of W0."""
    return cfg["scale"] * (params["B"] @ params["A"])


def merge(params, w0, cfg):
    """Return W0 + s B A, the weight that gives the same outputs with no extra cost at inference (eq-lora-merge)."""
    return np.asarray(w0, dtype=np.float64) + delta(params, cfg)


def unmerge(params, w, cfg):
    """Return W - s B A, which recovers W0 from a merged weight, up to rounding."""
    return np.asarray(w, dtype=np.float64) - delta(params, cfg)
