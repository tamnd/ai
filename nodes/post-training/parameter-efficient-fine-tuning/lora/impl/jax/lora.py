# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/lora/ (ai://NEW-lora).
"""A linear layer with a low-rank adapter (LoRA) as pure functions, the JAX reference.

apply gives y = x W0^T + b0 + s (x A^T) B^T, eq-lora-forward of the page, with s = alpha / r (eq-lora-scale) or
s = alpha / sqrt(r) for the rank-stabilized variant. Only the adapter {A, B} is a parameter; W0 and b0 are passed in
as constants, so jax.grad of a loss with respect to the adapter gives eq-lora-grads, and the tests check it.
"""

import math

import jax
import jax.numpy as jnp


def config(d_in, d_out, r=8, alpha=None, rank_stabilized=False):
    """Return the configuration of the layer. alpha defaults to r, so that s = 1."""
    d_in, d_out, r = int(d_in), int(d_out), int(r)
    if min(d_in, d_out) < 1:
        raise ValueError(f"the sizes must be positive, got d_in={d_in} and d_out={d_out}")
    if not 1 <= r <= min(d_in, d_out):
        raise ValueError(f"r must be in [1, min(d_in, d_out)] = [1, {min(d_in, d_out)}], got {r}")
    alpha = float(r if alpha is None else alpha)
    scale = alpha / math.sqrt(r) if rank_stabilized else alpha / r
    return {"d_in": d_in, "d_out": d_out, "r": r, "alpha": alpha, "rank_stabilized": bool(rank_stabilized), "scale": scale}


def init(key, cfg, std=None, dtype=jnp.float32):
    """Return the adapter: A Gaussian with standard deviation std (default 1/sqrt(d_in)), and B zero."""
    std = 1.0 / math.sqrt(cfg["d_in"]) if std is None else float(std)
    a = std * jax.random.normal(key, (cfg["r"], cfg["d_in"]), dtype=dtype)
    b = jnp.zeros((cfg["d_out"], cfg["r"]), dtype=dtype)
    return {"A": a, "B": b}


def num_trainable(cfg):
    """The number of trainable entries of one adapter: r (d_in + d_out), eq-lora-count."""
    return cfg["r"] * (cfg["d_in"] + cfg["d_out"])


def apply(params, w0, x, cfg, b0=None):
    """Return y = x W0^T + b0 + s (x A^T) B^T for x of shape (n, d_in). W0 is not differentiated."""
    w0 = jax.lax.stop_gradient(w0)
    y = x @ w0.T + cfg["scale"] * ((x @ params["A"].T) @ params["B"].T)
    if b0 is not None:
        y = y + jax.lax.stop_gradient(b0)
    return y


def delta(params, cfg):
    """The update s B A, of the shape of W0."""
    return cfg["scale"] * (params["B"] @ params["A"])


def merge(params, w0, cfg):
    """Return W0 + s B A, which gives the same outputs with no extra cost at inference (eq-lora-merge)."""
    return w0 + delta(params, cfg)
