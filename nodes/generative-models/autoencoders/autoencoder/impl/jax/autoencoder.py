# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/autoencoder/ (ai://JC1Q9N).
"""The autoencoder of the page in JAX: pure init and apply for eq-encoder, eq-decoder and eq-layer, the losses
eq-cross-entropy and eq-squared-error, the corruptions of eq-dae, and eq-cae by jax.jacrev; jax.grad does the rest.

The parameter tree is that of the NumPy reference, {"enc_W": [...], "enc_b": [...], "dec_W": [...], "dec_b": [...]},
with W of shape (n_in, n_out) and a layer computing h @ W + b; with tied weights the tree has no "dec_W".
"""

import math

import jax
import jax.numpy as jnp

ACTIVATIONS = {"sigmoid": jax.nn.sigmoid, "relu": jax.nn.relu, "tanh": jnp.tanh, "linear": lambda a: a}


def config(d_in=6, d_hidden=5, n_hidden=1, d_code=2, act="sigmoid", code_act="linear", out="sigmoid", tied=False):
    return from_widths([d_in] + [d_hidden] * n_hidden + [d_code], act, code_act, out, tied)


def from_widths(widths, act="sigmoid", code_act="linear", out="sigmoid", tied=False):
    widths = tuple(int(w) for w in widths)
    if len(widths) < 2 or min(widths) < 1 or act not in ACTIVATIONS or code_act not in ACTIVATIONS:
        raise ValueError("at least two positive widths, and activations among ACTIVATIONS")
    if out not in ("sigmoid", "linear"):
        raise ValueError("out is sigmoid or linear")
    return {"widths": widths, "act": act, "code_act": code_act, "out": out, "tied": bool(tied)}


def param_count(cfg):
    w = cfg["widths"]
    weights = sum(a * b for a, b in zip(w[:-1], w[1:]))
    return (1 if cfg["tied"] else 2) * weights + sum(w[1:]) + sum(w[:-1])


def init(key, cfg, gain=1.0, dtype=jnp.float64):
    w, K = cfg["widths"], len(cfg["widths"]) - 1
    keys = iter(jax.random.split(key, 2 * K))

    def glorot(n_in, n_out):
        r = gain * math.sqrt(6.0 / (n_in + n_out))
        return jax.random.uniform(next(keys), (n_in, n_out), dtype, -r, r)

    p = {"enc_W": [glorot(w[i], w[i + 1]) for i in range(K)], "enc_b": [jnp.zeros(w[i + 1], dtype) for i in range(K)],
         "dec_b": [jnp.zeros(w[K - j - 1], dtype) for j in range(K)]}  # fmt: skip
    if not cfg["tied"]:
        p["dec_W"] = [glorot(w[K - j], w[K - j - 1]) for j in range(K)]
    return p


def dec_weights(p, cfg):
    return [W.T for W in reversed(p["enc_W"])] if cfg["tied"] else p["dec_W"]


def encode(p, cfg, x):
    K = len(p["enc_W"])
    h = x  # (B, D)
    for i, (W, b) in enumerate(zip(p["enc_W"], p["enc_b"])):
        h = ACTIVATIONS[cfg["code_act"] if i == K - 1 else cfg["act"]](h @ W + b)  # eq: eq-layer
    return h  # (B, C)  eq: eq-encoder


def apply(p, cfg, x):
    """The code y, the output before its activation a, and the reconstruction z of a (B, D) batch."""
    y = encode(p, cfg, x)
    Ws = dec_weights(p, cfg)
    h = y
    for j, (W, b) in enumerate(zip(Ws, p["dec_b"])):
        a = h @ W + b
        h = ACTIVATIONS[cfg["act"]](a) if j < len(Ws) - 1 else a
    z = jax.nn.sigmoid(h) if cfg["out"] == "sigmoid" else h  # eq: eq-decoder
    return {"y": y, "a": h, "z": z}


def loss(cfg, target, a, weight=None):
    """eq-cross-entropy from the logits, softplus(a) - x a, or eq-squared-error; summed over the components and
    averaged over the batch."""
    e = jax.nn.softplus(a) - target * a if cfg["out"] == "sigmoid" else (target - a) ** 2
    if weight is not None:
        e = e * weight
    return e.sum() / target.shape[0]


def corrupt(key, x, kind, level, low=0.0, high=1.0):
    """The corruptions of eq-dae; returns x~ and the boolean array of the corrupted components."""
    k1, k2 = jax.random.split(key)
    if kind == "gaussian":
        return x + level * jax.random.normal(k1, x.shape, x.dtype), jnp.ones(x.shape, bool)
    B, D = x.shape
    k = int(round(level * D))
    ranks = jnp.argsort(jnp.argsort(jax.random.uniform(k1, (B, D)), axis=1), axis=1)
    mask = ranks < k  # exactly k components of each example
    if kind == "mask":
        return jnp.where(mask, 0.0, x), mask
    if kind == "salt_pepper":
        return jnp.where(mask, jnp.where(jax.random.bernoulli(k2, 0.5, (B, D)), high, low), x), mask
    raise ValueError(kind)


def jacobian_penalty(p, cfg, x):
    """eq-cae: ||J_f(x)||_F^2 of the encoder, by jax.jacrev, averaged over the batch."""
    J = jax.vmap(jax.jacrev(lambda v: encode(p, cfg, v[None])[0]))(x)  # (B, C, D)
    return jnp.sum(J**2) / x.shape[0]


def contractive_penalty(p, x):
    """eq-cae-sigmoid, the closed form for one sigmoid layer."""
    W, b = p["enc_W"][0], p["enc_b"][0]
    h = jax.nn.sigmoid(x @ W + b)
    return jnp.sum((h * (1 - h)) ** 2 * jnp.sum(W**2, 0)) / x.shape[0]


def objective(p, cfg, x, key=None, noise=None, contraction=0.0, weight_decay=0.0):
    """The training objective of one batch: denoising when noise = (kind, level), with the contractive penalty and
    weight decay when their lambdas are nonzero."""
    xin = corrupt(key, x, *noise)[0] if noise is not None else x
    value = loss(cfg, x, apply(p, cfg, xin)["a"])
    if contraction:
        value = value + contraction * jacobian_penalty(p, cfg, xin)
    if weight_decay:
        value = value + weight_decay * sum(jnp.sum(W**2) for k in ("enc_W", "dec_W") for W in p.get(k, []))
    return value
