# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/multilayer-perceptron/ (ai://YA59M7).
"""The multilayer perceptron, the JAX reference: pure init and apply for eq-layer and eq-output, jax.grad for the backward pass.

The parameter tree is {"W": [...], "b": [...]}, with W_l of shape (n_out, n_in), the tree of the NumPy reference.
"""

import math

import jax
import jax.numpy as jnp

ACTIVATIONS = {
    "relu": jax.nn.relu,
    "tanh": jnp.tanh,
    "sigmoid": jax.nn.sigmoid,
    "gelu": lambda z: z * 0.5 * (1.0 + jax.scipy.special.erf(z / math.sqrt(2.0))),
    "identity": lambda z: z,
}


def config(d_in=2, d_hidden=4, d_out=1, n_hidden=1, activation="relu"):
    if min(d_in, d_hidden, d_out, n_hidden) < 1 or activation not in ACTIVATIONS:
        raise ValueError("the widths and n_hidden must be at least 1, and the activation one of ACTIVATIONS")
    return {"d_in": d_in, "d_hidden": d_hidden, "d_out": d_out, "n_hidden": n_hidden, "activation": activation}


def widths(cfg):
    return [cfg["d_in"]] + [cfg["d_hidden"]] * cfg["n_hidden"] + [cfg["d_out"]]


def param_count(cfg):
    n = widths(cfg)
    return sum(a * b + b for a, b in zip(n[:-1], n[1:], strict=True))


def init(key, cfg, dtype=jnp.float64):
    n = widths(cfg)
    keys = jax.random.split(key, len(n) - 1)
    W = [
        jax.random.normal(k, (n_out, n_in), dtype) * math.sqrt(2.0 / n_in)
        for k, n_in, n_out in zip(keys, n[:-1], n[1:], strict=True)
    ]  # eq: eq-he
    return {"W": W, "b": [jnp.zeros(n_out, dtype) for n_out in n[1:]]}


def apply(params, x, cfg):
    act = ACTIVATIONS[cfg["activation"]]
    h = x  # (B, I)
    for W, b in zip(params["W"][:-1], params["b"][:-1], strict=True):
        h = act(h @ W.T + b)  # (B, H)  eq: eq-layer
    return h @ params["W"][-1].T + params["b"][-1]  # (B, O)  eq: eq-output


def mse_loss(params, x, t, cfg):
    y = apply(params, x, cfg)
    return 0.5 * jnp.sum((y - t) ** 2) / y.shape[0]  # eq: eq-loss


def sgd_step(params, grads, lr):
    return jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)  # eq: eq-sgd
