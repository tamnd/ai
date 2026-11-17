# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/residual-connection/ (ai://F5K1AF).
"""A stack of residual MLP blocks as pure functions, the JAX reference.

Each block computes x + F(x) with F(x) = W2 act(W1 x + b1) + b2 (eq-residual, eq-branch). jax.grad and jax.jacfwd
give the backward pass and the Jacobian, and the tests compare them with the NumPy reference on the same numbers.
"""

import jax
import jax.numpy as jnp


def config(d, n, hidden=None, activation="relu", residual=True):
    """Return the configuration: width d, n blocks, branch width hidden, the activation, and the shortcut on or off."""
    if activation not in ("relu", "tanh"):
        raise ValueError(f"unknown activation {activation}")
    return {"d": d, "n": n, "hidden": hidden or d, "activation": activation, "residual": residual}


def init(key, cfg, gain=1.0, zero_last=False):
    """Draw W1 with the He scale sqrt(2 / d) and W2 with gain / sqrt(hidden); zero_last sets every W2 to zero."""
    d, m = cfg["d"], cfg["hidden"]
    params = []
    for k in jax.random.split(key, cfg["n"]):
        k1, k2 = jax.random.split(k)
        w1 = jax.random.normal(k1, (m, d)) * jnp.sqrt(2.0 / d)
        w2 = jax.random.normal(k2, (d, m)) * (gain / jnp.sqrt(m))
        if zero_last:
            w2 = jnp.zeros((d, m))
        params.append({"W1": w1, "b1": jnp.zeros(m), "W2": w2, "b2": jnp.zeros(d)})
    return params


def branch(p, x, activation):
    """The residual branch F(x) on a batch x of shape (B, d) or a vector of shape (d,)."""
    z = x @ p["W1"].T + p["b1"]  # eq: eq-branch
    h = jnp.tanh(z) if activation == "tanh" else jax.nn.relu(z)
    return h @ p["W2"].T + p["b2"]


def apply(params, x, cfg):
    """Run the stack on x."""
    for p in params:
        f = branch(p, x, cfg["activation"])
        x = x + f if cfg["residual"] else f  # eq: eq-residual
    return x


def highway(p, x, activation, t=None, c=None, gate=None):
    """A highway layer y = H(x) T(x) + x C(x) (eq-highway); without t, T = sigmoid(x WT^T + bT) and C = 1 - T."""
    h = branch(p, x, activation)
    if t is None:
        t = jax.nn.sigmoid(x @ gate["WT"].T + gate["bT"])
    if c is None:
        c = 1.0 - t
    return h * t + x * c
