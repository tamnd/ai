# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/rotary-position-embedding/ (ai://0V8E5T).
"""Rotary position embedding (RoPE) as pure functions, the JAX reference.

apply computes eq-rope-efficient with the angles of eq-theta and eq-pi. jax.grad gives the backward pass,
and the tests compare it with the rotation by the opposite angles of the NumPy backward.
"""

import jax.numpy as jnp


def config(d, base=10000.0, scale=1.0):
    """Return the configuration. d is the even head width, base is b, and scale is the factor s of position interpolation."""
    d = int(d)
    if d < 2 or d % 2 != 0:
        raise ValueError(f"d must be even and at least 2, got {d}")
    if base <= 1.0 or scale <= 0.0:
        raise ValueError(f"base must be greater than 1 and scale positive, got {base} and {scale}")
    return {"d": d, "base": float(base), "scale": float(scale)}


def init(key, cfg):
    """RoPE has no parameters."""
    del key, cfg
    return {}


def frequencies(cfg):
    i = jnp.arange(cfg["d"] // 2, dtype=jnp.float64)
    return cfg["base"] ** (-2.0 * i / cfg["d"])  # eq: eq-theta


def apply(params, x, positions, cfg):
    """Rotate x, whose last two axes are (positions, d)."""
    del params
    if x.shape[-1] != cfg["d"]:
        raise ValueError(f"the last axis of x has length {x.shape[-1]}, and d is {cfg['d']}")
    m = jnp.asarray(positions, dtype=x.dtype) / cfg["scale"]  # eq: eq-pi
    a = jnp.outer(m, frequencies(cfg))
    cos = jnp.repeat(jnp.cos(a), 2, axis=-1)
    sin = jnp.repeat(jnp.sin(a), 2, axis=-1)
    rot = jnp.stack([-x[..., 1::2], x[..., 0::2]], axis=-1).reshape(x.shape)
    return x * cos + rot * sin  # eq: eq-rope-efficient
