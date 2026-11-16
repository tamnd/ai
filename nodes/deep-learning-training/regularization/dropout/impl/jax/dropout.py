# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/dropout/ (ai://Y8VNMB).
"""Inverted dropout as pure functions, the JAX reference.

The forward pass is eq-inverted of the page. jax.grad gives the backward pass, and the tests compare it with the NumPy backward.
"""

import jax


def config(p=0.5):
    """Return the configuration of the layer. p is the probability that a unit is set to zero."""
    p = float(p)
    if not 0.0 <= p < 1.0:
        raise ValueError(f"p must be in [0, 1), got {p}")
    return {"p": p}


def init(key, cfg):
    """Dropout has no parameters."""
    del key, cfg
    return {}


def apply(params, x, cfg, key=None, training=True, mask=None):
    """Apply dropout to x. Give key to draw a new mask, or mask to use a fixed one.

    Split the key before each call. The same key gives the same mask.
    """
    del params
    p = cfg["p"]
    if not training or p == 0.0:
        return x
    if mask is None:
        if key is None:
            raise ValueError("training needs key or mask")
        mask = jax.random.bernoulli(key, 1.0 - p, x.shape).astype(x.dtype)
    if mask.shape != x.shape:
        raise ValueError(f"the mask has the shape {mask.shape}, and x has the shape {x.shape}")
    return x * mask / (1.0 - p)
