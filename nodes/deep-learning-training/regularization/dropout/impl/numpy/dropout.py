# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/dropout/ (ai://Y8VNMB).
"""Inverted dropout, the NumPy reference.

The forward pass is eq-inverted of the page. The backward pass uses the mask of the forward pass.
"""

import numpy as np


def config(p=0.5):
    """Return the configuration of the layer. p is the probability that a unit is set to zero."""
    p = float(p)
    if not 0.0 <= p < 1.0:
        raise ValueError(f"p must be in [0, 1), got {p}")
    return {"p": p}


def init(rng, cfg):
    """Dropout has no parameters."""
    del rng, cfg
    return {}


def sample_mask(rng, shape, p):
    """Draw a mask: each entry is 1 with probability 1 - p and 0 with probability p."""
    return (rng.random(shape) >= p).astype(np.float64)


def forward(params, x, cfg, rng=None, training=True, mask=None):
    """Apply dropout to x. Return (y, cache).

    In training, y = mask * x / (1 - p) (eq-inverted). Give rng to draw a new mask, or mask to use a fixed one.
    At inference, y = x.
    """
    del params
    x = np.asarray(x, dtype=np.float64)
    p = cfg["p"]
    if not training or p == 0.0:
        return x.copy(), {"scale": None}
    if mask is None:
        if rng is None:
            raise ValueError("training needs rng or mask")
        mask = sample_mask(rng, x.shape, p)
    mask = np.asarray(mask, dtype=np.float64)
    if mask.shape != x.shape:
        raise ValueError(f"the mask has the shape {mask.shape}, and x has the shape {x.shape}")
    scale = mask / (1.0 - p)
    return x * scale, {"scale": scale}


def backward(params, cache, dy):
    """Return (dx, dparams) for the output gradient dy.

    dx = mask * dy / (1 - p), with the mask of the forward pass. At inference, dx = dy.
    """
    del params
    dy = np.asarray(dy, dtype=np.float64)
    if cache["scale"] is None:
        return dy.copy(), {}
    return dy * cache["scale"], {}
