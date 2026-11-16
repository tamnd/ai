# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/convolutional-neural-network/ (ai://NEW-convolutional-neural-network).
"""The convolutional layer and the small network of the page in JAX; the NumPy reference is the oracle.

conv2d calls jax.lax.conv_general_dilated, which computes the same cross-correlation as eq-conv; circular padding
is done first with jnp.pad. The parameter tree is the one of the NumPy reference.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

KERNEL = 5
POOL = 2


def config(channels=1, c1=6, c2=16, f1=120, f2=84, classes=10, size=32):
    if min(channels, c1, c2, f1, f2, classes) < 1 or size <= 12 or (size - 12) % 4:
        raise ValueError("all widths must be at least 1, and size - 12 a positive multiple of 4")
    return {"channels": channels, "c1": c1, "c2": c2, "f1": f1, "f2": f2, "classes": classes, "size": size}


def conv2d(x, w, b=None, stride=1, pad=0, mode="zeros"):
    """x (B, C, H, W) and w (D, C, k, k) give (B, D, H', W')."""
    if mode not in ("zeros", "circular"):
        raise ValueError("mode is zeros or circular")
    if mode == "circular" and pad:
        x = jnp.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="wrap")  # (B, C, H + 2p, W + 2p)
        pad = 0
    y = jax.lax.conv_general_dilated(x, w, (stride, stride), [(pad, pad), (pad, pad)],
                                     dimension_numbers=("NCHW", "OIHW", "NCHW"))  # eq: eq-conv  (B, D, H', W')
    return y if b is None else y + b[None, :, None, None]


def max_pool(x):
    return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 1, POOL, POOL), (1, 1, POOL, POOL), "VALID")  # eq: eq-maxpool


def apply(params, x, cfg):
    """Return (logits (B, K), cache) for images x (B, C, R, R)."""
    p = params
    h1 = jax.nn.relu(conv2d(x, p["conv1"]["w"], p["conv1"]["b"]))  # (B, C1, R - 4, R - 4)
    q1 = max_pool(h1)  # (B, C1, (R - 4) / 2, (R - 4) / 2)
    h2 = jax.nn.relu(conv2d(q1, p["conv2"]["w"], p["conv2"]["b"]))  # (B, C2, (R - 12) / 2, (R - 12) / 2)
    q2 = max_pool(h2)  # (B, C2, (R - 12) / 4, (R - 12) / 4)
    z1 = jax.nn.relu(q2.reshape(q2.shape[0], -1) @ p["fc1"]["w"] + p["fc1"]["b"])  # (B, F1)
    z2 = jax.nn.relu(z1 @ p["fc2"]["w"] + p["fc2"]["b"])  # (B, F2)
    logits = z2 @ p["fc3"]["w"] + p["fc3"]["b"]  # (B, K)
    return logits, {"conv1": h1, "pool1": q1, "conv2": h2, "pool2": q2, "fc1": z1, "fc2": z2}
