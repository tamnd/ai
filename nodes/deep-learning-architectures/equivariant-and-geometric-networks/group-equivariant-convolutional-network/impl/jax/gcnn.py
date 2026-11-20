# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/group-equivariant-convolutional-network/ (ai://1REXCC).
"""The group equivariant convolutional network for the group p4, the JAX reference.

The same model as the NumPy reference. Both equivariant layers build the augmented filter bank and call
lax.conv_general_dilated once, a cross-correlation as on the page (Cohen and Welling, section 7); jax.grad
differentiates through the bank. Run it in float64 (jax.config.update("jax_enable_x64", True)) to match the fixture.
"""

import jax
import jax.numpy as jnp
from jax import lax

NAMES = ("W1", "b1", "W2", "b2", "Wc", "bc")


def config(c_in=1, c1=4, c2=4, k=3, n_classes=3, mode="zero"):
    if min(c_in, c1, c2, n_classes) < 1 or k < 1 or k % 2 == 0:
        raise ValueError("the channel counts and n_classes must be at least 1, and k odd")
    if mode not in ("zero", "circular"):
        raise ValueError("mode is zero or circular")
    return {"c_in": c_in, "c1": c1, "c2": c2, "k": k, "n_classes": n_classes, "mode": mode}


def param_count(cfg):
    c_in, c1, c2, k, m = cfg["c_in"], cfg["c1"], cfg["c2"], cfg["k"], cfg["n_classes"]
    return c1 * c_in * k * k + c1 + 4 * c2 * c1 * k * k + c2 + m * c2 + m


def init(key, cfg):
    c_in, c1, c2, k, m = cfg["c_in"], cfg["c1"], cfg["c2"], cfg["k"], cfg["n_classes"]
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "W1": jax.random.normal(k1, (c1, c_in, k, k)) / jnp.sqrt(c_in * k * k),
        "b1": jnp.zeros(c1),
        "W2": jax.random.normal(k2, (c2, c1, 4, k, k)) / jnp.sqrt(4 * c1 * k * k),
        "b2": jnp.zeros(c2),
        "Wc": jax.random.normal(k3, (m, c2)) / jnp.sqrt(c2),
        "bc": jnp.zeros(m),
    }


def rotate(x, r=1):
    return jnp.rot90(x, r, axes=(-2, -1))


def rotate_p4(f, r=1):
    """A p4 map (B, C, 4, n, n) rotated by r times 90 degrees: each plane rotates and plane s moves to s + r."""
    return jnp.roll(rotate(f, r), r, axis=2)


def lift_bank(W):
    O, C, k, _ = W.shape  # noqa: E741
    return jnp.stack([rotate(W, r) for r in range(4)], axis=1).reshape(4 * O, C, k, k)


def gconv_bank(W):
    """W (O, C, 4, k, k) to (4 O, 4 C, k, k): filter (o, r) on input plane (c, s) is W[o, c, s - r] turned r times."""
    O, C, _, k, _ = W.shape  # noqa: E741
    rows = [rotate(jnp.roll(W, r, axis=2), r) for r in range(4)]
    return jnp.stack(rows, axis=1).reshape(4 * O, 4 * C, k, k)


def corr2d(x, w, mode):
    p = w.shape[-1] // 2
    if mode == "circular":
        x = jnp.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="wrap")
        pad = "VALID"
    else:
        pad = ((p, p), (p, p))
    return lax.conv_general_dilated(x, w, (1, 1), pad, dimension_numbers=("NCHW", "OIHW", "NCHW"))


def lift(x, W, mode="zero"):
    B, _, n, _ = x.shape
    return corr2d(x, lift_bank(W), mode).reshape(B, W.shape[0], 4, n, n)


def gconv(f, W, mode="zero"):
    B, C, _, n, _ = f.shape
    return corr2d(f.reshape(B, 4 * C, n, n), gconv_bank(W), mode).reshape(B, W.shape[0], 4, n, n)


def forward(params, x, cfg):
    mode = cfg["mode"]
    h1 = jax.nn.relu(lift(x, params["W1"], mode) + params["b1"][:, None, None, None])
    h2 = jax.nn.relu(gconv(h1, params["W2"], mode) + params["b2"][:, None, None, None])
    g = h2.max(axis=2).mean(axis=(2, 3))
    return g @ params["Wc"].T + params["bc"]


def loss_fn(params, x, labels, cfg):
    logp = jax.nn.log_softmax(forward(params, x, cfg), axis=1)
    return -jnp.take_along_axis(logp, labels[:, None], axis=1).mean()
