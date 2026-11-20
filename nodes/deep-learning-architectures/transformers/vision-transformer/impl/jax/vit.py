# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/vision-transformer/ (ai://DV9G9W).
"""The Vision Transformer of Dosovitskiy et al. (2021) in JAX, as pure functions of the parameters.

The parameters have the names and the layout of the NumPy reference, so the shared fixture loads as it is. The
patches are cut by a reshape in raster order, each flattened (row, column, channel); a class token goes in front,
learned position embeddings are added, and L pre-norm encoder layers with an exact GELU follow, then the final
layer normalization and the head on the class token, or on the average of the patches with pool="gap". The
gradient of eq-loss is left to jax.grad.
"""

import math

import jax
import jax.numpy as jnp

EPS = 1e-6  # of every layer normalization
POOLS = ("cls", "gap")
POSITIONS = ("learned", "none")


def config(
    image_size=8,
    patch_size=2,
    channels=3,
    d_model=8,
    n_heads=2,
    d_ff=16,
    n_layers=2,
    n_classes=5,
    head_hidden=0,
    pool="cls",
    pos="learned",
):
    """Return the config of the network; the defaults are the example config of shapes.yaml."""
    if min(image_size, patch_size, channels, d_model, n_heads, d_ff, n_layers) < 1 or head_hidden < 0:
        raise ValueError("the sizes must be at least 1")
    if image_size % patch_size:
        raise ValueError("the patch size must divide the image size")
    if d_model % n_heads:
        raise ValueError("the number of heads must divide d_model")
    if n_classes < 2:
        raise ValueError("a classifier needs at least two classes")
    if pool not in POOLS or pos not in POSITIONS:
        raise ValueError(f"pool must be one of {POOLS} and pos one of {POSITIONS}")
    return {
        "image_size": image_size,
        "patch_size": patch_size,
        "channels": channels,
        "d_model": d_model,
        "n_heads": n_heads,
        "d_ff": d_ff,
        "n_layers": n_layers,
        "n_classes": n_classes,
        "head_hidden": head_hidden,
        "pool": pool,
        "pos": pos,
    }


def n_tokens(cfg):
    return (cfg["image_size"] // cfg["patch_size"]) ** 2 + (cfg["pool"] == "cls")


def param_shapes(cfg):
    """The name and shape of every parameter, in the order of the NumPy reference."""
    P, C, D, F, K, Hh = (cfg[k] for k in ("patch_size", "channels", "d_model", "d_ff", "n_classes", "head_hidden"))
    shapes = {"embed_W": (P * P * C, D), "embed_b": (D,)}
    if cfg["pool"] == "cls":
        shapes["cls"] = (D,)
    if cfg["pos"] == "learned":
        shapes["pos"] = (n_tokens(cfg), D)
    for l in range(cfg["n_layers"]):  # noqa: E741
        shapes |= {
            f"l{l}_ln1_g": (D,), f"l{l}_ln1_b": (D,),
            f"l{l}_qkv_W": (D, 3 * D), f"l{l}_qkv_b": (3 * D,),
            f"l{l}_out_W": (D, D), f"l{l}_out_b": (D,),
            f"l{l}_ln2_g": (D,), f"l{l}_ln2_b": (D,),
            f"l{l}_fc1_W": (D, F), f"l{l}_fc1_b": (F,),
            f"l{l}_fc2_W": (F, D), f"l{l}_fc2_b": (D,),
        }  # fmt: skip
    shapes |= {"ln_g": (D,), "ln_b": (D,)}
    if Hh:
        shapes |= {"pre_W": (D, Hh), "pre_b": (Hh,), "head_W": (Hh, K), "head_b": (K,)}
    else:
        shapes |= {"head_W": (D, K), "head_b": (K,)}
    return shapes


def param_count(cfg):
    return sum(math.prod(s) for s in param_shapes(cfg).values())


def init(key, cfg, scale=0.1):
    """Random parameters: weights and biases N(0, scale^2), gains of the normalizations 1 + N(0, scale^2)."""
    shapes = param_shapes(cfg)
    keys = jax.random.split(key, len(shapes))
    return {
        name: scale * jax.random.normal(k, shape) + (1.0 if name.endswith("_g") else 0.0)
        for k, (name, shape) in zip(keys, shapes.items(), strict=True)
    }


def patchify(images, P):
    """(B, S, S, C) -> (B, N, P * P * C): the patches in raster order, each in the order (row, column, channel)."""
    B, S, _, C = images.shape
    G = S // P
    return images.reshape(B, G, P, G, P, C).transpose(0, 1, 3, 2, 4, 5).reshape(B, G * G, P * P * C)


def layer_norm(x, g, b):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + EPS) * g + b


def layer(p, l, z, H):  # noqa: E741
    """One pre-norm encoder layer, eq-attn and eq-mlp."""
    B, n, D = z.shape
    h = layer_norm(z, p[f"l{l}_ln1_g"], p[f"l{l}_ln1_b"])
    q, k, v = (h @ p[f"l{l}_qkv_W"] + p[f"l{l}_qkv_b"]).reshape(B, n, 3, H, D // H).transpose(2, 0, 3, 1, 4)
    a = jax.nn.softmax(q @ k.transpose(0, 1, 3, 2) / math.sqrt(D // H), axis=-1)
    z = z + (a @ v).transpose(0, 2, 1, 3).reshape(B, n, D) @ p[f"l{l}_out_W"] + p[f"l{l}_out_b"]
    h = layer_norm(z, p[f"l{l}_ln2_g"], p[f"l{l}_ln2_b"])
    u = jax.nn.gelu(h @ p[f"l{l}_fc1_W"] + p[f"l{l}_fc1_b"], approximate=False)
    return z + u @ p[f"l{l}_fc2_W"] + p[f"l{l}_fc2_b"]


def forward(params, cfg, images):
    """The class scores (B, K) of a batch of images (B, S, S, C)."""
    p = params
    z = patchify(images, cfg["patch_size"]) @ p["embed_W"] + p["embed_b"]
    if cfg["pool"] == "cls":
        z = jnp.concatenate([jnp.broadcast_to(p["cls"], (z.shape[0], 1, z.shape[-1])), z], axis=1)
    if cfg["pos"] == "learned":
        z = z + p["pos"]
    for l in range(cfg["n_layers"]):  # noqa: E741
        z = layer(p, l, z, cfg["n_heads"])
    z = layer_norm(z, p["ln_g"], p["ln_b"])
    y = z[:, 0] if cfg["pool"] == "cls" else z.mean(1)
    if cfg["head_hidden"]:
        y = jnp.tanh(y @ p["pre_W"] + p["pre_b"])
    return y @ p["head_W"] + p["head_b"]


def loss(params, cfg, images, labels):
    """eq-loss: the cross-entropy of the labels, averaged over the batch."""
    logp = jax.nn.log_softmax(forward(params, cfg, images), axis=-1)
    return -jnp.take_along_axis(logp, labels[:, None], axis=1).mean()
