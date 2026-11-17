# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/deep-sets/ (ai://NEW-deep-sets).
"""Deep Sets of Zaheer et al. (2017) in JAX, as pure functions of the parameters.

The parameters have the names and the layout of the NumPy reference, so the shared fixture loads as it is. A batch of
sets is a (B, N, D0) array with a (B, N) mask; each equivariant layer is act(h Lambda - 1 pool(h) Gamma + 1 beta),
the pool skipping the padding, and head="set" pools the set and applies rho, head="element" scores each element. The
gradient of eq-loss is left to jax.grad.
"""

import jax
import jax.numpy as jnp

POOLS = ("sum", "mean", "max")
ACTS = ("tanh", "relu")
HEADS = ("set", "element")


def config(d_in=3, width=8, n_layers=2, n_classes=5, pool="max", readout="sum", act="tanh", interact=True,
           head="set"):  # fmt: skip
    """Return the config of the network; the defaults are the example config of shapes.yaml."""
    if min(d_in, width, n_layers) < 1:
        raise ValueError("the sizes must be at least 1")
    if head == "set" and n_classes < 1:
        raise ValueError("the set head needs at least one output; n_classes=1 gives a regression")
    if pool not in POOLS or readout not in POOLS:
        raise ValueError(f"pool and readout must be one of {POOLS}")
    if act not in ACTS or head not in HEADS:
        raise ValueError(f"act must be one of {ACTS} and head one of {HEADS}")
    return {"d_in": d_in, "width": width, "n_layers": n_layers, "n_classes": n_classes, "pool": pool,
            "readout": readout, "act": act, "interact": bool(interact), "head": head}  # fmt: skip


def param_shapes(cfg):
    w = [cfg["d_in"]] + [cfg["width"]] * cfg["n_layers"]
    shapes = {}
    for l in range(cfg["n_layers"]):
        shapes[f"eq{l}_Lambda"] = (w[l], w[l + 1])
        if cfg["interact"]:
            shapes[f"eq{l}_Gamma"] = (w[l], w[l + 1])
        shapes[f"eq{l}_beta"] = (w[l + 1],)
    H, K = cfg["width"], cfg["n_classes"]
    if cfg["head"] == "set":
        shapes |= {"rho_W": (H, H), "rho_b": (H,), "out_W": (H, K), "out_b": (K,)}
    else:
        shapes["out_Lambda"] = (H, 1)
        if cfg["interact"]:
            shapes["out_Gamma"] = (H, 1)
        shapes["out_beta"] = (1,)
    return shapes


def param_count(cfg):
    n = 0
    for shape in param_shapes(cfg).values():
        k = 1
        for s in shape:
            k *= s
        n += k
    return n


def init(key, cfg, scale=0.1):
    """Random parameters, every one N(0, scale^2)."""
    shapes = param_shapes(cfg)
    keys = jax.random.split(key, len(shapes))
    return {name: scale * jax.random.normal(k, shape) for k, (name, shape) in zip(keys, shapes.items())}


def pool(h, mask, kind):
    """Pool (B, N, D) over the elements of each set to (B, D), feature by feature, skipping the padding."""
    m = mask[..., None]
    if kind == "max":
        return jnp.where(m > 0, h, -jnp.inf).max(1)
    s = (h * m).sum(1)
    return s / m.sum(1) if kind == "mean" else s


def activate(z, act):
    return jnp.tanh(z) if act == "tanh" else jax.nn.relu(z)


def equivariant(params, prefix, h, mask, kind):
    """eq-equivariant before the activation."""
    z = h @ params[prefix + "Lambda"] + params[prefix + "beta"]
    if prefix + "Gamma" in params:
        z = z - (pool(h, mask, kind) @ params[prefix + "Gamma"])[:, None, :]
    return z


def features(params, cfg, x, mask):
    h = x * mask[..., None]
    for l in range(cfg["n_layers"]):
        h = activate(equivariant(params, f"eq{l}_", h, mask, cfg["pool"]), cfg["act"]) * mask[..., None]
    return h


def forward(params, cfg, x, mask=None):
    """(B, K) class scores for head="set", (B, N) element scores (padding at -inf) for head="element"."""
    mask = jnp.ones(x.shape[:2], x.dtype) if mask is None else jnp.asarray(mask, x.dtype)
    h = features(params, cfg, x, mask)
    if cfg["head"] == "set":
        r = activate(pool(h, mask, cfg["readout"]) @ params["rho_W"] + params["rho_b"], cfg["act"])
        return r @ params["out_W"] + params["out_b"]
    return jnp.where(mask > 0, equivariant(params, "out_", h, mask, cfg["pool"])[..., 0], -jnp.inf)


def loss(params, cfg, x, labels, mask=None):
    """eq-loss: the cross-entropy of the labels, averaged over the batch."""
    logp = jax.nn.log_softmax(forward(params, cfg, x, mask), -1)
    return -jnp.take_along_axis(logp, labels[:, None], 1).mean()
