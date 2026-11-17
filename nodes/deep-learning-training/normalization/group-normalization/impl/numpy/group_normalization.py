# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/group-normalization/ (ai://QFEJMP).
"""Group normalization, the NumPy reference.

The input has the layout (N, C, *S): a batch axis, a channel axis and any number of spatial axes, (H, W) for images
and (T, H, W) for video. forward is eq-gn-set with eq-norm and eq-affine; backward is eq-gn-grad. normalize_over is
the general form eq-norm over a set of eq-sets or eq-gn-set. There is no running state
and no training mode: group normalization computes the same function in training and at test time.
"""

import numpy as np

KINDS = ("batch", "layer", "instance", "group")


def config(num_channels, num_groups=32, eps=1e-5, affine=True):
    """Return the configuration of a group normalization layer with C channels in G groups."""
    c, g = int(num_channels), int(num_groups)
    if c < 1 or g < 1:
        raise ValueError(f"the channels and the groups must be positive, got C = {c} and G = {g}")
    if c % g != 0:
        raise ValueError(f"the number of channels C = {c} must be divisible by the number of groups G = {g}")
    if eps < 0:
        raise ValueError(f"eps must be non-negative, got {eps}")
    return {"C": c, "G": g, "eps": float(eps), "affine": bool(affine)}


def init(cfg):
    """Return the parameters: a scale gamma of ones and a shift beta of zeros, one of each per channel."""
    if not cfg["affine"]:
        return {}
    return {"gamma": np.ones(cfg["C"]), "beta": np.zeros(cfg["C"])}


def _channel_shape(x):
    """The shape that broadcasts a vector of C entries along the channel axis of x."""
    return (1, x.shape[1]) + (1,) * (x.ndim - 2)


def normalize_over(x, groups, over_batch=False, eps=1e-5):
    """Return (xhat, mean, var): x normalized by the mean and the variance of each set S_i.

    The channels are split into `groups` contiguous groups. A set holds the entries of one group of one example,
    or of one group across the whole batch when over_batch is true. groups = C with over_batch gives batch
    normalization in training mode, groups = 1 layer normalization, groups = C instance normalization, and
    1 < groups < C group normalization.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim < 2:
        raise ValueError(f"x must have the layout (N, C, *S), got shape {x.shape}")
    n, c = x.shape[0], x.shape[1]
    if groups < 1 or c % groups != 0:
        raise ValueError(f"C = {c} channels cannot be split into {groups} groups")
    xg = x.reshape((n, groups, c // groups) + x.shape[2:])
    axes = tuple(range(2, xg.ndim))
    if over_batch:
        axes = (0,) + axes
    # eq: eq-norm, the mean and the biased variance over the set, then (x - mean) / sqrt(var + eps)
    mean = xg.mean(axis=axes, keepdims=True)
    var = ((xg - mean) ** 2).mean(axis=axes, keepdims=True)
    xhat = (xg - mean) / np.sqrt(var + eps)
    return xhat.reshape(x.shape), mean, var


def normalize(x, kind, num_groups=None, eps=1e-5):
    """Normalize x as batch (training mode), layer, instance or group normalization does, with no affine map."""
    x = np.asarray(x, dtype=np.float64)
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    c = x.shape[1]
    if kind == "group" and num_groups is None:
        raise ValueError("group normalization needs num_groups")
    # eq: eq-sets and eq-gn-set as choices of the groups and of the batch axis
    groups = {"batch": c, "layer": 1, "instance": c, "group": num_groups}[kind]
    return normalize_over(x, groups, over_batch=(kind == "batch"), eps=eps)[0]


def forward(params, x, cfg):
    """Return (y, cache): x normalized within each group of channels of each example, then scaled and shifted."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim < 2 or x.shape[1] != cfg["C"]:
        raise ValueError(f"x must have the layout (N, {cfg['C']}, *S), got shape {x.shape}")
    # eq: eq-gn-set, the set of one group of C/G channels of one example
    xhat, _, var = normalize_over(x, cfg["G"], over_batch=False, eps=cfg["eps"])
    if "gamma" in params:
        # eq: eq-affine, a scale and a shift per channel
        y = xhat * params["gamma"].reshape(_channel_shape(x)) + params["beta"].reshape(_channel_shape(x))
    else:
        y = xhat
    return y, {"xhat": xhat, "std": np.sqrt(var + cfg["eps"]), "G": cfg["G"]}


def backward(params, cache, dy):
    """Return (dx, grads) for the output gradient dy. The statistics depend on x, and the gradient flows through them."""
    dy = np.asarray(dy, dtype=np.float64)
    xhat, std, g = cache["xhat"], cache["std"], cache["G"]
    n, c = dy.shape[0], dy.shape[1]
    u = dy * params["gamma"].reshape(_channel_shape(dy)) if "gamma" in params else dy
    shape = (n, g, c // g) + dy.shape[2:]
    ug, hg = u.reshape(shape), xhat.reshape(shape)
    axes = tuple(range(2, len(shape)))
    # eq: eq-gn-grad, dx = (u - mean(u) - xhat * mean(u * xhat)) / std, with the means over the group
    dx = (ug - ug.mean(axis=axes, keepdims=True) - hg * (ug * hg).mean(axis=axes, keepdims=True)) / std
    grads = {}
    if "gamma" in params:
        other = (0,) + tuple(range(2, dy.ndim))
        grads["gamma"] = np.sum(dy * xhat, axis=other)
        grads["beta"] = np.sum(dy, axis=other)
    return dx.reshape(dy.shape), grads
