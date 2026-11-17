# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/layer-normalization/ (ai://NEW-layer-normalization).
"""Layer normalization and RMSNorm, the NumPy reference.

As in PyTorch, the statistics are taken over the last axes of the input, the axes of normalized_shape, separately
for each position of the leading axes: each example, or each token of each sequence. forward is eq-ln-stats and
eq-ln-forward, with the biased variance and eps added to it; backward is eq-ln-backward. rms_forward and
rms_backward are RMSNorm, eq-rms-forward and eq-rms-backward. There is no training mode: the same computation runs
at training and at test time, and no state is kept. post_ln and pre_ln are the two placements of eq-post-ln and
eq-pre-ln in a residual block.
"""

import numpy as np


def _shape(normalized_shape):
    shape = (int(normalized_shape),) if np.isscalar(normalized_shape) else tuple(int(n) for n in normalized_shape)
    if not shape or min(shape) < 1:
        raise ValueError(f"normalized_shape must be a non-empty shape of positive sizes, got {normalized_shape}")
    return shape


def config(normalized_shape, eps=1e-5, elementwise_affine=True, bias=True):
    """Return the configuration of layer normalization, with the PyTorch defaults."""
    if eps < 0:
        raise ValueError(f"eps must be non-negative, got {eps}")
    return {"shape": _shape(normalized_shape), "eps": float(eps), "affine": bool(elementwise_affine),
            "bias": bool(elementwise_affine and bias)}


def rms_config(normalized_shape, eps=None, elementwise_affine=True):
    """Return the configuration of RMSNorm. eps=None is the machine epsilon of float64, the PyTorch default for
    float64 inputs."""
    eps = np.finfo(np.float64).eps if eps is None else eps
    if eps < 0:
        raise ValueError(f"eps must be non-negative, got {eps}")
    return {"shape": _shape(normalized_shape), "eps": float(eps), "affine": bool(elementwise_affine), "bias": False}


def init(rng, cfg):
    """Return the parameters: a gain of ones and a bias of zeros of normalized_shape, one pair per element."""
    del rng
    p = {}
    if cfg["affine"]:
        p["gain"] = np.ones(cfg["shape"])
    if cfg["bias"]:
        p["bias"] = np.zeros(cfg["shape"])
    return p


def _axes(x, cfg):
    k = len(cfg["shape"])
    if x.ndim < k or x.shape[x.ndim - k:] != cfg["shape"]:
        raise ValueError(f"the last axes of x must be {cfg['shape']}, got {x.shape}")
    return tuple(range(x.ndim - k, x.ndim)), tuple(range(x.ndim - k))


def _affine(params, z):
    y = z * params["gain"] if "gain" in params else z
    return y + params["bias"] if "bias" in params else y


def _param_grads(params, dy, z, lead):
    grads = {}
    if "gain" in params:
        grads["gain"] = np.sum(dy * z, axis=lead)
    if "bias" in params:
        grads["bias"] = np.sum(dy, axis=lead)
    return grads


def forward(params, x, cfg):
    """Return (y, cache) for layer normalization over the axes of normalized_shape."""
    x = np.asarray(x, dtype=np.float64)
    axes, lead = _axes(x, cfg)
    mean = x.mean(axis=axes, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=axes, keepdims=True)  # biased: divide by H, as in eq-ln-stats
    inv = 1.0 / np.sqrt(var + cfg["eps"])
    xhat = (x - mean) * inv
    return _affine(params, xhat), {"z": xhat, "inv": inv, "axes": axes, "lead": lead, "n": int(np.prod(cfg["shape"]))}


def backward(params, cache, dy):
    """Return (dx, grads) for the output gradient dy, eq-ln-backward. The statistics depend on x, and the gradient
    flows through them."""
    dy = np.asarray(dy, dtype=np.float64)
    xhat, axes, n = cache["z"], cache["axes"], cache["n"]
    u = dy * params["gain"] if "gain" in params else dy
    su = np.sum(u, axis=axes, keepdims=True)
    sux = np.sum(u * xhat, axis=axes, keepdims=True)
    dx = cache["inv"] / n * (n * u - su - xhat * sux)
    return dx, _param_grads(params, dy, xhat, cache["lead"])


def rms_forward(params, x, cfg):
    """Return (y, cache) for RMSNorm: x divided by the root mean square over normalized_shape, then the gain."""
    x = np.asarray(x, dtype=np.float64)
    axes, lead = _axes(x, cfg)
    inv = 1.0 / np.sqrt(np.mean(x ** 2, axis=axes, keepdims=True) + cfg["eps"])
    z = x * inv
    return _affine(params, z), {"z": z, "inv": inv, "axes": axes, "lead": lead, "n": int(np.prod(cfg["shape"]))}


def rms_backward(params, cache, dy):
    """Return (dx, grads) for RMSNorm, eq-rms-backward."""
    dy = np.asarray(dy, dtype=np.float64)
    z, axes, n = cache["z"], cache["axes"], cache["n"]
    u = dy * params["gain"] if "gain" in params else dy
    dx = cache["inv"] * (u - z * np.sum(u * z, axis=axes, keepdims=True) / n)
    return dx, _param_grads(params, dy, z, cache["lead"])


def post_ln(x, sublayer, norm):
    """A Post-LN residual block, eq-post-ln: the normalization after the residual sum."""
    return norm(x + sublayer(x))


def pre_ln(x, sublayer, norm):
    """A Pre-LN residual block, eq-pre-ln: the normalization inside the residual branch."""
    return x + sublayer(norm(x))
