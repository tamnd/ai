# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/layer-normalization/ (ai://RA5R92).
"""Layer normalization and RMSNorm as pure functions, the JAX reference.

The statistics are taken over the last len(shape) axes, separately for each position of the leading axes. layer_norm
is eq-ln-stats and eq-ln-forward, with the biased variance and eps added to it; rms_norm is eq-rms-forward. There is
no state and no training flag. post_ln and pre_ln are the placements of eq-post-ln and eq-pre-ln. jax.grad gives the
backward pass, and the tests compare it with eq-ln-backward and eq-rms-backward.
"""

import jax.numpy as jnp


def _config(shape, eps, affine, bias):
    shape = (int(shape),) if isinstance(shape, int) else tuple(int(n) for n in shape)
    if not shape or min(shape) < 1:
        raise ValueError(f"shape must be a non-empty shape of positive sizes, got {shape}")
    if eps is not None and eps < 0:
        raise ValueError(f"eps must be non-negative, got {eps}")
    return {"shape": shape, "eps": None if eps is None else float(eps), "affine": bool(affine),
            "bias": bool(affine and bias)}


def config(shape, eps=1e-5, elementwise_affine=True, bias=True):
    """Return the configuration of layer normalization over the trailing axes of the given shape."""
    if eps is None:
        raise ValueError("layer normalization needs a value of eps")
    return _config(shape, eps, elementwise_affine, bias)


def rms_config(shape, eps=None, elementwise_affine=True):
    """Return the configuration of RMSNorm. eps=None is the machine epsilon of the input dtype."""
    return _config(shape, eps, elementwise_affine, False)


def init(key, cfg):
    """Return the parameters: a gain of ones and a bias of zeros, one per normalized element."""
    del key
    p = {}
    if cfg["affine"]:
        p["gain"] = jnp.ones(cfg["shape"])
    if cfg["bias"]:
        p["bias"] = jnp.zeros(cfg["shape"])
    return p


def _axes(x, cfg):
    k = len(cfg["shape"])
    if x.ndim < k or tuple(x.shape[x.ndim - k:]) != cfg["shape"]:
        raise ValueError(f"the last axes of x must be {cfg['shape']}, got {x.shape}")
    return tuple(range(x.ndim - k, x.ndim))


def _affine(params, z):
    y = z * params["gain"] if "gain" in params else z
    return y + params["bias"] if "bias" in params else y


def layer_norm(params, x, cfg):
    """Return the layer normalization of x."""
    axes = _axes(x, cfg)
    mean = jnp.mean(x, axis=axes, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=axes, keepdims=True)
    return _affine(params, (x - mean) / jnp.sqrt(var + cfg["eps"]))


def rms_norm(params, x, cfg):
    """Return RMSNorm of x: x over its root mean square on the normalized axes, times the gain."""
    axes = _axes(x, cfg)
    eps = jnp.finfo(x.dtype).eps if cfg["eps"] is None else cfg["eps"]
    return _affine(params, x / jnp.sqrt(jnp.mean(x ** 2, axis=axes, keepdims=True) + eps))


def post_ln(x, sublayer, norm):
    """A Post-LN residual block: norm(x + sublayer(x))."""
    return norm(x + sublayer(x))


def pre_ln(x, sublayer, norm):
    """A Pre-LN residual block: x + sublayer(norm(x))."""
    return x + sublayer(norm(x))
