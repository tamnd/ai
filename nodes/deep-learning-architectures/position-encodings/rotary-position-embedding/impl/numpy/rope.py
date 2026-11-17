# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/rotary-position-embedding/ (ai://NEW-rotary-position-embedding).
"""Rotary position embedding (RoPE), the NumPy reference.

Each pair of dimensions (2i, 2i + 1) of a query or a key at position m is rotated by the angle m theta_i.
forward computes eq-rope-efficient with the angles of eq-theta and eq-pi; rotation_matrix builds
eq-rope-matrix; rotate_complex computes eq-rope-complex; scores computes eq-score; ntk_aware_base is eq-ntk.
The rotation is orthogonal, so backward rotates the output gradient by the opposite angles.
"""

import numpy as np


def config(d, base=10000.0, scale=1.0):
    """Return the configuration. d is the even head width, base is b, and scale is the factor s of position interpolation."""
    d = int(d)
    if d < 2 or d % 2 != 0:
        raise ValueError(f"d must be even and at least 2, got {d}")
    if base <= 1.0:
        raise ValueError(f"base must be greater than 1, got {base}")
    if scale <= 0.0:
        raise ValueError(f"scale must be positive, got {scale}")
    return {"d": d, "base": float(base), "scale": float(scale)}


def init(rng, cfg):
    """RoPE has no parameters."""
    del rng, cfg
    return {}


def frequencies(cfg):
    """Return theta_i = base^(-2i/d) for i = 0, ..., d/2 - 1, an array of length d/2."""
    i = np.arange(cfg["d"] // 2, dtype=np.float64)
    return cfg["base"] ** (-2.0 * i / cfg["d"])  # eq: eq-theta


def angles(positions, cfg):
    """Return the angles (m / s) theta_i, with the shape of positions plus one axis of length d/2."""
    m = np.asarray(positions, dtype=np.float64) / cfg["scale"]  # eq: eq-pi
    return np.multiply.outer(m, frequencies(cfg))


def rotate_pairs(x):
    """Map (x_1, x_2, x_3, x_4, ...) to (-x_2, x_1, -x_4, x_3, ...), the second vector of eq-rope-efficient."""
    y = np.empty_like(x)
    y[..., 0::2] = -x[..., 1::2]
    y[..., 1::2] = x[..., 0::2]
    return y


def forward(params, x, positions, cfg):
    """Rotate x, whose last two axes are (positions, d). Return (y, cache)."""
    del params
    x = np.asarray(x, dtype=np.float64)
    if x.shape[-1] != cfg["d"]:
        raise ValueError(f"the last axis of x has length {x.shape[-1]}, and d is {cfg['d']}")
    a = angles(positions, cfg)
    cos = np.repeat(np.cos(a), 2, axis=-1)
    sin = np.repeat(np.sin(a), 2, axis=-1)
    y = x * cos + rotate_pairs(x) * sin  # eq: eq-rope-efficient
    return y, {"cos": cos, "sin": sin}


def backward(params, cache, dy):
    """Return (dx, dparams). dx = R^T dy, the rotation by the opposite angles."""
    del params
    dy = np.asarray(dy, dtype=np.float64)
    return dy * cache["cos"] - rotate_pairs(dy) * cache["sin"], {}


def rotation_matrix(m, cfg):
    """Return the d x d block-diagonal matrix R_m of eq-rope-matrix."""
    d = cfg["d"]
    r = np.zeros((d, d))
    for i, a in enumerate(angles(m, cfg)):
        c, s = np.cos(a), np.sin(a)
        r[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = [[c, -s], [s, c]]  # eq: eq-rope-matrix
    return r


def rotate_complex(x, positions, cfg):
    """Rotate x with complex numbers: pair (x_2i, x_2i+1) is z = x_2i + i x_2i+1, multiplied by exp(i m theta_i)."""
    x = np.asarray(x, dtype=np.float64)
    z = (x[..., 0::2] + 1j * x[..., 1::2]) * np.exp(1j * angles(positions, cfg))  # eq: eq-rope-complex
    y = np.empty_like(x)
    y[..., 0::2] = z.real
    y[..., 1::2] = z.imag
    return y


def scores(q, k, pos_q, pos_k, cfg):
    """Return the matrix of dot products of the rotated queries and keys, eq-score, without the 1/sqrt(d) scale."""
    rq, _ = forward({}, q, pos_q, cfg)
    rk, _ = forward({}, k, pos_k, cfg)
    return rq @ np.swapaxes(rk, -1, -2)  # eq: eq-score


def ntk_aware_base(base, s, d):
    """Return the base b' = b s^(d / (d - 2)) of the NTK-aware change, eq-ntk."""
    if d <= 2:
        raise ValueError(f"d must be greater than 2, got {d}")
    return base * s ** (d / (d - 2))  # eq: eq-ntk
