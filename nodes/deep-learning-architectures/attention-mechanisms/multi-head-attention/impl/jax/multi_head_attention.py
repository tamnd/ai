# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/multi-head-attention/ (ai://NEW-multi-head-attention).
"""Multi-head attention as pure functions, the JAX reference.

It computes eq-scores, eq-softmax, eq-attention, eq-heads and eq-multihead of the page with jax.numpy, as the
NumPy reference does, and leaves the gradient to jax.grad. init and apply use the parameter dict of the NumPy
reference, so the shared fixture loads as it is. A mask is boolean and True means "may attend".
"""

import math

import jax
import jax.numpy as jnp


def config(d_model=512, n_heads=8, d_k=None, d_v=None):
    """Return the configuration. By default d_k = d_v = d_model / n_heads."""
    if (d_k is None or d_v is None) and d_model % n_heads != 0:
        raise ValueError(f"d_model = {d_model} is not a multiple of n_heads = {n_heads}")
    d_k = d_model // n_heads if d_k is None else int(d_k)
    d_v = d_model // n_heads if d_v is None else int(d_v)
    if min(d_model, n_heads, d_k, d_v) < 1:
        raise ValueError("every size must be at least 1")
    return {"d_model": int(d_model), "n_heads": int(n_heads), "d_k": d_k, "d_v": d_v}


def init(key, cfg):
    """Draw W_q, W_k, W_v and W_o with entries of variance 1 / fan-in; there are no biases."""
    M, H, Dk, Dv = cfg["d_model"], cfg["n_heads"], cfg["d_k"], cfg["d_v"]
    kq, kk, kv, ko = jax.random.split(key, 4)
    return {
        "W_q": jax.random.normal(kq, (M, H * Dk)) * M**-0.5,  # (M, H*Dk)
        "W_k": jax.random.normal(kk, (M, H * Dk)) * M**-0.5,  # (M, H*Dk)
        "W_v": jax.random.normal(kv, (M, H * Dv)) * M**-0.5,  # (M, H*Dv)
        "W_o": jax.random.normal(ko, (H * Dv, M)) * (H * Dv) ** -0.5,  # (H*Dv, M)
    }


def causal_mask(T, S=None):
    """Query t may attend to key s when s <= t + (S - T)."""
    S = T if S is None else S
    return jnp.tril(jnp.ones((T, S), dtype=bool), k=S - T)  # (T, S)


def padding_mask(lengths, S):
    """Key s of sequence b is real when s < lengths[b]."""
    return (jnp.arange(S)[None, :] < jnp.asarray(lengths)[:, None])[:, None, None, :]  # (B, 1, 1, S)


def attention(q, k, v, mask=None):
    """Scaled dot-product attention; a row with every key masked gives a zero output and a zero gradient."""
    scores = q @ jnp.swapaxes(k, -1, -2) / math.sqrt(q.shape[-1])  # eq: scores  (B, H, T, S)
    if mask is None:
        a = jax.nn.softmax(scores, axis=-1)  # eq: softmax
        return a @ v, a  # eq: attention  (B, H, T, Dv)
    mask = jnp.broadcast_to(mask, scores.shape)
    top = jax.lax.stop_gradient(jnp.max(jnp.where(mask, scores, -jnp.inf), axis=-1, keepdims=True))
    top = jnp.where(jnp.isfinite(top), top, 0.0)
    e = jnp.where(mask, jnp.exp(jnp.where(mask, scores - top, 0.0)), 0.0)  # no inf reaches the gradient
    z = jnp.sum(e, axis=-1, keepdims=True)
    a = e / jnp.where(z > 0, z, 1.0)  # (B, H, T, S)
    return a @ v, a


def _split(x, H):
    B, T, HD = x.shape
    return x.reshape(B, T, H, HD // H).transpose(0, 2, 1, 3)  # (B, H, T, D)


def apply(params, x, cfg, memory=None, mask=None):
    """Return (y, a): y is (B, T, M), a is (B, H, T, S). memory=None is self-attention."""
    H = cfg["n_heads"]
    src = x if memory is None else memory
    q = _split(x @ params["W_q"], H)  # eq: heads  (B, H, T, Dk)
    k = _split(src @ params["W_k"], H)  # (B, H, S, Dk)
    v = _split(src @ params["W_v"], H)  # (B, H, S, Dv)
    o, a = attention(q, k, v, mask)  # (B, H, T, Dv)
    B, _, T, Dv = o.shape
    concat = o.transpose(0, 2, 1, 3).reshape(B, T, H * Dv)  # (B, T, H*Dv)
    return concat @ params["W_o"], a  # eq: multihead  (B, T, M)


def param_count(cfg):
    """2 M H d_k + 2 M H d_v weights."""
    M, H = cfg["d_model"], cfg["n_heads"]
    return 2 * M * H * cfg["d_k"] + 2 * M * H * cfg["d_v"]
