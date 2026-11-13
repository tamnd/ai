# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/transformer/ (ai://4W3NTQ).
"""The original encoder-decoder Transformer as pure functions, the JAX reference.

It computes eq-pe, eq-attention, eq-multihead and eq-ffn of the page with jax.numpy, as the NumPy reference
does, and leaves the gradient to jax.grad. init and apply use the parameter tree of the NumPy reference, so
the shared fixture loads as it is. Dropout is left out, as in the NumPy reference.
"""

import math

import jax
import jax.numpy as jnp

EPS = 1e-6


def config(d_model=512, n_heads=8, d_ff=2048, n_layers=6, vocab=37000):
    """Return the config of the model. The width of each head is d_model / n_heads."""
    if min(d_model, n_heads, d_ff, n_layers, vocab) < 1 or d_model % n_heads:
        raise ValueError("the sizes must be at least 1, and n_heads must divide d_model")
    return {
        "d_model": d_model,
        "n_heads": n_heads,
        "d_head": d_model // n_heads,
        "d_ff": d_ff,
        "n_layers": n_layers,
        "vocab": vocab,
    }


def init(key, cfg):
    """Return the parameter tree of the NumPy reference: weights with variance 1 / n_in, zero biases, unit scales."""
    M, F = cfg["d_model"], cfg["d_ff"]
    keys = iter(jax.random.split(key, 1 + 16 * cfg["n_layers"]))  # 6 weights per encoder layer, 10 per decoder layer

    def linear(n_in, n_out):
        return jax.random.normal(next(keys), (n_in, n_out)) * n_in**-0.5, jnp.zeros(n_out)

    def attn():
        p = {}
        for name in ("q", "k", "v", "o"):
            p["W_" + name], p["b_" + name] = linear(M, M)
        return p

    def ffn():
        (W_1, b_1), (W_2, b_2) = linear(M, F), linear(F, M)
        return {"W_1": W_1, "b_1": b_1, "W_2": W_2, "b_2": b_2}

    def norm():
        return {"gamma": jnp.ones(M), "beta": jnp.zeros(M)}

    embed = jax.random.normal(next(keys), (cfg["vocab"], M)) * M**-0.5  # (V, M)
    enc = [{"attn": attn(), "ffn": ffn(), "norm1": norm(), "norm2": norm()} for _ in range(cfg["n_layers"])]
    dec = [
        {"attn": attn(), "cross": attn(), "ffn": ffn(), "norm1": norm(), "norm2": norm(), "norm3": norm()}
        for _ in range(cfg["n_layers"])
    ]
    return {"embed": embed, "enc": enc, "dec": dec}


def positional_encoding(T, M):
    """The sinusoids of eq-pe: sin on the even features, cos on the odd ones."""
    # eq: pe
    pos = jnp.arange(T)[:, None]  # (T, 1)
    j = jnp.arange((M + 1) // 2)[None, :]
    angle = pos / 10000.0 ** (2 * j / M)  # (T, ceil(M / 2))
    pe = jnp.zeros((T, M))  # (T, M)
    return pe.at[:, 0::2].set(jnp.sin(angle)).at[:, 1::2].set(jnp.cos(angle[:, : M // 2]))


def attention(q, k, v, allowed=None):
    """Scaled dot-product attention, eq-attention. A query that may attend to no key gets a zero output."""
    # eq: attention
    scores = q @ jnp.swapaxes(k, -1, -2) / math.sqrt(q.shape[-1])  # (B, H, T, S)
    if allowed is None:
        allowed = jnp.ones(scores.shape, dtype=bool)
    allowed = jnp.broadcast_to(allowed, scores.shape)
    top = jnp.max(jnp.where(allowed, scores, -jnp.inf), axis=-1, keepdims=True)
    top = jax.lax.stop_gradient(jnp.where(jnp.isfinite(top), top, 0.0))
    e = jnp.where(allowed, jnp.exp(jnp.where(allowed, scores - top, 0.0)), 0.0)  # subtract the maximum, as 8.3.1 asks
    total = e.sum(axis=-1, keepdims=True)
    weights = e / jnp.where(total > 0, total, 1.0)  # (B, H, T, S)
    return weights @ v, weights  # (B, H, T, D), (B, H, T, S)


def multihead(p, x, memory, n_heads, allowed=None):
    """Multi-head attention, eq-multihead. Queries come from x, keys and values from memory."""
    # eq: multihead
    B, T, M = x.shape
    S = memory.shape[1]

    def heads(y, n):
        return y.reshape(B, n, n_heads, M // n_heads).transpose(0, 2, 1, 3)  # (B, H, n, D)

    q = heads(x @ p["W_q"] + p["b_q"], T)  # (B, H, T, D)
    k = heads(memory @ p["W_k"] + p["b_k"], S)  # (B, H, S, D)
    v = heads(memory @ p["W_v"] + p["b_v"], S)  # (B, H, S, D)
    out, weights = attention(q, k, v, allowed)
    y = out.transpose(0, 2, 1, 3).reshape(B, T, M) @ p["W_o"] + p["b_o"]  # (B, T, M)
    return y, {"q": q, "k": k, "v": v, "weights": weights}


def ffn(p, x):
    """The position-wise feedforward network, eq-ffn."""
    # eq: ffn
    hidden = jnp.maximum(0.0, x @ p["W_1"] + p["b_1"])  # (B, T, F)
    return hidden @ p["W_2"] + p["b_2"], hidden  # (B, T, M), (B, T, F)


def layer_norm(p, x):
    """Normalize each position over its features, then scale and shift."""
    mean = x.mean(axis=-1, keepdims=True)  # (B, T, 1)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)  # (B, T, 1)
    return p["gamma"] * (x - mean) / jnp.sqrt(var + EPS) + p["beta"]


def embed(E, tokens):
    """Row v of the embedding for token v, times sqrt(d_model), plus the positions."""
    M = E.shape[1]
    return E[tokens] * math.sqrt(M) + positional_encoding(tokens.shape[1], M)  # (B, T, M)


def apply(params, src, tgt, cfg, src_pad=None):
    """Return (logits, cache) for source tokens src (B, S) and target tokens tgt (B, T)."""
    H = cfg["n_heads"]
    src_allowed = None if src_pad is None else ~src_pad[:, None, None, :]  # (B, 1, 1, S)
    E = params["embed"]
    x = embed(E, src)  # (B, S, M)
    enc_cache = []
    for p in params["enc"]:
        a, c = multihead(p["attn"], x, x, H, src_allowed)
        x = layer_norm(p["norm1"], x + a)  # post-norm: LayerNorm(x + Sublayer(x))
        f, c["hidden"] = ffn(p["ffn"], x)
        x = layer_norm(p["norm2"], x + f)  # (B, S, M)
        enc_cache.append(c)
    y = embed(E, tgt)  # (B, T, M)
    T = tgt.shape[1]
    causal = jnp.tril(jnp.ones((T, T), dtype=bool))  # (T, T): position t sees the positions s <= t
    dec_cache = []
    for p in params["dec"]:
        a, self_cache = multihead(p["attn"], y, y, H, causal)
        y = layer_norm(p["norm1"], y + a)
        c, cross_cache = multihead(p["cross"], y, x, H, src_allowed)
        y = layer_norm(p["norm2"], y + c)
        f, hidden = ffn(p["ffn"], y)
        y = layer_norm(p["norm3"], y + f)  # (B, T, M)
        dec_cache.append({"self": self_cache, "cross": cross_cache, "hidden": hidden})
    logits = y @ E.T  # (B, T, V): the output layer shares the embedding
    return logits, {"memory": x, "enc": enc_cache, "dec": dec_cache}


def param_count(cfg):
    """The number of parameters, from the sizes alone."""
    M, F = cfg["d_model"], cfg["d_ff"]
    attn, ffn_, norm = 4 * (M * M + M), 2 * M * F + F + M, 2 * M
    return cfg["vocab"] * M + cfg["n_layers"] * ((attn + ffn_ + 2 * norm) + (2 * attn + ffn_ + 3 * norm))
