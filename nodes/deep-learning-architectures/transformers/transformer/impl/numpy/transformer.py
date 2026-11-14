# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/transformer/ (ai://4W3NTQ).
"""The original encoder-decoder Transformer, the NumPy reference.

It computes eq-pe, eq-attention, eq-multihead and eq-ffn of the page, with post-norm layers and one embedding
matrix shared by the encoder, the decoder and the output layer. It is the forward pass at inference: dropout
and label smoothing belong to training, and the page of dropout has its own code.
"""

import numpy as np

EPS = 1e-6  # the eps of layer normalization; the paper does not give one


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


def _linear(rng, n_in, n_out):
    """A weight with variance 1 / n_in and a zero bias. The paper does not give the initialization."""
    return rng.normal(0.0, n_in**-0.5, (n_in, n_out)), np.zeros(n_out)  # (n_in, n_out), (n_out,)


def _attn_params(rng, M):
    p = {}
    for name in ("q", "k", "v", "o"):
        p["W_" + name], p["b_" + name] = _linear(rng, M, M)
    return p


def _norm_params(M):
    return {"gamma": np.ones(M), "beta": np.zeros(M)}  # (M,)


def init(rng, cfg):
    """Return the parameter tree: the shared embedding, then a list of encoder and of decoder layers."""
    M, F = cfg["d_model"], cfg["d_ff"]

    def ffn():
        W_1, b_1 = _linear(rng, M, F)
        W_2, b_2 = _linear(rng, F, M)
        return {"W_1": W_1, "b_1": b_1, "W_2": W_2, "b_2": b_2}

    def layer(attns, n_norms):  # the attentions in order, then the feedforward network, then the norms
        p = {name: _attn_params(rng, M) for name in attns}
        p["ffn"] = ffn()
        return p | {f"norm{i}": _norm_params(M) for i in range(1, n_norms + 1)}

    enc = [layer(("attn",), 2) for _ in range(cfg["n_layers"])]
    dec = [layer(("attn", "cross"), 3) for _ in range(cfg["n_layers"])]
    return {"embed": rng.normal(0.0, M**-0.5, (cfg["vocab"], M)), "enc": enc, "dec": dec}  # embed: (V, M)


def positional_encoding(T, M):
    """The sinusoids of eq-pe: sin on the even features, cos on the odd ones."""
    # eq: pe
    pos = np.arange(T)[:, None]  # (T, 1)
    j = np.arange((M + 1) // 2)[None, :]
    angle = pos / 10000.0 ** (2 * j / M)  # (T, ceil(M / 2))
    pe = np.zeros((T, M))  # (T, M)
    pe[:, 0::2] = np.sin(angle)
    pe[:, 1::2] = np.cos(angle[:, : M // 2])
    return pe


def attention(q, k, v, allowed=None):
    """Scaled dot-product attention, eq-attention, over the last two axes.

    allowed is a boolean array that broadcasts to the scores: False is the -inf of the mask M.
    A query that may attend to no key gets zero weights and a zero output.
    """
    # eq: attention
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(q.shape[-1])  # (B, H, T, S)
    if allowed is None:
        allowed = np.ones(scores.shape, dtype=bool)
    allowed = np.broadcast_to(allowed, scores.shape)
    top = np.max(np.where(allowed, scores, -np.inf), axis=-1, keepdims=True)
    top = np.where(np.isfinite(top), top, 0.0)
    e = np.where(allowed, np.exp(scores - top), 0.0)  # subtract the maximum, as 8.3.1 asks
    total = e.sum(axis=-1, keepdims=True)
    weights = e / np.where(total > 0, total, 1.0)  # (B, H, T, S)
    return weights @ v, weights  # (B, H, T, D), (B, H, T, S)


def multihead(p, x, memory, n_heads, allowed=None):
    """Multi-head attention, eq-multihead. Queries come from x, keys and values from memory."""
    # eq: multihead
    B, T, M = x.shape
    S = memory.shape[1]
    D = M // n_heads

    def heads(y, n):
        return y.reshape(B, n, n_heads, D).transpose(0, 2, 1, 3)  # (B, H, n, D)

    q = heads(x @ p["W_q"] + p["b_q"], T)  # (B, H, T, D)
    k = heads(memory @ p["W_k"] + p["b_k"], S)  # (B, H, S, D)
    v = heads(memory @ p["W_v"] + p["b_v"], S)  # (B, H, S, D)
    out, weights = attention(q, k, v, allowed)
    y = out.transpose(0, 2, 1, 3).reshape(B, T, M) @ p["W_o"] + p["b_o"]  # (B, T, M)
    return y, {"q": q, "k": k, "v": v, "weights": weights}


def ffn(p, x):
    """The position-wise feedforward network, eq-ffn."""
    # eq: ffn
    hidden = np.maximum(0.0, x @ p["W_1"] + p["b_1"])  # (B, T, F)
    return hidden @ p["W_2"] + p["b_2"], hidden  # (B, T, M), (B, T, F)


def layer_norm(p, x):
    """Normalize each position over its features, then scale and shift."""
    mean = x.mean(axis=-1, keepdims=True)  # (B, T, 1)
    var = x.var(axis=-1, keepdims=True)  # (B, T, 1)
    return p["gamma"] * (x - mean) / np.sqrt(var + EPS) + p["beta"]


def embed(E, tokens):
    """Row v of the embedding for token v, times sqrt(d_model), plus the positions."""
    M = E.shape[1]
    return E[tokens] * np.sqrt(M) + positional_encoding(tokens.shape[1], M)  # (B, T, M)


def encoder_layer(p, x, n_heads, src_allowed=None):
    a, cache = multihead(p["attn"], x, x, n_heads, src_allowed)
    x = layer_norm(p["norm1"], x + a)  # post-norm: LayerNorm(x + Sublayer(x))
    f, cache["hidden"] = ffn(p["ffn"], x)
    return layer_norm(p["norm2"], x + f), cache  # (B, S, M)


def decoder_layer(p, y, memory, n_heads, src_allowed=None):
    T = y.shape[1]
    causal = np.tril(np.ones((T, T), dtype=bool))  # (T, T): position t sees the positions s <= t
    a, self_cache = multihead(p["attn"], y, y, n_heads, causal)
    y = layer_norm(p["norm1"], y + a)
    c, cross_cache = multihead(p["cross"], y, memory, n_heads, src_allowed)
    y = layer_norm(p["norm2"], y + c)
    f, hidden = ffn(p["ffn"], y)
    return layer_norm(p["norm3"], y + f), {"self": self_cache, "cross": cross_cache, "hidden": hidden}


def forward(params, src, tgt, cfg, src_pad=None):
    """Return (logits, cache) for source tokens src (B, S) and target tokens tgt (B, T).

    logits[b, t] scores the token after tgt[b, :t + 1]. src_pad (B, S) marks padding of the source that no
    query may attend to. The cache holds the internals of each layer, for the shape tests.
    """
    src_allowed = None if src_pad is None else ~src_pad[:, None, None, :]  # (B, 1, 1, S)
    E = params["embed"]
    x = embed(E, src)  # (B, S, M)
    enc_cache = []
    for p in params["enc"]:
        x, c = encoder_layer(p, x, cfg["n_heads"], src_allowed)
        enc_cache.append(c)
    y = embed(E, tgt)  # (B, T, M)
    dec_cache = []
    for p in params["dec"]:
        y, c = decoder_layer(p, y, x, cfg["n_heads"], src_allowed)
        dec_cache.append(c)
    logits = y @ E.T  # (B, T, V): the output layer shares the embedding
    return logits, {"memory": x, "enc": enc_cache, "dec": dec_cache}


def param_count(cfg):
    """The number of parameters, from the sizes alone."""
    M, F = cfg["d_model"], cfg["d_ff"]
    attn = 4 * (M * M + M)
    ffn_ = M * F + F + F * M + M
    norm = 2 * M
    return cfg["vocab"] * M + cfg["n_layers"] * ((attn + ffn_ + 2 * norm) + (2 * attn + ffn_ + 3 * norm))


def flops(cfg, B, S, T):
    """FLOPs of forward with a multiply-add as 2, without softmax, normalization and activations."""
    M, F, V = cfg["d_model"], cfg["d_ff"], cfg["vocab"]

    def mm(rows, n_in, n_out):  # a product of a (rows, n_in) matrix and a (n_in, n_out) matrix
        return 2 * rows * n_in * n_out

    def attn(n_q, n_kv):  # four projections, the scores and the weighted sum of the values
        return 2 * mm(B * n_q, M, M) + 2 * mm(B * n_kv, M, M) + 2 * mm(B * n_q, M, n_kv)

    def ffn_(n):
        return mm(B * n, M, F) + mm(B * n, F, M)

    layer = attn(S, S) + ffn_(S) + attn(T, T) + attn(T, S) + ffn_(T)
    return cfg["n_layers"] * layer + mm(B * T, M, V)
