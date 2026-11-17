# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/multi-head-attention/ (ai://NEW-multi-head-attention).
"""Multi-head attention, forward and backward, the NumPy reference.

The forward pass is eq-scores, eq-softmax, eq-attention, eq-heads and eq-multihead of the page.
The backward pass is eq-grad-out, eq-grad-v, eq-grad-weights, eq-grad-scores and eq-grad-qk of the derivation.
A mask is a boolean array that broadcasts to (B, H, T, S): True means that the query may attend to the key.
"""

import numpy as np


def config(d_model=512, n_heads=8, d_k=None, d_v=None):
    """Return the configuration. By default d_k = d_v = d_model / n_heads, as in Vaswani et al. (2017)."""
    if d_k is None or d_v is None:
        if d_model % n_heads != 0:
            raise ValueError(f"d_model = {d_model} is not a multiple of n_heads = {n_heads}")
    d_k = d_model // n_heads if d_k is None else int(d_k)
    d_v = d_model // n_heads if d_v is None else int(d_v)
    if min(d_model, n_heads, d_k, d_v) < 1:
        raise ValueError("every size must be at least 1")
    return {"d_model": int(d_model), "n_heads": int(n_heads), "d_k": d_k, "d_v": d_v}


def init(rng, cfg):
    """Draw the four projections with entries of variance 1 / fan-in. There are no biases, as in eq-heads."""
    M, H, Dk, Dv = cfg["d_model"], cfg["n_heads"], cfg["d_k"], cfg["d_v"]
    return {
        "W_q": rng.normal(0.0, M**-0.5, (M, H * Dk)),  # (M, H*Dk), the H matrices W^Q_r side by side
        "W_k": rng.normal(0.0, M**-0.5, (M, H * Dk)),  # (M, H*Dk)
        "W_v": rng.normal(0.0, M**-0.5, (M, H * Dv)),  # (M, H*Dv)
        "W_o": rng.normal(0.0, (H * Dv) ** -0.5, (H * Dv, M)),  # (H*Dv, M)
    }


def causal_mask(T, S=None):
    """Query t may attend to key s when s <= t + (S - T); with S = T that is s <= t."""
    S = T if S is None else S
    return np.tril(np.ones((T, S), dtype=bool), k=S - T)  # (T, S)


def padding_mask(lengths, S):
    """Key s of sequence b is real when s < lengths[b]; the other keys are padding."""
    lengths = np.asarray(lengths)
    return (np.arange(S)[None, :] < lengths[:, None])[:, None, None, :]  # (B, 1, 1, S)


def softmax(scores):
    """Softmax over the last axis. Masked entries are -inf; a row with every entry masked gives zeros."""
    m = np.max(scores, axis=-1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)  # an all-masked row has the max -inf
    e = np.exp(scores - m)
    z = np.sum(e, axis=-1, keepdims=True)
    return np.divide(e, z, out=np.zeros_like(e), where=z > 0)


def attention(q, k, v, mask=None):
    """Scaled dot-product attention. q: (B, H, T, Dk), k: (B, H, S, Dk), v: (B, H, S, Dv). Return (y, a)."""
    d_k = q.shape[-1]
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d_k)  # eq: scores  (B, H, T, S)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    a = softmax(scores)  # eq: softmax  (B, H, T, S)
    return a @ v, a  # eq: attention  (B, H, T, Dv)


def split_heads(x, n_heads):
    """(B, T, H*D) -> (B, H, T, D)."""
    B, T, HD = x.shape
    return x.reshape(B, T, n_heads, HD // n_heads).transpose(0, 2, 1, 3)


def merge_heads(x):
    """(B, H, T, D) -> (B, T, H*D), the concatenation of the heads."""
    B, H, T, D = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, T, H * D)


def forward(params, x, cfg, memory=None, mask=None):
    """Multi-head attention of the queries x: (B, T, M) over memory: (B, S, M).

    With memory=None, the keys and values come from x as well: that is self-attention.
    Return (y, cache), with y: (B, T, M) and the attention weights in cache["a"]: (B, H, T, S).
    """
    x = np.asarray(x, dtype=np.float64)
    src = x if memory is None else np.asarray(memory, dtype=np.float64)
    H = cfg["n_heads"]
    q = split_heads(x @ params["W_q"], H)  # eq: heads  (B, H, T, Dk)
    k = split_heads(src @ params["W_k"], H)  # (B, H, S, Dk)
    v = split_heads(src @ params["W_v"], H)  # (B, H, S, Dv)
    o, a = attention(q, k, v, mask)  # (B, H, T, Dv)
    concat = merge_heads(o)  # (B, T, H*Dv)
    y = concat @ params["W_o"]  # eq: multihead  (B, T, M)
    cache = {"x": x, "src": src, "self": memory is None, "q": q, "k": k, "v": v, "a": a, "concat": concat}
    return y, cache


def backward(params, cache, dy):
    """Return (dx, dmemory, dparams) for the output gradient dy: (B, T, M).

    For self-attention, dx holds the gradient through the queries, the keys and the values, and dmemory is None.
    """
    dy = np.asarray(dy, dtype=np.float64)
    q, k, v, a = cache["q"], cache["k"], cache["v"], cache["a"]
    H, d_k = q.shape[1], q.shape[-1]
    dW_o = np.einsum("btc,btm->cm", cache["concat"], dy)  # eq: grad-out  (H*Dv, M)
    do = split_heads(dy @ params["W_o"].T, H)  # (B, H, T, Dv)
    dv = np.swapaxes(a, -1, -2) @ do  # eq: grad-v  (B, H, S, Dv)
    da = do @ np.swapaxes(v, -1, -2)  # eq: grad-weights  (B, H, T, S)
    ds = a * (da - np.sum(da * a, axis=-1, keepdims=True))  # eq: grad-scores  (B, H, T, S)
    dq = ds @ k / np.sqrt(d_k)  # eq: grad-qk  (B, H, T, Dk)
    dk = np.swapaxes(ds, -1, -2) @ q / np.sqrt(d_k)  # (B, H, S, Dk)
    dq, dk, dv = merge_heads(dq), merge_heads(dk), merge_heads(dv)  # (B, T, H*Dk), (B, S, H*Dk), (B, S, H*Dv)
    x, src = cache["x"], cache["src"]
    dparams = {
        "W_q": np.einsum("btm,btc->mc", x, dq),
        "W_k": np.einsum("bsm,bsc->mc", src, dk),
        "W_v": np.einsum("bsm,bsc->mc", src, dv),
        "W_o": dW_o,
    }
    dx = dq @ params["W_q"].T  # (B, T, M)
    dsrc = dk @ params["W_k"].T + dv @ params["W_v"].T  # (B, S, M)
    if cache["self"]:
        return dx + dsrc, None, dparams
    return dx, dsrc, dparams


def param_count(cfg):
    """The number of weights: 2 M H d_k + 2 M H d_v, which is 4 M^2 when d_k = d_v = M / H."""
    M, H = cfg["d_model"], cfg["n_heads"]
    return 2 * M * H * cfg["d_k"] + 2 * M * H * cfg["d_v"]


def flops(cfg, B, T, S=None):
    """The FLOPs of the forward pass, a multiply-add counted as 2, with the softmax and the mask left out."""
    S = T if S is None else S
    M, H, Dk, Dv = cfg["d_model"], cfg["n_heads"], cfg["d_k"], cfg["d_v"]
    projections = 2 * B * T * M * H * Dk + 2 * B * S * M * H * (Dk + Dv) + 2 * B * T * H * Dv * M
    scores_and_sum = 2 * B * H * T * S * (Dk + Dv)
    return projections + scores_and_sum
