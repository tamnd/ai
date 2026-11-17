# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/qk-layernorm/ (ai://NEW-qk-layernorm).
"""QK-layernorm, the z-loss and the attention-entropy diagnostic, the NumPy reference.

attention is eq-attention of the page, with eq-qk-layernorm when qk_norm is on. z_loss is eq-z-loss,
z_loss_grad is eq-z-loss-grad, and attention_entropy is eq-entropy. Everything works in float64.
"""

import numpy as np


def layer_norm(x, gain=None, bias=None, eps=0.0):
    """LayerNorm over the last axis: (x - mean) / sqrt(var + eps) * gain + bias. eq: eq-layernorm"""
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    y = (x - mu) / np.sqrt(var + eps)
    if gain is not None:
        y = y * np.asarray(gain, dtype=np.float64)
    if bias is not None:
        y = y + np.asarray(bias, dtype=np.float64)
    return y


def softmax(z, axis=-1):
    """A softmax that subtracts the maximum first, so that large logits do not overflow."""
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def logsumexp(z, axis=-1):
    """log sum_j exp(z_j) along an axis, computed stably."""
    z = np.asarray(z, dtype=np.float64)
    m = z.max(axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.exp(z - m).sum(axis=axis))


def split_heads(x, heads):
    """(..., n, heads * dh) -> (..., heads, n, dh)."""
    *lead, n, width = x.shape
    if width % heads != 0:
        raise ValueError(f"the width {width} is not a multiple of heads={heads}")
    x = x.reshape(*lead, n, heads, width // heads)
    return np.swapaxes(x, -3, -2)


def merge_heads(x):
    """(..., heads, n, dh) -> (..., n, heads * dh)."""
    x = np.swapaxes(x, -3, -2)
    *lead, n, heads, dh = x.shape
    return x.reshape(*lead, n, heads * dh)


def attention(
    x,
    wq,
    wk,
    wv,
    heads=1,
    qk_norm=False,
    gain_q=None,
    gain_k=None,
    eps=0.0,
    causal=False,
):
    """Multi-head self-attention on x of shape (..., n, d_model). Return a dict.

    The keys are out (..., n, heads * dh), weights (..., heads, n, n) and logits (..., heads, n, n).
    With qk_norm, each query and each key is LayerNorm'd over its head dimension dh before the dot
    product (eq-qk-layernorm); gain_q and gain_k have shape (dh,) and default to ones. The logits are
    q . k / sqrt(dh) in both cases (eq-attention).
    """
    x = np.asarray(x, dtype=np.float64)
    q = split_heads(x @ np.asarray(wq, dtype=np.float64), heads)
    k = split_heads(x @ np.asarray(wk, dtype=np.float64), heads)
    v = split_heads(x @ np.asarray(wv, dtype=np.float64), heads)
    dh = q.shape[-1]
    if qk_norm:  # eq: eq-qk-layernorm
        q = layer_norm(q, gain_q, None, eps)
        k = layer_norm(k, gain_k, None, eps)
    logits = q @ np.swapaxes(k, -1, -2) / np.sqrt(dh)  # eq: eq-attention
    if causal:
        n = logits.shape[-1]
        logits = np.where(np.tril(np.ones((n, n), dtype=bool)), logits, -np.inf)
    weights = softmax(logits)
    out = merge_heads(weights @ v)
    return {"out": out, "weights": weights, "logits": logits, "q": q, "k": k}


def logit_bound(dh, gain_q=None, gain_k=None):
    """The bound sqrt(dh) * max|gain_q| * max|gain_k| on |logit| under QK-layernorm with eps = 0 (eq-bound)."""
    gq = 1.0 if gain_q is None else float(np.max(np.abs(gain_q)))
    gk = 1.0 if gain_k is None else float(np.max(np.abs(gain_k)))
    return float(np.sqrt(dh)) * gq * gk


def z_loss(logits, alpha=1e-4):
    """alpha * (log Z)^2, averaged over all leading positions (eq-z-loss). Z sums exp over the last axis."""
    lz = logsumexp(logits)
    return float(alpha * np.mean(lz**2))


def z_loss_grad(logits, alpha=1e-4):
    """The gradient of z_loss with respect to the logits: 2 alpha log Z softmax(z) / count (eq-z-loss-grad)."""
    logits = np.asarray(logits, dtype=np.float64)
    lz = logsumexp(logits)[..., None]
    count = max(lz.size, 1)
    return 2.0 * alpha * lz * softmax(logits) / count


def attention_entropy(weights, axis=-1):
    """The Shannon entropy in nats of each attention row, with 0 log 0 = 0 (eq-entropy)."""
    p = np.asarray(weights, dtype=np.float64)
    logp = np.log(p, out=np.zeros_like(p), where=p > 0)
    return -np.sum(p * logp, axis=axis)


def max_logit(logits):
    """The largest finite |logit|, the quantity that Dehghani et al. and Wortsman et al. track."""
    z = np.asarray(logits, dtype=np.float64)
    return float(np.max(np.abs(z[np.isfinite(z)])))
