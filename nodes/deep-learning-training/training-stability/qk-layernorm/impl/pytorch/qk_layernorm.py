# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/qk-layernorm/ (ai://NEW-qk-layernorm).
"""QK-layernorm attention, the z-loss and the attention-entropy diagnostic, the PyTorch reference.

The functions mirror the NumPy reference. Autograd gives the gradients; the tests compare the z-loss
gradient with eq-z-loss-grad and the attention with torch.nn.functional.scaled_dot_product_attention.
"""

import math

import torch
import torch.nn.functional as F


def split_heads(x, heads):
    """(..., n, heads * dh) -> (..., heads, n, dh)."""
    *lead, n, width = x.shape
    if width % heads != 0:
        raise ValueError(f"the width {width} is not a multiple of heads={heads}")
    return x.reshape(*lead, n, heads, width // heads).transpose(-3, -2)


def merge_heads(x):
    """(..., heads, n, dh) -> (..., n, heads * dh)."""
    x = x.transpose(-3, -2)
    return x.reshape(*x.shape[:-2], x.shape[-2] * x.shape[-1])


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
    """Multi-head self-attention, with LayerNorm on each query and key over dh when qk_norm is on.

    Return a dict with out, weights, logits, q and k, as the NumPy reference does.
    """
    q = split_heads(x @ wq, heads)
    k = split_heads(x @ wk, heads)
    v = split_heads(x @ wv, heads)
    dh = q.shape[-1]
    if qk_norm:  # eq: eq-qk-layernorm
        q = F.layer_norm(q, (dh,), gain_q, None, eps)
        k = F.layer_norm(k, (dh,), gain_k, None, eps)
    logits = q @ k.transpose(-1, -2) / math.sqrt(dh)  # eq: eq-attention
    if causal:
        n = logits.shape[-1]
        keep = torch.ones(n, n, dtype=torch.bool, device=logits.device).tril()
        logits = logits.masked_fill(~keep, float("-inf"))
    weights = torch.softmax(logits, dim=-1)
    return {
        "out": merge_heads(weights @ v),
        "weights": weights,
        "logits": logits,
        "q": q,
        "k": k,
    }


class QKNormAttention(torch.nn.Module):
    """Multi-head self-attention with optional QK-layernorm and learned gains, no biases."""

    def __init__(self, d_model, heads, qk_norm=True, eps=1e-6):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError(f"d_model={d_model} is not a multiple of heads={heads}")
        self.heads, self.qk_norm, self.eps = heads, qk_norm, eps
        self.wq = torch.nn.Parameter(torch.randn(d_model, d_model) / math.sqrt(d_model))
        self.wk = torch.nn.Parameter(torch.randn(d_model, d_model) / math.sqrt(d_model))
        self.wv = torch.nn.Parameter(torch.randn(d_model, d_model) / math.sqrt(d_model))
        dh = d_model // heads
        self.gain_q = torch.nn.Parameter(torch.ones(dh)) if qk_norm else None
        self.gain_k = torch.nn.Parameter(torch.ones(dh)) if qk_norm else None

    def forward(self, x, causal=False):
        return attention(
            x,
            self.wq,
            self.wk,
            self.wv,
            self.heads,
            self.qk_norm,
            self.gain_q,
            self.gain_k,
            self.eps,
            causal,
        )


def logit_bound(dh, gain_q=None, gain_k=None):
    """sqrt(dh) * max|gain_q| * max|gain_k|, the bound of eq-bound."""
    gq = 1.0 if gain_q is None else float(torch.as_tensor(gain_q).abs().max())
    gk = 1.0 if gain_k is None else float(torch.as_tensor(gain_k).abs().max())
    return math.sqrt(dh) * gq * gk


def z_loss(logits, alpha=1e-4):
    """alpha * (log Z)^2 averaged over the leading positions (eq-z-loss). Differentiable."""
    return alpha * torch.logsumexp(logits, dim=-1).pow(2).mean()


def z_loss_grad(logits, alpha=1e-4):
    """The closed form of eq-z-loss-grad: 2 alpha log Z softmax(z) / count."""
    lz = torch.logsumexp(logits, dim=-1, keepdim=True)
    count = max(lz.numel(), 1)
    return 2.0 * alpha * lz * torch.softmax(logits, dim=-1) / count


def attention_entropy(weights, dim=-1):
    """The entropy in nats of each attention row, with 0 log 0 = 0 (eq-entropy)."""
    return torch.special.entr(weights).sum(dim=dim)


def max_logit(logits):
    """The largest finite |logit|."""
    z = logits.detach()
    return float(z[torch.isfinite(z)].abs().max())
