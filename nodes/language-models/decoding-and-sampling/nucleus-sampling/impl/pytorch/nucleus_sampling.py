# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/nucleus-sampling/ (ai://NEW-nucleus-sampling).
"""Nucleus (top-p) sampling in PyTorch, batched over the leading axes.

The filters act on logits with the vocabulary on the last axis and set the logits of removed tokens to -inf, so a
softmax afterwards gives the renormalized distribution P' of eq-renormalize. top_p_filter keeps the nucleus of
eq-top-p-set: after a stable descending sort, a token is removed when the mass of the tokens before it already
reaches p, so the token that crosses p stays. top_k_filter is eq-top-k-set and min_p_filter is eq-min-p.
sample_next divides by the temperature first, eq-temperature, then filters and draws with a torch.Generator.
"""

import torch


def top_p_filter(logits, p):
    """Logits with every token outside the nucleus V(p) set to -inf, eq-top-p-set."""
    if not 0.0 < p <= 1.0:
        raise ValueError(f"p must be in (0, 1], got {p}")
    sorted_logits, order = torch.sort(logits, dim=-1, descending=True, stable=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    mass_before = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
    remove_sorted = mass_before >= p
    remove_sorted[..., 0] = False
    remove = torch.zeros_like(remove_sorted).scatter(-1, order, remove_sorted)
    return logits.masked_fill(remove, float("-inf"))


def top_k_filter(logits, k):
    """Logits with every token outside the k most probable set to -inf, eq-top-k-set."""
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    k = min(k, logits.shape[-1])
    _, order = torch.sort(logits, dim=-1, descending=True, stable=True)
    remove = torch.ones_like(logits, dtype=torch.bool).scatter(
        -1, order[..., :k], False
    )
    return logits.masked_fill(remove, float("-inf"))


def min_p_filter(logits, p_base):
    """Logits with every token below p_base times the largest probability set to -inf, eq-min-p."""
    if not 0.0 < p_base <= 1.0:
        raise ValueError(f"p_base must be in (0, 1], got {p_base}")
    probs = torch.softmax(logits, dim=-1)
    threshold = p_base * probs.max(dim=-1, keepdim=True).values
    return logits.masked_fill(probs < threshold, float("-inf"))


def sample_next(logits, generator, temperature=1.0, top_p=None, top_k=None, min_p=None):
    """Draw one token per row: temperature, then the filters given, then a draw from the renormalized softmax.

    Returns the tokens, shape logits.shape[:-1], and the distributions they were drawn from."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    z = logits / temperature
    if top_k is not None:
        z = top_k_filter(z, top_k)
    if top_p is not None:
        z = top_p_filter(z, top_p)
    if min_p is not None:
        z = min_p_filter(z, min_p)
    probs = torch.softmax(z, dim=-1)
    flat = probs.reshape(-1, probs.shape[-1])
    tokens = torch.multinomial(flat, 1, generator=generator).reshape(probs.shape[:-1])
    return tokens, probs
