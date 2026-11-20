# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/nucleus-sampling/ (ai://NEW-nucleus-sampling).
"""Nucleus (top-p) sampling and its relatives in NumPy.

Each truncation rule takes a probability vector over the vocabulary and returns a boolean mask of the tokens it
keeps; renormalize then divides the kept probabilities by their mass p', eq-renormalize. top_p_mask is the nucleus of
eq-top-p-set, the shortest prefix of the tokens sorted by probability whose mass reaches p, including the token that
crosses p. top_k_mask is eq-top-k-set, min_p_mask is eq-min-p, eta_mask is eq-eta and typical_mask is eq-typical.
Sorting is stable, so equal probabilities are ordered by token index and every rule is deterministic. tempered is
eq-temperature. sample_next applies temperature, then the rules, then draws one token with a numpy Generator.
"""

import numpy as np


def softmax(logits):
    """softmax over the last axis, shifted by the maximum for stability."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def tempered(logits, t):
    """softmax(logits / t), eq-temperature; t must be positive."""
    if t <= 0:
        raise ValueError(f"temperature must be positive, got {t}")
    return softmax(np.asarray(logits, dtype=np.float64) / t)


def _sorted_order(probs):
    """Token indices by decreasing probability, ties by increasing index."""
    return np.argsort(-probs, kind="stable")


def top_p_mask(probs, p):
    """The nucleus V(p): the smallest sorted prefix with total probability at least p, eq-top-p-set."""
    if not 0.0 < p <= 1.0:
        raise ValueError(f"p must be in (0, 1], got {p}")
    probs = np.asarray(probs, dtype=np.float64)
    order = _sorted_order(probs)
    sorted_probs = probs[order]
    mass_before = np.cumsum(sorted_probs) - sorted_probs
    keep_sorted = mass_before < p
    keep_sorted[0] = True
    mask = np.zeros(probs.shape, dtype=bool)
    mask[order] = keep_sorted
    return mask


def top_k_mask(probs, k):
    """The k most probable tokens, eq-top-k-set."""
    probs = np.asarray(probs, dtype=np.float64)
    if not 1 <= k:
        raise ValueError(f"k must be at least 1, got {k}")
    mask = np.zeros(probs.shape, dtype=bool)
    mask[_sorted_order(probs)[:k]] = True
    return mask


def min_p_mask(probs, p_base):
    """Tokens with probability at least p_base times the largest probability, eq-min-p."""
    if not 0.0 < p_base <= 1.0:
        raise ValueError(f"p_base must be in (0, 1], got {p_base}")
    probs = np.asarray(probs, dtype=np.float64)
    return probs >= p_base * probs.max()


def entropy(probs):
    """Shannon entropy in nats; zero probabilities contribute nothing."""
    probs = np.asarray(probs, dtype=np.float64)
    nz = probs[probs > 0]
    return float(-(nz * np.log(nz)).sum())


def eta_mask(probs, epsilon, alpha=None):
    """Tokens with probability above eta = min(epsilon, alpha exp(-h)), eq-eta; alpha defaults to sqrt(epsilon).

    The most probable token is always kept, so the set is never empty."""
    probs = np.asarray(probs, dtype=np.float64)
    if alpha is None:
        alpha = np.sqrt(epsilon)
    eta = min(epsilon, alpha * np.exp(-entropy(probs)))
    mask = probs > eta
    mask[_sorted_order(probs)[0]] = True
    return mask


def typical_mask(probs, tau):
    """Locally typical set, eq-typical: tokens sorted by |H + log p|, taken until their mass reaches tau."""
    if not 0.0 < tau <= 1.0:
        raise ValueError(f"tau must be in (0, 1], got {tau}")
    probs = np.asarray(probs, dtype=np.float64)
    h = entropy(probs)
    with np.errstate(divide="ignore"):
        dist = np.abs(h + np.log(probs))
    order = np.argsort(dist, kind="stable")
    sorted_probs = probs[order]
    mass_before = np.cumsum(sorted_probs) - sorted_probs
    keep_sorted = mass_before < tau
    keep_sorted[0] = True
    mask = np.zeros(probs.shape, dtype=bool)
    mask[order] = keep_sorted
    return mask


def renormalize(probs, mask):
    """P restricted to the mask and divided by its mass p', eq-renormalize."""
    probs = np.asarray(probs, dtype=np.float64)
    kept = np.where(mask, probs, 0.0)
    total = kept.sum()
    if total <= 0:
        raise ValueError("the mask keeps no probability mass")
    return kept / total


def nucleus_distribution(probs, p):
    """The renormalized nucleus distribution P' of eq-renormalize."""
    return renormalize(probs, top_p_mask(probs, p))


def total_variation(a, b):
    """Total variation distance, half the L1 distance."""
    return 0.5 * float(np.abs(np.asarray(a) - np.asarray(b)).sum())


def sample_next(logits, rng, temperature=1.0, top_p=None, top_k=None, min_p=None):
    """Draw one token: temperature first, then each truncation rule given, then renormalize and sample.

    The rules are intersected, which is how they are combined in practice; the most probable token is in every set,
    so the intersection is never empty. Returns the token index and the distribution it was drawn from."""
    probs = tempered(logits, temperature)
    mask = np.ones(probs.shape, dtype=bool)
    if top_k is not None:
        mask &= top_k_mask(probs, top_k)
    if top_p is not None:
        mask &= top_p_mask(probs, top_p)
    if min_p is not None:
        mask &= min_p_mask(probs, min_p)
    dist = renormalize(probs, mask)
    return int(rng.choice(len(dist), p=dist)), dist
