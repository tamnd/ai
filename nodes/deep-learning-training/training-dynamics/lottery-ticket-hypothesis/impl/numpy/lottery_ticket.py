# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/lottery-ticket-hypothesis/ (ai://NEW-lottery-ticket-hypothesis).
"""The lottery ticket procedure in NumPy: magnitude pruning masks, one round of iterative magnitude pruning with a
reset or a rewind, the pruning schedule, and the error barrier on the linear path between two networks.

A network is a list of weight arrays and a mask is a list of 0/1 arrays of the same shapes. magnitude_masks removes a
fraction of the weights that are still alive, those of smallest magnitude, in each layer (scope "layer") or over all
layers together (scope "global"); a weight that is pruned stays pruned. imp_round makes the next mask from trained
weights and sets the surviving weights back to theta_0 (a reset) or to theta_k (a rewind). remaining_fraction and
rounds_to_reach give the schedule (1 - p)^n of eq-remaining. interpolate and error_barrier give the error
E_alpha = E(alpha W1 + (1 - alpha) W2) on the linear path and the barrier sup_alpha E_alpha - (E(W1) + E(W2)) / 2 of
eq-barrier."""

import math

import numpy as np


def remaining_fraction(rate, rounds):
    """The fraction of weights left after `rounds` rounds that each prune `rate` of the survivors: (1 - rate)^rounds.

    eq: eq-remaining"""
    if not 0.0 <= rate < 1.0:
        raise ValueError("rate must be in [0, 1)")
    if rounds < 0:
        raise ValueError("rounds must be at least 0")
    return (1.0 - rate) ** rounds


def rounds_to_reach(rate, target):
    """The least number of rounds n with (1 - rate)^n <= target, for 0 < rate < 1 and 0 < target <= 1.

    eq: eq-rounds"""
    if not 0.0 < rate < 1.0:
        raise ValueError("rate must be in (0, 1)")
    if not 0.0 < target <= 1.0:
        raise ValueError("target must be in (0, 1]")
    # Count up rather than take ceil(log(target) / log(1 - rate)), whose rounding can be off by one.
    n = 0
    while remaining_fraction(rate, n) > target:
        n += 1
    return n


def _prune_count(rate, alive):
    """The number of weights to remove: rate * alive, rounded half up."""
    return int(math.floor(rate * alive + 0.5))


def magnitude_masks(weights, masks, rate, scope="layer"):
    """New masks that remove the fraction `rate` of the alive weights with the smallest |w|.

    With scope "layer", each layer loses round(rate * alive) of its own alive weights, and `rate` may be one number or
    one number per layer (Frankle and Carbin prune the output layer at half the rate). With scope "global", the alive
    weights of all layers are ranked together and round(rate * all alive) are removed. Ties are broken by position, so
    exactly that many are removed. A weight with mask 0 stays at 0."""
    if len(weights) != len(masks):
        raise ValueError("weights and masks must have the same length")
    for w, m in zip(weights, masks, strict=True):
        if np.shape(w) != np.shape(m):
            raise ValueError("each mask must have the shape of its weights")
    if scope == "layer":
        rates = list(rate) if np.ndim(rate) else [rate] * len(weights)
        if len(rates) != len(weights):
            raise ValueError("give one rate per layer")
        return [_prune(np.abs(w), m, r) for w, m, r in zip(weights, masks, rates, strict=True)]
    if scope == "global":
        if np.ndim(rate):
            raise ValueError("global pruning takes one rate")
        flat_w = np.concatenate([np.abs(np.asarray(w, dtype=np.float64)).ravel() for w in weights])
        flat_m = np.concatenate([np.asarray(m).ravel() for m in masks])
        pruned = _prune(flat_w, flat_m, rate)
        out, start = [], 0
        for m in masks:
            out.append(pruned[start : start + m.size].reshape(m.shape))
            start += m.size
        return out
    raise ValueError("scope must be 'layer' or 'global'")


def _prune(magnitude, mask, rate):
    """Set to 0 the mask entries of the round(rate * alive) alive weights of smallest magnitude."""
    if not 0.0 <= rate < 1.0:
        raise ValueError("rate must be in [0, 1)")
    mask = np.asarray(mask)
    new = (mask != 0).astype(np.float64).ravel()
    alive = np.flatnonzero(new)
    k = _prune_count(rate, alive.size)
    order = np.argsort(np.asarray(magnitude, dtype=np.float64).ravel()[alive], kind="stable")
    new[alive[order[:k]]] = 0.0
    return new.reshape(mask.shape)


def imp_round(trained, rewind, masks, rate, scope="layer"):
    """One round of iterative magnitude pruning: prune the trained weights, then set the survivors to `rewind`.

    `rewind` is theta_0 for the reset of Frankle and Carbin (2019) or theta_k for rewinding to step k as in Frankle et
    al. (2020). Returns the new masks and the masked starting weights m * rewind."""
    new_masks = magnitude_masks(trained, masks, rate, scope)
    return new_masks, [np.asarray(r, dtype=np.float64) * m for r, m in zip(rewind, new_masks, strict=True)]


def sparsity(masks):
    """The fraction of weights that are alive, ||m||_0 / |theta| (P_m in Frankle and Carbin)."""
    return float(sum(np.count_nonzero(m) for m in masks) / sum(np.size(m) for m in masks))


def interpolate(w1, w2, alpha):
    """The network alpha * W1 + (1 - alpha) * W2, layer by layer."""
    return [
        alpha * np.asarray(a, dtype=np.float64) + (1.0 - alpha) * np.asarray(b, dtype=np.float64)
        for a, b in zip(w1, w2, strict=True)
    ]


def error_barrier(error, w1, w2, num=30):
    """The error barrier on the linear path: max over the alphas of E(alpha W1 + (1 - alpha) W2) minus the mean of
    E(W1) and E(W2). `error` maps a network to a number; the alphas are num evenly spaced points of [0, 1], ends
    included (Frankle et al. use 30). Returns (barrier, alphas, errors).

    eq: eq-barrier"""
    if num < 2:
        raise ValueError("num must be at least 2")
    alphas = np.linspace(0.0, 1.0, num)
    errors = np.array([float(error(interpolate(w1, w2, a))) for a in alphas])
    mean_ends = 0.5 * (float(error(w1)) + float(error(w2)))
    return float(errors.max() - mean_ends), alphas, errors
