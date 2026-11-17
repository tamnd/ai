# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/lottery-ticket-hypothesis/ (ai://NEW-lottery-ticket-hypothesis).
"""The lottery ticket procedure in PyTorch, for the weights of a torch.nn.Module or any list of tensors.

magnitude_masks removes the fraction `rate` of the alive weights of smallest magnitude, per layer or over all layers.
imp_round prunes trained weights and loads the survivors of theta_0 (a reset) or theta_k (a rewind) into a module.
error_barrier walks the line between two state dicts and gives sup_alpha E_alpha - (E(W1) + E(W2)) / 2 (eq-barrier).
The NumPy reference holds the same rules; the tests share its fixture."""

import math

import torch


def _prune(magnitude, mask, rate):
    """Zero the mask entries of the round(rate * alive) alive weights of smallest magnitude; ties go by position."""
    if not 0.0 <= rate < 1.0:
        raise ValueError("rate must be in [0, 1)")
    new = (mask != 0).to(torch.float64).flatten()
    alive = torch.nonzero(new).flatten()
    k = int(math.floor(rate * alive.numel() + 0.5))
    order = torch.sort(magnitude.flatten().to(torch.float64)[alive], stable=True).indices
    new[alive[order[:k]]] = 0.0
    return new.reshape(mask.shape)


def magnitude_masks(weights, masks, rate, scope="layer"):
    """New 0/1 masks (float64) that prune the fraction `rate` of the alive weights with the smallest |w|.

    scope "layer" prunes each layer by its own rate (one number, or one per layer); scope "global" ranks all alive
    weights together. A weight with mask 0 stays at 0."""
    if len(weights) != len(masks):
        raise ValueError("weights and masks must have the same length")
    weights = [w.detach() for w in weights]
    if any(w.shape != m.shape for w, m in zip(weights, masks, strict=True)):
        raise ValueError("each mask must have the shape of its weights")
    if scope == "layer":
        rates = list(rate) if isinstance(rate, (list, tuple)) else [rate] * len(weights)
        if len(rates) != len(weights):
            raise ValueError("give one rate per layer")
        return [_prune(w.abs(), m, r) for w, m, r in zip(weights, masks, rates, strict=True)]
    if scope == "global":
        if isinstance(rate, (list, tuple)):
            raise ValueError("global pruning takes one rate")
        flat = _prune(torch.cat([w.abs().flatten() for w in weights]), torch.cat([m.flatten() for m in masks]), rate)
        return [
            part.reshape(m.shape) for part, m in zip(torch.split(flat, [m.numel() for m in masks]), masks, strict=True)
        ]
    raise ValueError("scope must be 'layer' or 'global'")


@torch.no_grad()
def imp_round(module, rewind, masks, rate, names, scope="layer"):
    """One round of IMP on `module`, whose parameters `names` hold trained weights.

    Prunes those weights by magnitude, then loads m * rewind[name] into each, where `rewind` is a state dict of theta_0
    (reset) or theta_k (rewind). Returns the new masks, in the order of `names`."""
    params = dict(module.named_parameters())
    new = magnitude_masks([params[n] for n in names], masks, rate, scope)
    for n, m in zip(names, new, strict=True):
        params[n].copy_(rewind[n].to(params[n].dtype) * m.to(params[n].dtype))
    return new


def interpolate(state1, state2, alpha):
    """The state dict alpha * W1 + (1 - alpha) * W2."""
    return {k: alpha * state1[k] + (1.0 - alpha) * state2[k] for k in state1}


@torch.no_grad()
def error_barrier(error, state1, state2, num=30):
    """max over num evenly spaced alpha in [0, 1] of E(alpha W1 + (1 - alpha) W2), minus (E(W1) + E(W2)) / 2.

    `error` maps a state dict to a number. Returns (barrier, alphas, errors).

    eq: eq-barrier"""
    if num < 2:
        raise ValueError("num must be at least 2")
    alphas = torch.linspace(0.0, 1.0, num, dtype=torch.float64)
    errors = torch.tensor([float(error(interpolate(state1, state2, float(a)))) for a in alphas], dtype=torch.float64)
    mean_ends = 0.5 * (float(error(state1)) + float(error(state2)))
    return float(errors.max()) - mean_ends, alphas, errors
