# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/label-smoothing/ (ai://NEW-label-smoothing).
"""Label smoothing in NumPy.

smooth_labels builds the target q' = (1 - eps) one_hot(y) + eps u of eq-smoothed-target, with u uniform unless a prior
is given. smoothed_cross_entropy is the cross-entropy with that target, and decomposed_loss computes the same number
as the pair of losses of eq-decomposition. The gradient in the logits is softmax(z) - q'. confidence_penalty_loss is
the confidence penalty of Pereyra et al., optimal_logit_gap the gap of eq-logit-gap, and label_dropout draws the noisy
labels whose expected loss label smoothing computes.
"""

import numpy as np


def _check_eps(eps):
    if not 0.0 <= eps <= 1.0:
        raise ValueError(f"eps must be in [0, 1], got {eps}")


def log_softmax(z):
    """log softmax over the last axis, shifted by the maximum for stability."""
    z = np.asarray(z, dtype=float)
    m = z.max(axis=-1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=-1, keepdims=True))


def smooth_labels(y, num_classes, eps, prior=None):
    """The smoothed targets (1 - eps) one_hot(y) + eps * prior for integer labels y, shape (..., num_classes).

    prior defaults to the uniform distribution 1 / num_classes, the choice of Szegedy et al.; a unigram prior gives
    the unigram smoothing of Pereyra et al."""
    _check_eps(eps)
    if prior is None:
        prior = np.full(num_classes, 1.0 / num_classes)
    prior = np.asarray(prior, dtype=float)
    if prior.shape != (num_classes,) or np.any(prior < 0) or not np.isclose(prior.sum(), 1.0):
        raise ValueError("prior must be a distribution over the classes")
    one_hot = np.eye(num_classes)[y]
    return (1.0 - eps) * one_hot + eps * prior


def smoothed_cross_entropy(logits, y, eps, prior=None, reduction="mean"):
    """Cross-entropy of softmax(logits) against smooth_labels(y, K, eps, prior), per example or reduced."""
    logits = np.asarray(logits, dtype=float)
    q = smooth_labels(y, logits.shape[-1], eps, prior)
    per = -(q * log_softmax(logits)).sum(axis=-1)
    return _reduce(per, reduction)


def decomposed_loss(logits, y, eps, prior=None, reduction="mean"):
    """The same loss as the pair (1 - eps) NLL + eps H(u, p) of Szegedy et al."""
    logits = np.asarray(logits, dtype=float)
    k = logits.shape[-1]
    lp = log_softmax(logits)
    u = np.full(k, 1.0 / k) if prior is None else np.asarray(prior, dtype=float)
    nll = -np.take_along_axis(lp, np.asarray(y)[..., None], axis=-1)[..., 0]
    h_up = -(u * lp).sum(axis=-1)
    return _reduce((1.0 - eps) * nll + eps * h_up, reduction)


def kl_form(logits, y, eps, reduction="mean"):
    """(1 - eps) NLL + eps KL(u || p) + eps log K for uniform u: the same loss again."""
    logits = np.asarray(logits, dtype=float)
    lp = log_softmax(logits)
    nll = -np.take_along_axis(lp, np.asarray(y)[..., None], axis=-1)[..., 0]
    k = logits.shape[-1]
    return _reduce((1.0 - eps) * nll + eps * kl_from_uniform(logits) + eps * np.log(k), reduction)


def kl_from_uniform(logits):
    """D_KL(u || softmax(logits)) for the uniform u over the last axis: mean_k log(1 / (K p_k))."""
    lp = log_softmax(logits)
    return -(lp + np.log(lp.shape[-1])).mean(axis=-1)


def smoothed_cross_entropy_grad(logits, y, eps, prior=None, reduction="mean"):
    """The gradient of smoothed_cross_entropy in the logits: softmax(logits) - q', divided by N for the mean."""
    logits = np.asarray(logits, dtype=float)
    g = np.exp(log_softmax(logits)) - smooth_labels(y, logits.shape[-1], eps, prior)
    if reduction == "mean":
        return g / (g.size // g.shape[-1])
    if reduction in ("sum", "none"):
        return g
    raise ValueError(f"unknown reduction {reduction!r}")


def entropy(logits):
    """The entropy H(p) of softmax(logits) over the last axis."""
    lp = log_softmax(logits)
    return -(np.exp(lp) * lp).sum(axis=-1)


def confidence_penalty_loss(logits, y, beta, reduction="mean"):
    """The confidence penalty of Pereyra et al.: NLL - beta H(p)."""
    lp = log_softmax(logits)
    nll = -np.take_along_axis(lp, np.asarray(y)[..., None], axis=-1)[..., 0]
    return _reduce(nll - beta * entropy(logits), reduction)


def optimal_logit_gap(num_classes, eps):
    """The gap z_y - z_k (k != y) at the minimum of the smoothed loss of one example: log(1 + K (1 - eps) / eps).
    It is infinite for eps = 0, where the loss has no minimizer with finite logits."""
    _check_eps(eps)
    if eps == 0.0:
        return np.inf
    return float(np.log1p(num_classes * (1.0 - eps) / eps))


def eps_from_wrong_class_mass(eps_wrong, num_classes):
    """The eps of uniform smoothing that equals the convention with 1 - eps_wrong on the label and
    eps_wrong / (K - 1) on each other class: eps_wrong K / (K - 1)."""
    return eps_wrong * num_classes / (num_classes - 1)


def label_dropout(y, num_classes, eps, rng):
    """Keep each label with probability 1 - eps, else replace it by a uniform draw: the label noise whose expected
    cross-entropy is the smoothed cross-entropy."""
    _check_eps(eps)
    y = np.asarray(y)
    replace = rng.random(y.shape) < eps
    return np.where(replace, rng.integers(num_classes, size=y.shape), y)


def _reduce(per, reduction):
    if reduction == "mean":
        return float(per.mean())
    if reduction == "sum":
        return float(per.sum())
    if reduction == "none":
        return per
    raise ValueError(f"unknown reduction {reduction!r}")
