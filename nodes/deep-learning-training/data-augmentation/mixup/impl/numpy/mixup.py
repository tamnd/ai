# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/mixup/ (ai://8X8XEV).
"""Mixup and CutMix in NumPy.

sample_lambda draws the mixing weight from Beta(alpha, alpha). mixup_batch mixes a minibatch with a permutation of
itself as in eq-mix, and mixup_targets builds the matching soft labels. mixed_cross_entropy is the two-label form of
eq-mixed-loss and soft_cross_entropy the soft-target form; they agree. cutmix_box samples and clips the CutMix box,
cutmix_lambda is the area-adjusted weight of eq-cutmix-lambda, and cutmix_batch pastes the boxes. beta_variance is
eq-beta-moments.
"""

import math

import numpy as np


def _check_lam(lam):
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")


def _check_perm(perm, n):
    perm = np.asarray(perm)
    if perm.shape != (n,) or not np.array_equal(np.sort(perm), np.arange(n)):
        raise ValueError("perm must be a permutation of the batch indices")
    return perm


def sample_lambda(alpha, rng):
    """One mixing weight lam ~ Beta(alpha, alpha), alpha > 0."""
    if not alpha > 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    return float(rng.beta(alpha, alpha))


def beta_variance(alpha):
    """Var[lam] = 1 / (4 (2 alpha + 1)) for lam ~ Beta(alpha, alpha); the mean is 1/2."""
    # eq: eq-beta-moments
    return 1.0 / (4.0 * (2.0 * alpha + 1.0))


def one_hot(y, num_classes):
    return np.eye(num_classes)[np.asarray(y)]


def mixup_batch(x, y, lam, perm):
    """Mix the batch x with x[perm]: returns lam x + (1 - lam) x[perm], y and y[perm].

    The inputs are mixed in floating point, so integer images are not truncated."""
    _check_lam(lam)
    x = np.asarray(x)
    perm = _check_perm(perm, x.shape[0])
    xf = x.astype(np.result_type(x.dtype, np.float64))
    # eq: eq-mix
    mixed = lam * xf + (1.0 - lam) * xf[perm]
    y = np.asarray(y)
    return mixed, y, y[perm]


def mixup_targets(y, num_classes, lam, perm):
    """The soft labels lam one_hot(y) + (1 - lam) one_hot(y[perm]) of eq-mix."""
    _check_lam(lam)
    y = np.asarray(y)
    perm = _check_perm(perm, y.shape[0])
    e = one_hot(y, num_classes)
    # eq: eq-mix
    return lam * e + (1.0 - lam) * e[perm]


def log_softmax(z):
    z = np.asarray(z, dtype=float)
    m = z.max(axis=-1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=-1, keepdims=True))


def soft_cross_entropy(logits, target):
    """Mean over the batch of -sum_k target_k log softmax(logits)_k."""
    return float(-(np.asarray(target, dtype=float) * log_softmax(logits)).sum(axis=-1).mean())


def mixed_cross_entropy(logits, y_a, y_b, lam):
    """lam CE(logits, y_a) + (1 - lam) CE(logits, y_b), the two-label form of eq-mixed-loss."""
    _check_lam(lam)
    lp = log_softmax(logits)
    rows = np.arange(lp.shape[0])
    # eq: eq-mixed-loss
    return float(lam * -lp[rows, y_a].mean() + (1.0 - lam) * -lp[rows, y_b].mean())


def cutmix_box(height, width, lam, center):
    """The CutMix box (x1, x2, y1, y2) for the weight lam and the centre (cx, cy).

    The box has width W sqrt(1 - lam) and height H sqrt(1 - lam) (Equation (2) of Yun et al.); its edges are
    clipped to the image and rounded. Columns are x in [x1, x2), rows are y in [y1, y2)."""
    _check_lam(lam)
    cx, cy = center
    side = math.sqrt(1.0 - lam)
    rw, rh = width * side, height * side
    x1 = int(round(min(max(cx - rw / 2, 0.0), width)))
    x2 = int(round(min(max(cx + rw / 2, 0.0), width)))
    y1 = int(round(min(max(cy - rh / 2, 0.0), height)))
    y2 = int(round(min(max(cy + rh / 2, 0.0), height)))
    return x1, x2, y1, y2


def sample_cutmix_box(height, width, lam, rng):
    """cutmix_box with the centre drawn uniformly over the image, r_x ~ Unif(0, W) and r_y ~ Unif(0, H)."""
    return cutmix_box(height, width, lam, (rng.uniform(0, width), rng.uniform(0, height)))


def cutmix_lambda(box, height, width):
    """The share of pixels kept from the first image, 1 - (x2 - x1)(y2 - y1) / (W H)."""
    x1, x2, y1, y2 = box
    # eq: eq-cutmix-lambda
    return 1.0 - (x2 - x1) * (y2 - y1) / (width * height)


def cutmix_batch(x, y, box, perm):
    """Paste the box of x[perm] into x, for x of shape (N, ..., H, W).

    Returns the mixed batch, y, y[perm] and the adjusted weight of eq-cutmix-lambda."""
    x = np.asarray(x)
    perm = _check_perm(perm, x.shape[0])
    height, width = x.shape[-2], x.shape[-1]
    x1, x2, y1, y2 = box
    mixed = x.copy()
    mixed[..., y1:y2, x1:x2] = x[perm][..., y1:y2, x1:x2]
    y = np.asarray(y)
    return mixed, y, y[perm], cutmix_lambda(box, height, width)
