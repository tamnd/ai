# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/mixup/ (ai://NEW-mixup).
"""Mixup and CutMix in PyTorch.

sample_lambda draws the mixing weight from Beta(alpha, alpha). mixup_batch mixes a minibatch with a permutation of
itself as in eq-mix, and mixup_targets builds the matching soft labels. mixed_cross_entropy is the two-label form of
eq-mixed-loss. cutmix_box samples and clips the CutMix box, cutmix_lambda is the area-adjusted weight of
eq-cutmix-lambda, and cutmix_batch pastes the boxes. beta_variance is eq-beta-moments. Mixup is a module that mixes in
training mode only.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def _check_lam(lam):
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")


def _check_perm(perm, n):
    perm = torch.as_tensor(perm, dtype=torch.long)
    if perm.shape != (n,) or not torch.equal(torch.sort(perm).values, torch.arange(n)):
        raise ValueError("perm must be a permutation of the batch indices")
    return perm


def sample_lambda(alpha, generator=None):
    """One mixing weight lam ~ Beta(alpha, alpha), alpha > 0, as a Python float.

    With a generator, the draw is seeded from it so that a seeded run is repeatable; the global RNG is untouched."""
    if not alpha > 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    beta = torch.distributions.Beta(torch.tensor(float(alpha)), torch.tensor(float(alpha)))
    if generator is None:
        return float(beta.sample())
    seed = int(torch.randint(2**62, (1,), generator=generator))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return float(beta.sample())


def beta_variance(alpha):
    """Var[lam] = 1 / (4 (2 alpha + 1)) for lam ~ Beta(alpha, alpha); the mean is 1/2."""
    # eq: eq-beta-moments
    return 1.0 / (4.0 * (2.0 * alpha + 1.0))


def mixup_batch(x, y, lam, perm):
    """Mix the batch x with x[perm]: returns lam x + (1 - lam) x[perm], y and y[perm].

    Integer inputs are converted to the default floating dtype before mixing."""
    _check_lam(lam)
    x = torch.as_tensor(x)
    perm = _check_perm(perm, x.shape[0])
    if not torch.is_floating_point(x):
        x = x.to(torch.get_default_dtype())
    # eq: eq-mix
    mixed = lam * x + (1.0 - lam) * x[perm]
    y = torch.as_tensor(y)
    return mixed, y, y[perm]


def mixup_targets(y, num_classes, lam, perm):
    """The soft labels lam one_hot(y) + (1 - lam) one_hot(y[perm]) of eq-mix."""
    _check_lam(lam)
    y = torch.as_tensor(y, dtype=torch.long)
    perm = _check_perm(perm, y.shape[0])
    e = F.one_hot(y, num_classes).to(torch.get_default_dtype())
    # eq: eq-mix
    return lam * e + (1.0 - lam) * e[perm]


def mixed_cross_entropy(logits, y_a, y_b, lam):
    """lam CE(logits, y_a) + (1 - lam) CE(logits, y_b), the two-label form of eq-mixed-loss."""
    _check_lam(lam)
    # eq: eq-mixed-loss
    return lam * F.cross_entropy(logits, y_a) + (1.0 - lam) * F.cross_entropy(logits, y_b)


def cutmix_box(height, width, lam, center):
    """The CutMix box (x1, x2, y1, y2) for the weight lam and the centre (cx, cy).

    The box has width W sqrt(1 - lam) and height H sqrt(1 - lam); its edges are clipped to the image and rounded.
    Columns are x in [x1, x2), rows are y in [y1, y2)."""
    _check_lam(lam)
    cx, cy = (float(c) for c in center)
    side = math.sqrt(1.0 - lam)
    rw, rh = width * side, height * side
    x1 = int(round(min(max(cx - rw / 2, 0.0), width)))
    x2 = int(round(min(max(cx + rw / 2, 0.0), width)))
    y1 = int(round(min(max(cy - rh / 2, 0.0), height)))
    y2 = int(round(min(max(cy + rh / 2, 0.0), height)))
    return x1, x2, y1, y2


def sample_cutmix_box(height, width, lam, generator=None):
    """cutmix_box with the centre drawn uniformly over the image."""
    u = torch.rand(2, dtype=torch.float64, generator=generator)
    return cutmix_box(height, width, lam, (float(u[0]) * width, float(u[1]) * height))


def cutmix_lambda(box, height, width):
    """The share of pixels kept from the first image, 1 - (x2 - x1)(y2 - y1) / (W H)."""
    x1, x2, y1, y2 = box
    # eq: eq-cutmix-lambda
    return 1.0 - (x2 - x1) * (y2 - y1) / (width * height)


def cutmix_batch(x, y, box, perm):
    """Paste the box of x[perm] into a copy of x, for x of shape (N, ..., H, W).

    Returns the mixed batch, y, y[perm] and the adjusted weight of eq-cutmix-lambda."""
    x = torch.as_tensor(x)
    perm = _check_perm(perm, x.shape[0])
    height, width = x.shape[-2], x.shape[-1]
    x1, x2, y1, y2 = box
    mixed = x.clone()
    mixed[..., y1:y2, x1:x2] = x[perm][..., y1:y2, x1:x2]
    y = torch.as_tensor(y)
    return mixed, y, y[perm], cutmix_lambda(box, height, width)


class Mixup(nn.Module):
    """Mixes a batch in training mode and passes it through unchanged in evaluation mode.

    forward(x, y) returns (x_mixed, y_a, y_b, lam); train with mixed_cross_entropy(model(x_mixed), y_a, y_b, lam).
    In evaluation mode it returns (x, y, y, 1.0), for which that loss is the ordinary cross-entropy."""

    def __init__(self, alpha=0.2, generator=None):
        super().__init__()
        if not alpha > 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self.alpha = alpha
        self.generator = generator

    def forward(self, x, y):
        y = torch.as_tensor(y)
        if not self.training:
            return x, y, y, 1.0
        lam = sample_lambda(self.alpha, self.generator)
        perm = torch.randperm(x.shape[0], generator=self.generator)
        mixed, y_a, y_b = mixup_batch(x, y, lam, perm)
        return mixed, y_a, y_b, lam
