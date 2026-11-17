# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/rms-normalization/ (ai://NEW-rms-normalization).
"""RMSNorm and pRMSNorm as a torch.nn.Module, the PyTorch reference.

The forward pass is eq-rms, or eq-prms when p < 1, over the last axis. Autograd gives the backward pass, and the
tests compare it with eq-rms-grad and with the NumPy reference.
"""

import math

import torch


def partial_count(n, p):
    """Return k = ceil(n p), the number of leading entries that pRMSNorm uses for its statistic."""
    n, p = int(n), float(p)
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    if not 0.0 < p <= 1.0:
        raise ValueError(f"p must be in (0, 1], got {p}")
    return max(1, math.ceil(round(n * p, 9)))  # round first, so that 10 * 0.3 gives 3 and not 4


class RMSNorm(torch.nn.Module):
    """Divide the last axis by the root mean square of its first k = ceil(n p) entries, then apply a learned gain."""

    def __init__(self, n, eps=1e-6, p=1.0, elementwise_affine=True):
        super().__init__()
        if eps < 0:
            raise ValueError(f"eps must be non-negative, got {eps}")
        self.n, self.k, self.eps = int(n), partial_count(n, p), float(eps)
        self.gain = torch.nn.Parameter(torch.ones(self.n)) if elementwise_affine else None

    def forward(self, x):
        if x.dim() < 1 or x.shape[-1] != self.n:
            raise ValueError(f"the last axis of x must have size {self.n}, got shape {tuple(x.shape)}")
        # eq: eq-rms (k = n) and eq-prms (k = ceil(n p))
        r = torch.sqrt(x[..., : self.k].pow(2).mean(dim=-1, keepdim=True) + self.eps)
        z = x / r
        return z * self.gain if self.gain is not None else z


def input_gradient(x, dy, gain=None, eps=1e-6, p=1.0):
    """The closed-form input gradient of eq-rms-grad, used to check autograd."""
    k = partial_count(x.shape[-1], p)
    u = dy * gain if gain is not None else dy
    r = torch.sqrt(x[..., :k].pow(2).mean(dim=-1, keepdim=True) + eps)
    # eq: eq-rms-grad
    xu = (x * u).sum(dim=-1, keepdim=True)
    mask = (torch.arange(x.shape[-1]) < k).to(x.dtype)
    return u / r - mask * x * xu / (k * r ** 3)
