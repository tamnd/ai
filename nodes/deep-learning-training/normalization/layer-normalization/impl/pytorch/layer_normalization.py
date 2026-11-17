# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/layer-normalization/ (ai://NEW-layer-normalization).
"""Layer normalization and RMSNorm as PyTorch modules, written from tensor operations.

LayerNorm follows torch.nn.LayerNorm: the statistics are the mean and the biased variance over the last axes, those
of normalized_shape, eps is added to the variance (eq-ln-forward), and the gain and the bias have one element per
normalized element. RMSNorm follows torch.nn.RMSNorm (eq-rms-forward), with eps=None for the machine epsilon. Neither
module keeps a state or has a training mode. Residual is a residual block with the normalization placed after the sum
(Post-LN, eq-post-ln) or at the start of the branch (Pre-LN, eq-pre-ln). Autograd gives the backward pass, and the
tests compare it with eq-ln-backward and with the torch modules.
"""

import torch
from torch import nn


def _shape(normalized_shape):
    shape = (int(normalized_shape),) if isinstance(normalized_shape, int) else tuple(int(n) for n in normalized_shape)
    if not shape or min(shape) < 1:
        raise ValueError(f"normalized_shape must be a non-empty shape of positive sizes, got {normalized_shape}")
    return shape


def _check(x, shape):
    k = len(shape)
    if x.dim() < k or tuple(x.shape[x.dim() - k:]) != shape:
        raise ValueError(f"the last axes of x must be {shape}, got {tuple(x.shape)}")
    return tuple(range(x.dim() - k, x.dim()))


class LayerNorm(nn.Module):
    """Layer normalization over the last len(normalized_shape) axes."""

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, bias=True):
        super().__init__()
        self.normalized_shape, self.eps = _shape(normalized_shape), float(eps)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape)) if elementwise_affine else None
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape)) if elementwise_affine and bias else None

    def forward(self, x):
        dims = _check(x, self.normalized_shape)
        mean = x.mean(dim=dims, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=dims, keepdim=True)  # biased, eq-ln-stats
        y = (x - mean) / torch.sqrt(var + self.eps)
        if self.weight is not None:
            y = y * self.weight
        if self.bias is not None:
            y = y + self.bias
        return y


class RMSNorm(nn.Module):
    """RMSNorm over the last len(normalized_shape) axes. eps=None uses the machine epsilon of float64 for float64
    inputs and of float32 otherwise, as torch.nn.RMSNorm does."""

    def __init__(self, normalized_shape, eps=None, elementwise_affine=True):
        super().__init__()
        self.normalized_shape, self.eps = _shape(normalized_shape), eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape)) if elementwise_affine else None

    def forward(self, x):
        dims = _check(x, self.normalized_shape)
        eps = self.eps
        if eps is None:
            eps = torch.finfo(torch.float64 if x.dtype == torch.float64 else torch.float32).eps
        y = x / torch.sqrt((x ** 2).mean(dim=dims, keepdim=True) + eps)
        return y * self.weight if self.weight is not None else y


class Residual(nn.Module):
    """x -> norm(x + sublayer(x)) for placement "post", x -> x + sublayer(norm(x)) for placement "pre"."""

    def __init__(self, sublayer, norm, placement="pre"):
        super().__init__()
        if placement not in ("pre", "post"):
            raise ValueError(f"placement must be 'pre' or 'post', got {placement!r}")
        self.sublayer, self.norm, self.placement = sublayer, norm, placement

    def forward(self, x):
        if self.placement == "post":
            return self.norm(x + self.sublayer(x))
        return x + self.sublayer(self.norm(x))
