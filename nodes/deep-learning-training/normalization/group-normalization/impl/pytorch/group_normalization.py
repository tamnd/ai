# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/group-normalization/ (ai://QFEJMP).
"""Group normalization as a torch.nn.Module, the PyTorch reference.

The input has the layout (N, C, *S). The forward pass is eq-gn-set with eq-norm and eq-affine. normalize_over is the
general form of eq-norm, which gives batch, layer, instance and group normalization by the choice of the set.
Autograd gives the backward pass, and the tests compare it with eq-gn-grad and with torch.nn.GroupNorm,
torch.nn.InstanceNorm2d and torch.nn.LayerNorm.
"""

import torch


def normalize_over(x, groups, over_batch=False, eps=1e-5):
    """Normalize x by the mean and the biased variance of each group of channels, per example or across the batch."""
    if x.dim() < 2:
        raise ValueError(f"x must have the layout (N, C, *S), got shape {tuple(x.shape)}")
    n, c = x.shape[0], x.shape[1]
    if groups < 1 or c % groups != 0:
        raise ValueError(f"C = {c} channels cannot be split into {groups} groups")
    xg = x.reshape(n, groups, c // groups, *x.shape[2:])
    dims = list(range(2, xg.dim()))
    if over_batch:
        dims = [0] + dims
    # eq: eq-norm
    mean = xg.mean(dim=dims, keepdim=True)
    var = (xg - mean).pow(2).mean(dim=dims, keepdim=True)
    return ((xg - mean) / torch.sqrt(var + eps)).reshape(x.shape)


class GroupNorm(torch.nn.Module):
    """Normalize each group of C/G channels of each example, then scale and shift each channel."""

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        if num_groups < 1 or num_channels < 1 or num_channels % num_groups != 0:
            raise ValueError(f"C = {num_channels} channels cannot be split into G = {num_groups} groups")
        if eps < 0:
            raise ValueError(f"eps must be non-negative, got {eps}")
        self.num_groups, self.num_channels, self.eps = int(num_groups), int(num_channels), float(eps)
        self.weight = torch.nn.Parameter(torch.ones(self.num_channels)) if affine else None
        self.bias = torch.nn.Parameter(torch.zeros(self.num_channels)) if affine else None

    def forward(self, x):
        if x.dim() < 2 or x.shape[1] != self.num_channels:
            raise ValueError(f"x must have the layout (N, {self.num_channels}, *S), got shape {tuple(x.shape)}")
        # eq: eq-gn-set
        z = normalize_over(x, self.num_groups, eps=self.eps)
        if self.weight is None or self.bias is None:
            return z
        shape = (1, -1) + (1,) * (x.dim() - 2)
        # eq: eq-affine
        return z * self.weight.reshape(shape) + self.bias.reshape(shape)


def input_gradient(x, dy, num_groups, gamma=None, eps=1e-5):
    """The closed-form input gradient of eq-gn-grad, used to check autograd."""
    n = x.shape[0]
    shape = (1, -1) + (1,) * (x.dim() - 2)
    u = dy * gamma.reshape(shape) if gamma is not None else dy
    xg = x.reshape(n, num_groups, -1)
    ug = u.reshape(n, num_groups, -1)
    mean = xg.mean(dim=-1, keepdim=True)
    std = torch.sqrt((xg - mean).pow(2).mean(dim=-1, keepdim=True) + eps)
    xhat = (xg - mean) / std
    # eq: eq-gn-grad
    dx = (ug - ug.mean(dim=-1, keepdim=True) - xhat * (ug * xhat).mean(dim=-1, keepdim=True)) / std
    return dx.reshape(x.shape)
