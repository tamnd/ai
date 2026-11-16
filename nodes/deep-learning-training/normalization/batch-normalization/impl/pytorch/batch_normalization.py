# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/batch-normalization/ (ai://NEW-batch-normalization).
"""Batch normalization as a PyTorch module, written from tensor operations.

The module follows torch.nn.BatchNorm1d and torch.nn.BatchNorm2d: the input has the shape (N, C) or (N, C, ...),
training normalizes with the biased batch variance (eq-bn-forward), and the running variance is updated with the
unbiased one (eq-running). In evaluation mode the layer uses the running averages (eq-inference). Autograd gives the
backward pass, and the tests compare it with eq-bn-backward and with torch.nn.BatchNorm2d.
"""

import torch
from torch import nn


class BatchNorm(nn.Module):
    """Batch normalization over all axes except the channel axis 1. momentum is the PyTorch convention, or None for a
    cumulative average."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        if num_features < 1:
            raise ValueError(f"num_features must be positive, got {num_features}")
        self.num_features, self.eps, self.momentum = int(num_features), float(eps), momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        if x.dim() < 2 or x.shape[1] != self.num_features:
            raise ValueError(f"x must have the shape (N, {self.num_features}, ...), got {tuple(x.shape)}")
        axes = [0] + list(range(2, x.dim()))
        shape = [1, -1] + [1] * (x.dim() - 2)
        if self.training:
            count = x.numel() // x.shape[1]
            if count < 2:
                raise ValueError(f"training needs more than 1 value per channel, got {count}")
            mean = x.mean(dim=axes)
            var = ((x - mean.view(shape)) ** 2).mean(dim=axes)  # biased, as in Algorithm 1
            with torch.no_grad():
                self.num_batches_tracked += 1
                alpha = 1.0 / float(self.num_batches_tracked) if self.momentum is None else self.momentum
                self.running_mean.mul_(1 - alpha).add_(alpha * mean.detach())
                self.running_var.mul_(1 - alpha).add_(alpha * var.detach() * count / (count - 1))
        else:
            mean, var = self.running_mean, self.running_var
        xhat = (x - mean.view(shape)) / torch.sqrt(var.view(shape) + self.eps)
        return self.weight.view(shape) * xhat + self.bias.view(shape)

    @torch.no_grad()
    def fold(self, weight, bias=None):
        """Return (weight', bias') of the linear or convolutional layer before this one with the layer folded in."""
        a = self.weight / torch.sqrt(self.running_var + self.eps)
        bias = torch.zeros_like(self.running_mean) if bias is None else bias
        w2 = weight * a.view([-1] + [1] * (weight.dim() - 1))
        return w2, a * (bias - self.running_mean) + self.bias
