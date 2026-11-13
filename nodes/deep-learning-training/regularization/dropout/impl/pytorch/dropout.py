# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/dropout/ (ai://Y8VNMB).
"""Inverted dropout as a torch.nn.Module, the PyTorch reference.

The forward pass is eq-inverted of the page. Autograd gives the backward pass, and the tests compare it with the NumPy backward.
"""

import torch


class Dropout(torch.nn.Module):
    """Set each entry to zero with probability p in training, and scale the others by 1 / (1 - p)."""

    def __init__(self, p=0.5):
        super().__init__()
        p = float(p)
        if not 0.0 <= p < 1.0:
            raise ValueError(f"p must be in [0, 1), got {p}")
        self.p = p

    @classmethod
    def from_params(cls, params, p=0.5):
        """Dropout has no parameters, so params must be empty."""
        if params:
            raise ValueError("dropout has no parameters")
        return cls(p)

    def forward(self, x, mask=None, generator=None):
        """Apply dropout. Give mask to use a fixed mask, or generator to draw one from a seeded generator."""
        if not self.training or self.p == 0.0:
            return x
        if mask is None:
            keep = torch.full_like(x, 1.0 - self.p)
            mask = torch.bernoulli(keep, generator=generator)
        if mask.shape != x.shape:
            raise ValueError(f"the mask has the shape {tuple(mask.shape)}, and x has the shape {tuple(x.shape)}")
        return x * mask / (1.0 - self.p)
