# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/rotary-position-embedding/ (ai://0V8E5T).
"""Rotary position embedding (RoPE) as a torch.nn.Module, the PyTorch reference.

The forward pass is eq-rope-efficient with the angles of eq-theta and eq-pi. Autograd gives the backward pass,
and the tests compare it with the rotation by the opposite angles of the NumPy backward.
"""

import torch


class RotaryEmbedding(torch.nn.Module):
    """Rotate each pair of dimensions (2i, 2i + 1) of x at position m by the angle (m / scale) theta_i."""

    def __init__(self, d, base=10000.0, scale=1.0):
        super().__init__()
        if d < 2 or d % 2 != 0:
            raise ValueError(f"d must be even and at least 2, got {d}")
        if base <= 1.0 or scale <= 0.0:
            raise ValueError(f"base must be greater than 1 and scale positive, got {base} and {scale}")
        self.d, self.base, self.scale = int(d), float(base), float(scale)

    @classmethod
    def from_params(cls, params, d, base=10000.0, scale=1.0):
        """RoPE has no parameters, so params must be empty."""
        if params:
            raise ValueError("RoPE has no parameters")
        return cls(d, base, scale)

    def frequencies(self, dtype=torch.float64):
        i = torch.arange(self.d // 2, dtype=dtype)
        return self.base ** (-2.0 * i / self.d)  # eq: eq-theta

    def forward(self, x, positions):
        """Rotate x, whose last two axes are (positions, d)."""
        if x.shape[-1] != self.d:
            raise ValueError(f"the last axis of x has length {x.shape[-1]}, and d is {self.d}")
        m = torch.as_tensor(positions, dtype=x.dtype) / self.scale  # eq: eq-pi
        a = torch.outer(m, self.frequencies(x.dtype))
        cos = torch.repeat_interleave(torch.cos(a), 2, dim=-1)
        sin = torch.repeat_interleave(torch.sin(a), 2, dim=-1)
        rot = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).flatten(-2)
        return x * cos + rot * sin  # eq: eq-rope-efficient
