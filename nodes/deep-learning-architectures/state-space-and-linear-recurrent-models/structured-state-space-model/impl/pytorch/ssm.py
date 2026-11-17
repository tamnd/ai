# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/structured-state-space-model/ (ai://MDN3YW).
"""A diagonal state space layer in PyTorch, with the same parameters and equations as the NumPy reference.

The convolutional mode trains in parallel with the FFT; the recurrent mode and step run one position at a time.
"""

import math

import torch
from torch import nn


class DiagonalSSM(nn.Module):
    """H channels, each a single-input single-output SSM with N complex diagonal entries in conjugate pairs."""

    def __init__(self, h=2, n=4, method="zoh", dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        if method not in ("zoh", "bilinear"):
            raise ValueError(f"method must be zoh or bilinear, got {method!r}")
        self.h, self.n, self.method = h, n, method
        k = torch.arange(n, dtype=torch.float64)
        self.log_dt = nn.Parameter(torch.rand(h, dtype=torch.float64) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        self.log_a_re = nn.Parameter(torch.full((h, n), math.log(0.5), dtype=torch.float64))
        self.a_im = nn.Parameter((math.pi * k).repeat(h, 1))
        self.b = nn.Parameter(torch.ones(h, n, dtype=torch.complex128))
        self.c = nn.Parameter(torch.randn(h, n, dtype=torch.complex128))
        self.d = nn.Parameter(torch.randn(h, dtype=torch.float64))

    @classmethod
    def from_params(cls, params, method="zoh"):
        """Build the layer from the NumPy parameter dictionary."""
        h, n = params["c"].shape
        m = cls(h, n, method)
        with torch.no_grad():
            for name in ("log_dt", "log_a_re", "a_im", "b", "c", "d"):
                getattr(m, name).copy_(torch.as_tensor(params[name]))
        return m

    def discrete(self):
        """Return (Ab, Bb), each (H, N), by zero-order hold or the bilinear rule."""
        a = torch.complex(-torch.exp(self.log_a_re), self.a_im)
        dt = torch.exp(self.log_dt)[:, None]
        if self.method == "zoh":
            ab = torch.exp(dt * a)
            return ab, (ab - 1.0) / a * self.b
        den = 1.0 - dt * a / 2.0
        return (1.0 + dt * a / 2.0) / den, dt * self.b / den

    def kernel(self, length):
        """K[h, l] = 2 Re sum_n C Bb Ab^l, shape (H, L)."""
        ab, bb = self.discrete()
        powers = ab[:, :, None] ** torch.arange(length, dtype=torch.float64)
        return 2.0 * torch.einsum("hn,hnl->hl", self.c * bb, powers).real

    def forward(self, u, mode="conv"):
        """u (B, L, H) to y (B, L, H); mode is conv (FFT) or scan (the recurrence)."""
        if mode == "scan":
            x = torch.zeros(u.shape[0], self.h, self.n, dtype=torch.complex128)
            ys = []
            for t in range(u.shape[1]):
                x, y_t = self.step(x, u[:, t])
                ys.append(y_t)
            return torch.stack(ys, dim=1)
        if mode != "conv":
            raise ValueError(f"mode must be conv or scan, got {mode!r}")
        length = u.shape[1]
        k = self.kernel(length)
        uf = torch.fft.rfft(u, n=2 * length, dim=1)
        kf = torch.fft.rfft(k.T, n=2 * length, dim=0)
        y = torch.fft.irfft(uf * kf, n=2 * length, dim=1)[:, :length]
        return y + self.d * u

    def step(self, x, u_t):
        """One step for inference: x (B, H, N), u_t (B, H) to the new state and y_t (B, H)."""
        ab, bb = self.discrete()
        x = ab * x + bb * u_t[..., None]
        return x, 2.0 * torch.einsum("hn,bhn->bh", self.c, x).real + self.d * u_t
