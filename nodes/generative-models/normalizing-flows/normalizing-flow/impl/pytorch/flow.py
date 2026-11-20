# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/normalizing-flow/ (ai://NEW-normalizing-flow).
"""The normalizing flow of the page in PyTorch, with autograd for every backward pass: a stack of affine coupling
layers (eq-coupling) with alternating masks, a standard normal base density and the exact log-likelihood of
eq-change-of-variables; also the planar flow with the invertibility fix of eq-planar and one masked autoregressive
layer (eq-autoregressive).

The weights have the shape (n_in, n_out) and a layer computes h @ W + b, as in the NumPy reference, so from_params
loads the same arrays. The direction x -> z is forward and evaluates densities; inverse samples.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

PARTS = ("W1", "b1", "Ws", "bs", "Wt", "bt")


def config(d_data=3, d_hidden=5, n_layers=3):
    if min(d_data, d_hidden, n_layers) < 1 or d_data < 2:
        raise ValueError("d_data must be at least 2 and every size positive")
    return {"d_data": int(d_data), "d_hidden": int(d_hidden), "n_layers": int(n_layers)}


def param_shapes(cfg):
    D, H = cfg["d_data"], cfg["d_hidden"]
    one = {"W1": (D, H), "b1": (H,), "Ws": (H, D), "bs": (D,), "Wt": (H, D), "bt": (D,)}
    return {f"c{k}_{n}": s for k in range(cfg["n_layers"]) for n, s in one.items()}


def param_count(cfg):
    return sum(math.prod(s) for s in param_shapes(cfg).values())


def flops_forward(cfg, batch):
    return 6 * batch * cfg["n_layers"] * cfg["d_data"] * cfg["d_hidden"]


class Coupling(nn.Module):
    """One affine coupling layer: the coordinates where mask = 1 pass through and set s and t for the others."""

    mask: torch.Tensor

    def __init__(self, mask, d_hidden):
        super().__init__()
        D = len(mask)
        self.register_buffer("mask", mask)
        self.W1 = nn.Parameter(torch.randn(D, d_hidden) / math.sqrt(D))
        self.b1 = nn.Parameter(torch.zeros(d_hidden))
        self.Ws, self.bs = nn.Parameter(torch.zeros(d_hidden, D)), nn.Parameter(torch.zeros(D))
        self.Wt, self.bt = nn.Parameter(torch.zeros(d_hidden, D)), nn.Parameter(torch.zeros(D))

    def scale_shift(self, x):
        h = torch.tanh((x * self.mask) @ self.W1 + self.b1)
        return (h @ self.Ws + self.bs) * (1 - self.mask), (h @ self.Wt + self.bt) * (1 - self.mask)

    def forward(self, x):
        s, t = self.scale_shift(x)
        return x * torch.exp(s) + t, s.sum(-1)

    def inverse(self, y):
        s, t = self.scale_shift(y)  # the kept coordinates of y are those of x
        return (y - t) * torch.exp(-s)


class Flow(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        i = torch.arange(cfg["d_data"])
        self.layers = nn.ModuleList(
            Coupling(((i + k) % 2 == 0).to(torch.get_default_dtype()), cfg["d_hidden"]) for k in range(cfg["n_layers"])
        )

    @classmethod
    def from_params(cls, params, cfg):
        model = cls(cfg)
        with torch.no_grad():
            for k, layer in enumerate(model.layers):
                for n in PARTS:
                    getattr(layer, n).copy_(torch.as_tensor(params[f"c{k}_{n}"]))
        return model

    def forward(self, x):
        """f: x -> z and log |det df/dx| per row (eq-logdet)."""
        ld = torch.zeros(x.shape[0], dtype=x.dtype)
        for layer in self.layers:
            x, ldk = layer(x)
            ld = ld + ldk
        return x, ld

    def inverse(self, z):
        for layer in reversed(self.layers):
            assert isinstance(layer, Coupling)
            z = layer.inverse(z)
        return z

    def log_prob(self, x):
        z, ld = self(x)
        return -0.5 * (z**2).sum(-1) - 0.5 * z.shape[-1] * math.log(2 * math.pi) + ld

    def nll(self, x):
        return -self.log_prob(x).mean()

    @torch.no_grad()
    def sample(self, n, temperature=1.0, generator=None):
        z = torch.randn(n, self.cfg["d_data"], generator=generator)
        return self.inverse(temperature * z)


def u_hat(w, u):
    """eq-planar: w.u_hat = -1 + softplus(w.u) > -1."""
    wu = w @ u
    return u + (F.softplus(wu) - 1 - wu) * w / (w @ w)


def planar(z, w, u, b):
    a = torch.tanh(z @ w + b)
    return z + a[:, None] * u, torch.log(torch.abs(1 + (1 - a**2) * (w @ u)))


def ar_params(x, q):
    m = torch.triu(torch.ones(x.shape[-1], x.shape[-1], dtype=x.dtype), 1)
    return x @ (q["Wm"] * m) + q["bm"], x @ (q["Wa"] * m) + q["ba"]


def ar_density(q, x):
    mu, al = ar_params(x, q)
    return (x - mu) * torch.exp(-al), -al.sum(-1)


def ar_sample(q, u, passes=None):
    x = torch.zeros_like(u)
    for _ in range(u.shape[-1] if passes is None else passes):
        mu, al = ar_params(x, q)
        x = u * torch.exp(al) + mu
    return x


def train(model, sample_data, steps, batch=256, lr=3e-3, generator=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    trace = []
    for _ in range(steps):
        loss = model.nll(sample_data(batch, generator))
        opt.zero_grad()
        loss.backward()
        opt.step()
        trace.append(loss.item())
    return trace
