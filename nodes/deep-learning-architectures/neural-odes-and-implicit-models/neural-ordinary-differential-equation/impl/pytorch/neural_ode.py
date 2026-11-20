# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-ordinary-differential-equation/ (ai://WNCNQE).
"""The neural ordinary differential equation, the PyTorch reference: eq-field, eq-augment and RK4 steps of eq-ode.

Autograd through the steps gives the gradient of the discrete map (discretize, then optimize), as backprop of the
NumPy reference does. AdjointODE is a torch.autograd.Function whose backward solves the adjoint system eq-adjoint
and eq-adjoint-theta backwards in time with the same steps (optimize, then discretize), so it stores nothing of
the forward solve but its output. The weights have the shapes of the page and of the NumPy reference.
"""

import torch
from torch import nn

NAMES = ("W1", "u", "b1", "W2", "b2")


def config(d=2, d_hidden=8, d_state=None, n_steps=4, T=1.0):
    d_state = d if d_state is None else d_state
    if min(d, d_hidden, n_steps) < 1 or d_state < d or T <= 0:
        raise ValueError("d, d_hidden and n_steps must be at least 1, d_state at least d and T positive")
    return {"d": d, "d_hidden": d_hidden, "d_state": d_state, "n_steps": n_steps, "T": float(T)}


def param_count(cfg):
    D, H = cfg["d_state"], cfg["d_hidden"]
    return 2 * D * H + 2 * H + D  # eq: eq-params


def rk4(f, h, t0, t1, n):
    dt = (t1 - t0) / n
    for k in range(n):
        t = t0 + k * dt
        k1 = f(h, t)
        k2 = f(h + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = f(h + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = f(h + dt * k3, t + dt)
        h = h + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return h


def field(params, h, t):
    W1, u, b1, W2, b2 = params
    return torch.tanh(h @ W1.T + t * u + b1) @ W2.T + b2  # eq: eq-field


class AdjointODE(torch.autograd.Function):
    """h(T) from h(0), with the backward pass of Algorithm 1 of Chen et al. instead of autograd through the steps."""

    @staticmethod
    def forward(ctx, h0, T, n, *params):
        with torch.no_grad():
            y = rk4(lambda h, t: field(params, h, t), h0, 0.0, T, n)
        ctx.save_for_backward(y, *params)
        ctx.T, ctx.n = T, n
        return y

    @staticmethod
    def backward(ctx, dy):
        y, *params = ctx.saved_tensors
        D = y.shape[1]
        sizes = [p.numel() for p in params]

        def aug(z, t):
            h, a = z[0], z[1]
            with torch.enable_grad():
                hh = h.detach().requires_grad_(True)
                ps = [p.detach().requires_grad_(True) for p in params]
                f = field(ps, hh, t)
                vjp = torch.autograd.grad(f, [hh, *ps], a)  # a^T df/dh and a^T df/dtheta
            return (f.detach(), -vjp[0], -torch.cat([v.reshape(-1) for v in vjp[1:]]))

        def step(z, t, dt):
            k1 = aug(z, t)
            k2 = aug(tuple(zi + 0.5 * dt * ki for zi, ki in zip(z, k1, strict=True)), t + 0.5 * dt)
            k3 = aug(tuple(zi + 0.5 * dt * ki for zi, ki in zip(z, k2, strict=True)), t + 0.5 * dt)
            k4 = aug(tuple(zi + dt * ki for zi, ki in zip(z, k3, strict=True)), t + dt)
            return tuple(
                zi + dt / 6.0 * (a + 2.0 * b + 2.0 * c + e) for zi, a, b, c, e in zip(z, k1, k2, k3, k4, strict=True)
            )

        z = (y, dy, torch.zeros(sum(sizes), dtype=y.dtype))
        dt = -ctx.T / ctx.n
        for k in range(ctx.n):
            z = step(z, ctx.T + k * dt, dt)  # eq: eq-adjoint and eq-adjoint-theta, from T back to 0
        _, a0, g = z
        grads, i = [], 0
        for p, s in zip(params, sizes, strict=True):
            grads.append(g[i : i + s].reshape(p.shape))
            i += s
        assert a0.shape[1] == D
        return (a0, None, None, *grads)


class NeuralODE(nn.Module):
    def __init__(self, cfg, generator=None):
        super().__init__()
        self.cfg = cfg
        D, H = cfg["d_state"], cfg["d_hidden"]
        self.W1 = nn.Parameter(torch.randn(H, D, generator=generator) / D**0.5)
        self.u = nn.Parameter(torch.randn(H, generator=generator))
        self.b1 = nn.Parameter(torch.zeros(H))
        self.W2 = nn.Parameter(torch.randn(D, H, generator=generator) / H**0.5)
        self.b2 = nn.Parameter(torch.zeros(D))

    @classmethod
    def from_params(cls, params, cfg):
        model = cls(cfg)
        with torch.no_grad():
            for k in NAMES:
                getattr(model, k).copy_(torch.as_tensor(params[k]))
        return model

    def params(self):
        return [getattr(self, k) for k in NAMES]

    def augment(self, x):
        return torch.cat([x, x.new_zeros(x.shape[0], self.cfg["d_state"] - self.cfg["d"])], 1)  # eq: eq-augment

    def forward(self, x, adjoint=False):
        h0 = self.augment(x)  # (B, D)
        if adjoint:
            return AdjointODE.apply(h0, self.cfg["T"], self.cfg["n_steps"], *self.params())
        return rk4(lambda h, t: field(self.params(), h, t), h0, 0.0, self.cfg["T"], self.cfg["n_steps"])  # (B, D)


def mse_loss(y, t):
    return 0.5 * ((y - t) ** 2).sum() / y.shape[0]
