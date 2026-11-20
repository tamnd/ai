# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/flow-matching/ (ai://NEW-flow-matching).
"""Flow matching in PyTorch, with autograd for the backward pass: the conditional optimal-transport path
(eq-ot-path) and its target (eq-cfm-target), a velocity module VelocityModel trained on the conditional flow matching
loss (eq-cfm-loss), the Euler and midpoint samplers of the ODE (eq-ode), the divergence of the field by autograd and the
log-likelihood by the instantaneous change of variables (eq-change-of-variables), minibatch optimal transport pairing
(eq-ot-coupling) and logit-normal times (eq-logit-normal).

The weights have the shape (n_in, n_out) and a layer computes h @ W + b, as in the NumPy reference, so from_params
loads the same arrays. Time runs from noise at t = 0 to data at t = 1.
"""

import math

import numpy as np
import torch
from torch import nn


def config(d_data=3, d_hidden=5, d_time=4, sigma_min=0.0):
    if min(d_data, d_hidden, d_time) < 1:
        raise ValueError("every size must be positive")
    if not 0.0 <= sigma_min < 1.0:
        raise ValueError("sigma_min must lie in [0, 1)")
    return {"d_data": int(d_data), "d_hidden": int(d_hidden), "d_time": int(d_time), "sigma_min": float(sigma_min)}


def param_shapes(cfg):
    D, H, E = cfg["d_data"], cfg["d_hidden"], cfg["d_time"]
    return {"W1": (D + E, H), "b1": (H,), "W2": (H, D), "b2": (D,)}


def param_count(cfg):
    return sum(math.prod(s) for s in param_shapes(cfg).values())


def flops_forward(cfg, batch):
    D, H, E = cfg["d_data"], cfg["d_hidden"], cfg["d_time"]
    return 2 * batch * H * (2 * D + E)


def time_features(t, d_time):
    """eq-time-features: entry 2k is sin((k + 1) pi t), entry 2k + 1 is cos((k + 1) pi t)."""
    j = torch.arange(d_time)
    a = t.double()[:, None] * (torch.div(j, 2, rounding_mode="floor") + 1) * math.pi
    return torch.where(j % 2 == 0, torch.sin(a), torch.cos(a))


class VelocityModel(nn.Module):
    """v_theta(x, t) = tanh([x, e(t)] W1 + b1) W2 + b2; everything but W1 starts at 0, so v starts at 0."""

    def __init__(self, cfg):
        super().__init__()
        D, H, E = cfg["d_data"], cfg["d_hidden"], cfg["d_time"]
        self.cfg = cfg
        self.W1 = nn.Parameter(torch.randn(D + E, H) / math.sqrt(D + E))
        self.b1 = nn.Parameter(torch.zeros(H))
        self.W2 = nn.Parameter(torch.zeros(H, D))
        self.b2 = nn.Parameter(torch.zeros(D))

    def forward(self, x, t):
        inp = torch.cat([x, time_features(t, self.cfg["d_time"])], 1)
        return torch.tanh(inp @ self.W1 + self.b1) @ self.W2 + self.b2

    @classmethod
    def from_params(cls, params, cfg):
        model = cls(cfg)
        with torch.no_grad():
            for k, v in params.items():
                getattr(model, k).copy_(torch.as_tensor(v))
        return model


def divergence(v_fn, x, t):
    """The trace of the Jacobian of v_fn in x, by one backward pass per coordinate: exact, and cheap for small D."""
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        v = v_fn(x, t)
        cols = [torch.autograd.grad(v[:, i].sum(), x, retain_graph=True)[0][:, i] for i in range(x.shape[1])]
        return torch.stack(cols, 1).sum(1)


def _c(t):
    return t.double().reshape(-1, 1)


def path(cfg, x0, x1, t):
    """eq-ot-path: x_t = (1 - (1 - sigma_min) t) x_0 + t x_1."""
    return (1.0 - (1.0 - cfg["sigma_min"]) * _c(t)) * x0 + _c(t) * x1


def target(cfg, x0, x1):
    """eq-cfm-target: x_1 - (1 - sigma_min) x_0."""
    return x1 - (1.0 - cfg["sigma_min"]) * x0


def cond_field(cfg, x, x1, t):
    """eq-cond-field: (x_1 - (1 - sigma_min) x) / (1 - (1 - sigma_min) t)."""
    c = 1.0 - cfg["sigma_min"]
    return (x1 - c * x) / (1.0 - c * _c(t))


def cfm_loss(model, x0, x1, t):
    """eq-cfm-loss: the mean over rows of ||v_theta(x_t, t) - (x_1 - (1 - sigma_min) x_0)||^2."""
    cfg = model.cfg
    return ((model(path(cfg, x0, x1, t), t) - target(cfg, x0, x1)) ** 2).sum(1).mean()


@torch.no_grad()
def euler_sample(v_fn, x, n_steps, t0=0.0, t1=1.0):
    h = (t1 - t0) / n_steps
    for i in range(n_steps):
        x = x + h * v_fn(x, torch.full((len(x),), t0 + i * h))
    return x


@torch.no_grad()
def midpoint_sample(v_fn, x, n_steps, t0=0.0, t1=1.0):
    h = (t1 - t0) / n_steps
    for i in range(n_steps):
        t = torch.full((len(x),), t0 + i * h)
        x = x + h * v_fn(x + 0.5 * h * v_fn(x, t), t + 0.5 * h)
    return x


def log_likelihood(v_fn, x1, n_steps, log_p0):
    """eq-change-of-variables, run back from t = 1 to t = 0 with midpoint steps and the divergence by autograd."""
    h = 1.0 / n_steps
    x, acc = x1.detach(), torch.zeros(len(x1))
    for i in range(n_steps):
        t = torch.full((len(x),), 1.0 - i * h)
        with torch.no_grad():
            xm = x - 0.5 * h * v_fn(x, t)
        acc = acc + h * divergence(v_fn, xm, t - 0.5 * h).detach()
        with torch.no_grad():
            x = x - h * v_fn(xm, t - 0.5 * h)
    return log_p0(x) - acc


def assignment(cost):
    """The permutation col minimizing sum_i cost[i, col[i]]: the Hungarian algorithm with potentials, O(n^3)."""
    cost = np.asarray(cost, float)
    n = len(cost)
    u, v = np.zeros(n + 1), np.zeros(n + 1)
    row, way = np.zeros(n + 1, int), np.zeros(n + 1, int)
    for i in range(1, n + 1):
        row[0], j0 = i, 0
        minv, used = np.full(n + 1, np.inf), np.zeros(n + 1, bool)
        while True:
            used[j0] = True
            cur = cost[row[j0] - 1] - u[row[j0]] - v[1:]
            better = ~used[1:] & (cur < minv[1:])
            minv[1:][better], way[1:][better] = cur[better], j0
            free = np.nonzero(~used[1:])[0] + 1
            j1 = free[np.argmin(minv[free])]
            delta = minv[j1]
            u[row[used]] += delta
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
            if row[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            row[j0], j0 = row[j1], j1
    col = np.zeros(n, int)
    col[row[1:] - 1] = np.arange(n)
    return torch.as_tensor(col)


def ot_pairs(x0, x1):
    """eq-ot-coupling on a minibatch: reorder x_1 so that the pairs minimize the summed squared distance."""
    return x0, x1[assignment(torch.cdist(x0, x1).pow(2).cpu().numpy())]


def logit_normal(n, m=0.0, s=1.0, generator=None):
    """eq-logit-normal: t = sigmoid(m + s z), z ~ N(0, 1)."""
    return torch.sigmoid(m + s * torch.randn(n, generator=generator))


def train(model, sample_data, steps, batch=256, lr=3e-3, coupling="independent", t_sampler=None, generator=None):
    """Conditional flow matching with Adam; returns the losses."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    trace = []
    for _ in range(steps):
        x1 = sample_data(batch, generator)
        x0 = torch.randn(x1.shape, generator=generator)
        if coupling == "ot":
            x0, x1 = ot_pairs(x0, x1)
        t = torch.rand(batch, generator=generator) if t_sampler is None else t_sampler(batch, generator)
        loss = cfm_loss(model, x0, x1, t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        trace.append(loss.item())
    return trace
