# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/diffusion-model/ (ai://NEW-diffusion-model).
"""The denoising diffusion model of the page in PyTorch, with autograd for the backward pass: the linear schedule, the
forward marginal (eq-forward-marginal), the posterior (eq-posterior), a noise-prediction module EpsModel trained on the
simple loss (eq-simple-loss), the ancestral step and sampler (eq-ddpm-step) and the DDIM step and sampler
(eq-ddim-step), and classifier-free guidance (eq-guidance).

The weights have the shape (n_in, n_out) and a layer computes h @ W + b, as in the NumPy reference, so from_params
loads the same arrays. Steps are integer tensors t = 1..T, and alpha_bar_0 = 1.
"""

import math

import torch
from torch import nn


def config(d_data=3, d_hidden=5, d_time=4, n_steps=10, beta_1=0.02, beta_T=0.5):
    if min(d_data, d_hidden, d_time, n_steps) < 1:
        raise ValueError("every size must be positive")
    if not 0 < beta_1 <= beta_T < 1:
        raise ValueError("the variances must satisfy 0 < beta_1 <= beta_T < 1")
    return {"d_data": int(d_data), "d_hidden": int(d_hidden), "d_time": int(d_time), "n_steps": int(n_steps),
            "beta_1": float(beta_1), "beta_T": float(beta_T)}  # fmt: skip


def param_shapes(cfg):
    D, H, E = cfg["d_data"], cfg["d_hidden"], cfg["d_time"]
    return {"W1": (D + E, H), "b1": (H,), "W2": (H, D), "b2": (D,)}


def param_count(cfg):
    return sum(math.prod(s) for s in param_shapes(cfg).values())


def flops_forward(cfg, batch):
    D, H, E = cfg["d_data"], cfg["d_hidden"], cfg["d_time"]
    return 2 * batch * H * (2 * D + E)


def schedule(cfg):
    """beta, alpha, alpha_bar and beta_tilde as float64 tensors of length T, and alpha_bar with alpha_bar_0 = 1 in
    front, of length T + 1."""
    beta = torch.linspace(cfg["beta_1"], cfg["beta_T"], cfg["n_steps"], dtype=torch.float64)
    alpha = 1.0 - beta
    abar = torch.cumprod(alpha, 0)
    abar0 = torch.cat([torch.ones(1, dtype=torch.float64), abar])
    return {"beta": beta, "alpha": alpha, "abar": abar, "abar0": abar0,
            "beta_tilde": (1.0 - abar0[:-1]) / (1.0 - abar) * beta}  # fmt: skip


def time_embedding(t, d_time):
    """eq-time-embedding: entry 2i is sin(t w_i), entry 2i + 1 is cos(t w_i), w_i = 10000^(-2i / d_time)."""
    i = torch.arange(d_time)
    w = 10000.0 ** (-2.0 * torch.div(i, 2, rounding_mode="floor").double() / d_time)
    a = t.double()[:, None] * w
    return torch.where(i % 2 == 0, torch.sin(a), torch.cos(a))


class EpsModel(nn.Module):
    """eps_theta(x_t, t) = tanh([x_t, e(t)] W1 + b1) W2 + b2; the output layer starts at 0."""

    def __init__(self, cfg):
        super().__init__()
        D, H, E = cfg["d_data"], cfg["d_hidden"], cfg["d_time"]
        self.cfg = cfg
        self.W1 = nn.Parameter(torch.randn(D + E, H) / math.sqrt(D + E))
        self.b1 = nn.Parameter(torch.zeros(H))
        self.W2 = nn.Parameter(torch.zeros(H, D))
        self.b2 = nn.Parameter(torch.zeros(D))

    def forward(self, x, t):
        inp = torch.cat([x, time_embedding(t, self.cfg["d_time"])], 1)
        return torch.tanh(inp @ self.W1 + self.b1) @ self.W2 + self.b2

    @classmethod
    def from_params(cls, params, cfg):
        model = cls(cfg)
        with torch.no_grad():
            for k, v in params.items():
                getattr(model, k).copy_(torch.as_tensor(v))
        return model


def _col(v, t):
    return v[t - 1][:, None]


def q_sample(cfg, x0, t, eps):
    """eq-forward-marginal: x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps."""
    ab = _col(schedule(cfg)["abar"], t)
    return ab.sqrt() * x0 + (1.0 - ab).sqrt() * eps


def posterior(cfg, xt, x0, t):
    """eq-posterior: the mean and variance of q(x_{t-1} | x_t, x_0)."""
    s = schedule(cfg)
    b, a, ab, abp = (
        _col(s["beta"], t),
        _col(s["alpha"], t),
        _col(s["abar"], t),
        _col(s["abar0"][:-1], t),
    )
    mean = abp.sqrt() * b / (1.0 - ab) * x0 + a.sqrt() * (1.0 - abp) / (1.0 - ab) * xt
    return mean, _col(s["beta_tilde"], t)


def reverse_mean(cfg, xt, t, eps):
    s = schedule(cfg)
    b, a, ab = _col(s["beta"], t), _col(s["alpha"], t), _col(s["abar"], t)
    return (xt - b / (1.0 - ab).sqrt() * eps) / a.sqrt()


def ddpm_step(cfg, xt, t, eps, z):
    """eq-ddpm-step with sigma_t^2 = beta_t."""
    return reverse_mean(cfg, xt, t, eps) + _col(schedule(cfg)["beta"], t).sqrt() * z


def ddim_step(cfg, xt, t, s, eps, z, eta=0.0):
    """eq-ddim-step from step t to step s < t, with the sigma of eq-ddim-sigma."""
    ab0 = schedule(cfg)["abar0"]
    at, as_ = ab0[t][:, None], ab0[s][:, None]
    sig = eta * ((1.0 - as_) / (1.0 - at) * (1.0 - at / as_)).sqrt()
    x0 = (xt - (1.0 - at).sqrt() * eps) / at.sqrt()
    return as_.sqrt() * x0 + (1.0 - as_ - sig**2).sqrt() * eps + sig * z


def simple_loss(model, x0, t, eps):
    """eq-simple-loss: the mean over rows of ||eps - eps_theta(x_t, t)||^2."""
    return ((model(q_sample(model.cfg, x0, t, eps), t) - eps) ** 2).sum(1).mean()


def guided_eps(eps_cond, eps_uncond, w):
    """eq-guidance."""
    return (1.0 + w) * eps_cond - w * eps_uncond


@torch.no_grad()
def ddpm_sample(model, n, generator=None):
    cfg = model.cfg
    x = torch.randn(n, cfg["d_data"], generator=generator)
    for t in range(cfg["n_steps"], 0, -1):
        tt = torch.full((n,), t)
        z = torch.randn(x.shape, generator=generator) if t > 1 else torch.zeros_like(x)
        x = ddpm_step(cfg, x, tt, model(x, tt), z)
    return x


@torch.no_grad()
def ddim_sample(model, x_T, n_sample):
    """The deterministic sampler (eta = 0) along n_sample evenly spaced steps ending at T."""
    cfg = model.cfg
    tau = torch.unique(torch.round(torch.linspace(1, cfg["n_steps"], n_sample)).long())
    prev = torch.cat([torch.zeros(1, dtype=torch.long), tau[:-1]])
    x = x_T
    for t, s in zip(tau.flip(0).tolist(), prev.flip(0).tolist(), strict=True):
        tt, ss = torch.full((len(x),), t), torch.full((len(x),), s)
        x = ddim_step(cfg, x, tt, ss, model(x, tt), torch.zeros_like(x))
    return x


def train(model, sample_data, steps, batch=256, lr=3e-3, generator=None):
    """Algorithm 1 of Ho et al. with Adam; returns the losses."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    T, trace = model.cfg["n_steps"], []
    for _ in range(steps):
        x0 = sample_data(batch, generator)
        t = torch.randint(1, T + 1, (batch,), generator=generator)
        loss = simple_loss(model, x0, t, torch.randn(x0.shape, generator=generator))
        opt.zero_grad()
        loss.backward()
        opt.step()
        trace.append(loss.item())
    return trace
