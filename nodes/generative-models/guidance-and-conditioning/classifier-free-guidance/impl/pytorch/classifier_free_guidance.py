# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/classifier-free-guidance/ (ai://NEW-classifier-free-guidance).
"""Classifier-free guidance in PyTorch.

- The exact models of a Gaussian mixture, one Gaussian per class: the conditional and unconditional noise predictors,
  the noisy classifier log p(c | z_t), and its gradient by autograd.
- Guidance: eq-cfg, classifier guidance (eq-cg), the rescale of eq-rescale with torch's std as in Lin and colleagues'
  Algorithm 2, static and dynamic thresholding (eq-dyn) with torch.quantile, and the guidance interval.
- Deterministic DDIM with a guided noise predictor that counts network evaluations.
- The denoiser as an nn.Module with a class embedding of C + 1 rows, the last the null class, trained with
  torch.optim.Adam while the class is dropped with probability p_uncond (Algorithm 1 of Ho and Salimans).
"""

import math

import torch
from torch import nn

torch.set_default_dtype(torch.float64)


def alpha_bars(T, b0=1e-4, b1=2e-2):
    return torch.cumprod(1.0 - torch.linspace(b0, b1, T), 0)


def q_sample(x0, abar, eps):
    a = abar.reshape((-1,) + (1,) * (x0.dim() - 1))
    return a.sqrt() * x0 + (1.0 - a).sqrt() * eps


def edm_sigma(abar):
    abar = torch.as_tensor(abar)
    return ((1.0 - abar) / abar).sqrt()


def _resp(zt, abar, means, s2, weights):
    a = abar.reshape(-1, 1)
    v = a * s2 + 1.0 - a
    r = zt[:, None, :] - a.sqrt()[:, None] * means[None]
    logw = weights.log()[None] - 0.5 * (r**2).sum(-1) / v
    return a, v, r, torch.log_softmax(logw, 1), torch.logsumexp(logw, 1)


def exact_eps(zt, abar, means, s2, weights):
    a, v, r, logpost, _ = _resp(zt, abar, means, s2, weights)
    return (1.0 - a).sqrt() * torch.einsum("bk,bkd->bd", logpost.exp(), r) / v


def exact_eps_cond(zt, abar, means, s2, c):
    return exact_eps(zt, abar, means[c : c + 1], s2, torch.ones(1))


def log_marginal(zt, abar, means, s2, weights):
    a, v, _, _, lse = _resp(zt, abar, means, s2, weights)
    return lse - 0.5 * zt.shape[1] * torch.log(2.0 * math.pi * v[:, 0])


def log_classifier(zt, abar, means, s2, weights):
    return _resp(zt, abar, means, s2, weights)[3]


def classifier_grad(zt, abar, means, s2, weights, c):
    """grad log p(c | z_t) by autograd."""
    z = zt.detach().requires_grad_(True)
    lc = log_classifier(z, abar, means, s2, weights)
    (g,) = torch.autograd.grad(lc[torch.arange(len(z)), c].sum(), z)
    return g


def guide(e_cond, e_null, s):
    return e_null + s * (e_cond - e_null)


def guide_ho(e_cond, e_null, w):
    return (1.0 + w) * e_cond - w * e_null


def classifier_guided_eps(e, grad_log_classifier, abar, s):
    a = abar.reshape((-1,) + (1,) * (e.dim() - 1))
    return e - s * (1.0 - a).sqrt() * grad_log_classifier


def eps_to_x0(zt, e, abar):
    a = abar.reshape((-1,) + (1,) * (zt.dim() - 1))
    return (zt - (1.0 - a).sqrt() * e) / a.sqrt()


def rescale(out_cond, out_cfg, phi):
    """Lin and colleagues' Algorithm 2: cfg * (phi * std_pos / std_cfg + 1 - phi), with torch's (unbiased) std."""
    dims = list(range(1, out_cfg.dim()))
    factor = out_cond.std(dim=dims, keepdim=True) / out_cfg.std(dim=dims, keepdim=True)
    return out_cfg * (phi * factor + (1.0 - phi))


def static_threshold(x0):
    return x0.clamp(-1.0, 1.0)


def dynamic_threshold(x0, p):
    s = torch.quantile(x0.abs().reshape(len(x0), -1), p / 100.0, dim=1).clamp(min=1.0)
    s = s.reshape((-1,) + (1,) * (x0.dim() - 1))
    return torch.maximum(torch.minimum(x0, s), -s) / s


def interval_scale(sigma, s, lo, hi):
    sigma = torch.as_tensor(sigma)
    return torch.where((sigma > lo) & (sigma <= hi), torch.as_tensor(float(s)), torch.as_tensor(1.0))


def gaussian_eps(zt, abar, m, v):
    a = abar.reshape((-1,) + (1,) * (zt.dim() - 1))
    return (1.0 - a).sqrt() * (zt - a.sqrt() * m) / (a * v + 1.0 - a)


def ddim_sample(eps_fn, zT, abar, steps, threshold=None):
    z = zT
    for i, t in enumerate(steps):
        a = float(abar[t])
        a_prev = float(abar[steps[i + 1]]) if i + 1 < len(steps) else 1.0
        e = eps_fn(z, t)
        x0 = (z - math.sqrt(1.0 - a) * e) / math.sqrt(a)
        if threshold is not None:
            x0 = threshold(x0)
            e = (z - math.sqrt(a) * x0) / math.sqrt(1.0 - a)
        z = math.sqrt(a_prev) * x0 + math.sqrt(1.0 - a_prev) * e
    return z


def guided_sampler(eps_cond, eps_null, s, abar, interval=None, phi=0.0):
    count = {"nfe": 0}

    def eps(z, t):
        sc = s if interval is None else float(interval_scale(edm_sigma(abar[t]), s, *interval))
        e_c = eps_cond(z, t)
        count["nfe"] += 1
        if sc == 1.0:
            return e_c
        e_u = eps_null(z, t)
        count["nfe"] += 1
        out = guide(e_c, e_u, sc)
        return rescale(e_c, out, phi) if phi else out

    return eps, count


def time_embedding(t, E):
    """(len(t), E): sines at (E + 1) // 2 frequencies, then cosines at the first E // 2."""
    n = (E + 1) // 2
    a = torch.as_tensor(t, dtype=torch.float64)[:, None] * torch.exp(-math.log(10000.0) * torch.arange(n) / n)
    return torch.cat([a.sin(), a[:, : E // 2].cos()], -1)


def param_count(D, H, E, C):
    return D * H + E * H + (C + 1) * H + H + H * D + D


def flops_forward(D, H, E, B):
    """Multiply-adds of the three matrix products, counted as 2; the embedding lookup, biases and tanh are left out."""
    return 2 * B * (D * H + E * H + H * D)


class Denoiser(nn.Module):
    """eps(z, t, y) = z + tanh(z W1 + emb(t) Wt + E[y] + b1) W2 + b2, with E[C] the null class."""

    W1: nn.Parameter
    Wt: nn.Parameter
    emb: nn.Parameter
    b1: nn.Parameter
    W2: nn.Parameter
    b2: nn.Parameter

    def __init__(self, D, H, E, C, params=None):
        super().__init__()
        self.D, self.H, self.E, self.C = D, H, E, C
        shapes = {"W1": (D, H), "Wt": (E, H), "emb": (C + 1, H), "b1": (H,), "W2": (H, D), "b2": (D,)}
        for k, s in shapes.items():
            if params is not None:
                v = torch.as_tensor(params[k], dtype=torch.float64).clone()
            elif k.startswith("b"):
                v = torch.zeros(s)
            else:
                v = torch.randn(s) / math.sqrt(s[0] if k != "emb" else 1)
            setattr(self, k, nn.Parameter(v))

    def forward(self, z, t, y):
        t = torch.as_tensor(t).expand(len(z))
        y = torch.as_tensor(y).expand(len(z))
        h = torch.tanh(z @ self.W1 + time_embedding(t, self.E) @ self.Wt + self.emb[y] + self.b1)
        return z + h @ self.W2 + self.b2


def eps_loss(model, zt, t, y, eps):
    return ((model(zt, t, y) - eps) ** 2).sum(1).mean()


def drop_condition(y, u, p_uncond, C):
    return torch.where(u < p_uncond, torch.as_tensor(C), y)


def train(model, x0, y, abar, steps, gen, batch=256, lr=3e-3, p_uncond=0.1):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    trace = []
    for _ in range(steps):
        i = torch.randint(0, len(x0), (batch,), generator=gen)
        t = torch.randint(0, len(abar), (batch,), generator=gen)
        eps = torch.randn(x0[i].shape, generator=gen)
        yi = drop_condition(y[i], torch.rand(batch, generator=gen), p_uncond, model.C)
        loss = eps_loss(model, q_sample(x0[i], abar[t], eps), t, yi, eps)
        opt.zero_grad()
        loss.backward()
        opt.step()
        trace.append(loss.item())
    return torch.tensor(trace)
