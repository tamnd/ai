# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/consistency-model/ (ai://NEW-consistency-model).
"""Consistency models in PyTorch.

- The noise levels of Karras and colleagues, the boundary-preserving scales c_skip, c_out and c_in (eq-param), and
  the exact score and probability-flow ODE of a mixture of Gaussians, with the closed-form consistency function of
  one Gaussian (eq-gauss-cf).
- The model as an nn.Module, f(x, t) = c_skip(t) x + c_out(t) F(x, t), and its time derivative along the ODE by
  torch.func.jvp, the quantity that the continuous-time losses of Song and colleagues drive to zero.
- The distillation and training losses (eq-cd, eq-ct) with the squared l2 or Pseudo-Huber metric (eq-huber), the
  target network as a deep copy updated by torch's in-place EMA (eq-ema) and kept out of autograd, and training with
  torch.optim.Adam; consistency training follows Song and Dhariwal's improved recipe.
- The sampler of Algorithm 1, the toy of Proposition 1 of Song and Dhariwal, and the target of progressive
  distillation (eq-pd).
"""

import copy
import math

import torch
from torch import nn

torch.set_default_dtype(torch.float64)

SIGMA_DATA, EPS, T_MAX, RHO = 0.5, 0.002, 80.0, 7.0


def karras_steps(N, eps=EPS, T=T_MAX, rho=RHO):
    i = torch.arange(N, dtype=torch.float64)
    return (eps ** (1 / rho) + i / (N - 1) * (T ** (1 / rho) - eps ** (1 / rho))) ** rho


def c_skip(t, eps=EPS, sd=SIGMA_DATA):
    return sd**2 / ((t - eps) ** 2 + sd**2)


def c_out(t, eps=EPS, sd=SIGMA_DATA):
    return sd * (t - eps) / torch.sqrt(sd**2 + torch.as_tensor(t) ** 2)


def c_in(t, sd=SIGMA_DATA):
    return 1.0 / torch.sqrt(torch.as_tensor(t) ** 2 + sd**2)


def _col(t, n):
    return torch.as_tensor(t, dtype=torch.float64).expand(n)[:, None]


def score(x, t, means, s2, weights):
    v = s2 + _col(t, len(x)) ** 2
    r = x[:, None, :] - means[None]
    post = torch.softmax(weights.log()[None] - 0.5 * (r**2).sum(-1) / v, 1)
    return -torch.einsum("bk,bkd->bd", post, r) / v


def ode_drift(x, t, score_fn):
    return -_col(t, len(x)) * score_fn(x, t)


def teacher_step(x, t_from, t_to, score_fn, solver="heun"):
    h = _col(t_to, len(x)) - _col(t_from, len(x))
    d1 = ode_drift(x, t_from, score_fn)
    x1 = x + h * d1
    if solver == "euler":
        return x1
    return x + 0.5 * h * (d1 + ode_drift(x1, t_to, score_fn))


def solve_ode(x, t_from, t_to, score_fn, n=200):
    ts = karras_steps(n + 1, eps=t_to, T=t_from).flip(0)
    for a, b in zip(ts[:-1], ts[1:], strict=True):
        x = teacher_step(x, a, b, score_fn)
    return x


def gaussian_consistency(x, t, mu, s2, eps=EPS):
    return mu + (x - mu) * torch.sqrt((s2 + eps**2) / (s2 + _col(t, len(x)) ** 2))


def param_count(D, H, E):
    return D * H + E * H + H + H * D + D


def flops_forward(D, H, E, B):
    return 2 * B * (D * H + E * H + H * D)


def time_embedding(t, E):
    n = (E + 1) // 2
    a = (torch.log(t) / 4.0)[:, None] * 2.0 ** torch.arange(n, dtype=torch.float64)
    return torch.cat([a.sin(), a[:, : E // 2].cos()], -1)


class ConsistencyModel(nn.Module):
    """f(x, t) = c_skip(t) x + c_out(t) (tanh(c_in(t) x W1 + emb(t) Wt + b1) W2 + b2)."""

    W1: nn.Parameter
    Wt: nn.Parameter
    b1: nn.Parameter
    W2: nn.Parameter
    b2: nn.Parameter

    def __init__(self, D, H, E, params=None):
        super().__init__()
        self.D, self.H, self.E = D, H, E
        for k, s in {"W1": (D, H), "Wt": (E, H), "b1": (H,), "W2": (H, D), "b2": (D,)}.items():
            if params is not None:
                v = torch.as_tensor(params[k], dtype=torch.float64).clone()
            else:
                v = torch.zeros(s) if k.startswith("b") else torch.randn(s) / math.sqrt(s[0])
            setattr(self, k, nn.Parameter(v))

    def forward(self, x, t):
        t = torch.as_tensor(t, dtype=torch.float64).expand(len(x))
        h = torch.tanh(c_in(t)[:, None] * x @ self.W1 + time_embedding(t, self.E) @ self.Wt + self.b1)
        return c_skip(t)[:, None] * x + c_out(t)[:, None] * (h @ self.W2 + self.b2)


def total_derivative(f, x, t, score_fn):
    """d/dt f(x_t, t) along the ODE through (x, t): the Jacobian-vector product of f with the tangent (dx/dt, 1)."""
    t = torch.as_tensor(t, dtype=torch.float64).expand(len(x)).clone()
    d = torch.func.jvp(f, (x, t), (ode_drift(x, t, score_fn), torch.ones_like(t)))[1]
    return d


def metric(a, b, c=None):
    d2 = ((a - b) ** 2).sum(1)
    return d2 if c is None else torch.sqrt(d2 + c**2) - c


def consistency_loss(model, target, x_hi, t_hi, x_lo, t_lo, weight=1.0, c=None):
    """mean lambda d(f_theta(x_hi, t_hi), f_theta-(x_lo, t_lo)), the target computed without a graph."""
    with torch.no_grad():
        tgt = target(x_lo, t_lo)
    return (torch.as_tensor(weight) * metric(model(x_hi, t_hi), tgt, c)).mean()


def cd_pair(x0, z, t_lo, t_hi, score_fn, solver="heun"):
    x_hi = x0 + _col(t_hi, len(x0)) * z
    return x_hi, teacher_step(x_hi, t_hi, t_lo, score_fn, solver)


def ct_pair(x0, z, t_lo, t_hi):
    return x0 + _col(t_hi, len(x0)) * z, x0 + _col(t_lo, len(x0)) * z


@torch.no_grad()
def ema_(target, model, mu):
    """eq-ema in place: theta- <- mu theta- + (1 - mu) theta."""
    for pm, p in zip(target.parameters(), model.parameters(), strict=True):
        pm.lerp_(p, 1.0 - mu)


def cm_schedule(k, K, s0=2, s1=150, mu0=0.9):
    N = math.ceil(math.sqrt(k / K * ((s1 + 1) ** 2 - s0**2) + s0**2) - 1) + 1
    return N, math.exp(s0 * math.log(mu0) / N)


def ict_steps(k, K, s0=10, s1=1280):
    Kp = math.floor(K / (math.log2(math.floor(s1 / s0)) + 1))
    return min(s0 * 2 ** math.floor(k / Kp), s1) + 1


def lognormal_probs(ts, p_mean=-1.1, p_std=2.0):
    e = torch.erf((torch.log(ts) - p_mean) / (math.sqrt(2.0) * p_std))
    q = e[1:] - e[:-1]
    return q / q.sum()


def train_cd(model, x0, score_fn, steps, gen, N=18, batch=256, lr=2e-3, mu=0.95, c=None, solver="heun"):
    """Consistency distillation (Algorithm 2): returns the target network and the loss trace."""
    ts, target = karras_steps(N), copy.deepcopy(model).requires_grad_(False)
    opt, trace = torch.optim.Adam(model.parameters(), lr=lr), []
    for _ in range(steps):
        x = x0[torch.randint(0, len(x0), (batch,), generator=gen)]
        n = torch.randint(0, N - 1, (batch,), generator=gen)
        x_hi, x_lo = cd_pair(x, torch.randn(x.shape, generator=gen), ts[n], ts[n + 1], score_fn, solver)
        loss = consistency_loss(model, target, x_hi, ts[n + 1], x_lo, ts[n], 1.0, c)
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema_(target, model, mu)
        trace.append(loss.item())
    return target, torch.tensor(trace)


def train_ct(model, x0, steps, gen, batch=256, lr=2e-3, s0=10, s1=160, c=None):
    """Improved consistency training: the target is the student itself under no_grad, N doubled on a curriculum,
    lognormal levels, the weight 1 / (t_{i+1} - t_i) and the Pseudo-Huber metric."""
    opt, trace = torch.optim.Adam(model.parameters(), lr=lr), []
    if c is None:
        c = 0.00054 * math.sqrt(x0.shape[1])
    for k in range(steps):
        ts = karras_steps(ict_steps(k, steps, s0, s1))
        x = x0[torch.randint(0, len(x0), (batch,), generator=gen)]
        n = torch.multinomial(lognormal_probs(ts), batch, replacement=True, generator=gen)
        x_hi, x_lo = ct_pair(x, torch.randn(x.shape, generator=gen), ts[n], ts[n + 1])
        loss = consistency_loss(model, model, x_hi, ts[n + 1], x_lo, ts[n], 1.0 / (ts[n + 1] - ts[n]), c)
        opt.zero_grad()
        loss.backward()
        opt.step()
        trace.append(loss.item())
    return torch.tensor(trace)


@torch.no_grad()
def sample(f, xT, taus=(), noise=None, eps=EPS, T=T_MAX):
    """Algorithm 1 with the noise given, shape (len(taus),) + x_T.shape."""
    x = f(xT, T)
    for k, tau in enumerate(taus):
        assert noise is not None
        x = f(x + math.sqrt(tau**2 - eps**2) * noise[k], tau)
    return x


def toy_model(x, t, theta, smin):
    return smin / t * x + (1 - smin / t) * theta


def toy_ct_loss(theta, theta_minus, xi, ts, z):
    """The CT loss of the toy as a tensor of theta, the target detached; its gradient comes from autograd."""
    smin, lo, hi = ts[0], ts[:-1, None], ts[1:, None]
    diff = toy_model(xi + hi * z, hi, theta, smin) - toy_model(xi + lo * z, lo, theta_minus.detach(), smin)
    return (diff**2).mean()


def vp(lam):
    a = 1.0 / torch.sqrt(1.0 + torch.exp(-torch.as_tensor(lam, dtype=torch.float64)))
    return a, torch.sqrt(1.0 - a**2)


def ddim_step(z, x_pred, a_t, s_t, a_s, s_s):
    return a_s * x_pred + s_s * (z - a_t * x_pred) / s_t


def pd_target(z_t, z_s2, a_t, s_t, a_s2, s_s2):
    r = s_s2 / s_t
    return (z_s2 - r * z_t) / (a_s2 - r * a_t)
