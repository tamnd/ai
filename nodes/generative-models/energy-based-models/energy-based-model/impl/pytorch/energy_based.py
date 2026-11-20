# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/energy-based-model/ (ai://NEW-energy-based-model).
"""Energy-based models in PyTorch, with autograd doing the differentiation the NumPy reference writes by hand.

- `RBM`, a restricted Boltzmann machine as an nn.Module whose forward pass is the free energy of eq-free-energy. The
  energy of eq-rbm-energy, the conditionals of eq-rbm-conditionals, block Gibbs sampling, log Z by enumeration for
  small models. The exact log-likelihood gradient of eq-ml-grad is the autograd gradient of `nll`; the contrastive
  divergence estimate of eq-cd is the autograd gradient of `cd_loss`, the mean free energy of the data minus that of
  the (detached) negative samples.
- `EnergyNet`, a continuous energy E(x) = tanh(x W1 + b1) . w2 + gamma |x|^2 / 2 as an nn.Module. Its input gradient
  (the negative score), its Laplacian and the score-matching objective of eq-score-matching come from
  torch.autograd.grad with create_graph, so the objective can itself be differentiated in the parameters. Langevin
  dynamics (eq-langevin) and training against Langevin samples from a replay buffer use the same input gradient.
- Noise-contrastive estimation (eq-nce) of a one-dimensional unnormalized Gaussian, fitted by Newton's method with
  the autograd Hessian, and the energy of a classifier's logits (eq-jem).

Vectors are rows: a batch of B binary vectors is a (B, D) tensor, and W has the shape (D, H), as in the NumPy
reference.
"""

import itertools
import math

import torch
from torch import nn

torch.set_default_dtype(torch.float64)


def config(d_visible=4, d_hidden=3):
    if d_visible < 1 or d_hidden < 1:
        raise ValueError("an RBM needs at least one visible and one hidden unit")
    return {"d_visible": int(d_visible), "d_hidden": int(d_hidden)}


def param_shapes(cfg):
    D, H = cfg["d_visible"], cfg["d_hidden"]
    return {"W": (D, H), "b": (D,), "c": (H,)}


def param_count(cfg):
    return sum(math.prod(s) for s in param_shapes(cfg).values())


def flops_forward(cfg, batch):
    """eq-flops: one full Gibbs step v -> h -> v on B vectors, the products v W and h W^T, a multiply-add as 2."""
    return 4 * batch * cfg["d_visible"] * cfg["d_hidden"]


def all_binary(D):
    return torch.tensor(list(itertools.product([0.0, 1.0], repeat=D)))


class RBM(nn.Module):
    """E(v, h) = -v W h - b.v - c.h over v in {0,1}^D and h in {0,1}^H; forward(v) is the free energy F(v)."""

    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg or config()
        D, H = cfg["d_visible"], cfg["d_hidden"]
        self.W = nn.Parameter(torch.zeros(D, H))
        self.b = nn.Parameter(torch.zeros(D))
        self.c = nn.Parameter(torch.zeros(H))

    @classmethod
    def from_params(cls, params):
        D, H = params["W"].shape
        model = cls(config(D, H))
        with torch.no_grad():
            for k, p in model.named_params().items():
                p.copy_(torch.as_tensor(params[k]))
        return model

    def named_params(self):
        return {"W": self.W, "b": self.b, "c": self.c}

    def energy(self, v, h):
        """eq-rbm-energy for rows v (B, D) and h (B, H)."""
        return -((v @ self.W) * h).sum(1) - v @ self.b - h @ self.c

    def forward(self, v):
        """eq-free-energy: F(v) = -b.v - sum_j softplus(c_j + (v W)_j), so p(v) = exp(-F(v)) / Z."""
        return -v @ self.b - nn.functional.softplus(self.c + v @ self.W).sum(1)

    def p_h(self, v):
        """eq-rbm-conditionals: p(h_j = 1 | v) = sigmoid(c_j + (v W)_j), independent over j."""
        return torch.sigmoid(self.c + v @ self.W)

    def p_v(self, h):
        return torch.sigmoid(self.b + h @ self.W.T)

    @torch.no_grad()
    def gibbs_step(self, v, u_h, u_v):
        """One block Gibbs step v -> h -> v' with the uniforms u_h (B, H) and u_v (B, D)."""
        h = (u_h < self.p_h(v)).to(v.dtype)
        return h, (u_v < self.p_v(h)).to(v.dtype)

    @torch.no_grad()
    def gibbs_chain(self, v, k, g):
        B, (D, H) = len(v), self.W.shape
        for _ in range(k):
            _, v = self.gibbs_step(v, torch.rand(B, H, generator=g), torch.rand(B, D, generator=g))
        return v

    def log_partition(self):
        """log Z = logsumexp over the 2^D visible vectors of -F(v); differentiable in the parameters."""
        return torch.logsumexp(-self(all_binary(self.W.shape[0])), 0)

    def log_prob(self, v):
        return -self(v) - self.log_partition()


def nll(model, data):
    """The mean negative log-likelihood; its negative gradient is eq-ml-grad, computed exactly through log Z."""
    return -model.log_prob(data).mean()


def cd_loss(model, pos, neg):
    """mean F(pos) - mean F(neg) with neg held fixed: its negative gradient is the estimate of eq-ml-grad with the
    model expectation replaced by an average over neg, the CD-k estimate of eq-cd when neg are k Gibbs steps from
    pos, the PCD estimate when they are persistent chains."""
    return model(pos).mean() - model(neg.detach()).mean()


def train_rbm(model, data, steps, g, k=1, persistent=False, n_chains=100, batch=100, lr=0.05):
    """Gradient descent on cd_loss with CD-k or PCD negatives; returns the trace of the exact mean log-likelihood."""
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    chains = data[torch.randint(0, len(data), (n_chains,), generator=g)].clone()
    trace = []
    for _ in range(steps):
        pos = data[torch.randint(0, len(data), (batch,), generator=g)]
        if persistent:
            chains = model.gibbs_chain(chains, k, g)
            neg = chains
        else:
            neg = model.gibbs_chain(pos, k, g)
        opt.zero_grad()
        cd_loss(model, pos, neg).backward()
        opt.step()
        with torch.no_grad():
            trace.append(model.log_prob(data).mean().item())
    return trace


# A continuous energy and the derivatives in its input.


class EnergyNet(nn.Module):
    """E(x) = tanh(x W1 + b1) . w2 + gamma |x|^2 / 2 for x in R^d; the quadratic term has no parameter."""

    def __init__(self, d=2, hidden=32, gamma=0.1):
        super().__init__()
        self.W1 = nn.Parameter(torch.zeros(d, hidden))
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.w2 = nn.Parameter(torch.zeros(hidden))
        self.gamma = float(gamma)

    @classmethod
    def from_params(cls, params, gamma=0.1):
        d, H = params["W1"].shape
        model = cls(d, H, gamma)
        with torch.no_grad():
            for k, p in model.named_params().items():
                p.copy_(torch.as_tensor(params[k]))
        return model

    def reset(self, g, scale=1.0):
        """The initialization of the NumPy reference: W1 ~ N(0, scale^2 / d), b1 ~ N(0, 1/4), w2 = 0."""
        d, H = self.W1.shape
        with torch.no_grad():
            self.W1.copy_(scale / math.sqrt(d) * torch.randn(d, H, generator=g))
            self.b1.copy_(0.5 * torch.randn(H, generator=g))
            self.w2.zero_()
        return self

    def named_params(self):
        return {"W1": self.W1, "b1": self.b1, "w2": self.w2}

    def forward(self, x):
        return torch.tanh(x @ self.W1 + self.b1) @ self.w2 + 0.5 * self.gamma * (x * x).sum(1)


def grad_x(f, x, create_graph=False):
    """dE/dx row by row for an energy f mapping (B, d) to (B,): the rows do not interact, so the gradient of the sum."""
    x = x if x.requires_grad else x.detach().requires_grad_(True)
    (g,) = torch.autograd.grad(f(x).sum(), x, create_graph=create_graph)
    return g


def laplacian(f, x, create_graph=False):
    """sum_i d^2E/dx_i^2 row by row: d backward passes through the input gradient, one per coordinate."""
    x = x.detach().requires_grad_(True)
    g = grad_x(f, x, create_graph=True)
    lap = torch.zeros(len(x))
    for i in range(x.shape[1]):
        (gi,) = torch.autograd.grad(g[:, i].sum(), x, create_graph=create_graph, retain_graph=True)
        lap = lap + gi[:, i]
    return lap


def sm_objective(f, x):
    """eq-score-matching with psi = -dE/dx: the mean of -Laplacian E + |dE/dx|^2 / 2, differentiable in f's parameters."""
    x = x.detach().requires_grad_(True)
    g = grad_x(f, x, create_graph=True)
    return (-laplacian(f, x, create_graph=True) + 0.5 * (g * g).sum(1)).mean()


def gaussian_energy(mu, M):
    """E(x) = (x - mu) M (x - mu)^T / 2 as a function of x."""
    return lambda x: 0.5 * (((x - mu) @ M) * (x - mu)).sum(1)


def langevin_step(f, x, step, z, noise=None):
    """eq-langevin with the standard normal draws z: x - (step/2) dE/dx + noise z, noise = sqrt(step) unless given."""
    s = math.sqrt(step) if noise is None else noise
    return (x - 0.5 * step * grad_x(f, x) + s * z).detach()


def langevin(f, x, n_steps, step, g, noise=None):
    for _ in range(n_steps):
        x = langevin_step(f, x, step, torch.randn(x.shape, generator=g), noise)
    return x


def train_ebm(model, data, steps, g, batch=128, k=20, step=0.02, noise=0.05, alpha=0.1, lr=0.01, box=3.0,
              buffer_size=4096, reinit=0.05):  # fmt: skip
    """Algorithm 1 of Du and Mordatch: chains from a replay buffer (or fresh uniform noise on [-box, box]^d with
    probability reinit), k Langevin steps, then Adam on mean E(x+) - mean E(x-) + alpha (mean E(x+)^2 + mean E(x-)^2)
    with the samples fixed. Returns the buffer and the trace of mean E(x+) - mean E(x-)."""
    d = data.shape[1]
    buf = box * (2 * torch.rand(buffer_size, d, generator=g) - 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    trace = []
    for _ in range(steps):
        pos = data[torch.randint(0, len(data), (batch,), generator=g)]
        idx = torch.randint(0, buffer_size, (batch,), generator=g)
        x0 = buf[idx].clone()
        fresh = torch.rand(batch, generator=g) < reinit
        x0[fresh] = box * (2 * torch.rand(int(fresh.sum()), d, generator=g) - 1)
        neg = langevin(model, x0, k, step, g, noise).clamp(-box, box)
        buf[idx] = neg
        ep, en = model(pos), model(neg)
        loss = ep.mean() - en.mean() + alpha * ((ep * ep).mean() + (en * en).mean())
        opt.zero_grad()
        loss.backward()
        opt.step()
        trace.append((ep.mean() - en.mean()).item())
    return buf, trace


# Noise-contrastive estimation of log p_m(x) = theta . (-x^2/2, x, 1), with Gaussian noise (eq-nce, nu = 1).


def nce_features(x):
    return torch.stack([-0.5 * x * x, x, torch.ones_like(x)], 1)


def gauss_logpdf(x, mean, std):
    return -0.5 * ((x - mean) / std) ** 2 - math.log(std) - 0.5 * math.log(2 * math.pi)


def nce_objective(theta, x, y, noise_mean, noise_std):
    """J = mean log sigmoid(G(x)) + mean log(1 - sigmoid(G(y))), G(u) = log p_m(u) - log p_n(u)."""
    gx = nce_features(x) @ theta - gauss_logpdf(x, noise_mean, noise_std)
    gy = nce_features(y) @ theta - gauss_logpdf(y, noise_mean, noise_std)
    return nn.functional.logsigmoid(gx).mean() + nn.functional.logsigmoid(-gy).mean()


def nce_fit(x, y, noise_mean, noise_std, iters=50):
    """Newton's method on the concave J with the autograd gradient and Hessian, from theta = (1, 0, 0)."""
    theta = torch.tensor([1.0, 0.0, 0.0])

    def neg_j(t):
        return -nce_objective(t, x, y, noise_mean, noise_std)

    for _ in range(iters):
        g = torch.autograd.functional.jacobian(neg_j, theta)
        Hm = torch.autograd.functional.hessian(neg_j, theta)
        assert isinstance(g, torch.Tensor) and isinstance(Hm, torch.Tensor)
        theta = theta - torch.linalg.solve(Hm, g)
    return theta


def gaussian_natural(mean, std):
    """The theta of nce_features for N(mean, std^2), with its log-normalizer in the third entry."""
    return torch.tensor(
        [1 / std**2, mean / std**2, -0.5 * mean**2 / std**2 - math.log(std) - 0.5 * math.log(2 * math.pi)]
    )


# A classifier as an energy-based model (eq-jem).


def jem_energy(logits):
    """E(x) = -logsumexp_y f(x)[y] for the (B, K) logits of a classifier."""
    return -torch.logsumexp(logits, 1)


def jem_class_probs(logits):
    """p(y | x) = softmax(f(x)), the same for logits shifted by any constant."""
    return torch.softmax(logits, 1)
