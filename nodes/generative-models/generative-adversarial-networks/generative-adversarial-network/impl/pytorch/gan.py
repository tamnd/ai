# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/generative-adversarial-network/ (ai://NEW-generative-adversarial-network).
"""The generative adversarial network of the page in PyTorch, with autograd for every backward pass: the generator of
eq-generator, the discriminator or critic of eq-discriminator, and the five losses of the NumPy reference, "minimax",
"nonsat", "wgan", "wgan_gp" and "r1". The penalties of "wgan_gp" and "r1" take the gradient of the critic in its
input with create_graph=True, so that their own gradients in the critic's weights are the second derivatives.

The weights have the shape (n_in, n_out) and a layer computes h @ W + b, as in the NumPy reference, so from_params
loads the same arrays. The discriminator's loss holds the fakes fixed with detach(), as Algorithm 1 of Goodfellow et
al. (2014) does: a discriminator step changes the discriminator only.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

LOSSES = ("minimax", "nonsat", "wgan", "wgan_gp", "r1")
GEN, DIS = ("gen_W1", "gen_b1", "gen_W2", "gen_b2"), ("dis_W1", "dis_b1", "dis_w2", "dis_b2")


def config(d_noise=2, d_hidden=5, d_data=3, d_critic=4):
    if min(d_noise, d_hidden, d_data, d_critic) < 1:
        raise ValueError("every size must be positive")
    return {"d_noise": int(d_noise), "d_hidden": int(d_hidden), "d_data": int(d_data), "d_critic": int(d_critic)}


def param_shapes(cfg):
    J, H, D, K = cfg["d_noise"], cfg["d_hidden"], cfg["d_data"], cfg["d_critic"]
    return {"gen_W1": (J, H), "gen_b1": (H,), "gen_W2": (H, D), "gen_b2": (D,),
            "dis_W1": (D, K), "dis_b1": (K,), "dis_w2": (K,), "dis_b2": (1,)}  # fmt: skip


def param_count(cfg):
    s = param_shapes(cfg)
    return sum(math.prod(s[k]) for k in GEN), sum(math.prod(s[k]) for k in DIS)


class GAN(nn.Module):
    def __init__(self, cfg, generator=None):
        super().__init__()
        self.cfg = cfg
        self.p = nn.ParameterDict()
        for k, s in param_shapes(cfg).items():
            weight = k in ("gen_W1", "gen_W2", "dis_W1", "dis_w2")
            w = torch.randn(s, generator=generator) / math.sqrt(s[0]) if weight else torch.zeros(s)
            self.p[k] = nn.Parameter(w)

    @classmethod
    def from_params(cls, params, cfg):
        """A model holding copies of the arrays of a NumPy parameter dict, the keys of param_shapes."""
        m = cls(cfg)
        with torch.no_grad():
            for k in m.p:
                m.p[k].copy_(torch.as_tensor(params[k]))
        return m

    def gen_params(self):
        return [self.p[k] for k in GEN]

    def dis_params(self):
        return [self.p[k] for k in DIS]

    def generate(self, z):
        """eq-generator: G(z) = tanh(z U1 + c1) U2 + c2."""
        return torch.tanh(z @ self.p["gen_W1"] + self.p["gen_b1"]) @ self.p["gen_W2"] + self.p["gen_b2"]

    def critic(self, x):
        """eq-discriminator: the logit a(x) = tanh(x W1 + b1) w2 + b2, one per row."""
        return torch.tanh(x @ self.p["dis_W1"] + self.p["dis_b1"]) @ self.p["dis_w2"] + self.p["dis_b2"]

    def input_grad(self, x):
        """The gradient of a in its input at each row of x, kept in the graph so that it can be differentiated."""
        x = x.detach().requires_grad_(True)
        (g,) = torch.autograd.grad(self.critic(x).sum(), x, create_graph=True)
        return g

    def dis_loss(self, x, z, kind="nonsat", alpha=None, lam=10.0, gamma=1.0):
        """The discriminator's or critic's loss on data x and noise z, the fakes held fixed."""
        if kind not in LOSSES:
            raise ValueError(kind)
        fake = self.generate(z).detach()
        ar, af = self.critic(x), self.critic(fake)
        if kind in ("wgan", "wgan_gp"):
            loss = af.mean() - ar.mean()
        else:
            loss = F.softplus(-ar).mean() + F.softplus(af).mean()
        if kind == "wgan_gp":
            if alpha is None:
                raise ValueError("wgan_gp needs alpha")
            xh = alpha[:, None] * x + (1 - alpha[:, None]) * fake
            loss = loss + lam * ((self.input_grad(xh).norm(dim=-1) - 1) ** 2).mean()
        elif kind == "r1":
            loss = loss + 0.5 * gamma * (self.input_grad(x) ** 2).sum(-1).mean()
        return loss

    def gen_loss(self, z, kind="nonsat"):
        """The generator's loss: log(1 - D(G(z))), -log D(G(z)) or -a(G(z))."""
        if kind not in LOSSES:
            raise ValueError(kind)
        af = self.critic(self.generate(z))
        if kind == "minimax":
            return -F.softplus(af).mean()
        if kind in ("nonsat", "r1"):
            return F.softplus(-af).mean()
        return -af.mean()


def train(model, sample_data, steps, kind="nonsat", batch=64, n_dis=1, lr=1e-3, betas=(0.5, 0.999), lam=10.0,
          gamma=1.0, generator=None):  # fmt: skip
    """Alternating Adam steps, n_dis on the discriminator per step on the generator; returns their losses."""
    od = torch.optim.Adam(model.dis_params(), lr=lr, betas=betas)
    og = torch.optim.Adam(model.gen_params(), lr=lr, betas=betas)
    J, trace = model.cfg["d_noise"], []
    dt = model.p["gen_W1"].dtype
    for _ in range(steps):
        for _ in range(n_dis):
            x = sample_data(batch, generator)
            z = torch.randn(batch, J, generator=generator, dtype=dt)
            alpha = torch.rand(batch, generator=generator, dtype=dt)
            od.zero_grad()
            ld = model.dis_loss(x, z, kind, alpha, lam, gamma)
            ld.backward()
            od.step()
        og.zero_grad()
        lg = model.gen_loss(torch.randn(batch, J, generator=generator, dtype=dt), kind)
        lg.backward()
        og.step()
        trace.append((ld.item(), lg.item()))
    return trace
