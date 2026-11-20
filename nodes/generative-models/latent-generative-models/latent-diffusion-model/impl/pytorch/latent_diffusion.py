# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/latent-diffusion-model/ (ai://NEW-latent-diffusion-model).
"""A latent diffusion model in PyTorch, the gradients by autograd.

The same pieces as the NumPy reference: the principal component autoencoder that downsamples by f (eq-autoencoder),
the KL penalty and the latent scale of eq-scale; vector quantization (eq-vq) with the straight-through estimator
written as z_e + (z_q - z_e).detach(), the codebook and commitment terms of eq-vq-loss with detach in place of the
stop-gradient, the k-means step and the EMA codebook of eq-ema; the linear schedule, forward noising, the exact noise
predictor for a Gaussian mixture, DDIM and classifier-free guidance; and the denoiser as an nn.Module, conditioned
through the cross-attention of eq-cross-attn and trained on eq-ldm with torch.optim.Adam. Everything is float64.
"""

import math

import torch
from torch import nn

torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------------------------------------------------------------
# The first stage


def patches(x: torch.Tensor, f: int) -> torch.Tensor:
    return x.reshape(x.shape[0], -1, f)


def fit_autoencoder(x: torch.Tensor, f: int, c: int) -> dict[str, torch.Tensor]:
    """The linear autoencoder of the top c principal components of the patches of length f of the signals x."""
    P = patches(x, f).reshape(-1, f)
    mean = P.mean(0)
    cov = (P - mean).T @ (P - mean) / P.shape[0]
    _, vecs = torch.linalg.eigh(cov)
    We = vecs.flip(1)[:, :c]
    return {"We": We, "be": -mean @ We, "Wd": We.T.clone(), "bd": mean}


def encode(ae: dict[str, torch.Tensor], x: torch.Tensor, f: int) -> torch.Tensor:
    return patches(x, f) @ ae["We"] + ae["be"]


def decode(ae: dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
    return (z @ ae["Wd"] + ae["bd"]).reshape(z.shape[0], -1)


def kl_to_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * (mu**2 + logvar.exp() - 1.0 - logvar).reshape(mu.shape[0], -1).sum(1)


def latent_scale(z: torch.Tensor) -> float:
    return float(z.std(unbiased=False))


# ---------------------------------------------------------------------------------------------------------------------
# Vector quantization


def quantize(z: torch.Tensor, codebook: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    d = (z**2).sum(-1, keepdim=True) - 2.0 * z @ codebook.T + (codebook**2).sum(-1)
    idx = d.argmin(-1)
    return idx, codebook[idx]


def vq_terms(z_e: torch.Tensor, codebook: torch.Tensor, idx: torch.Tensor, beta: float):
    """The codebook term |sg(z_e) - e|^2 and beta times the commitment term |z_e - sg(e)|^2, averaged over vectors."""
    e = codebook[idx]
    return ((z_e.detach() - e) ** 2).sum(-1).mean(), beta * ((z_e - e.detach()) ** 2).sum(-1).mean()


def straight_through(z_e: torch.Tensor, z_q: torch.Tensor) -> torch.Tensor:
    """z_q in the forward pass, the identity in the backward pass: the decoder's gradient is copied onto z_e."""
    return z_e + (z_q - z_e).detach()


def vq_grads(z_e: torch.Tensor, codebook: torch.Tensor, idx: torch.Tensor, g_zq: torch.Tensor, beta: float):
    """The gradients of <g_zq, z_q> through the straight-through estimator plus the two terms, by autograd."""
    z_e, codebook = z_e.detach().requires_grad_(), codebook.detach().requires_grad_()
    sq, commit = vq_terms(z_e, codebook, idx, beta)
    loss = (g_zq * straight_through(z_e, codebook[idx])).sum() + sq + commit
    g_e, g_cb = torch.autograd.grad(loss, (z_e, codebook))
    return g_e, g_cb


def kmeans_step(z: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    z = z.reshape(-1, codebook.shape[1])
    idx, _ = quantize(z, codebook)
    n = torch.bincount(idx, minlength=len(codebook)).to(z.dtype)
    s = torch.zeros_like(codebook).index_add_(0, idx, z)
    return torch.where(n[:, None] > 0, s / n.clamp(min=1.0)[:, None], codebook)


def ema_init(codebook: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"N": torch.zeros(len(codebook)), "m": torch.zeros_like(codebook), "e": codebook.clone()}


def ema_update(state: dict[str, torch.Tensor], z_e: torch.Tensor, idx: torch.Tensor, gamma: float):
    z, idx = z_e.reshape(-1, state["e"].shape[1]), idx.reshape(-1)
    n = torch.bincount(idx, minlength=len(state["e"])).to(z.dtype)
    s = torch.zeros_like(state["m"]).index_add_(0, idx, z)
    N = gamma * state["N"] + (1 - gamma) * n
    m = gamma * state["m"] + (1 - gamma) * s
    e = torch.where(N[:, None] > 0, m / torch.where(N > 0, N, 1.0)[:, None], state["e"])
    return {"N": N, "m": m, "e": e}


# ---------------------------------------------------------------------------------------------------------------------
# Diffusion in the latent space


def alpha_bars(T: int, b0: float = 1e-4, b1: float = 2e-2) -> torch.Tensor:
    return torch.cumprod(1.0 - torch.linspace(b0, b1, T), 0)


def q_sample(z0: torch.Tensor, abar: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    a = abar.reshape((-1,) + (1,) * (z0.dim() - 1))
    return a.sqrt() * z0 + (1.0 - a).sqrt() * eps


def snr(abar, var=1.0):
    return abar * var / (1.0 - abar)


def exact_eps(zt: torch.Tensor, abar: torch.Tensor, means: torch.Tensor, s2: float, weights: torch.Tensor):
    a = abar.reshape(-1, 1)
    v = a * s2 + 1.0 - a
    r = zt[:, None, :] - a.sqrt()[:, None] * means[None]
    w = torch.softmax(weights.log()[None] - 0.5 * (r**2).sum(-1) / v, 1)
    z0 = means[None] + (a.sqrt() * s2 / v)[:, None] * r
    return (zt - a.sqrt() * torch.einsum("bk,bkd->bd", w, z0)) / (1.0 - a).sqrt()


def ddim_sample(eps_fn, zT: torch.Tensor, abar: torch.Tensor, steps) -> torch.Tensor:
    z = zT
    steps = [int(t) for t in steps]
    for i, t in enumerate(steps):
        a = float(abar[t])
        a_prev = float(abar[steps[i + 1]]) if i + 1 < len(steps) else 1.0
        e = eps_fn(z, t)
        z0 = (z - math.sqrt(1.0 - a) * e) / math.sqrt(a)
        z = math.sqrt(a_prev) * z0 + math.sqrt(1.0 - a_prev) * e
    return z


def guided(e_cond, e_null, s):
    return e_null + s * (e_cond - e_null)


# ---------------------------------------------------------------------------------------------------------------------
# The denoiser


def config(N, c, H, E, M, d, C):
    return {"positions": N, "channels": c, "hidden": H, "temb": E, "ctx_tokens": M, "attn": d, "classes": C}


def param_shapes(cfg):
    D, H, E, M, d, C = (cfg["positions"] * cfg["channels"], cfg["hidden"], cfg["temb"], cfg["ctx_tokens"],
                        cfg["attn"], cfg["classes"])  # fmt: skip
    return {"W1": (D, H), "b1": (H,), "Wt": (E, H), "ctx": (C + 1, M, d), "Wq": (H, d), "Wk": (d, d),
            "Wv": (d, H), "W2": (H, D), "b2": (D,)}  # fmt: skip


def param_count(cfg):
    return sum(math.prod(s) for s in param_shapes(cfg).values())


def flops_forward(cfg, B):
    D, H, E, M, d = cfg["positions"] * cfg["channels"], cfg["hidden"], cfg["temb"], cfg["ctx_tokens"], cfg["attn"]
    return 2 * B * (D * H + E * H + H * d + M * d * d + M * d * H + M * d + M * H + H * D)


def time_embedding(t: torch.Tensor, E: int) -> torch.Tensor:
    w = torch.exp(-math.log(10000.0) * torch.arange(E // 2) / (E // 2))
    a = t.to(torch.float64)[:, None] * w
    return torch.cat([a.sin(), a.cos()], -1)


class Denoiser(nn.Module):
    """The noise predictor z + tanh(tanh(z W1 + emb(t) Wt + b1) + attention) W2 + b2, one parameter per NumPy key."""

    W1: nn.Parameter
    b1: nn.Parameter
    Wt: nn.Parameter
    ctx: nn.Parameter
    Wq: nn.Parameter
    Wk: nn.Parameter
    Wv: nn.Parameter
    W2: nn.Parameter
    b2: nn.Parameter

    def __init__(self, cfg, generator: torch.Generator | None = None):
        super().__init__()
        self.cfg = cfg
        for k, s in param_shapes(cfg).items():
            if k.startswith("b"):
                v = torch.zeros(s)
            else:
                v = torch.randn(s, generator=generator) / (math.sqrt(s[-2]) if k != "ctx" else 1.0)
            setattr(self, k, nn.Parameter(v))

    def attend(self, h: torch.Tensor, y: torch.Tensor):
        q, Cx = h @ self.Wq, self.ctx[y]
        K, V = Cx @ self.Wk, Cx @ self.Wv
        w = torch.softmax(torch.einsum("bd,bmd->bm", q, K) / math.sqrt(self.cfg["attn"]), 1)
        return torch.einsum("bm,bmh->bh", w, V), w

    def forward(self, z: torch.Tensor, t, y: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        zf = z.reshape(B, -1)
        t = torch.as_tensor(t).expand(B) if torch.as_tensor(t).dim() == 0 else torch.as_tensor(t)
        h = torch.tanh(zf @ self.W1 + time_embedding(t, self.cfg["temb"]) @ self.Wt + self.b1)
        a, _ = self.attend(h, y)
        return zf + torch.tanh(h + a) @ self.W2 + self.b2


def eps_loss(model: Denoiser, zt, t, y, eps) -> torch.Tensor:
    return ((model(zt, t, y) - eps.reshape(len(eps), -1)) ** 2).sum(1).mean()


def train(model: Denoiser, z0, y, abar, steps, generator, batch=256, lr=3e-3, p_null=0.1) -> list[float]:
    """Adam on eq-ldm with random times and noise, dropping the class to the null class with probability p_null."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    trace = []
    for _ in range(steps):
        i = torch.randint(0, len(z0), (batch,), generator=generator)
        t = torch.randint(0, len(abar), (batch,), generator=generator)
        eps = torch.randn(z0[i].shape, generator=generator)
        drop = torch.rand(batch, generator=generator) < p_null
        yi = torch.where(drop, model.cfg["classes"], y[i])
        loss = eps_loss(model, q_sample(z0[i], abar[t], eps), t, yi, eps)
        opt.zero_grad()
        loss.backward()
        opt.step()
        trace.append(float(loss))
    return trace
