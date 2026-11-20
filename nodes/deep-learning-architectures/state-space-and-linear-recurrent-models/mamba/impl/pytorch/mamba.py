# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/mamba/ (ai://2WYA9X).
"""Mamba of Gu and Dao (2023) in PyTorch: the Mamba block and a language model of them, as torch.nn.Modules.

The parameters and their names are those of the NumPy reference (impl/numpy/mamba.py), with linear maps stored as
(in, out) matrices, so a dict of arrays from it or from the fixtures loads with from_params. The selective scan runs
as a loop over the positions; autograd gives the backward pass. selective=False is the ablation of the paper's
Table 7 with no selective parameter: Delta = softplus(dt_b), and B and C are learned (I, N) matrices that are the same
at every position, which makes the SSM time-invariant (an S4D layer with real A).
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def config(vocab=11, d_model=8, n_layers=2, d_state=4, d_conv=3, expand=2, dt_rank=None, disc="simple", eps=1e-5):
    R = math.ceil(d_model / 16) if dt_rank is None else dt_rank
    return {"vocab": vocab, "d_model": d_model, "n_layers": n_layers, "d_state": d_state, "d_conv": d_conv,
            "expand": expand, "d_inner": expand * d_model, "dt_rank": R, "disc": disc, "eps": eps}  # fmt: skip


def rmsnorm(x, w, eps):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * w


class MambaBlock(nn.Module):
    """The mixer: in_proj, causal depthwise conv and SiLU, the selective SSM with the skip D, the gate, out_proj."""

    def __init__(self, cfg, selective=True, mode="loop"):
        super().__init__()
        self.cfg, self.selective, self.mode = cfg, selective, mode
        D, I, N, K, R = (cfg[k] for k in ("d_model", "d_inner", "d_state", "d_conv", "dt_rank"))
        z = torch.zeros
        self.norm = nn.Parameter(torch.ones(D))
        self.in_proj = nn.Parameter(torch.randn(D, 2 * I) / math.sqrt(D))
        self.conv_w = nn.Parameter(torch.randn(I, K) / math.sqrt(K))
        self.conv_b = nn.Parameter(z(I))
        self.x_proj = nn.Parameter(torch.randn(I, R + 2 * N) / math.sqrt(I))
        self.dt_w = nn.Parameter(torch.empty(R, I).uniform_(-(R**-0.5), R**-0.5))
        dt = torch.exp(torch.empty(I).uniform_(math.log(1e-3), math.log(1e-1)))
        self.dt_b = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.log(torch.arange(1.0, N + 1).repeat(I, 1)))
        self.D = nn.Parameter(torch.ones(I))
        self.out_proj = nn.Parameter(torch.randn(I, D) / math.sqrt(I))
        if not selective:
            self.B_lti = nn.Parameter(torch.randn(I, N) / math.sqrt(N))
            self.C_lti = nn.Parameter(torch.randn(I, N) / math.sqrt(N))

    def ssm_inputs(self, xa):
        """Delta (B, T, I) and B, C as (B, T, 1, N) or, without selection, (1, 1, I, N)."""
        N, R = self.cfg["d_state"], self.cfg["dt_rank"]
        if not self.selective:
            return F.softplus(self.dt_b).expand(xa.shape), self.B_lti[None, None], self.C_lti[None, None]
        dbl = xa @ self.x_proj
        dt = F.softplus(dbl[..., :R] @ self.dt_w + self.dt_b)
        return dt, dbl[..., None, R : R + N], dbl[..., None, R + N :]

    def forward(self, u):
        I, K = self.cfg["d_inner"], self.cfg["d_conv"]
        T = u.shape[1]
        x, z = (u @ self.in_proj).split(I, -1)
        xc = F.conv1d(x.transpose(1, 2), self.conv_w[:, None], self.conv_b, padding=K - 1, groups=I)[..., :T]
        xa = F.silu(xc.transpose(1, 2))
        dt, Bm, C = self.ssm_inputs(xa)
        A = -torch.exp(self.A_log)
        dA = torch.exp(dt[..., None] * A)
        dB = dt[..., None] * Bm if self.cfg["disc"] == "simple" else (dA - 1.0) / A * Bm
        if self.mode == "loop":
            h = u.new_zeros(u.shape[0], I, A.shape[1])
            ys = []
            for t in range(T):
                h = dA[:, t] * h + dB[:, min(t, dB.shape[1] - 1)] * xa[:, t, :, None]
                ys.append((h * C[:, min(t, C.shape[1] - 1)]).sum(-1))
            y = torch.stack(ys, 1)
        else:
            y = quadratic_scan(dt, A, dB.expand(dt.shape + A.shape[-1:]), xa, C.expand(dt.shape + A.shape[-1:]))
        return ((y + self.D * xa) * F.silu(z)) @ self.out_proj


def quadratic_scan(dt, A, dB, x, C):
    """eq-quadratic: the same outputs as the scan, all at once. Because A_bar_t = exp(Delta_t A), the product of
    A_bar from s + 1 to t is exp(A (S_t - S_s)) with S the running sum of Delta, so y_t = sum_{s <= t} C_t .
    exp(A (S_t - S_s)) B_bar_s x_s: a masked (T, T) product per channel, O(T^2) work but no loop over positions.
    The exponent is never positive, since A < 0 and S is increasing, so nothing overflows."""
    S = dt.cumsum(1)  # (B, T, I)
    diff = S[:, :, None, :] - S[:, None, :, :]  # (B, t, s, I)
    causal = torch.ones(dt.shape[1], dt.shape[1], dtype=dt.dtype, device=dt.device).tril()[None, :, :, None]
    diff = diff * causal  # zero above the diagonal, where s > t, so that the exponent stays finite there
    decay = torch.exp(diff[..., None] * A) * causal[..., None]  # (B, t, s, I, N), zero above the diagonal
    return ((decay * (dB * x[..., None])[:, None]).sum(2) * C).sum(-1)


class MambaLM(nn.Module):
    """Embedding, n_layers residual blocks r <- r + mixer(rmsnorm(r)), a final RMSNorm and the tied head."""

    def __init__(self, cfg, selective=True, mode="loop"):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Parameter(torch.randn(cfg["vocab"], cfg["d_model"]) * 0.1)
        self.norm_f = nn.Parameter(torch.ones(cfg["d_model"]))
        self.blocks = nn.ModuleList(MambaBlock(cfg, selective, mode) for _ in range(cfg["n_layers"]))

    @classmethod
    def from_params(cls, cfg, params):
        """Build the model from a dict of arrays named as in the NumPy reference (l0_in_proj, ..., embed)."""
        m = cls(cfg).double()
        with torch.no_grad():
            m.embed.copy_(torch.as_tensor(params["embed"]))
            m.norm_f.copy_(torch.as_tensor(params["norm_f"]))
            for l, b in enumerate(m.blocks):
                for name, t in b.named_parameters():
                    t.copy_(torch.as_tensor(params[f"l{l}_{name}"]))
        return m

    def grads(self):
        """The gradients under the names of the NumPy reference."""
        g = {"embed": self.embed.grad, "norm_f": self.norm_f.grad}
        for l, b in enumerate(self.blocks):
            g |= {f"l{l}_{n}": t.grad for n, t in b.named_parameters()}
        return g

    def forward(self, tokens):
        r = self.embed[tokens]
        for b in self.blocks:
            r = r + b(rmsnorm(r, b.norm, self.cfg["eps"]))
        return rmsnorm(r, self.norm_f, self.cfg["eps"]) @ self.embed.T


def param_count(cfg):
    D, I, N, K, R, V = (cfg[k] for k in ("d_model", "d_inner", "d_state", "d_conv", "dt_rank", "vocab"))
    return V * D + D + cfg["n_layers"] * (3 * I * D + I * (K + 1) + I * (R + 2 * N) + (R + 1) * I + I * N + I + D)


def nll(logits, targets):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
