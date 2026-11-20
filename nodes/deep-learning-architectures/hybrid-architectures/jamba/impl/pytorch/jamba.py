# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/jamba/ (ai://2V1JK1).
"""Jamba of Lieber et al. (2024) in PyTorch: the layers of a Jamba block and the language model, as torch.nn.Modules.

The parameters and their names are those of the NumPy reference (impl/numpy/jamba.py), with linear maps stored as
(in, out) matrices, so a dict of arrays from it or from the fixtures loads with from_params. The selective scan of
the Mamba layers runs as a loop over the positions; autograd gives the backward pass. The kinds argument of JambaLM
replaces the layout of the paper, attention at layer l / 2 of each block, by any list of "attn" and "mamba", which is
how the tests build the all-attention and all-Mamba models of the paper's ablations.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def config(vocab=13, d_model=8, n_blocks=1, block_len=4, moe_every=2, n_experts=4, top_k=2, d_ff=12, n_heads=2,
           n_kv=1, d_state=4, d_conv=3, expand=2, dt_rank=2, aux=0.01, eps=1e-6):  # fmt: skip
    return {"vocab": vocab, "d_model": d_model, "n_blocks": n_blocks, "block_len": block_len, "moe_every": moe_every,
            "n_experts": n_experts, "top_k": top_k, "d_ff": d_ff, "n_heads": n_heads, "n_kv": n_kv,
            "head_dim": d_model // n_heads, "d_state": d_state, "d_conv": d_conv, "expand": expand,
            "d_inner": expand * d_model, "dt_rank": dt_rank, "n_layers": n_blocks * block_len, "aux": aux,
            "eps": eps}  # fmt: skip


def layers(cfg):
    """eq-layout: attention at layer l // 2 of every block, a mixture of experts at every e-th layer from e - 1."""
    l, e = cfg["block_len"], cfg["moe_every"]
    return [("attn" if i % l == l // 2 else "mamba", "moe" if i % e == e - 1 else "mlp") for i in range(cfg["n_layers"])]


def rmsnorm(x, w, eps):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * w


def p(*shape, fan=None):
    return nn.Parameter(torch.randn(*shape) / math.sqrt(fan or shape[-2]))


class Attention(nn.Module):
    """eq-attn: causal self-attention with H query heads in G groups that share keys and values, and no positional
    encoding."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        D, H, G, A = cfg["d_model"], cfg["n_heads"], cfg["n_kv"], cfg["head_dim"]
        self.wq, self.wk, self.wv, self.wo = p(D, H * A), p(D, G * A), p(D, G * A), p(H * A, D)

    def forward(self, u):
        B, T, _ = u.shape
        G, A = self.cfg["n_kv"], self.cfg["head_dim"]
        q = (u @ self.wq).view(B, T, G, -1, A)
        k, v = (u @ self.wk).view(B, T, G, A), (u @ self.wv).view(B, T, G, A)
        s = torch.einsum("btgqd,bsgd->bgqts", q, k) / math.sqrt(A)
        s = s.masked_fill(~torch.ones(T, T, dtype=torch.bool, device=u.device).tril(), float("-inf"))
        o = torch.einsum("bgqts,bsgd->btgqd", s.softmax(-1), v)
        return o.reshape(B, T, -1) @ self.wo


class Mamba(nn.Module):
    """eq-mamba: in_proj, causal depthwise conv and SiLU, x_proj to Delta's low-rank input, B and C, each through an
    RMSNorm (Section 6.4 of the paper), the selective scan with the skip D, the gate SiLU(z) and out_proj."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        D, I, N, K, R = (cfg[k] for k in ("d_model", "d_inner", "d_state", "d_conv", "dt_rank"))
        self.in_proj, self.conv_w, self.conv_b = p(D, 2 * I), p(I, K, fan=K), nn.Parameter(torch.zeros(I))
        self.x_proj = p(I, R + 2 * N)
        self.dt_norm, self.B_norm, self.C_norm = (nn.Parameter(torch.ones(n)) for n in (R, N, N))
        self.dt_w = nn.Parameter(torch.empty(R, I).uniform_(-(R**-0.5), R**-0.5))
        dt = torch.exp(torch.empty(I).uniform_(math.log(1e-3), math.log(1e-1)))
        self.dt_b = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.log(torch.arange(1.0, N + 1).repeat(I, 1)))
        self.D = nn.Parameter(torch.ones(I))
        self.out_proj = p(I, D)

    def forward(self, u):
        I, N, R, K, eps = (self.cfg[k] for k in ("d_inner", "d_state", "dt_rank", "d_conv", "eps"))
        T = u.shape[1]
        x, z = (u @ self.in_proj).split(I, -1)
        xp = F.pad(x, (0, 0, K - 1, 0))  # the causal depthwise convolution as K shifted products, cheaper than conv1d
        xa = F.silu(self.conv_b + sum(self.conv_w[:, k] * xp[:, k : k + T] for k in range(K)))
        dtl, Bm, C = (xa @ self.x_proj).split([R, N, N], -1)
        dt = F.softplus(rmsnorm(dtl, self.dt_norm, eps) @ self.dt_w + self.dt_b)
        Bm, C = rmsnorm(Bm, self.B_norm, eps), rmsnorm(C, self.C_norm, eps)
        dA = torch.exp(dt[..., None] * -torch.exp(self.A_log))
        dBx = dt[..., None] * Bm[:, :, None] * xa[..., None]
        h, ys = u.new_zeros(u.shape[0], I, N), []
        for t in range(T):  # h_t = A_bar_t h_{t-1} + B_bar_t x_t, y_t = C_t h_t
            h = dA[:, t] * h + dBx[:, t]
            ys.append(torch.einsum("bin,bn->bi", h, C[:, t]))
        return ((torch.stack(ys, 1) + self.D * xa) * F.silu(z)) @ self.out_proj


def swiglu(x, wg, wu, wd):
    return (F.silu(x @ wg) * (x @ wu)) @ wd


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        D, Fh = cfg["d_model"], cfg["d_ff"]
        self.w_gate, self.w_up, self.w_down = p(D, Fh), p(D, Fh), p(Fh, D)

    def forward(self, v):
        return swiglu(v, self.w_gate, self.w_up, self.w_down), v.new_zeros(())


class MoE(nn.Module):
    """eq-moe and eq-aux: n SwiGLU experts; each token runs through the K with the largest router probabilities,
    weighted by them, and the layer returns the load-balancing term alpha n sum_j f_j P_j as well."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        D, Fh, n = cfg["d_model"], cfg["d_ff"], cfg["n_experts"]
        self.router = p(D, n)
        self.e_gate, self.e_up, self.e_down = p(n, D, Fh), p(n, D, Fh), p(n, Fh, D)

    def forward(self, v):
        n, K = self.cfg["n_experts"], self.cfg["top_k"]
        x = v.reshape(-1, v.shape[-1])
        probs = (x @ self.router).softmax(-1)
        w, top = probs.topk(K, -1)
        out = torch.zeros_like(x)
        for j in range(n):
            rows, slot = torch.nonzero(top == j, as_tuple=True)
            out = out.index_add(0, rows, w[rows, slot, None] * swiglu(x[rows], self.e_gate[j], self.e_up[j],
                                                                      self.e_down[j]))  # fmt: skip
        f = torch.bincount(top.reshape(-1), minlength=n).to(x.dtype) / top.numel()
        return out.view_as(v), self.cfg["aux"] * n * (f * probs.mean(0)).sum()


class Layer(nn.Module):
    """eq-layer: r <- r + mixer(rmsnorm(r)), then r <- r + ffn(rmsnorm(r))."""

    def __init__(self, cfg, mixer, ffn):
        super().__init__()
        self.eps = cfg["eps"]
        self.norm1, self.norm2 = nn.Parameter(torch.ones(cfg["d_model"])), nn.Parameter(torch.ones(cfg["d_model"]))
        self.mixer = Attention(cfg) if mixer == "attn" else Mamba(cfg)
        self.ffn = MoE(cfg) if ffn == "moe" else MLP(cfg)

    def forward(self, r):
        r = r + self.mixer(rmsnorm(r, self.norm1, self.eps))
        out, aux = self.ffn(rmsnorm(r, self.norm2, self.eps))
        return r + out, aux


class JambaLM(nn.Module):
    """The embedding, the layers, a final RMSNorm and an untied output layer. forward returns the scores and the
    sum of the load-balancing terms."""

    def __init__(self, cfg, kinds=None):
        super().__init__()
        self.cfg = cfg
        lay = layers(cfg) if kinds is None else [(m, f) for m, (_, f) in zip(kinds, layers(cfg), strict=True)]
        self.embed = nn.Parameter(torch.randn(cfg["vocab"], cfg["d_model"]))
        self.layers = nn.ModuleList(Layer(cfg, m, f) for m, f in lay)
        self.norm_f, self.head = nn.Parameter(torch.ones(cfg["d_model"])), p(cfg["d_model"], cfg["vocab"])

    def named_like_numpy(self):
        """The parameters under the names of the NumPy reference: l{i}_norm1, l{i}_wq, l{i}_e_gate, ..., embed."""
        out = {"embed": self.embed, "norm_f": self.norm_f, "head": self.head}
        for i, l in enumerate(self.layers):
            out |= {f"l{i}_{n.split('.')[-1]}": t for n, t in l.named_parameters()}
        return out

    @classmethod
    def from_params(cls, cfg, params):
        m = cls(cfg).double()
        with torch.no_grad():
            for k, t in m.named_like_numpy().items():
                t.copy_(torch.as_tensor(params[k]))
        return m

    def forward(self, tokens):
        r, aux = self.embed[tokens], 0.0
        for l in self.layers:
            r, a = l(r)
            aux = aux + a
        return rmsnorm(r, self.norm_f, self.cfg["eps"]) @ self.head, aux


def lm_loss(model, tokens):
    """eq-loss: the cross-entropy of tokens[:, 1:] given tokens[:, :-1], plus the load-balancing terms."""
    logits, aux = model(tokens[:, :-1])
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1)) + aux
