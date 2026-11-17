# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/linear-attention/ (ai://NEW-linear-attention).
"""Linear attention in PyTorch: the layer, the constant-memory causal product of Algorithm 1 as an autograd
Function, the layer as an RNN, and a small causal transformer to compare softmax and linear attention on a copy task.

The weights have the names and the layout of the NumPy reference (rows are vectors, heads side by side, no biases),
so the shared fixture loads as it is. CausalProduct computes the numerators phi(Q_i)^T sum_{j <= i} phi(K_j) V_j^T
with a loop over the positions and a running (C, M) sum, and its backward is the two loops of Algorithm 1; it stores
only its inputs. causal="cumsum" materializes every S_i instead, which is what plain autograd would keep for the
backward pass, N C M numbers per head, and causal="quadratic" builds the masked N x N matrix of similarities, the
same numbers again; TinyLM trains with it, its sequences being short.
"""

import math

import torch
from torch import nn


def elu_plus_one(x):
    return torch.nn.functional.elu(x) + 1


class CausalProduct(torch.autograd.Function):
    """num_i = phi(Q_i)^T S_i with S_i = sum_{j <= i} phi(K_j) V_j^T, for (..., N, C) and (..., N, M) inputs."""

    @staticmethod
    def forward(ctx, fQ, fK, V):
        ctx.save_for_backward(fQ, fK, V)
        S = fK.new_zeros(fK.shape[:-2] + (fK.shape[-1], V.shape[-1]))
        out = []
        for i in range(fQ.shape[-2]):
            S = S + fK[..., i, :, None] * V[..., i, None, :]
            out.append(torch.einsum("...c,...cm->...m", fQ[..., i, :], S))
        return torch.stack(out, -2)

    @staticmethod
    def backward(ctx, G):
        fQ, fK, V = ctx.saved_tensors
        N = fQ.shape[-2]
        dQ, dK, dV = torch.empty_like(fQ), torch.empty_like(fK), torch.empty_like(V)
        S = fK.new_zeros(fK.shape[:-2] + (fK.shape[-1], V.shape[-1]))
        for i in range(N):
            S = S + fK[..., i, :, None] * V[..., i, None, :]
            dQ[..., i, :] = torch.einsum("...m,...cm->...c", G[..., i, :], S)
        S = torch.zeros_like(S)
        for i in reversed(range(N)):
            S = S + fQ[..., i, :, None] * G[..., i, None, :]
            dV[..., i, :] = torch.einsum("...cm,...c->...m", S, fK[..., i, :])
            dK[..., i, :] = torch.einsum("...cm,...m->...c", S, V[..., i, :])
        return dQ, dK, dV


def causal_product_cumsum(fQ, fK, V):
    """The same numerators by materializing S_1..S_N, (..., N, C, M)."""
    S = torch.cumsum(fK[..., :, None] * V[..., None, :], -3)
    return torch.einsum("...nc,...ncm->...nm", fQ, S)


def linear_attention(fQ, fK, V, causal=False, impl="algorithm1"):
    if not causal:
        return (fQ @ (fK.transpose(-1, -2) @ V)) / (fQ @ fK.sum(-2)[..., None])
    if impl == "quadratic":  # the parallel form tril(phi(Q) phi(K)^T) V: N^2 work, fastest for short sequences
        A = (fQ @ fK.transpose(-1, -2)).tril()
        return (A @ V) / A.sum(-1, keepdim=True)
    num = CausalProduct.apply(fQ, fK, V) if impl == "algorithm1" else causal_product_cumsum(fQ, fK, V)
    return num / (fQ * torch.cumsum(fK, -2)).sum(-1, keepdim=True)


def softmax_attention(Q, K, V, causal=False):
    s = Q @ K.transpose(-1, -2) / math.sqrt(Q.shape[-1])
    if causal:
        N = s.shape[-1]
        s = s.masked_fill(~torch.ones(N, N, dtype=torch.bool, device=s.device).tril(), float("-inf"))
    return torch.softmax(s, -1) @ V


class LinearAttention(nn.Module):
    """h heads of attention between the projections W_q, W_k, W_v and W_o; kind is "linear" (elu + 1) or
    "softmax", so the same module serves both sides of the comparison."""

    def __init__(self, d_model=8, n_heads=2, causal=True, kind="linear", impl="algorithm1"):
        super().__init__()
        self.h, self.causal, self.kind, self.impl = n_heads, causal, kind, impl
        d = d_model // n_heads
        for name, shape in (("W_q", (d_model, n_heads * d)), ("W_k", (d_model, n_heads * d)),
                            ("W_v", (d_model, n_heads * d)), ("W_o", (n_heads * d, d_model))):  # fmt: skip
            self.register_parameter(name, nn.Parameter(torch.randn(*shape) / math.sqrt(shape[0])))

    @classmethod
    def from_params(cls, params, **kw):
        F, HD = params["W_q"].shape
        m = cls(F, kw.pop("n_heads", 2), **kw).double()
        with torch.no_grad():
            for k, v in params.items():
                getattr(m, k).copy_(torch.as_tensor(v))
        return m

    def split(self, x):
        B, N, _ = x.shape
        return x.view(B, N, self.h, -1).transpose(1, 2)

    def forward(self, x):
        q, k, v = (self.split(x @ w) for w in (self.W_q, self.W_k, self.W_v))
        if self.kind == "softmax":
            a = softmax_attention(q, k, v, self.causal)
        else:
            a = linear_attention(elu_plus_one(q), elu_plus_one(k), v, self.causal, self.impl)
        return a.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1) @ self.W_o

    def initial_state(self, B):
        D = self.W_q.shape[1] // self.h
        z = self.W_q.new_zeros(B, self.h, D)
        return self.W_q.new_zeros(B, self.h, D, self.W_v.shape[1] // self.h), z

    def step(self, state, x_i):
        """eq-rnn: one token (B, F) in, one output out, and the state (s, z) updated in place of a KV cache."""
        s, z = state
        B = x_i.shape[0]
        q, k, v = ((x_i @ w).view(B, self.h, -1) for w in (self.W_q, self.W_k, self.W_v))
        fq, fk = elu_plus_one(q), elu_plus_one(k)
        s, z = s + fk[..., :, None] * v[..., None, :], z + fk
        a = torch.einsum("bhc,bhcm->bhm", fq, s) / (fq * z).sum(-1, keepdim=True)
        return a.reshape(B, -1) @ self.W_o, (s, z)


class TinyLM(nn.Module):
    """A pre-norm causal transformer: token and learned position embeddings, n_layers blocks of attention and a
    4x feedforward, then a linear read-out. kind selects softmax or linear attention in every block."""

    def __init__(self, vocab, d_model=32, n_heads=4, n_layers=2, max_len=64, kind="linear"):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, d_model), nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleDict({
                "ln1": nn.LayerNorm(d_model), "att": LinearAttention(d_model, n_heads, True, kind, "quadratic"),
                "ln2": nn.LayerNorm(d_model),
                "ff": nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)),
            }))  # fmt: skip
        self.ln, self.out = nn.LayerNorm(d_model), nn.Linear(d_model, vocab)

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1], device=x.device))
        for b in self.blocks:
            h = h + b["att"](b["ln1"](h))
            h = h + b["ff"](b["ln2"](h))
        return self.out(self.ln(h))
