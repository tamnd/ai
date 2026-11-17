# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/multi-head-attention/ (ai://NEW-multi-head-attention).
"""Multi-head attention as a torch.nn.Module, the PyTorch reference.

It computes eq-scores, eq-softmax, eq-attention, eq-heads and eq-multihead of the page with torch operations,
as the NumPy reference does, and leaves the gradient to autograd. The parameters have the names and shapes of
the NumPy reference, so from_params loads the shared fixture. A mask is boolean and True means "may attend",
as in the NumPy reference (torch.nn.MultiheadAttention uses the opposite convention).
"""

import math

import torch


def config(d_model=512, n_heads=8, d_k=None, d_v=None):
    """Return the configuration. By default d_k = d_v = d_model / n_heads."""
    if (d_k is None or d_v is None) and d_model % n_heads != 0:
        raise ValueError(f"d_model = {d_model} is not a multiple of n_heads = {n_heads}")
    d_k = d_model // n_heads if d_k is None else int(d_k)
    d_v = d_model // n_heads if d_v is None else int(d_v)
    if min(d_model, n_heads, d_k, d_v) < 1:
        raise ValueError("every size must be at least 1")
    return {"d_model": int(d_model), "n_heads": int(n_heads), "d_k": d_k, "d_v": d_v}


def causal_mask(T, S=None):
    """Query t may attend to key s when s <= t + (S - T)."""
    S = T if S is None else S
    return torch.ones(T, S, dtype=torch.bool).tril(diagonal=S - T)  # (T, S)


def padding_mask(lengths, S):
    """Key s of sequence b is real when s < lengths[b]."""
    lengths = torch.as_tensor(lengths)
    return (torch.arange(S)[None, :] < lengths[:, None])[:, None, None, :]  # (B, 1, 1, S)


def attention(q, k, v, mask=None):
    """Scaled dot-product attention; a row with every key masked gives a zero output."""
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])  # eq: scores  (B, H, T, S)
    if mask is None:
        a = torch.softmax(scores, dim=-1)  # eq: softmax  (torch.softmax subtracts the maximum)
        return a @ v, a  # eq: attention
    mask = torch.as_tensor(mask, device=scores.device).expand(scores.shape)
    scores = scores.masked_fill(~mask, -math.inf)
    top = scores.amax(dim=-1, keepdim=True).nan_to_num(0.0, neginf=0.0).detach()
    e = torch.exp(scores - top).masked_fill(~mask, 0.0)
    z = e.sum(dim=-1, keepdim=True)
    a = e / torch.where(z > 0, z, torch.ones_like(z))  # (B, H, T, S)
    return a @ v, a  # (B, H, T, Dv)


class MultiHeadAttention(torch.nn.Module):
    """y = Concat(head_1, ..., head_h) W_o with head_r = Attention(x W_q^r, m W_k^r, m W_v^r)."""

    def __init__(self, cfg, generator=None):
        super().__init__()
        self.cfg = cfg
        M, H, Dk, Dv = cfg["d_model"], cfg["n_heads"], cfg["d_k"], cfg["d_v"]

        def weight(n_in, n_out):
            return torch.nn.Parameter(torch.randn(n_in, n_out, generator=generator) * n_in**-0.5)

        self.W_q, self.W_k = weight(M, H * Dk), weight(M, H * Dk)  # (M, H*Dk)
        self.W_v, self.W_o = weight(M, H * Dv), weight(H * Dv, M)  # (M, H*Dv), (H*Dv, M)

    @classmethod
    def from_params(cls, params, cfg):
        """Build the module from the NumPy parameter dict (W_q, W_k, W_v, W_o)."""
        m = cls(cfg)
        with torch.no_grad():
            for name in ("W_q", "W_k", "W_v", "W_o"):
                getattr(m, name).copy_(torch.as_tensor(params[name]))
        return m

    def _split(self, x):
        B, T, HD = x.shape
        H = self.cfg["n_heads"]
        return x.reshape(B, T, H, HD // H).transpose(1, 2)  # (B, H, T, D)

    def forward(self, x, memory=None, mask=None):
        """Return (y, a): y is (B, T, M) and a holds the weights, (B, H, T, S). memory=None is self-attention."""
        src = x if memory is None else memory
        q = self._split(x @ self.W_q)  # eq: heads  (B, H, T, Dk)
        k = self._split(src @ self.W_k)  # (B, H, S, Dk)
        v = self._split(src @ self.W_v)  # (B, H, S, Dv)
        o, a = attention(q, k, v, mask)  # (B, H, T, Dv)
        B, H, T, Dv = o.shape
        concat = o.transpose(1, 2).reshape(B, T, H * Dv)  # (B, T, H*Dv)
        return concat @ self.W_o, a  # eq: multihead  (B, T, M)


def param_count(cfg):
    """2 M H d_k + 2 M H d_v weights."""
    M, H = cfg["d_model"], cfg["n_heads"]
    return 2 * M * H * cfg["d_k"] + 2 * M * H * cfg["d_v"]
