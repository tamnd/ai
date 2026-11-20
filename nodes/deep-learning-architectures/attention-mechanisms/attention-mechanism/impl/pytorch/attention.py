# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/attention-mechanism/ (ai://B95BYJ).
"""RNNsearch of Bahdanau, Cho and Bengio (2015) and the attention layer of Luong, Pham and Manning (2015) in PyTorch.

The parameters, their names and shapes are those of the NumPy reference (impl/numpy/attention.py): rows are vectors,
a linear map is x @ W, the three gate matrices of a GRU sit side by side in the order (z, r, candidate) and there are
no biases. A dict of arrays from the reference or the fixtures loads with RNNsearch.from_params; autograd gives the
backward pass. attention=False is the paper's RNNencdec, with c_i fixed to the two final encoder states.
"""

import math

import torch
from torch import nn


def config(src_vocab=7, tgt_vocab=6, embed=3, hidden=4, align=5, maxout=3, attention=True):
    return {"src_vocab": src_vocab, "tgt_vocab": tgt_vocab, "embed": embed, "hidden": hidden, "align": align,
            "maxout": maxout, "attention": attention}  # fmt: skip


def shapes(cfg):
    Kx, Ky, m, n, a, l = (cfg[k] for k in ("src_vocab", "tgt_vocab", "embed", "hidden", "align", "maxout"))
    return {
        "E_x": (Kx, m), "fwd_W": (m, 3 * n), "fwd_U": (n, 3 * n), "bwd_W": (m, 3 * n), "bwd_U": (n, 3 * n),
        "E_y": (Ky, m), "dec_W": (m, 3 * n), "dec_U": (n, 3 * n), "dec_C": (2 * n, 3 * n), "W_s": (n, n),
        "W_a": (n, a), "U_a": (2 * n, a), "v_a": (a,),
        "U_o": (n, 2 * l), "V_o": (m, 2 * l), "C_o": (2 * n, 2 * l), "W_o": (l, Ky),
    }  # fmt: skip


def gru_step(x, h, W, U, c=None, C=None):
    n = h.shape[-1]
    a = x @ W if c is None else x @ W + c @ C
    zr = torch.sigmoid(a[:, : 2 * n] + h @ U[:, : 2 * n])
    z, r = zr[:, :n], zr[:, n:]
    return (1 - z) * h + z * torch.tanh(a[:, 2 * n :] + (r * h) @ U[:, 2 * n :])


class RNNsearch(nn.Module):
    # The parameters are registered by name in __init__; the annotations tell the type checker they are tensors.
    E_x: torch.Tensor
    fwd_W: torch.Tensor
    fwd_U: torch.Tensor
    bwd_W: torch.Tensor
    bwd_U: torch.Tensor
    E_y: torch.Tensor
    dec_W: torch.Tensor
    dec_U: torch.Tensor
    dec_C: torch.Tensor
    W_s: torch.Tensor
    W_a: torch.Tensor
    U_a: torch.Tensor
    v_a: torch.Tensor
    U_o: torch.Tensor
    V_o: torch.Tensor
    C_o: torch.Tensor
    W_o: torch.Tensor

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        for k, s in shapes(cfg).items():
            fan = s[0] if len(s) == 2 and not k.startswith("E_") else 1
            self.register_parameter(k, nn.Parameter(torch.randn(s) / math.sqrt(fan)))

    @classmethod
    def from_params(cls, cfg, params):
        m = cls(cfg).double()
        with torch.no_grad():
            for k, t in m.named_parameters():
                t.copy_(torch.as_tensor(params[k]))
        return m

    def encode(self, src, mask):
        """eq-annot: the annotations h_j = [fwd h_j; bwd h_j], (B, S, 2n). A padded position keeps the state."""
        B, S = src.shape
        ex, m = self.E_x[src], mask[..., None].to(self.E_x.dtype)
        h, hf = ex.new_zeros(B, self.cfg["hidden"]), []
        for t in range(S):
            h = m[:, t] * gru_step(ex[:, t], h, self.fwd_W, self.fwd_U) + (1 - m[:, t]) * h
            hf.append(h)
        h, hb = ex.new_zeros(B, self.cfg["hidden"]), [None] * S
        for t in reversed(range(S)):
            h = m[:, t] * gru_step(ex[:, t], h, self.bwd_W, self.bwd_U) + (1 - m[:, t]) * h
            hb[t] = h
        return torch.cat([torch.stack(hf, 1), torch.stack(hb, 1)], -1)

    def align(self, s, H, HU, mask):
        """eq-score, eq-weights and eq-context."""
        e = torch.tanh((s @ self.W_a)[:, None] + HU) @ self.v_a
        alpha = torch.softmax(e.masked_fill(~mask, float("-inf")), -1)
        return torch.einsum("bs,bsk->bk", alpha, H), alpha

    def start(self, src, mask):
        n = self.cfg["hidden"]
        H = self.encode(src, mask)
        fixed = torch.cat([H[:, -1, :n], H[:, 0, n:]], -1)
        return H, H @ self.U_a, torch.tanh(H[:, 0, n:] @ self.W_s), fixed

    def step(self, s, e, H, HU, mask, fixed):
        """One decoder step from s_{i-1} and the embedding e of y_{i-1}: the logits of y_i, s_i and alpha_i."""
        if self.cfg["attention"]:
            c, alpha = self.align(s, H, HU, mask)
        else:
            c, alpha = fixed, torch.zeros(mask.shape, dtype=s.dtype)
        tt = s @ self.U_o + e @ self.V_o + c @ self.C_o
        logits = torch.maximum(tt[:, 0::2], tt[:, 1::2]) @ self.W_o  # eq-output, maxout over pairs
        return logits, gru_step(e, s, self.dec_W, self.dec_U, c, self.dec_C), alpha

    def forward(self, src, tgt, src_mask=None):
        """Teacher-forced logits (B, T, K_y) and alignments (B, T, S); y_0 has an all-zero embedding."""
        src_mask = torch.ones(src.shape, dtype=torch.bool) if src_mask is None else src_mask
        H, HU, s, fixed = self.start(src, src_mask)
        ey = torch.cat([H.new_zeros(tgt.shape[0], 1, self.cfg["embed"]), self.E_y[tgt[:, :-1]]], 1)
        logits, alphas = [], []
        for i in range(tgt.shape[1]):
            lg, s, alpha = self.step(s, ey[:, i], H, HU, src_mask, fixed)
            logits.append(lg)
            alphas.append(alpha)
        return torch.stack(logits, 1), torch.stack(alphas, 1)

    @torch.no_grad()
    def greedy(self, src, max_len):
        """Greedy decoding of a batch for max_len steps (no stopping); the tokens (B, max_len) and alphas."""
        mask = torch.ones(src.shape, dtype=torch.bool)
        H, HU, s, fixed = self.start(src, mask)
        e, ys, alphas = H.new_zeros(src.shape[0], self.cfg["embed"]), [], []
        for _ in range(max_len):
            lg, s, alpha = self.step(s, e, H, HU, mask, fixed)
            y = lg.argmax(-1)
            e = self.E_y[y]
            ys.append(y)
            alphas.append(alpha)
        return torch.stack(ys, 1), torch.stack(alphas, 1)


def nll(logits, tgt, tgt_mask=None):
    lp = torch.log_softmax(logits, -1).gather(-1, tgt[..., None])[..., 0]
    return -(lp if tgt_mask is None else lp[tgt_mask]).mean()


class LuongAttention(nn.Module):
    """Luong et al.'s Section 3: score in {dot, general, concat, location}, global or local-p attention, and the
    attentional state h~_t = tanh([c_t; h_t] W_c). Parameters follow the NumPy reference: W_a is (n, n) for general,
    (2n, n_a) for concat with the vector v_a, (n, S_max) for location; local-p adds W_p (n, n) and v_p (n,)."""

    def __init__(self, n, score="general", local_D=None, n_a=None, max_len=None):
        super().__init__()
        self.score, self.D = score, local_D
        r = lambda *s: nn.Parameter(torch.randn(*s) / math.sqrt(s[0]))  # noqa: E731
        if score == "general":
            self.W_a = r(n, n)
        elif score == "concat":
            self.W_a, self.v_a = r(2 * n, n_a or n), r(n_a or n)
        elif score == "location":
            self.W_a = r(n, max_len)
        if local_D is not None:
            self.W_p, self.v_p = r(n, n), r(n)
        self.W_c = r(2 * n, n)

    def scores(self, h, Hs):
        if self.score == "dot":
            return torch.einsum("bk,bsk->bs", h, Hs)
        if self.score == "general":
            return torch.einsum("bk,bsk->bs", h @ self.W_a, Hs)
        if self.score == "concat":
            n = h.shape[-1]
            return torch.tanh((h @ self.W_a[:n])[:, None] + Hs @ self.W_a[n:]) @ self.v_a
        return (h @ self.W_a)[:, : Hs.shape[1]]

    def forward(self, h, Hs, lengths):
        """Return h~_t, the weights a_t (B, S) and p_t (None for global attention)."""
        pos = torch.arange(Hs.shape[1], dtype=h.dtype)
        real = pos < lengths[:, None]
        p_t = None
        if self.D is None:
            a = torch.softmax(self.scores(h, Hs).masked_fill(~real, float("-inf")), -1)
        else:
            p_t = lengths.to(h.dtype) * torch.sigmoid(torch.tanh(h @ self.W_p) @ self.v_p)
            keep = real & ((pos - p_t[:, None]).abs() <= self.D)
            a = torch.softmax(self.scores(h, Hs).masked_fill(~keep, float("-inf")), -1)
            a = a * torch.exp(-((pos - p_t[:, None]) ** 2) / (2 * (self.D / 2) ** 2))
        c = torch.einsum("bs,bsk->bk", a, Hs)
        return torch.tanh(torch.cat([c, h], -1) @ self.W_c), a, p_t
