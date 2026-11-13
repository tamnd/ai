# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/transformer/ (ai://4W3NTQ).
"""The original encoder-decoder Transformer as a torch.nn.Module, the PyTorch reference.

It computes eq-pe, eq-attention, eq-multihead and eq-ffn of the page with torch operations, as the NumPy
reference does, and leaves the gradient to autograd. The names of the parameters are the paths of the NumPy
parameter tree, so from_params loads the shared fixture. Dropout is left out, as in the NumPy reference.
"""

import math

import torch

EPS = 1e-6


def config(d_model=512, n_heads=8, d_ff=2048, n_layers=6, vocab=37000):
    """Return the config of the model. The width of each head is d_model / n_heads."""
    if min(d_model, n_heads, d_ff, n_layers, vocab) < 1 or d_model % n_heads:
        raise ValueError("the sizes must be at least 1, and n_heads must divide d_model")
    return {"d_model": d_model, "n_heads": n_heads, "d_head": d_model // n_heads, "d_ff": d_ff,
            "n_layers": n_layers, "vocab": vocab}


def _param(*shape):
    return torch.nn.Parameter(torch.zeros(*shape))


def positional_encoding(T, M, dtype=None):
    """The sinusoids of eq-pe: sin on the even features, cos on the odd ones."""
    # eq: pe
    pos = torch.arange(T, dtype=dtype)[:, None]  # (T, 1)
    j = torch.arange((M + 1) // 2, dtype=dtype)[None, :]
    angle = pos / 10000.0 ** (2 * j / M)  # (T, ceil(M / 2))
    pe = torch.zeros(T, M, dtype=dtype)  # (T, M)
    pe[:, 0::2] = torch.sin(angle)
    pe[:, 1::2] = torch.cos(angle[:, : M // 2])
    return pe


def attention(q, k, v, allowed=None):
    """Scaled dot-product attention, eq-attention. A query that may attend to no key gets a zero output."""
    # eq: attention
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])  # (B, H, T, S)
    if allowed is None:
        weights = torch.softmax(scores, dim=-1)  # torch.softmax subtracts the maximum
        return weights @ v, weights
    allowed = allowed.expand(scores.shape)
    scores = scores.masked_fill(~allowed, -math.inf)
    top = scores.amax(dim=-1, keepdim=True).nan_to_num(0.0, neginf=0.0).detach()
    e = torch.exp(scores - top).masked_fill(~allowed, 0.0)  # subtract the maximum, as 8.3.1 asks
    total = e.sum(dim=-1, keepdim=True)
    weights = e / torch.where(total > 0, total, torch.ones_like(total))  # (B, H, T, S)
    return weights @ v, weights  # (B, H, T, D), (B, H, T, S)


class MultiHead(torch.nn.Module):
    """Multi-head attention, eq-multihead. Queries come from x, keys and values from memory."""

    def __init__(self, M, n_heads):
        super().__init__()
        self.n_heads = n_heads
        for name in ("q", "k", "v", "o"):
            setattr(self, "W_" + name, _param(M, M))
            setattr(self, "b_" + name, _param(M))

    def forward(self, x, memory, allowed=None):
        # eq: multihead
        B, T, M = x.shape
        S = memory.shape[1]
        H = self.n_heads

        def heads(y, n):
            return y.reshape(B, n, H, M // H).transpose(1, 2)  # (B, H, n, D)

        q = heads(x @ self.W_q + self.b_q, T)  # (B, H, T, D)
        k = heads(memory @ self.W_k + self.b_k, S)  # (B, H, S, D)
        v = heads(memory @ self.W_v + self.b_v, S)  # (B, H, S, D)
        out, weights = attention(q, k, v, allowed)
        y = out.transpose(1, 2).reshape(B, T, M) @ self.W_o + self.b_o  # (B, T, M)
        return y, {"q": q, "k": k, "v": v, "weights": weights}


class FFN(torch.nn.Module):
    """The position-wise feedforward network, eq-ffn."""

    def __init__(self, M, F):
        super().__init__()
        self.W_1, self.b_1, self.W_2, self.b_2 = _param(M, F), _param(F), _param(F, M), _param(M)

    def forward(self, x):
        # eq: ffn
        hidden = torch.relu(x @ self.W_1 + self.b_1)  # (B, T, F)
        return hidden @ self.W_2 + self.b_2, hidden  # (B, T, M), (B, T, F)


class Norm(torch.nn.Module):
    """Layer normalization over the features, with a scale and a shift."""

    def __init__(self, M):
        super().__init__()
        self.gamma, self.beta = torch.nn.Parameter(torch.ones(M)), _param(M)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # (B, T, 1)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)  # (B, T, 1)
        return self.gamma * (x - mean) / torch.sqrt(var + EPS) + self.beta


class EncoderLayer(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        M = cfg["d_model"]
        self.attn, self.ffn = MultiHead(M, cfg["n_heads"]), FFN(M, cfg["d_ff"])
        self.norm1, self.norm2 = Norm(M), Norm(M)

    def forward(self, x, src_allowed=None):
        a, cache = self.attn(x, x, src_allowed)
        x = self.norm1(x + a)  # post-norm: LayerNorm(x + Sublayer(x))
        f, cache["hidden"] = self.ffn(x)
        return self.norm2(x + f), cache  # (B, S, M)


class DecoderLayer(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        M = cfg["d_model"]
        self.attn, self.cross = MultiHead(M, cfg["n_heads"]), MultiHead(M, cfg["n_heads"])
        self.ffn = FFN(M, cfg["d_ff"])
        self.norm1, self.norm2, self.norm3 = Norm(M), Norm(M), Norm(M)

    def forward(self, y, memory, src_allowed=None):
        T = y.shape[1]
        causal = torch.ones(T, T, dtype=torch.bool, device=y.device).tril()  # (T, T): t sees s <= t
        a, self_cache = self.attn(y, y, causal)
        y = self.norm1(y + a)
        c, cross_cache = self.cross(y, memory, src_allowed)
        y = self.norm2(y + c)
        f, hidden = self.ffn(y)
        return self.norm3(y + f), {"self": self_cache, "cross": cross_cache, "hidden": hidden}


class Transformer(torch.nn.Module):
    """The encoder, the decoder and the output layer, with one embedding matrix for the three."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = torch.nn.Parameter(torch.randn(cfg["vocab"], cfg["d_model"]) * cfg["d_model"] ** -0.5)
        self.enc = torch.nn.ModuleList(EncoderLayer(cfg) for _ in range(cfg["n_layers"]))
        self.dec = torch.nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg["n_layers"]))
        with torch.no_grad():
            for name, p in self.named_parameters():
                if name.split(".")[-1] in ("W_q", "W_k", "W_v", "W_o", "W_1", "W_2"):
                    p.normal_(0.0, p.shape[0] ** -0.5)  # variance 1 / n_in, as in the NumPy init

    @classmethod
    def from_params(cls, params, cfg):
        """Build the model from a flat parameter tree, its paths joined by "/", as in fixtures/params-small.npz."""
        model = cls(cfg)
        state = {k.replace("/", "."): torch.as_tensor(v) for k, v in params.items()}
        model.load_state_dict(state, strict=True)
        return model

    def _embed(self, tokens):
        M = self.embed.shape[1]
        return self.embed[tokens] * math.sqrt(M) + positional_encoding(tokens.shape[1], M, self.embed.dtype)

    def forward(self, src, tgt, src_pad=None):
        """Return (logits, cache) for source tokens src (B, S) and target tokens tgt (B, T)."""
        src_allowed = None if src_pad is None else ~src_pad[:, None, None, :]  # (B, 1, 1, S)
        x = self._embed(src)  # (B, S, M)
        enc_cache = []
        for layer in self.enc:
            x, c = layer(x, src_allowed)
            enc_cache.append(c)
        y = self._embed(tgt)  # (B, T, M)
        dec_cache = []
        for layer in self.dec:
            y, c = layer(y, x, src_allowed)
            dec_cache.append(c)
        logits = y @ self.embed.T  # (B, T, V): the output layer shares the embedding
        return logits, {"memory": x, "enc": enc_cache, "dec": dec_cache}


def param_count(cfg):
    """The number of parameters, from the sizes alone."""
    M, F = cfg["d_model"], cfg["d_ff"]
    attn, ffn, norm = 4 * (M * M + M), 2 * M * F + F + M, 2 * M
    return cfg["vocab"] * M + cfg["n_layers"] * ((attn + ffn + 2 * norm) + (2 * attn + ffn + 3 * norm))
