# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/t5/ (ai://NEW-t5).
"""The T5 encoder-decoder Transformer, the PyTorch version, with the parameters of the NumPy reference.

One embedding for the source, the decoder inputs and the output layer; L pre-norm encoder layers and L pre-norm
decoder layers, the decoder layers with causal self-attention and attention to the encoder output; F.rms_norm with a
gain and no bias as the normalization; no bias in any projection; and a table of position biases, K buckets by H heads,
for each stack, shared by its layers and added to the attention scores through F.scaled_dot_product_attention.
decode() takes the cache of self-attention keys and values, so that generate() feeds one token at a time.
"""

import math
from typing import Any

import torch
import torch.nn.functional as F

EPS = 1e-6
POSITIONS = ("relative", "none")
KEYS = ("vocab", "d_model", "n_heads", "d_kv", "d_ff", "n_layers", "buckets", "max_distance")


def config(
    vocab=13, d_model=8, n_heads=2, d_kv=3, d_ff=16, n_layers=2, buckets=8, max_distance=16, pos="relative"
) -> dict[str, Any]:
    """Return the config of the network; the defaults are the example config of shapes.yaml."""
    if min(vocab, d_model, n_heads, d_kv, d_ff, n_layers) < 1:
        raise ValueError("the sizes must be at least 1")
    if pos not in POSITIONS:
        raise ValueError(f"pos must be one of {POSITIONS}")
    if pos == "relative" and (buckets < 4 or max_distance <= buckets // 2):
        raise ValueError("there must be at least 4 buckets, and max_distance must exceed half of them")
    vals = (vocab, d_model, n_heads, d_kv, d_ff, n_layers, buckets, max_distance)
    return dict(zip(KEYS, (int(v) for v in vals), strict=True)) | {"pos": pos}


def param_shapes(cfg):
    """The name and shape of every parameter, in the order of the NumPy reference."""
    V, M, F_, K, H = (cfg[k] for k in ("vocab", "d_model", "d_ff", "buckets", "n_heads"))
    I = H * cfg["d_kv"]  # noqa: E741
    shapes: dict[str, tuple[int, ...]] = {"emb": (V, M)}
    if cfg["pos"] == "relative":
        shapes |= {"enc_rel": (K, H), "dec_rel": (K, H)}
    for i in range(cfg["n_layers"]):
        shapes |= {f"e{i}_ln1": (M,), f"e{i}_q": (M, I), f"e{i}_k": (M, I), f"e{i}_v": (M, I), f"e{i}_o": (I, M)}
        shapes |= {f"e{i}_ln2": (M,), f"e{i}_wi": (M, F_), f"e{i}_wo": (F_, M)}
    shapes["enc_ln"] = (M,)
    for i in range(cfg["n_layers"]):
        shapes |= {f"d{i}_ln1": (M,), f"d{i}_sq": (M, I), f"d{i}_sk": (M, I), f"d{i}_sv": (M, I), f"d{i}_so": (I, M)}
        shapes |= {f"d{i}_ln2": (M,), f"d{i}_cq": (M, I), f"d{i}_ck": (M, I), f"d{i}_cv": (M, I), f"d{i}_co": (I, M)}
        shapes |= {f"d{i}_ln3": (M,), f"d{i}_wi": (M, F_), f"d{i}_wo": (F_, M)}
    shapes["dec_ln"] = (M,)
    return shapes


def param_count(cfg):
    return sum(math.prod(s) for s in param_shapes(cfg).values())


def bucket(offset, bidirectional, buckets, max_distance):
    """The bucket of each offset key - query, as in the NumPy reference."""
    out = torch.zeros_like(offset)
    if bidirectional:
        buckets //= 2
        out = out + (offset > 0).long() * buckets
        n = offset.abs()
    else:
        n = (-offset).clamp(min=0)
    exact = buckets // 2
    far = exact + torch.log(n.clamp(min=1) / exact) / math.log(max_distance / exact) * (buckets - exact)
    far = far.long().clamp(max=buckets - 1)
    return out + torch.where(n < exact, n, far)


def shift_right(tgt, start_id=0):
    return torch.cat([torch.full_like(tgt[:, :1], start_id), tgt[:, :-1]], 1)


def nll(logits, tgt):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))


class T5(torch.nn.Module):
    """The T5 encoder-decoder, its parameters named as in the NumPy reference."""

    def __init__(self, cfg, scale=0.5, generator=None):
        super().__init__()
        self.cfg = cfg
        self.p = torch.nn.ParameterDict()
        for name, shape in param_shapes(cfg).items():
            if len(shape) == 1:
                w = torch.ones(shape)
            elif name.endswith("_rel"):
                w = scale * torch.randn(shape, generator=generator)
            else:
                fan = shape[1] if name == "emb" else shape[0]
                w = scale / math.sqrt(fan) * torch.randn(shape, generator=generator)
            self.p[name] = torch.nn.Parameter(w)

    @classmethod
    def from_params(cls, params, cfg):
        model = cls(cfg)
        with torch.no_grad():
            for name, value in params.items():
                model.p[name].copy_(torch.as_tensor(value))
        return model

    def lookup(self, ids):
        if ids.min() < 0 or ids.max() >= self.cfg["vocab"]:
            raise ValueError("token id out of range")
        return self.p["emb"][ids]

    def bias(self, table, q_pos, k_pos, bidirectional):
        b = bucket(k_pos[None, :] - q_pos[:, None], bidirectional, self.cfg["buckets"], self.cfg["max_distance"])
        return self.p[table][b].permute(2, 0, 1)

    def norm(self, x, name):
        return F.rms_norm(x, (x.shape[-1],), self.p[name], EPS)

    def heads(self, x):
        B, n, _ = x.shape
        return x.reshape(B, n, self.cfg["n_heads"], self.cfg["d_kv"]).transpose(1, 2)

    def attention(self, w, xq, xkv, bias=None, past=None):
        """Heads of xq on xkv after the cached keys and values past; bias is a float mask (H, nq, nk) that
        scaled_dot_product_attention adds to the scores, -inf where a key is hidden."""
        p = self.p
        q, k, v = self.heads(xq @ p[w[0]]), self.heads(xkv @ p[w[1]]), self.heads(xkv @ p[w[2]])
        if past is not None:
            k, v = torch.cat([past[0], k], 2), torch.cat([past[1], v], 2)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        B, _, n, _ = o.shape
        return o.transpose(1, 2).reshape(B, n, -1) @ p[w[3]], (k, v)

    def encode(self, src):
        """The encoder output (B, S, M) of the source ids."""
        cfg, p = self.cfg, self.p
        z = self.lookup(src)
        pos = torch.arange(src.shape[1])
        bias = self.bias("enc_rel", pos, pos, True) if cfg["pos"] == "relative" else None
        for i in range(cfg["n_layers"]):
            h = self.norm(z, f"e{i}_ln1")
            z = z + self.attention((f"e{i}_q", f"e{i}_k", f"e{i}_v", f"e{i}_o"), h, h, bias)[0]
            z = z + F.relu(self.norm(z, f"e{i}_ln2") @ p[f"e{i}_wi"]) @ p[f"e{i}_wo"]
        return self.norm(z, "enc_ln")

    def decode(self, e, ids, past=None, start=0):
        """The logits (B, n, V) of the decoder inputs ids at positions start, ..., and the self-attention keys and
        values of every layer for all the positions so far."""
        cfg, p = self.cfg, self.p
        n = ids.shape[1]
        y = self.lookup(ids)
        q_pos, k_pos = start + torch.arange(n), torch.arange(start + n)
        hidden = torch.where(k_pos[None, :] <= q_pos[:, None], 0.0, float("-inf"))
        mask = hidden + self.bias("dec_rel", q_pos, k_pos, False) if cfg["pos"] == "relative" else hidden
        kv = []
        for i in range(cfg["n_layers"]):
            h = self.norm(y, f"d{i}_ln1")
            w = (f"d{i}_sq", f"d{i}_sk", f"d{i}_sv", f"d{i}_so")
            a, c = self.attention(w, h, h, mask, None if past is None else past[i])
            kv.append(c)
            y = y + a
            h = self.norm(y, f"d{i}_ln2")
            y = y + self.attention((f"d{i}_cq", f"d{i}_ck", f"d{i}_cv", f"d{i}_co"), h, e)[0]
            y = y + F.relu(self.norm(y, f"d{i}_ln3") @ p[f"d{i}_wi"]) @ p[f"d{i}_wo"]
        return self.norm(y, "dec_ln") @ p["emb"].T, kv

    def forward(self, src, tgt):
        """The logits (B, T, V) of every target token under teacher forcing, and the encoder output."""
        e = self.encode(src)
        return self.decode(e, shift_right(tgt))[0], e

    @torch.no_grad()
    def generate(self, src, n_new, start_id=0):
        """Greedy decoding with the cache: the (B, n_new) tokens after the start token."""
        e = self.encode(src)
        tok = torch.full((src.shape[0], 1), start_id, dtype=torch.long)
        kv, out = None, []
        for t in range(n_new):
            logits, kv = self.decode(e, tok, kv, start=t)
            tok = logits[:, -1].argmax(-1, keepdim=True)
            out.append(tok)
        return torch.cat(out, 1)
