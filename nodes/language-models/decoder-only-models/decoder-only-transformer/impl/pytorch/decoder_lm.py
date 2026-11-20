# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/decoder-only-transformer/ (ai://NEW-decoder-only-transformer).
"""The decoder-only Transformer language model in the form of GPT-2 and GPT-3, the PyTorch version, with the
parameters of the NumPy reference.

Token embeddings plus learned position embeddings, L pre-norm layers of causal multi-head self-attention and an MLP
with an exact GELU, a final layer normalization, and logits through the token embedding transposed (tied=True) or
a separate matrix head_W, without a bias. forward() takes the cache of keys and values of the tokens before, so that
decoding feeds one token at a time; generate() decodes greedily with it. encoder_layer() returns the
torch.nn.TransformerEncoderLayer with the weights of a layer, which run with a causal mask is a decoder layer.
"""

import math

import torch
import torch.nn.functional as F

EPS = 1e-5  # of every layer normalization, the value of GPT-2
POSITIONS = ("learned", "none")


def config(vocab=11, n_ctx=8, d_model=8, n_heads=2, d_ff=16, n_layers=2, tied=True, pos="learned"):
    """Return the config of the network; the defaults are the example config of shapes.yaml."""
    if min(vocab, n_ctx, d_model, n_heads, d_ff, n_layers) < 1:
        raise ValueError("the sizes must be at least 1")
    if vocab < 2:
        raise ValueError("a language model needs at least two tokens")
    if d_model % n_heads:
        raise ValueError("the number of heads must divide d_model")
    if pos not in POSITIONS:
        raise ValueError(f"pos must be one of {POSITIONS}")
    return {
        "vocab": vocab,
        "n_ctx": n_ctx,
        "d_model": d_model,
        "n_heads": n_heads,
        "d_ff": d_ff,
        "n_layers": n_layers,
        "tied": bool(tied),
        "pos": pos,
    }


def param_shapes(cfg):
    """The name and shape of every parameter, in the order of the NumPy reference."""
    V, T, D, F_ = (cfg[k] for k in ("vocab", "n_ctx", "d_model", "d_ff"))
    shapes = {"wte": (V, D)}
    if cfg["pos"] == "learned":
        shapes["wpe"] = (T, D)
    for li in range(cfg["n_layers"]):
        shapes |= {
            f"l{li}_ln1_g": (D,), f"l{li}_ln1_b": (D,),
            f"l{li}_qkv_W": (D, 3 * D), f"l{li}_qkv_b": (3 * D,),
            f"l{li}_out_W": (D, D), f"l{li}_out_b": (D,),
            f"l{li}_ln2_g": (D,), f"l{li}_ln2_b": (D,),
            f"l{li}_fc1_W": (D, F_), f"l{li}_fc1_b": (F_,),
            f"l{li}_fc2_W": (F_, D), f"l{li}_fc2_b": (D,),
        }  # fmt: skip
    shapes |= {"ln_f_g": (D,), "ln_f_b": (D,)}
    if not cfg["tied"]:
        shapes["head_W"] = (D, V)
    return shapes


def param_count(cfg):
    return sum(math.prod(s) for s in param_shapes(cfg).values())


def nll(logits, ids):
    """The cross-entropy of the next token, averaged over the B (n - 1) positions that have one."""
    V = logits.shape[-1]
    return F.cross_entropy(logits[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1))


class DecoderLM(torch.nn.Module):
    """The decoder-only Transformer, its parameters named as in the NumPy reference."""

    def __init__(self, cfg, scale=0.1, generator=None):
        super().__init__()
        self.cfg = cfg
        self.p = torch.nn.ParameterDict()
        for name, shape in param_shapes(cfg).items():
            w = torch.randn(shape, generator=generator) * scale
            self.p[name] = torch.nn.Parameter(w + 1.0 if name.endswith("_g") else w)

    @classmethod
    def from_params(cls, params, cfg):
        model = cls(cfg)
        with torch.no_grad():
            for name, value in params.items():
                model.p[name].copy_(torch.as_tensor(value))
        return model

    def embed(self, ids, start=0):
        cfg, p = self.cfg, self.p
        if start + ids.shape[1] > cfg["n_ctx"] and cfg["pos"] == "learned":
            raise ValueError(f"positions beyond the context of {cfg['n_ctx']} tokens have no embedding")
        z = p["wte"][ids]
        if cfg["pos"] == "learned":
            z = z + p["wpe"][start : start + ids.shape[1]]
        return z

    def attention(self, li, h, past):
        p, H = self.p, self.cfg["n_heads"]
        B, n, D = h.shape
        q, k, v = (h @ p[f"l{li}_qkv_W"] + p[f"l{li}_qkv_b"]).reshape(B, n, 3, H, D // H).permute(2, 0, 3, 1, 4)
        t = 0
        if past is not None:
            t = past[0].shape[2]
            k, v = torch.cat([past[0], k], 2), torch.cat([past[1], v], 2)
        allowed = torch.arange(t + n)[None, :] <= t + torch.arange(n)[:, None]
        s = (q @ k.transpose(-1, -2) / math.sqrt(D // H)).masked_fill(~allowed, float("-inf"))
        o = (torch.softmax(s, -1) @ v).transpose(1, 2).reshape(B, n, D)
        return o @ p[f"l{li}_out_W"] + p[f"l{li}_out_b"], (k, v)

    def layer(self, li, z, past=None):
        p = self.p
        D = z.shape[-1]
        a, kv = self.attention(li, F.layer_norm(z, (D,), p[f"l{li}_ln1_g"], p[f"l{li}_ln1_b"], EPS), past)
        z = z + a
        h = F.layer_norm(z, (D,), p[f"l{li}_ln2_g"], p[f"l{li}_ln2_b"], EPS)
        return z + F.gelu(h @ p[f"l{li}_fc1_W"] + p[f"l{li}_fc1_b"]) @ p[f"l{li}_fc2_W"] + p[f"l{li}_fc2_b"], kv

    def forward(self, ids, past=None):
        """The logits (B, n, V) of the next token at every position, and the keys and values of every layer for
        all the tokens so far."""
        p, cfg = self.p, self.cfg
        z = self.embed(ids, 0 if past is None else past[0][0].shape[2])
        kv = []
        for li in range(cfg["n_layers"]):
            z, c = self.layer(li, z, None if past is None else past[li])
            kv.append(c)
        z = F.layer_norm(z, (z.shape[-1],), p["ln_f_g"], p["ln_f_b"], EPS)
        W = p["wte"].T if cfg["tied"] else p["head_W"]
        return z @ W, kv

    @torch.no_grad()
    def generate(self, prompt, n_new):
        """Greedy decoding with the cache of keys and values: the (B, n_new) new tokens."""
        logits, kv = self(prompt)
        out = []
        for _ in range(n_new):
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            out.append(nxt)
            if len(out) < n_new:
                logits, kv = self(nxt, kv)
        return torch.cat(out, 1)


def encoder_layer(model, li):
    """torch.nn.TransformerEncoderLayer with the weights of layer li: pre-norm, exact GELU, no dropout. Run with a
    causal mask it computes the decoder layer."""
    p, cfg = model.p, model.cfg
    D = cfg["d_model"]
    layer = torch.nn.TransformerEncoderLayer(D, cfg["n_heads"], cfg["d_ff"], dropout=0.0, activation="gelu",
                                             layer_norm_eps=EPS, batch_first=True, norm_first=True)  # fmt: skip
    with torch.no_grad():
        layer.self_attn.in_proj_weight.copy_(p[f"l{li}_qkv_W"].T)
        layer.self_attn.in_proj_bias.copy_(p[f"l{li}_qkv_b"])
        layer.self_attn.out_proj.weight.copy_(p[f"l{li}_out_W"].T)
        layer.self_attn.out_proj.bias.copy_(p[f"l{li}_out_b"])
        layer.linear1.weight.copy_(p[f"l{li}_fc1_W"].T)
        layer.linear1.bias.copy_(p[f"l{li}_fc1_b"])
        layer.linear2.weight.copy_(p[f"l{li}_fc2_W"].T)
        layer.linear2.bias.copy_(p[f"l{li}_fc2_b"])
        layer.norm1.weight.copy_(p[f"l{li}_ln1_g"])
        layer.norm1.bias.copy_(p[f"l{li}_ln1_b"])
        layer.norm2.weight.copy_(p[f"l{li}_ln2_g"])
        layer.norm2.bias.copy_(p[f"l{li}_ln2_b"])
    return layer.eval()
