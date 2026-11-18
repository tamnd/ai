# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/end-to-end-memory-network/ (ai://VDRYRT).
"""An end-to-end memory network for question answering in PyTorch, the same model as the NumPy reference.

The story is a (B, N, J) tensor of word ids, the most recent sentence first, padded with the null word 0; the question
is (B, J). Adjacent tying keeps K + 1 embedding matrices E_0..E_K and K + 1 temporal matrices; layer-wise tying keeps
A, C, B, W, H, T_A and T_C. The gradient of the null rows is not masked here: call zero_null after each update, as
the constraint of Section 4.2 asks, or the padding starts to carry a learned vector.
"""

import torch

NULL = 0


def config(vocab=12, dim=6, hops=3, mem=5, sent_len=4, tying="adjacent", encoding="pe", temporal=True):
    if min(vocab, dim, hops, mem, sent_len) < 1 or vocab < 2:
        raise ValueError("the sizes must be positive and the vocabulary must hold the null word and one more")
    if tying not in ("adjacent", "layerwise") or encoding not in ("bow", "pe"):
        raise ValueError("tying is adjacent or layerwise, encoding is bow or pe")
    return {"vocab": vocab, "dim": dim, "hops": hops, "mem": mem, "sent_len": sent_len, "tying": tying,
            "encoding": encoding, "temporal": bool(temporal)}


def param_count(cfg):
    v, d, k, n = cfg["vocab"], cfg["dim"], cfg["hops"], cfg["mem"]
    t = n * d if cfg["temporal"] else 0
    if cfg["tying"] == "adjacent":
        return (k + 1) * (v * d + t)
    return 4 * v * d + d * d + 2 * t


def embedding_names(cfg):
    if cfg["tying"] == "adjacent":
        return [f"E{i}" for i in range(cfg["hops"] + 1)]
    return ["A", "C", "B", "W"]


def init(gen, cfg, sigma=0.1, dtype=torch.float64):
    v, d, k, n = cfg["vocab"], cfg["dim"], cfg["hops"], cfg["mem"]

    def normal(*shape):
        return sigma * torch.randn(*shape, generator=gen, dtype=dtype)

    p = {name: normal(v, d) for name in embedding_names(cfg)}
    if cfg["temporal"]:
        tnames = [f"T{i}" for i in range(k + 1)] if cfg["tying"] == "adjacent" else ["TA", "TC"]
        p.update({name: normal(n, d) for name in tnames})
    if cfg["tying"] == "layerwise":
        p["H"] = normal(d, d)
    zero_null(p, cfg)
    return p


@torch.no_grad()
def zero_null(params, cfg):
    """Set the null row of every embedding to zero in place."""
    for name in embedding_names(cfg):
        params[name][NULL] = 0.0


def position_encoding(sent_len, dim, dtype=torch.float64):
    j = torch.arange(1, sent_len + 1, dtype=dtype)[:, None] / sent_len
    k = torch.arange(1, dim + 1, dtype=dtype)[None, :] / dim
    return (1 - j) - k * (1 - 2 * j)


def encode(E, words, weights=None):
    vec = E[words]
    if weights is None:
        return vec.sum(dim=-2)
    return torch.einsum("...jd,jd->...d", vec, weights)


def plan(cfg):
    k, t = cfg["hops"], cfg["temporal"]
    if cfg["tying"] == "adjacent":
        pair = [(f"E{i}", f"T{i}" if t else None) for i in range(k + 1)]
        return pair[0][0], [(pair[i], pair[i + 1]) for i in range(k)], pair[k][0]
    a, c = ("A", "TA" if t else None), ("C", "TC" if t else None)
    return "B", [(a, c)] * k, "W"


def forward(params, story, query, cfg, linear=False, trace=False):
    n, j, d = cfg["mem"], cfg["sent_len"], cfg["dim"]
    if tuple(story.shape[1:]) != (n, j) or tuple(query.shape[1:]) != (j,):
        raise ValueError(f"the story must be (B, {n}, {j}) and the question (B, {j})")
    dtype = params[embedding_names(cfg)[0]].dtype
    lw = position_encoding(j, d, dtype) if cfg["encoding"] == "pe" else None
    q_name, hops, w_name = plan(cfg)
    u = encode(params[q_name], query, lw)
    cache = {}

    def memory(pair):
        if pair not in cache:
            e, t = pair
            x = encode(params[e], story, lw)
            cache[pair] = x + params[t] if t else x
        return cache[pair]

    attn = []
    for in_pair, out_pair in hops:
        scores = torch.einsum("bd,bnd->bn", u, memory(in_pair))
        p = scores if linear else torch.softmax(scores, dim=-1)
        attn.append(p)
        o = torch.einsum("bn,bnd->bd", p, memory(out_pair))
        u = (u @ params["H"].T if cfg["tying"] == "layerwise" else u) + o
    logits = u @ params[w_name].T
    return (logits, torch.stack(attn)) if trace else logits


def cross_entropy(logits, answer):
    """The summed cross-entropy of the answers; Section 4.2 does not average the cost over a batch."""
    return torch.nn.functional.cross_entropy(logits, answer, reduction="sum")


def clip_each(params, max_norm=40.0):
    """Scale the gradient of each weight matrix to an l2 norm of at most max_norm, as Section 4.2 does for the QA
    tasks (footnote 5: each matrix separately)."""
    for p in params.values():
        if p.grad is not None:
            norm = p.grad.norm()
            if norm > max_norm:
                p.grad.mul_(max_norm / norm)
