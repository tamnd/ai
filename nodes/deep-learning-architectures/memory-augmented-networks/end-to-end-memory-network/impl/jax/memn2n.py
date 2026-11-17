# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/end-to-end-memory-network/ (ai://NEW-end-to-end-memory-network).
"""An end-to-end memory network for question answering in JAX, the same model as the NumPy reference.

forward is a pure function of the parameter dict, so jax.jit, jax.grad and jax.vmap apply to it; the config is a
static argument. The null rows are held at zero by zero_null after each update.
"""

import jax
import jax.numpy as jnp

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


def init(key, cfg, sigma=0.1):
    v, d, k, n = cfg["vocab"], cfg["dim"], cfg["hops"], cfg["mem"]
    shapes = {name: (v, d) for name in embedding_names(cfg)}
    if cfg["temporal"]:
        tnames = [f"T{i}" for i in range(k + 1)] if cfg["tying"] == "adjacent" else ["TA", "TC"]
        shapes.update({name: (n, d) for name in tnames})
    if cfg["tying"] == "layerwise":
        shapes["H"] = (d, d)
    keys = jax.random.split(key, len(shapes))
    p = {name: sigma * jax.random.normal(kk, s) for kk, (name, s) in zip(keys, sorted(shapes.items()))}
    return zero_null(p, cfg)


def zero_null(params, cfg):
    out = dict(params)
    for name in embedding_names(cfg):
        out[name] = out[name].at[NULL].set(0.0)
    return out


def position_encoding(sent_len, dim):
    j = jnp.arange(1, sent_len + 1)[:, None] / sent_len
    k = jnp.arange(1, dim + 1)[None, :] / dim
    return (1 - j) - k * (1 - 2 * j)


def encode(E, words, weights=None):
    vec = E[words]
    if weights is None:
        return vec.sum(axis=-2)
    return jnp.einsum("...jd,jd->...d", vec, weights)


def plan(cfg):
    k, t = cfg["hops"], cfg["temporal"]
    if cfg["tying"] == "adjacent":
        pair = [(f"E{i}", f"T{i}" if t else None) for i in range(k + 1)]
        return pair[0][0], [(pair[i], pair[i + 1]) for i in range(k)], pair[k][0]
    a, c = ("A", "TA" if t else None), ("C", "TC" if t else None)
    return "B", [(a, c)] * k, "W"


def forward(params, story, query, cfg, linear=False, trace=False):
    n, j, d = cfg["mem"], cfg["sent_len"], cfg["dim"]
    if story.shape[1:] != (n, j) or query.shape[1:] != (j,):
        raise ValueError(f"the story must be (B, {n}, {j}) and the question (B, {j})")
    lw = position_encoding(j, d) if cfg["encoding"] == "pe" else None
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
        scores = jnp.einsum("bd,bnd->bn", u, memory(in_pair))
        p = scores if linear else jax.nn.softmax(scores, axis=-1)
        attn.append(p)
        o = jnp.einsum("bn,bnd->bd", p, memory(out_pair))
        u = (u @ params["H"].T if cfg["tying"] == "layerwise" else u) + o
    logits = u @ params[w_name].T
    return (logits, jnp.stack(attn)) if trace else logits


def cross_entropy(logits, answer):
    logp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.take_along_axis(logp, answer[:, None], axis=-1).sum()
