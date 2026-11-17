# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/end-to-end-memory-network/ (ai://NEW-end-to-end-memory-network).
"""An end-to-end memory network for question answering, the NumPy reference.

A story is a (B, N, J) array of word ids: N sentences of J words for each problem of the batch, the most recent
sentence first, padded with the null word 0. A question is a (B, J) array. Each sentence is embedded twice, as a
memory vector m_i and an output vector c_i, by a bag of words or by the position encoding (eq-pe), plus a learned row
of a temporal matrix (eq-temporal). A hop attends over the memory vectors with a softmax of inner products with the
controller state u (eq-match), reads the weighted sum of the output vectors (eq-out) and adds it to u (eq-stack).
After K hops a linear map and a softmax over the vocabulary give the answer (eq-answer).

The weights follow the adjacent tying of Sukhbaatar et al.: K + 1 embedding matrices E_0..E_K of shape (V, d) with
A^k = E_{k-1}, C^k = E_k, B = A^1 = E_0 and W^T = C^K = E_K, and K + 1 temporal matrices tied the same way. The
layer-wise tying shares one A and one C over all hops and adds a linear map H between hops (eq-layerwise).
"""

import numpy as np

NULL = 0  # the word id of the null symbol, whose embedding is held at zero


def config(vocab=12, dim=6, hops=3, mem=5, sent_len=4, tying="adjacent", encoding="pe", temporal=True):
    """V = vocab words, embeddings of width d = dim, K = hops, N = mem sentences of J = sent_len words."""
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
        return (k + 1) * (v * d + t)  # eq: eq-params
    return 4 * v * d + d * d + 2 * t  # A, C, B, W, H and T_A, T_C


def flops(cfg, batch):
    """FLOPs of one forward pass, a multiply-add counted as 2, the products only: the position-weighted sums of the
    words of every sentence and of the question, then for every hop the inner products with the memory and the
    weighted sum of the outputs, and the answer map. Adjacent tying embeds the story K + 1 times, since
    C^k = A^{k+1}; layer-wise tying twice. The bag of words only adds, so its embedding counts no FLOPs."""
    v, d, k, n, j = cfg["vocab"], cfg["dim"], cfg["hops"], cfg["mem"], cfg["sent_len"]
    encodings = k + 1 if cfg["tying"] == "adjacent" else 2
    embed = 2 * j * d * (encodings * n + 1) if cfg["encoding"] == "pe" else 0
    hop_map = 2 * d * d if cfg["tying"] == "layerwise" else 0
    return batch * (embed + k * (4 * n * d + hop_map) + 2 * v * d)  # eq: eq-flops


def init(rng, cfg, sigma=0.1):
    """Gaussian weights of standard deviation sigma, as in Section 4.2, with the null rows at zero."""
    v, d, k, n = cfg["vocab"], cfg["dim"], cfg["hops"], cfg["mem"]
    names = [f"E{i}" for i in range(k + 1)] if cfg["tying"] == "adjacent" else ["A", "C", "B", "W"]
    p = {name: sigma * rng.normal(size=(v, d)) for name in names}
    if cfg["temporal"]:
        tnames = [f"T{i}" for i in range(k + 1)] if cfg["tying"] == "adjacent" else ["TA", "TC"]
        p.update({name: sigma * rng.normal(size=(n, d)) for name in tnames})
    if cfg["tying"] == "layerwise":
        p["H"] = sigma * rng.normal(size=(d, d))
    return zero_null(p, cfg)


def zero_null(params, cfg):
    """Set the null row of every embedding to zero, the constraint of Section 4.2; call it after each update."""
    out = dict(params)
    for name in embedding_names(cfg):
        out[name] = out[name].copy()
        out[name][NULL] = 0.0
    return out


def embedding_names(cfg):
    if cfg["tying"] == "adjacent":
        return [f"E{i}" for i in range(cfg["hops"] + 1)]
    return ["A", "C", "B", "W"]


def position_encoding(sent_len, dim):
    """l_{kj} = (1 - j/J) - (k/d)(1 - 2j/J) with 1-based j and k, as a (J, d) array (eq-pe)."""
    j = np.arange(1, sent_len + 1)[:, None] / sent_len
    k = np.arange(1, dim + 1)[None, :] / dim
    return (1 - j) - k * (1 - 2 * j)


def encode(E, words, weights=None):
    """Embed each sentence of words (..., J) as the sum of its word vectors, each scaled elementwise by its row of the
    (J, d) weights when given: the bag of words without weights, the position encoding with them."""
    vec = E[words]  # (..., J, d); the null row of E is zero
    if weights is None:
        return vec.sum(axis=-2)
    return np.einsum("...jd,jd->...d", vec, weights)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def hop(u, m, c, linear=False):
    """One memory layer: p_i = softmax(u . m_i) (eq-match), o = sum_i p_i c_i (eq-out). With linear=True the
    softmax is left out, as in the linear start of Section 4.2."""
    scores = np.einsum("bd,bnd->bn", u, m)
    p = scores if linear else softmax(scores)
    return np.einsum("bn,bnd->bd", p, c), p


def plan(cfg):
    """The names of the (embedding, temporal) pair for the question, and for the memory and output of each hop."""
    k, t = cfg["hops"], cfg["temporal"]
    if cfg["tying"] == "adjacent":
        pair = [(f"E{i}", f"T{i}" if t else None) for i in range(k + 1)]
        return pair[0][0], [(pair[i], pair[i + 1]) for i in range(k)], pair[k][0]
    a, c = ("A", "TA" if t else None), ("C", "TC" if t else None)
    return "B", [(a, c)] * k, "W"


def forward(params, story, query, cfg, linear=False, trace=False):
    """Logits over the vocabulary, (B, V); with trace, also the attention of every hop, (K, B, N)."""
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
            cache[pair] = x + params[t] if t else x  # eq: eq-temporal
        return cache[pair]

    attn = []
    for in_pair, out_pair in hops:
        o, p = hop(u, memory(in_pair), memory(out_pair), linear)
        attn.append(p)
        u = (u @ params["H"].T if cfg["tying"] == "layerwise" else u) + o  # eq: eq-stack
    logits = u @ params[w_name].T  # eq: eq-answer
    return (logits, np.stack(attn)) if trace else logits


def cross_entropy(logits, answer):
    """The summed cross-entropy of the answers, a (B,) array of word ids; Section 4.2 does not average it."""
    z = logits - logits.max(axis=-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    return float(-logp[np.arange(len(answer)), answer].sum())
