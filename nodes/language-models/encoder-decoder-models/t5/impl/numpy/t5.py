# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/t5/ (ai://NEW-t5).
"""The T5 encoder-decoder Transformer, the NumPy reference: forward, backward, the counts, span corruption, a cache of
keys and values for decoding, and greedy generation.

The source ids (B, S) are looked up in the shared embedding emb (V, M). L encoder layers follow, each
z' = z + SelfAttn(N(z)) and z = z' + FF(N(z')) with full attention (eq-enc); N rescales each token by the root of
its mean square and a learned gain, with no bias and no centring (eq-norm), and FF is two matrices with a ReLU between
them. A final N gives the encoder output e. The decoder reads the target shifted right by one, with the start token 0
in front, and each of its L layers adds causal self-attention, attention to e, and FF, each on a normalized input
(eq-dec). A final N and the embedding transposed give the logits (eq-logits), and the loss is the cross-entropy of
the T target tokens (eq-loss). No projection has a bias.

Position enters only as a scalar per head added to each attention score, looked up by the bucket of the offset between
key and query (eq-bias): one table of K buckets by H heads for the encoder and one for the decoder, shared by all the
layers of the stack. Attention to the encoder has none. pos="none" drops both tables. The queries, keys and values
of each head have d_kv features, and H d_kv need not equal M. Everything is float64.
"""

import math
from typing import Any

import numpy as np

EPS = 1e-6  # of every normalization
POSITIONS = ("relative", "none")


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
    keys = ("vocab", "d_model", "n_heads", "d_kv", "d_ff", "n_layers", "buckets", "max_distance")
    vals = (vocab, d_model, n_heads, d_kv, d_ff, n_layers, buckets, max_distance)
    return dict(zip(keys, (int(v) for v in vals), strict=True)) | {"pos": pos}


def param_shapes(cfg):
    """The name and shape of every parameter, in a fixed order."""
    V, M, F, K, H = (cfg[k] for k in ("vocab", "d_model", "d_ff", "buckets", "n_heads"))
    I = H * cfg["d_kv"]  # noqa: E741
    shapes: dict[str, tuple[int, ...]] = {"emb": (V, M)}
    if cfg["pos"] == "relative":
        shapes |= {"enc_rel": (K, H), "dec_rel": (K, H)}
    for i in range(cfg["n_layers"]):
        shapes |= {f"e{i}_ln1": (M,), f"e{i}_q": (M, I), f"e{i}_k": (M, I), f"e{i}_v": (M, I), f"e{i}_o": (I, M)}
        shapes |= {f"e{i}_ln2": (M,), f"e{i}_wi": (M, F), f"e{i}_wo": (F, M)}
    shapes["enc_ln"] = (M,)
    for i in range(cfg["n_layers"]):
        shapes |= {f"d{i}_ln1": (M,), f"d{i}_sq": (M, I), f"d{i}_sk": (M, I), f"d{i}_sv": (M, I), f"d{i}_so": (I, M)}
        shapes |= {f"d{i}_ln2": (M,), f"d{i}_cq": (M, I), f"d{i}_ck": (M, I), f"d{i}_cv": (M, I), f"d{i}_co": (I, M)}
        shapes |= {f"d{i}_ln3": (M,), f"d{i}_wi": (M, F), f"d{i}_wo": (F, M)}
    shapes["dec_ln"] = (M,)
    return shapes


def param_count(cfg):
    """eq-params: V M + 2 K H + L (4 M I + 2 M F + 2 M) + L (8 M I + 2 M F + 3 M) + 2 M, with I = H d_kv."""
    return sum(math.prod(s) for s in param_shapes(cfg).values())


def flops(cfg, B, S, T):
    """eq-flops: the FLOPs of the matrix products of one forward pass over B pairs of S source and T target tokens,
    a multiply-add counted as 2, the attention computed in full."""
    V, M, F, L = (cfg[k] for k in ("vocab", "d_model", "d_ff", "n_layers"))
    I = cfg["n_heads"] * cfg["d_kv"]  # noqa: E741
    enc = 8 * S * M * I + 4 * S * S * I + 4 * S * M * F
    dec = 8 * T * M * I + 4 * T * T * I + 4 * T * M * I + 4 * S * M * I + 4 * T * S * I + 4 * T * M * F
    return B * (L * (enc + dec) + 2 * T * M * V)


def init(rng, cfg, scale=0.5):
    """Normal weights of standard deviation scale / sqrt(fan-in), gains of one, and position tables of scale."""
    p = {}
    for name, shape in param_shapes(cfg).items():
        if len(shape) == 1:
            p[name] = np.ones(shape)
        elif name.endswith("_rel"):
            p[name] = scale * rng.standard_normal(shape)
        else:
            p[name] = scale / math.sqrt(shape[0] if name != "emb" else shape[1]) * rng.standard_normal(shape)
    return p


def bucket(offset, bidirectional, buckets, max_distance):
    """eq-bias: the bucket of the offset key - query. Bidirectional: half the buckets for keys after the query. Of
    the buckets of one side, the first half hold the offsets 0, 1, ... one each, and the second half split the rest
    on a log scale that reaches the last bucket at max_distance. Unidirectional: keys after the query share bucket 0,
    which the causal mask hides."""
    offset = np.asarray(offset)
    out = np.zeros(offset.shape, dtype=np.int64)
    if bidirectional:
        buckets //= 2
        out += (offset > 0) * buckets
        n = np.abs(offset)
    else:
        n = np.maximum(-offset, 0)
    exact = buckets // 2
    far = exact + (np.log(np.maximum(n, 1) / exact) / math.log(max_distance / exact) * (buckets - exact))
    far = np.minimum(far.astype(np.int64), buckets - 1)
    return out + np.where(n < exact, n, far)


def position_bias(table, cfg, q_pos, k_pos, bidirectional):
    """The (H, nq, nk) scores added in attention: table[bucket(k - q)] for query positions q_pos, key positions k_pos."""
    b = bucket(np.asarray(k_pos)[None, :] - np.asarray(q_pos)[:, None], bidirectional, cfg["buckets"],
               cfg["max_distance"])  # fmt: skip
    return table[b].transpose(2, 0, 1), b


def rms_norm(x, g):
    """eq-norm: x / sqrt(mean(x^2) + EPS) * g, over the last axis."""
    r = 1 / np.sqrt((x * x).mean(-1, keepdims=True) + EPS)
    return x * r * g, (x * r, r)


def rms_norm_backward(dy, g, c):
    xhat, r = c
    dxhat = dy * g
    return r * (dxhat - xhat * (dxhat * xhat).mean(-1, keepdims=True)), (dy * xhat).reshape(-1, g.size).sum(0)


def softmax(s):
    e = np.exp(s - s.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def split(x, H):
    B, n, width = x.shape
    return x.reshape(B, n, H, width // H).transpose(0, 2, 1, 3)


def merge(x):
    B, H, n, d = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, n, H * d)


def attention(p, w, xq, xkv, H, bias=None, mask=None, past=None):
    """eq-attn: heads of queries from xq (B, nq, M) on keys and values from xkv (B, nk, M), after the cached
    past = (k, v), plus bias (H, nq, nk), with the keys outside mask (nq, nk) removed. w names the four matrices."""
    q = split(xq @ p[w[0]], H)
    k, v = split(xkv @ p[w[1]], H), split(xkv @ p[w[2]], H)
    if past is not None:
        k, v = np.concatenate([past[0], k], 2), np.concatenate([past[1], v], 2)
    s = q @ k.transpose(0, 1, 3, 2) / math.sqrt(q.shape[-1])
    if bias is not None:
        s = s + bias
    if mask is not None:
        s = np.where(mask, s, -np.inf)
    a = softmax(s)
    o = merge(a @ v)
    return o @ p[w[3]], {"xq": xq, "xkv": xkv, "q": q, "k": k, "v": v, "weights": a, "o": o}


def attention_backward(p, w, dy, c):
    """The gradients of attention: (dxq, dxkv, dbias summed over the batch, grads of the four matrices)."""
    H, d = c["q"].shape[1], c["q"].shape[-1]
    g = {w[3]: np.einsum("bni,bnj->ij", c["o"], dy)}
    do = split(dy @ p[w[3]].T, H)
    a = c["weights"]
    da = do @ c["v"].transpose(0, 1, 3, 2)
    dv = merge(a.transpose(0, 1, 3, 2) @ do)
    ds = a * (da - (da * a).sum(-1, keepdims=True))  # zero where masked, since a is zero there
    dq, dk = merge(ds @ c["k"] / math.sqrt(d)), merge(ds.transpose(0, 1, 3, 2) @ c["q"] / math.sqrt(d))
    g[w[0]] = np.einsum("bni,bnj->ij", c["xq"], dq)
    g[w[1]] = np.einsum("bni,bnj->ij", c["xkv"], dk)
    g[w[2]] = np.einsum("bni,bnj->ij", c["xkv"], dv)
    dxkv = dk @ p[w[1]].T + dv @ p[w[2]].T
    return dq @ p[w[0]].T, dxkv, ds.sum(0), g


def lookup(p, cfg, ids):
    ids = np.asarray(ids)
    if ids.min() < 0 or ids.max() >= cfg["vocab"]:
        raise ValueError("token id out of range")
    return p["emb"][ids]


def encode(p, cfg, src):
    """eq-enc: the encoder output e (B, S, M) of the source ids src (B, S), and a cache for the backward pass."""
    H, S = cfg["n_heads"], np.shape(src)[1]
    z = lookup(p, cfg, src)
    cache = {"src": np.asarray(src), "layers": []}
    bias = None
    if cfg["pos"] == "relative":
        bias, cache["buckets"] = position_bias(p["enc_rel"], cfg, np.arange(S), np.arange(S), True)
    for i in range(cfg["n_layers"]):
        h1, n1 = rms_norm(z, p[f"e{i}_ln1"])
        a, ca = attention(p, (f"e{i}_q", f"e{i}_k", f"e{i}_v", f"e{i}_o"), h1, h1, H, bias)
        z1 = z + a
        h2, n2 = rms_norm(z1, p[f"e{i}_ln2"])
        u = h2 @ p[f"e{i}_wi"]
        z = z1 + np.maximum(u, 0) @ p[f"e{i}_wo"]
        cache["layers"].append({"n1": n1, "attn": ca, "n2": n2, "h2": h2, "u": u})
    e, cache["nf"] = rms_norm(z, p["enc_ln"])
    return e, cache


def causal(n_new, start=0):
    """True where the query at position start + i may see the key at position j."""
    return np.arange(start + n_new)[None, :] <= start + np.arange(n_new)[:, None]


def decode(p, cfg, e, ids, past=None, start=0):
    """eq-dec and eq-logits: the logits (B, n, V) of the decoder inputs ids (B, n) at positions start, ...,
    start + n - 1, given the encoder output e and the cached self-attention keys and values past of each layer."""
    H, n = cfg["n_heads"], np.shape(ids)[1]
    y = lookup(p, cfg, ids)
    cache = {"ids": np.asarray(ids), "layers": [], "kv": []}
    q_pos, k_pos = start + np.arange(n), np.arange(start + n)
    bias = None
    if cfg["pos"] == "relative":
        bias, cache["buckets"] = position_bias(p["dec_rel"], cfg, q_pos, k_pos, False)
    for i in range(cfg["n_layers"]):
        h1, n1 = rms_norm(y, p[f"d{i}_ln1"])
        w = (f"d{i}_sq", f"d{i}_sk", f"d{i}_sv", f"d{i}_so")
        a, ca = attention(p, w, h1, h1, H, bias, causal(n, start), None if past is None else past[i])
        cache["kv"].append((ca["k"], ca["v"]))
        y1 = y + a
        h2, n2 = rms_norm(y1, p[f"d{i}_ln2"])
        x, cx = attention(p, (f"d{i}_cq", f"d{i}_ck", f"d{i}_cv", f"d{i}_co"), h2, e, H)
        y2 = y1 + x
        h3, n3 = rms_norm(y2, p[f"d{i}_ln3"])
        u = h3 @ p[f"d{i}_wi"]
        y = y2 + np.maximum(u, 0) @ p[f"d{i}_wo"]
        cache["layers"].append({"n1": n1, "attn": ca, "n2": n2, "cross": cx, "n3": n3, "h3": h3, "u": u})
    final, cache["nf"] = rms_norm(y, p["dec_ln"])
    cache["final"] = final
    return final @ p["emb"].T, cache


def shift_right(tgt, start_id=0):
    """The decoder inputs of teacher forcing: the start token, then the target without its last token."""
    tgt = np.asarray(tgt)
    return np.concatenate([np.full((tgt.shape[0], 1), start_id), tgt[:, :-1]], axis=1)


def forward(p, cfg, src, tgt):
    """The logits (B, T, V) of every target token given the source and the target tokens before it."""
    e, ce = encode(p, cfg, src)
    logits, cd = decode(p, cfg, e, shift_right(tgt))
    return logits, {"enc": ce, "dec": cd, "e": e}


def nll(logits, tgt):
    """eq-loss: the mean over the B T target tokens of -log p(token)."""
    s = logits - logits.max(-1, keepdims=True)
    logp = s - np.log(np.exp(s).sum(-1, keepdims=True))
    return float(-np.take_along_axis(logp, np.asarray(tgt)[..., None], -1).mean())


def loss(p, cfg, src, tgt):
    return nll(forward(p, cfg, src, tgt)[0], tgt)


def backward(p, cfg, cache, logits, tgt):
    """The gradient of eq-loss with respect to every parameter, by hand through every layer."""
    g = {k: np.zeros_like(v) for k, v in p.items()}
    B, T, V = logits.shape
    prob = softmax(logits)
    prob[np.arange(B)[:, None], np.arange(T)[None, :], tgt] -= 1
    dlogits = prob / (B * T)
    cd, ce = cache["dec"], cache["enc"]
    g["emb"] += np.einsum("btv,btm->vm", dlogits, cd["final"])
    dy, g["dec_ln"] = rms_norm_backward(dlogits @ p["emb"], p["dec_ln"], cd["nf"])
    de = np.zeros_like(cache["e"])
    for i in reversed(range(cfg["n_layers"])):
        c = cd["layers"][i]
        g[f"d{i}_wo"] = np.einsum("bnf,bnm->fm", np.maximum(c["u"], 0), dy)
        du = (dy @ p[f"d{i}_wo"].T) * (c["u"] > 0)
        g[f"d{i}_wi"] = np.einsum("bnm,bnf->mf", c["h3"], du)
        dh3, g[f"d{i}_ln3"] = rms_norm_backward(du @ p[f"d{i}_wi"].T, p[f"d{i}_ln3"], c["n3"])
        dy2 = dy + dh3
        dh2, dex, _, gx = attention_backward(p, (f"d{i}_cq", f"d{i}_ck", f"d{i}_cv", f"d{i}_co"), dy2, c["cross"])
        g |= gx
        de += dex
        dn2, g[f"d{i}_ln2"] = rms_norm_backward(dh2, p[f"d{i}_ln2"], c["n2"])
        dy1 = dy2 + dn2
        dq, dkv, db, gs = attention_backward(p, (f"d{i}_sq", f"d{i}_sk", f"d{i}_sv", f"d{i}_so"), dy1, c["attn"])
        g |= gs
        if cfg["pos"] == "relative":
            for h in range(db.shape[0]):
                np.add.at(g["dec_rel"][:, h], cd["buckets"], db[h])
        dn1, g[f"d{i}_ln1"] = rms_norm_backward(dq + dkv, p[f"d{i}_ln1"], c["n1"])
        dy = dy1 + dn1
    np.add.at(g["emb"], cd["ids"], dy)
    dz, g["enc_ln"] = rms_norm_backward(de, p["enc_ln"], ce["nf"])
    for i in reversed(range(cfg["n_layers"])):
        c = ce["layers"][i]
        g[f"e{i}_wo"] = np.einsum("bnf,bnm->fm", np.maximum(c["u"], 0), dz)
        du = (dz @ p[f"e{i}_wo"].T) * (c["u"] > 0)
        g[f"e{i}_wi"] = np.einsum("bnm,bnf->mf", c["h2"], du)
        dh2, g[f"e{i}_ln2"] = rms_norm_backward(du @ p[f"e{i}_wi"].T, p[f"e{i}_ln2"], c["n2"])
        dz1 = dz + dh2
        dq, dkv, db, gs = attention_backward(p, (f"e{i}_q", f"e{i}_k", f"e{i}_v", f"e{i}_o"), dz1, c["attn"])
        g |= gs
        if cfg["pos"] == "relative":
            for h in range(db.shape[0]):
                np.add.at(g["enc_rel"][:, h], ce["buckets"], db[h])
        dn1, g[f"e{i}_ln1"] = rms_norm_backward(dq + dkv, p[f"e{i}_ln1"], c["n1"])
        dz = dz1 + dn1
    np.add.at(g["emb"], ce["src"], dz)
    return g


def generate(p, cfg, src, n_new, start_id=0):
    """Greedy decoding with the cache: n_new tokens (B, n_new) after the start token."""
    e, _ = encode(p, cfg, src)
    tok = np.full((np.shape(src)[0], 1), start_id)
    past, out = None, []
    for t in range(n_new):
        logits, c = decode(p, cfg, e, tok, past, start=t)
        past = c["kv"]
        tok = logits[:, -1].argmax(-1)[:, None]
        out.append(tok)
    return np.concatenate(out, axis=1)


def generate_without_cache(p, cfg, src, n_new, start_id=0):
    """Greedy decoding that runs the decoder on the whole prefix for every new token."""
    e, _ = encode(p, cfg, src)
    ids = np.full((np.shape(src)[0], 1), start_id)
    for _ in range(n_new):
        logits, _ = decode(p, cfg, e, ids)
        ids = np.concatenate([ids, logits[:, -1].argmax(-1)[:, None]], axis=1)
    return ids[:, 1:]


def random_spans_mask(n, density, mean_span, rng):
    """A noise mask of n tokens with round(n density) noise tokens in round(noise / mean_span) spans: the noise and
    the kept tokens are each cut into that many runs of random positive lengths, and the runs alternate, kept first."""
    noise = min(max(round(n * density), 1), n - 1)
    spans = max(round(noise / mean_span), 1)
    if n - noise < spans:
        raise ValueError("too few kept tokens to separate the spans")

    def cut(total):
        edges = np.sort(rng.choice(np.arange(1, total), spans - 1, replace=False))
        return np.diff(np.concatenate([[0], edges, [total]]))

    mask = np.zeros(n, dtype=bool)
    at = 0
    for keep, drop in zip(cut(n - noise), cut(noise), strict=True):
        mask[at + keep : at + keep + drop] = True
        at += keep + drop
    return mask


def span_corrupt(tokens, mask, sentinels):
    """eq-span: each run of masked tokens becomes one sentinel in the input; the target is each sentinel followed by
    its run, then one more sentinel. sentinels[j] is the id of the j-th sentinel."""
    tokens, mask = np.asarray(tokens), np.asarray(mask, dtype=bool)
    starts = mask & ~np.concatenate([[False], mask[:-1]])
    if starts.sum() + 1 > len(sentinels):
        raise ValueError("more spans than sentinels")
    inputs, targets, j = [], [], -1
    for t, m, s in zip(tokens, mask, starts, strict=True):
        if s:
            j += 1
            inputs.append(sentinels[j])
            targets.append(sentinels[j])
        if m:
            targets.append(t)
        else:
            inputs.append(t)
    targets.append(sentinels[j + 1])
    return np.array(inputs), np.array(targets)
