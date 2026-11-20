# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/decoder-only-transformer/ (ai://NEW-decoder-only-transformer).
"""The decoder-only Transformer language model in the form of GPT-2 and GPT-3, the NumPy reference: forward,
backward, the counts, a cache of keys and values for decoding, and greedy generation.

A sequence of token ids (B, n) is looked up in the token embedding wte (V, D) and the learned position embeddings
wpe (T, D) are added (eq-embed). L pre-norm layers follow, z' = z + MSA(LN(z)) and z = z' + MLP(LN(z')) (eq-layer),
the self-attention causal: position i only sees the positions j <= i (eq-mask), and the MLP has an exact GELU between
its two layers. A final layer normalization and the unembedding give the logits of the next token at every position
(eq-logits); the unembedding is wte transposed when tied=True and its own matrix head_W (D, V) otherwise, without a
bias in both cases. The loss is the cross-entropy of the next token, averaged over the B (n - 1) positions that have
one (eq-loss). pos="none" drops the position embeddings. Everything is float64.
"""

import math

import numpy as np

EPS = 1e-5  # of every layer normalization, the value of GPT-2
POSITIONS = ("learned", "none")
_erf = np.vectorize(math.erf, otypes=[float])


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
    """The name and shape of every parameter, in a fixed order."""
    V, T, D, F = (cfg[k] for k in ("vocab", "n_ctx", "d_model", "d_ff"))
    shapes = {"wte": (V, D)}
    if cfg["pos"] == "learned":
        shapes["wpe"] = (T, D)
    for li in range(cfg["n_layers"]):
        shapes |= {
            f"l{li}_ln1_g": (D,), f"l{li}_ln1_b": (D,),
            f"l{li}_qkv_W": (D, 3 * D), f"l{li}_qkv_b": (3 * D,),
            f"l{li}_out_W": (D, D), f"l{li}_out_b": (D,),
            f"l{li}_ln2_g": (D,), f"l{li}_ln2_b": (D,),
            f"l{li}_fc1_W": (D, F), f"l{li}_fc1_b": (F,),
            f"l{li}_fc2_W": (F, D), f"l{li}_fc2_b": (D,),
        }  # fmt: skip
    shapes |= {"ln_f_g": (D,), "ln_f_b": (D,)}
    if not cfg["tied"]:
        shapes["head_W"] = (D, V)
    return shapes


def param_count(cfg):
    """eq-params, in closed form."""
    V, T, D, F, L = (cfg[k] for k in ("vocab", "n_ctx", "d_model", "d_ff", "n_layers"))
    embed = V * D + T * D * (cfg["pos"] == "learned")
    layer = 4 * D * D + 2 * D * F + 9 * D + F
    head = 2 * D + (0 if cfg["tied"] else D * V)
    return embed + L * layer + head


def non_embedding_params(cfg):
    """The parameters of the layers, the N of Kaplan et al. (2020), biases and gains included."""
    D, F = cfg["d_model"], cfg["d_ff"]
    return cfg["n_layers"] * (4 * D * D + 2 * D * F + 9 * D + F)


def flops(cfg, B, n):
    """eq-flops: the FLOPs of the matrix products of a forward pass over B sequences of n tokens, a multiply-add
    counted as 2. The products of the attention are counted in full, the masked half included, as dense code does."""
    V, D, F, L = (cfg[k] for k in ("vocab", "d_model", "d_ff", "n_layers"))
    layer = 8 * n * D * D + 4 * n * n * D + 4 * n * D * F
    return B * (L * layer + 2 * n * D * V)


def kv_cache_size(cfg, n):
    """The numbers kept in the cache of keys and values for one sequence of n tokens: a key and a value of width D
    for every token in every layer."""
    return 2 * cfg["n_layers"] * cfg["d_model"] * n


def init(rng, cfg, scale=0.1, residual_scaled=False):
    """Random parameters: weights and biases N(0, scale^2), gains of the normalizations 1 + N(0, scale^2). With
    residual_scaled, the weights of the two projections that write into the residual stream in each layer, out_W and
    fc2_W, are divided by sqrt(2 L), the 1 / sqrt(N) of GPT-2 with N = 2 L residual layers."""
    params = {}
    for name, shape in param_shapes(cfg).items():
        params[name] = rng.normal(0.0, scale, shape) + (1.0 if name.endswith("_g") else 0.0)
        if residual_scaled and name.endswith(("_out_W", "_fc2_W")):
            params[name] /= math.sqrt(2 * cfg["n_layers"])
    return params


def unembedding(params, cfg):
    """The (D, V) matrix of eq-logits."""
    return params["wte"].T if cfg["tied"] else params["head_W"]


def layer_norm(x, g, b):
    mu = x.mean(-1, keepdims=True)
    rstd = 1.0 / np.sqrt(((x - mu) ** 2).mean(-1, keepdims=True) + EPS)
    xhat = (x - mu) * rstd
    return xhat * g + b, (xhat, rstd, g)


def layer_norm_backward(dy, cache):
    xhat, rstd, g = cache
    axes = tuple(range(dy.ndim - 1))
    dg, db = (dy * xhat).sum(axes), dy.sum(axes)
    dxhat = dy * g
    dx = rstd * (dxhat - dxhat.mean(-1, keepdims=True) - xhat * (dxhat * xhat).mean(-1, keepdims=True))
    return dx, dg, db


def gelu(x):
    """The exact GELU, x Phi(x), with Phi the standard normal distribution function."""
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def gelu_grad(x):
    return 0.5 * (1.0 + _erf(x / math.sqrt(2.0))) + x * np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def softmax(s):
    e = np.exp(s - s.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def log_softmax(s):
    m = s.max(-1, keepdims=True)
    return s - m - np.log(np.exp(s - m).sum(-1, keepdims=True))


def split_heads(x, H):
    B, n, D = x.shape
    return x.reshape(B, n, H, D // H).transpose(0, 2, 1, 3)


def merge_heads(x):
    B, H, n, d = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, n, H * d)


def causal_mask(n_new, n_past=0):
    """eq-mask: True where the query of new token i, at position n_past + i, may see the key at position j."""
    return np.arange(n_past + n_new)[None, :] <= n_past + np.arange(n_new)[:, None]


def msa(p, li, h, H, past=None):
    """eq-attn: causal multi-head self-attention of layer li on the normalized states h (B, n, D) of n new tokens,
    after the keys and values past = (k, v), each (B, H, t, D / H), of the t tokens before them."""
    D = h.shape[-1]
    qkv = h @ p[f"l{li}_qkv_W"] + p[f"l{li}_qkv_b"]
    q, k, v = (split_heads(qkv[..., i * D : (i + 1) * D], H) for i in range(3))
    t = 0
    if past is not None:
        t = past[0].shape[2]
        k, v = np.concatenate([past[0], k], axis=2), np.concatenate([past[1], v], axis=2)
    s = q @ k.transpose(0, 1, 3, 2) / math.sqrt(D // H)
    a = softmax(np.where(causal_mask(h.shape[1], t), s, -np.inf))
    o = merge_heads(a @ v)
    return o @ p[f"l{li}_out_W"] + p[f"l{li}_out_b"], {"h": h, "q": q, "k": k, "v": v, "weights": a, "o": o}


def msa_backward(p, li, dy, c):
    D = dy.shape[-1]
    d = D // c["q"].shape[1]
    g = {f"l{li}_out_W": np.einsum("bni,bnj->ij", c["o"], dy), f"l{li}_out_b": dy.sum((0, 1))}
    do = split_heads(dy @ p[f"l{li}_out_W"].T, c["q"].shape[1])
    a = c["weights"]
    da = do @ c["v"].transpose(0, 1, 3, 2)
    dv = a.transpose(0, 1, 3, 2) @ do
    ds = a * (da - (da * a).sum(-1, keepdims=True)) / math.sqrt(d)  # zero where masked, since a is zero there
    dq, dk = ds @ c["k"], ds.transpose(0, 1, 3, 2) @ c["q"]
    dqkv = np.concatenate([merge_heads(dq), merge_heads(dk), merge_heads(dv)], axis=-1)
    g[f"l{li}_qkv_W"] = np.einsum("bni,bnj->ij", c["h"], dqkv)
    g[f"l{li}_qkv_b"] = dqkv.sum((0, 1))
    return dqkv @ p[f"l{li}_qkv_W"].T, g


def embed(p, cfg, ids, start=0):
    """eq-embed: the states z_0 (B, n, D) of tokens ids (B, n) that sit at positions start, ..., start + n - 1."""
    ids = np.asarray(ids)
    if ids.min() < 0 or ids.max() >= cfg["vocab"]:
        raise ValueError("token id out of range")
    if start + ids.shape[1] > cfg["n_ctx"] and cfg["pos"] == "learned":
        raise ValueError(f"positions beyond the context of {cfg['n_ctx']} tokens have no embedding")
    z = p["wte"][ids]
    if cfg["pos"] == "learned":
        z = z + p["wpe"][start : start + ids.shape[1]]
    return z


def forward(params, cfg, ids, past=None):
    """The logits (B, n, V) of the next token at every position of ids (B, n), and the cache of the backward pass.
    With past, the list of (k, v) of every layer from an earlier call, the tokens continue that sequence; the
    cache then holds the keys and values of all the tokens so far under "kv"."""
    p, H = params, cfg["n_heads"]
    start = 0 if past is None else past[0][0].shape[2]
    z = embed(p, cfg, ids, start)
    cache = {"ids": np.asarray(ids), "tokens": z, "layers": [], "kv": []}
    for li in range(cfg["n_layers"]):
        h1, n1 = layer_norm(z, p[f"l{li}_ln1_g"], p[f"l{li}_ln1_b"])
        a, ca = msa(p, li, h1, H, None if past is None else past[li])
        z1 = z + a  # eq-layer
        h2, n2 = layer_norm(z1, p[f"l{li}_ln2_g"], p[f"l{li}_ln2_b"])
        u = h2 @ p[f"l{li}_fc1_W"] + p[f"l{li}_fc1_b"]
        gu = gelu(u)
        z = z1 + gu @ p[f"l{li}_fc2_W"] + p[f"l{li}_fc2_b"]  # eq-layer
        cache["layers"].append({"n1": n1, "attn": ca, "n2": n2, "h2": h2, "u": u, "gu": gu})
        cache["kv"].append((ca["k"], ca["v"]))
    zn, nf = layer_norm(z, p["ln_f_g"], p["ln_f_b"])
    cache |= {"final": z, "nf": nf, "y": zn}
    return zn @ unembedding(p, cfg), cache  # eq-logits


def nll(logits, ids):
    """eq-loss: the cross-entropy of the next token, averaged over the B (n - 1) positions that have one."""
    ids = np.asarray(ids)
    logp = log_softmax(logits[:, :-1])
    return float(-np.take_along_axis(logp, ids[:, 1:, None], -1).mean())


def loss(params, cfg, ids):
    return nll(forward(params, cfg, ids)[0], ids)


def backward(params, cfg, cache, ids):
    """The gradient of eq-loss with respect to every parameter."""
    p = params
    ids = np.asarray(ids)
    B, n = ids.shape
    y = cache["y"]
    W = unembedding(p, cfg)
    dlogits = softmax(y @ W)
    dlogits[:, -1] = 0.0  # the last position has no next token
    head, tgt = dlogits[:, :-1], ids[:, 1:, None]  # a view: writing to head writes to dlogits
    np.put_along_axis(head, tgt, np.take_along_axis(head, tgt, -1) - 1.0, -1)
    dlogits /= B * (n - 1)
    g = {}
    dW = np.einsum("bni,bnj->ij", y, dlogits)
    g["wte"] = dW.T.copy() if cfg["tied"] else np.zeros_like(p["wte"])
    if not cfg["tied"]:
        g["head_W"] = dW
    dz, g["ln_f_g"], g["ln_f_b"] = layer_norm_backward(dlogits @ W.T, cache["nf"])
    for li in reversed(range(cfg["n_layers"])):
        c = cache["layers"][li]
        g[f"l{li}_fc2_W"] = np.einsum("bni,bnj->ij", c["gu"], dz)
        g[f"l{li}_fc2_b"] = dz.sum((0, 1))
        du = (dz @ p[f"l{li}_fc2_W"].T) * gelu_grad(c["u"])
        g[f"l{li}_fc1_W"] = np.einsum("bni,bnj->ij", c["h2"], du)
        g[f"l{li}_fc1_b"] = du.sum((0, 1))
        dh2, g[f"l{li}_ln2_g"], g[f"l{li}_ln2_b"] = layer_norm_backward(du @ p[f"l{li}_fc1_W"].T, c["n2"])
        dz1 = dz + dh2
        dh1, ga = msa_backward(p, li, dz1, c["attn"])
        g |= ga
        dx1, g[f"l{li}_ln1_g"], g[f"l{li}_ln1_b"] = layer_norm_backward(dh1, c["n1"])
        dz = dz1 + dx1
    if cfg["pos"] == "learned":
        g["wpe"] = np.zeros_like(p["wpe"])
        g["wpe"][:n] = dz.sum(0)
    np.add.at(g["wte"], ids, dz)
    return {k: g[k] for k in param_shapes(cfg)}


def generate(params, cfg, prompt, n_new):
    """Greedy decoding: feed the prompt (B, n) once, then one token at a time with the cache of keys and values,
    each time appending the most likely next token. Returns the (B, n_new) new tokens."""
    logits, cache = forward(params, cfg, prompt)
    out = []
    for _ in range(n_new):
        nxt = logits[:, -1].argmax(-1)[:, None]
        out.append(nxt)
        if len(out) == n_new:
            break
        logits, cache = forward(params, cfg, nxt, past=cache["kv"])
    return np.concatenate(out, axis=1)


def generate_without_cache(params, cfg, prompt, n_new):
    """The same greedy decoding by running the whole sequence again for every new token: n_new full passes."""
    seq = np.asarray(prompt)
    for _ in range(n_new):
        nxt = forward(params, cfg, seq)[0][:, -1].argmax(-1)[:, None]
        seq = np.concatenate([seq, nxt], axis=1)
    return seq[:, np.asarray(prompt).shape[1] :]
