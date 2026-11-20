# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/decoder-only-transformer/ (ai://NEW-decoder-only-transformer).
"""The decoder-only Transformer language model in the form of GPT-2 and GPT-3, in JAX, as pure functions of the
parameters.

The parameters have the names and the layout of the NumPy reference, so the shared fixture loads as it is. forward()
runs a whole sequence with the causal mask; the gradient of eq-loss is left to jax.grad. Decoding uses a cache of
fixed size, allocated once for max_len tokens, so that every step has the same shapes and compiles once:
decode_step() writes the key and value of the new token at position t with dynamic_update_slice and masks the
places after t, and generate() runs the prompt and then the greedy steps through jax.lax.scan.
"""

import math

import jax
import jax.numpy as jnp

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
    return sum(math.prod(s) for s in param_shapes(cfg).values())


def init(key, cfg, scale=0.1):
    """Random parameters: weights and biases N(0, scale^2), gains of the normalizations 1 + N(0, scale^2)."""
    shapes = param_shapes(cfg)
    keys = jax.random.split(key, len(shapes))
    return {
        name: scale * jax.random.normal(k, shape) + (1.0 if name.endswith("_g") else 0.0)
        for k, (name, shape) in zip(keys, shapes.items(), strict=True)
    }


def layer_norm(x, g, b):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + EPS) * g + b


def unembedding(p, cfg):
    return p["wte"].T if cfg["tied"] else p["head_W"]


def attend(p, li, h, k_all, v_all, allowed, H):
    """Multi-head attention of the queries of h (B, n, D) over keys and values (B, H, t, D / H), where allowed
    (n, t) says which keys each query may see."""
    B, n, D = h.shape
    q = (h @ p[f"l{li}_qkv_W"][:, :D] + p[f"l{li}_qkv_b"][:D]).reshape(B, n, H, D // H).transpose(0, 2, 1, 3)
    s = jnp.where(allowed, q @ k_all.transpose(0, 1, 3, 2) / math.sqrt(D // H), -jnp.inf)
    o = (jax.nn.softmax(s, -1) @ v_all).transpose(0, 2, 1, 3).reshape(B, n, D)
    return o @ p[f"l{li}_out_W"] + p[f"l{li}_out_b"]


def keys_values(p, li, h, H):
    B, n, D = h.shape
    kv = h @ p[f"l{li}_qkv_W"][:, D:] + p[f"l{li}_qkv_b"][D:]
    k, v = kv[..., :D], kv[..., D:]
    return tuple(x.reshape(B, n, H, D // H).transpose(0, 2, 1, 3) for x in (k, v))


def mlp(p, li, z):
    h = layer_norm(z, p[f"l{li}_ln2_g"], p[f"l{li}_ln2_b"])
    u = jax.nn.gelu(h @ p[f"l{li}_fc1_W"] + p[f"l{li}_fc1_b"], approximate=False)
    return z + u @ p[f"l{li}_fc2_W"] + p[f"l{li}_fc2_b"]


def forward(params, cfg, ids):
    """The logits (B, n, V) of the next token at every position of ids (B, n)."""
    p, H = params, cfg["n_heads"]
    n = ids.shape[1]
    if n > cfg["n_ctx"] and cfg["pos"] == "learned":
        raise ValueError(f"positions beyond the context of {cfg['n_ctx']} tokens have no embedding")
    z = p["wte"][ids] + (p["wpe"][:n] if cfg["pos"] == "learned" else 0.0)
    allowed = jnp.tril(jnp.ones((n, n), bool))
    for li in range(cfg["n_layers"]):
        h = layer_norm(z, p[f"l{li}_ln1_g"], p[f"l{li}_ln1_b"])
        k, v = keys_values(p, li, h, H)
        z = mlp(p, li, z + attend(p, li, h, k, v, allowed, H))
    return layer_norm(z, p["ln_f_g"], p["ln_f_b"]) @ unembedding(p, cfg)


def loss(params, cfg, ids):
    """eq-loss: the cross-entropy of the next token, averaged over the B (n - 1) positions that have one."""
    logp = jax.nn.log_softmax(forward(params, cfg, ids)[:, :-1], -1)
    return -jnp.take_along_axis(logp, ids[:, 1:, None], -1).mean()


def init_cache(cfg, B, max_len):
    """Zero keys and values (L, 2, B, H, max_len, D / H) for up to max_len tokens."""
    H, D = cfg["n_heads"], cfg["d_model"]
    return jnp.zeros((cfg["n_layers"], 2, B, H, max_len, D // H))


def decode_step(params, cfg, cache, tok, t):
    """Feed the tokens tok (B,) at position t: the logits (B, V) of the next token and the cache with the keys and
    values of position t written in."""
    p, H = params, cfg["n_heads"]
    z = p["wte"][tok][:, None, :]
    if cfg["pos"] == "learned":
        z = z + jax.lax.dynamic_slice_in_dim(p["wpe"], t, 1)
    allowed = (jnp.arange(cache.shape[4]) <= t)[None, :]
    for li in range(cfg["n_layers"]):
        h = layer_norm(z, p[f"l{li}_ln1_g"], p[f"l{li}_ln1_b"])
        k, v = keys_values(p, li, h, H)
        cache = jax.lax.dynamic_update_slice(cache, jnp.stack([k, v])[None], (li, 0, 0, 0, t, 0))
        z = mlp(p, li, z + attend(p, li, h, cache[li, 0], cache[li, 1], allowed, H))
    return (layer_norm(z, p["ln_f_g"], p["ln_f_b"]) @ unembedding(p, cfg))[:, 0], cache


def generate(params, cfg, prompt, n_new):
    """Greedy decoding: the prompt (B, n) one token at a time into the cache, then n_new greedy tokens, all through
    jax.lax.scan. Returns the (B, n_new) new tokens."""
    B, n = prompt.shape
    cache = init_cache(cfg, B, n + n_new)

    def feed(carry, x):
        cache, _ = carry
        tok, t = x
        logits, cache = decode_step(params, cfg, cache, tok, t)
        return (cache, logits), None

    (cache, logits), _ = jax.lax.scan(feed, (cache, jnp.zeros((B, cfg["vocab"]))), (prompt.T, jnp.arange(n)))

    def step(carry, t):
        cache, logits = carry
        tok = logits.argmax(-1)
        logits, cache = decode_step(params, cfg, cache, tok, t)
        return (cache, logits), tok

    _, toks = jax.lax.scan(step, (cache, logits), jnp.arange(n, n + n_new))
    return toks.T
