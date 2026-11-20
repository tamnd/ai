# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/t5/ (ai://NEW-t5).
"""The T5 encoder-decoder Transformer in JAX, as pure functions of the parameters.

The parameters have the names and the layout of the NumPy reference, so the shared fixture loads as it is. forward()
runs the encoder on the source and the decoder on the shifted target with the causal mask; the gradient of eq-loss is
left to jax.grad. Decoding encodes the source once, computes the keys and values of the attention to the encoder once
per layer, and keeps the self-attention keys and values in a cache of fixed size, allocated for max_len tokens, so
that every step has the same shapes and compiles once: decode_step() writes the key and value of the token at
position t with dynamic_update_slice, computes the position biases of row t from the offsets, and masks the places
after t. generate() runs the greedy steps through jax.lax.scan.
"""

import math
from typing import Any

import jax
import jax.numpy as jnp

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
    return sum(math.prod(s) for s in param_shapes(cfg).values())


def init(key, cfg, scale=0.5):
    """Normal weights of standard deviation scale / sqrt(fan-in), gains of one, and position tables of scale."""
    shapes = param_shapes(cfg)
    keys = jax.random.split(key, len(shapes))
    p = {}
    for k, (name, shape) in zip(keys, shapes.items(), strict=True):
        if len(shape) == 1:
            p[name] = jnp.ones(shape)
        elif name.endswith("_rel"):
            p[name] = scale * jax.random.normal(k, shape)
        else:
            p[name] = scale / math.sqrt(shape[1] if name == "emb" else shape[0]) * jax.random.normal(k, shape)
    return p


def bucket(offset, bidirectional, buckets, max_distance):
    """The bucket of each offset key - query, as in the NumPy reference."""
    out = jnp.zeros_like(offset)
    if bidirectional:
        buckets //= 2
        out = out + (offset > 0) * buckets
        n = jnp.abs(offset)
    else:
        n = jnp.maximum(-offset, 0)
    exact = buckets // 2
    far = exact + jnp.log(jnp.maximum(n, 1) / exact) / math.log(max_distance / exact) * (buckets - exact)
    far = jnp.minimum(far.astype(offset.dtype), buckets - 1)
    return out + jnp.where(n < exact, n, far)


def bias(p, cfg, table, q_pos, k_pos, bidirectional):
    """The (H, nq, nk) position biases of the query positions q_pos on the key positions k_pos."""
    b = bucket(k_pos[None, :] - q_pos[:, None], bidirectional, cfg["buckets"], cfg["max_distance"])
    return p[table][b].transpose(2, 0, 1)


def rms_norm(x, g):
    """eq-norm: x / sqrt(mean(x^2) + EPS) * g."""
    return x * jax.lax.rsqrt((x * x).mean(-1, keepdims=True) + EPS) * g


def heads(x, cfg):
    B, n, _ = x.shape
    return x.reshape(B, n, cfg["n_heads"], cfg["d_kv"]).transpose(0, 2, 1, 3)


def attend(q, k, v, extra=None):
    """Softmax attention of the heads q (B, H, nq, d) on k, v (B, H, nk, d), plus extra (H, nq, nk) on the scores,
    -inf where a key is hidden; the heads merged into (B, nq, H d)."""
    s = q @ k.transpose(0, 1, 3, 2) / math.sqrt(q.shape[-1])
    if extra is not None:
        s = s + extra
    o = jax.nn.softmax(s, -1) @ v
    B, H, n, d = o.shape
    return o.transpose(0, 2, 1, 3).reshape(B, n, H * d)


def ff(p, pre, x):
    return jax.nn.relu(x @ p[pre + "wi"]) @ p[pre + "wo"]


def encode(p, cfg, src):
    """eq-enc: the encoder output (B, S, M) of the source ids."""
    z = p["emb"][src]
    pos = jnp.arange(src.shape[1])
    extra = bias(p, cfg, "enc_rel", pos, pos, True) if cfg["pos"] == "relative" else None
    for i in range(cfg["n_layers"]):
        h = rms_norm(z, p[f"e{i}_ln1"])
        q, k, v = (heads(h @ p[f"e{i}_{w}"], cfg) for w in "qkv")
        z = z + attend(q, k, v, extra) @ p[f"e{i}_o"]
        z = z + ff(p, f"e{i}_", rms_norm(z, p[f"e{i}_ln2"]))
    return rms_norm(z, p["enc_ln"])


def cross_keys_values(p, cfg, e):
    """The keys and values of the attention to the encoder output, one pair per layer, computed once."""
    return [(heads(e @ p[f"d{i}_ck"], cfg), heads(e @ p[f"d{i}_cv"], cfg)) for i in range(cfg["n_layers"])]


def decode(p, cfg, e, ids):
    """eq-dec and eq-logits: the logits (B, T, V) of the decoder inputs ids, the whole sequence at once."""
    return decoder(p, cfg, e, p["emb"][ids])


def decoder(p, cfg, e, y):
    """The decoder from its input embeddings y (B, T, M) to the logits."""
    pos = jnp.arange(y.shape[1])
    extra = jnp.where(pos[None, :] <= pos[:, None], 0.0, -jnp.inf)
    if cfg["pos"] == "relative":
        extra = extra + bias(p, cfg, "dec_rel", pos, pos, False)
    for i, (ck, cv) in enumerate(cross_keys_values(p, cfg, e)):
        h = rms_norm(y, p[f"d{i}_ln1"])
        q, k, v = (heads(h @ p[f"d{i}_s{w}"], cfg) for w in "qkv")
        y = y + attend(q, k, v, extra) @ p[f"d{i}_so"]
        y = y + attend(heads(rms_norm(y, p[f"d{i}_ln2"]) @ p[f"d{i}_cq"], cfg), ck, cv) @ p[f"d{i}_co"]
        y = y + ff(p, f"d{i}_", rms_norm(y, p[f"d{i}_ln3"]))
    return rms_norm(y, p["dec_ln"]) @ p["emb"].T


def shift_right(tgt, start_id=0):
    return jnp.concatenate([jnp.full_like(tgt[:, :1], start_id), tgt[:, :-1]], 1)


def forward(p, cfg, src, tgt):
    """The logits (B, T, V) of every target token under teacher forcing."""
    return decode(p, cfg, encode(p, cfg, src), shift_right(tgt))


def nll(logits, tgt):
    logp = jax.nn.log_softmax(logits, -1)
    return -jnp.take_along_axis(logp, tgt[..., None], -1).mean()


def loss(p, cfg, src, tgt):
    """eq-loss."""
    return nll(forward(p, cfg, src, tgt), tgt)


def init_cache(cfg, B, max_len):
    """Zeros for the self-attention keys and values of every layer, max_len positions each."""
    z = jnp.zeros((B, cfg["n_heads"], max_len, cfg["d_kv"]))
    return [(z, z) for _ in range(cfg["n_layers"])]


def decode_step(p, cfg, cross, cache, tok, t):
    """One decoder step for the tokens tok (B,) at position t: the logits (B, V) and the cache with the keys and
    values of position t written in. The shapes do not depend on t, so one compilation serves every step."""
    max_len = cache[0][0].shape[2]
    k_pos = jnp.arange(max_len)
    extra = jnp.where(k_pos <= t, 0.0, -jnp.inf)[None, None, :]
    if cfg["pos"] == "relative":
        extra = extra + bias(p, cfg, "dec_rel", jnp.reshape(t, (1,)), k_pos, False)
    y = p["emb"][tok][:, None]
    new = []
    for i, ((ck, cv), (K, V)) in enumerate(zip(cross, cache, strict=True)):
        h = rms_norm(y, p[f"d{i}_ln1"])
        q, k, v = (heads(h @ p[f"d{i}_s{w}"], cfg) for w in "qkv")
        K = jax.lax.dynamic_update_slice(K, k, (0, 0, t, 0))
        V = jax.lax.dynamic_update_slice(V, v, (0, 0, t, 0))
        new.append((K, V))
        y = y + attend(q, K, V, extra) @ p[f"d{i}_so"]
        y = y + attend(heads(rms_norm(y, p[f"d{i}_ln2"]) @ p[f"d{i}_cq"], cfg), ck, cv) @ p[f"d{i}_co"]
        y = y + ff(p, f"d{i}_", rms_norm(y, p[f"d{i}_ln3"]))
    return (rms_norm(y, p["dec_ln"]) @ p["emb"].T)[:, 0], new


def generate(p, cfg, src, n_new, start_id=0):
    """Greedy decoding through lax.scan: the (B, n_new) tokens after the start token."""
    e = encode(p, cfg, src)
    cross = cross_keys_values(p, cfg, e)
    B = src.shape[0]

    def step(carry, t):
        cache, tok = carry
        logits, cache = decode_step(p, cfg, cross, cache, tok, t)
        nxt = logits.argmax(-1)
        return (cache, nxt), nxt

    start = jnp.full((B,), start_id, dtype=src.dtype)
    _, out = jax.lax.scan(step, (init_cache(cfg, B, n_new), start), jnp.arange(n_new))
    return out.T
