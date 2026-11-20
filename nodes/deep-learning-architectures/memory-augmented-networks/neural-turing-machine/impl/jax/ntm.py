# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-turing-machine/ (ai://P3DWR1).
"""A Neural Turing Machine with a feedforward controller, the JAX reference.

The steps run in jax.lax.scan, with the state (memory, weightings, read vectors) as the carry: the weightings are
one array (R + W, B, N) and the read vectors one array (R, B, M). The squared lengths in the cosine have a floor
before the square root, as in the PyTorch reference, so that a row of zeros does not make the gradient NaN.
"""

import math

import jax
import jax.numpy as jnp

EPS = 1e-8
TINY = 1e-30
MEMORY_INIT = 1e-6
NAMES = ("Wc", "bc", "Wh", "bh", "Wy", "by")


def config(x_dim=4, y_dim=3, hidden=8, n_loc=6, width=4, n_read=1, n_write=1, shift=1):
    if min(x_dim, y_dim, hidden, n_loc, width, n_read) < 1 or n_write < 0 or shift < 0 or 2 * shift + 1 > n_loc:
        raise ValueError("the sizes must be positive, n_write >= 0, and 2 shift + 1 <= n_loc")
    return {
        "x_dim": x_dim,
        "y_dim": y_dim,
        "hidden": hidden,
        "n_loc": n_loc,
        "width": width,
        "n_read": n_read,
        "n_write": n_write,
        "shift": shift,
    }


def n_head_outputs(cfg):
    read = cfg["width"] + 3 + 2 * cfg["shift"] + 1
    return cfg["n_read"] * read + cfg["n_write"] * (read + 2 * cfg["width"])


def param_count(cfg):
    x, y, h, m, nr = cfg["x_dim"], cfg["y_dim"], cfg["hidden"], cfg["width"], cfg["n_read"]
    return h * (x + nr * m) + h + n_head_outputs(cfg) * (h + 1) + y * (h + nr * m) + y


def init(key, cfg, scale=1.0):
    x, y, h, m, nr = cfg["x_dim"], cfg["y_dim"], cfg["hidden"], cfg["width"], cfg["n_read"]
    p, z = n_head_outputs(cfg), x + nr * m
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "Wc": scale * jax.random.normal(k1, (h, z)) / math.sqrt(z),
        "bc": jnp.zeros(h),
        "Wh": scale * jax.random.normal(k2, (p, h)) / math.sqrt(h),
        "bh": jnp.zeros(p),
        "Wy": scale * jax.random.normal(k3, (y, h + nr * m)) / math.sqrt(h + nr * m),
        "by": jnp.zeros(y),
    }


def safe_norm(v):
    return jnp.sqrt(jnp.maximum((v * v).sum(-1), TINY))


def cosine(memory, key):
    dots = jnp.einsum("bnm,bm->bn", memory, key)
    return dots / (safe_norm(memory) * safe_norm(key)[:, None] + EPS)


def content_weights(memory, key, beta):
    return jax.nn.softmax(beta[:, None] * cosine(memory, key), axis=-1)


def interpolate(wc, w_prev, g):
    return g[:, None] * wc + (1.0 - g[:, None]) * w_prev


def shift(w, s):
    big_s = (s.shape[-1] - 1) // 2
    rolled = jnp.stack([jnp.roll(w, k, axis=-1) for k in range(-big_s, big_s + 1)], axis=1)
    return jnp.einsum("bkn,bk->bn", rolled, s)


def sharpen(w, gamma):
    p = w ** gamma[:, None]
    return p / p.sum(-1, keepdims=True)


def address(memory, w_prev, head):
    wc = content_weights(memory, head["key"], head["beta"])
    return sharpen(shift(interpolate(wc, w_prev, head["g"]), head["s"]), head["gamma"])


def read(memory, w):
    return jnp.einsum("bn,bnm->bm", w, memory)


def write(memory, ws, erases, adds):
    keep = jnp.ones_like(memory)
    for w, e in zip(ws, erases, strict=True):
        keep = keep * (1.0 - w[:, :, None] * e[:, None, :])
    out = memory * keep
    for w, a in zip(ws, adds, strict=True):
        out = out + w[:, :, None] * a[:, None, :]
    return out


def split_heads(o, cfg):
    m, k = cfg["width"], 2 * cfg["shift"] + 1
    heads, i = [], 0
    for j in range(cfg["n_read"] + cfg["n_write"]):
        hd = {
            "key": jnp.tanh(o[:, i : i + m]),
            "beta": jax.nn.softplus(o[:, i + m]),
            "g": jax.nn.sigmoid(o[:, i + m + 1]),
            "s": jax.nn.softmax(o[:, i + m + 2 : i + m + 2 + k], axis=-1),
            "gamma": 1.0 + jax.nn.softplus(o[:, i + m + 2 + k]),
        }
        i += m + 3 + k
        if j >= cfg["n_read"]:
            hd["erase"], hd["add"] = jax.nn.sigmoid(o[:, i : i + m]), jnp.tanh(o[:, i + m : i + 2 * m])
            i += 2 * m
        heads.append(hd)
    return heads


def initial_state(cfg, batch, dtype=jnp.float64):
    heads = cfg["n_read"] + cfg["n_write"]
    ws = jnp.zeros((heads, batch, cfg["n_loc"]), dtype).at[:, :, 0].set(1.0)
    memory = jnp.full((batch, cfg["n_loc"], cfg["width"]), MEMORY_INIT, dtype)
    return memory, ws, jnp.zeros((cfg["n_read"], batch, cfg["width"]), dtype)


def step(params, state, x_t, cfg):
    memory, ws, reads = state
    nr = cfg["n_read"]
    hid = jnp.tanh(jnp.concatenate([x_t] + list(reads), axis=-1) @ params["Wc"].T + params["bc"])
    heads = split_heads(hid @ params["Wh"].T + params["bh"], cfg)
    w_write = [address(memory, ws[nr + j], hd) for j, hd in enumerate(heads[nr:])]
    memory = write(memory, w_write, [hd["erase"] for hd in heads[nr:]], [hd["add"] for hd in heads[nr:]])
    w_read = [address(memory, ws[j], hd) for j, hd in enumerate(heads[:nr])]
    new_reads = jnp.stack([read(memory, w) for w in w_read])
    y = jnp.concatenate([hid] + list(new_reads), axis=-1) @ params["Wy"].T + params["by"]
    return (memory, jnp.stack(w_read + w_write), new_reads), y


def forward(params, x, cfg, trace=False):
    """x (B, T, X) to logits (B, T, Y); with trace, also the weightings (T, R + W, B, N) and the last memory."""

    def body(state, x_t):
        state, y = step(params, state, x_t, cfg)
        return state, (y, state[1])

    state, (ys, ws) = jax.lax.scan(body, initial_state(cfg, x.shape[0], x.dtype), jnp.swapaxes(x, 0, 1))
    logits = jnp.swapaxes(ys, 0, 1)
    return (logits, ws, state[0]) if trace else logits


def bits_per_sequence(logits, targets, mask):
    nats = jax.nn.softplus(logits) - targets * logits
    return (nats * mask[:, :, None]).sum() / logits.shape[0] / math.log(2.0)
