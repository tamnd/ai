# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/mamba/ (ai://2WYA9X).
"""Mamba of Gu and Dao (2023) in JAX, as pure functions of the parameters.

The parameters have the names and the layout of the NumPy reference, so the shared fixture loads as it is. The
selective scan h_t = A_bar_t h_{t-1} + B_bar_t x_t is a first-order linear recurrence, and the pairs (a, b) that map
h to a h + b compose associatively, (a1, b1) then (a2, b2) is (a1 a2, a2 b1 + b2), so jax.lax.associative_scan runs
it in O(log T) parallel steps; this is the parallel scan of the paper's Section 3.3.2 (Blelloch 1990), without its
kernel fusion. scan="sequential" runs the same recurrence with jax.lax.scan. step is one token of generation from a
constant-size state. The gradients are left to jax.grad.
"""

import math

import jax
import jax.numpy as jnp

DISCS = ("simple", "zoh")
SCANS = ("associative", "sequential")


def config(vocab=11, d_model=8, n_layers=2, d_state=4, d_conv=3, expand=2, dt_rank=None, disc="simple", eps=1e-5):
    """Return the config; the defaults are the example config of shapes.yaml, and dt_rank defaults to ceil(D / 16)
    as in the reference code."""
    if disc not in DISCS:
        raise ValueError(f"disc must be one of {DISCS}")
    R = math.ceil(d_model / 16) if dt_rank is None else dt_rank
    return {"vocab": vocab, "d_model": d_model, "n_layers": n_layers, "d_state": d_state, "d_conv": d_conv,
            "expand": expand, "d_inner": expand * d_model, "dt_rank": R, "disc": disc, "eps": eps}  # fmt: skip


def param_shapes(cfg):
    D, I, N, K, R, V = (cfg[k] for k in ("d_model", "d_inner", "d_state", "d_conv", "dt_rank", "vocab"))
    block = {"norm": (D,), "in_proj": (D, 2 * I), "conv_w": (I, K), "conv_b": (I,), "x_proj": (I, R + 2 * N),
             "dt_w": (R, I), "dt_b": (I,), "A_log": (I, N), "D": (I,), "out_proj": (I, D)}  # fmt: skip
    shapes = {"embed": (V, D), "norm_f": (D,)}
    for l in range(cfg["n_layers"]):
        shapes |= {f"l{l}_{k}": s for k, s in block.items()}
    return shapes


def param_count(cfg):
    D, I, N, K, R, V = (cfg[k] for k in ("d_model", "d_inner", "d_state", "d_conv", "dt_rank", "vocab"))
    return V * D + D + cfg["n_layers"] * (3 * I * D + I * (K + 1) + I * (R + 2 * N) + (R + 1) * I + I * N + I + D)


def init(key, cfg, scale=0.1, dt_min=1e-3, dt_max=1e-1):
    """As the NumPy init: N(0, scale^2) weights, A_log = log(1..N), D = 1, softplus(dt_b) log-uniform."""
    shapes = param_shapes(cfg)
    keys = jax.random.split(key, len(shapes) + cfg["n_layers"])
    p = {k: scale * jax.random.normal(kk, s) for kk, (k, s) in zip(keys, shapes.items())}
    I, N = cfg["d_inner"], cfg["d_state"]
    p["norm_f"] = jnp.ones(cfg["d_model"])
    for l in range(cfg["n_layers"]):
        dt = jnp.exp(jax.random.uniform(keys[len(shapes) + l], (I,), minval=math.log(dt_min), maxval=math.log(dt_max)))
        p[f"l{l}_dt_b"] = dt + jnp.log(-jnp.expm1(-dt))
        p[f"l{l}_A_log"] = jnp.log(jnp.tile(jnp.arange(1.0, N + 1), (I, 1)))
        p[f"l{l}_D"] = jnp.ones(I)
        p[f"l{l}_norm"] = jnp.ones(cfg["d_model"])
    return p


def rmsnorm(x, w, eps):
    return x * jax.lax.rsqrt((x * x).mean(-1, keepdims=True) + eps) * w


def causal_conv(x, w, b):
    K, T = w.shape[1], x.shape[1]
    xp = jnp.pad(x, ((0, 0), (K - 1, 0), (0, 0)))
    return b + sum(w[:, k] * xp[:, k : k + T] for k in range(K))


def discretize(dt, A, Bm, disc):
    """A_bar and B_bar, (..., I, N), from Delta (..., I), A (I, N) and B (..., N)."""
    dA = jnp.exp(dt[..., None] * A)
    if disc == "simple":
        return dA, dt[..., None] * Bm[..., None, :]
    return dA, (dA - 1.0) / A * Bm[..., None, :]


def combine(left, right):
    """The composition of h -> a1 h + b1 followed by h -> a2 h + b2."""
    a1, b1 = left
    a2, b2 = right
    return a1 * a2, a2 * b1 + b2


def selective_scan(dA, dB, x, C, scan="associative"):
    """eq-scan over the positions (axis 1); returns y (B, T, I) and the states (B, T, I, N)."""
    b = dB * x[..., None]
    if scan == "associative":
        _, hs = jax.lax.associative_scan(combine, (dA, b), axis=1)
    else:
        def body(h, ab):
            h = ab[0] * h + ab[1]
            return h, h

        _, hs = jax.lax.scan(body, jnp.zeros_like(b[:, 0]), (dA.swapaxes(0, 1), b.swapaxes(0, 1)))
        hs = hs.swapaxes(0, 1)
    return jnp.einsum("btin,btn->bti", hs, C), hs


def mixer(p, cfg, u, pre, scan="associative"):
    I, N, R = cfg["d_inner"], cfg["d_state"], cfg["dt_rank"]
    xz = u @ p[pre + "in_proj"]
    x, z = xz[..., :I], xz[..., I:]
    xa = jax.nn.silu(causal_conv(x, p[pre + "conv_w"], p[pre + "conv_b"]))
    dbl = xa @ p[pre + "x_proj"]
    dt = jax.nn.softplus(dbl[..., :R] @ p[pre + "dt_w"] + p[pre + "dt_b"])
    dA, dB = discretize(dt, -jnp.exp(p[pre + "A_log"]), dbl[..., R : R + N], cfg["disc"])
    y, hs = selective_scan(dA, dB, xa, dbl[..., R + N :], scan)
    return ((y + p[pre + "D"] * xa) * jax.nn.silu(z)) @ p[pre + "out_proj"], hs


def forward(p, cfg, tokens, scan="associative", states=False):
    """The (B, T, V) next-token scores of (B, T) token ids; with states=True also each block's SSM states."""
    r, hss = p["embed"][tokens], []
    for l in range(cfg["n_layers"]):
        out, hs = mixer(p, cfg, rmsnorm(r, p[f"l{l}_norm"], cfg["eps"]), f"l{l}_", scan)
        r, hss = r + out, hss + [hs]
    logits = rmsnorm(r, p["norm_f"], cfg["eps"]) @ p["embed"].T
    return (logits, hss) if states else logits


def nll(logits, targets):
    logp = jax.nn.log_softmax(logits, -1)
    return -jnp.take_along_axis(logp, targets[..., None], -1).mean()


def loss(p, cfg, tokens, scan="associative"):
    return nll(forward(p, cfg, tokens[:, :-1], scan), tokens[:, 1:])


def init_state(cfg, B):
    I = cfg["d_inner"]
    return [(jnp.zeros((B, cfg["d_conv"] - 1, I)), jnp.zeros((B, I, cfg["d_state"]))) for _ in range(cfg["n_layers"])]


def step(p, cfg, token, state):
    """One token of generation: (B,) ids and the state give the (B, V) scores and the next state."""
    I, N, R = cfg["d_inner"], cfg["d_state"], cfg["dt_rank"]
    r, new = p["embed"][token], []
    for l, (cs, h) in enumerate(state):
        q = f"l{l}_"
        xz = rmsnorm(r, p[q + "norm"], cfg["eps"]) @ p[q + "in_proj"]
        x, z = xz[:, :I], xz[:, I:]
        win = jnp.concatenate([cs, x[:, None]], 1)
        xa = jax.nn.silu(p[q + "conv_b"] + jnp.einsum("bki,ik->bi", win, p[q + "conv_w"]))
        dbl = xa @ p[q + "x_proj"]
        dt = jax.nn.softplus(dbl[:, :R] @ p[q + "dt_w"] + p[q + "dt_b"])
        dA, dB = discretize(dt, -jnp.exp(p[q + "A_log"]), dbl[:, R : R + N], cfg["disc"])
        h = dA * h + dB * xa[..., None]
        y = jnp.einsum("bin,bn->bi", h, dbl[:, R + N :]) + p[q + "D"] * xa
        r = r + (y * jax.nn.silu(z)) @ p[q + "out_proj"]
        new.append((win[:, 1:], h))
    return rmsnorm(r, p["norm_f"], cfg["eps"]) @ p["embed"].T, new
