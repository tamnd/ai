# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/jamba/ (ai://2V1JK1).
"""Jamba of Lieber et al. (2024) in JAX, as pure functions of the parameters.

The parameters have the names and the layout of the NumPy reference, so the shared fixture loads as it is. Two parts
differ in how they compute, not in what. The selective scan of the Mamba layers runs with jax.lax.associative_scan,
since the maps h -> a h + b compose associatively. The mixture of experts dispatches the tokens to the experts with
fixed-size arrays, as compiled code must: each expert has a buffer of C slots, a one-hot (M, K, n, C) tensor puts the
k-th choice of token m into slot c of expert j, the experts run as one batched product on their (n, C, D) buffers,
and the same tensor, weighted by the router probabilities, brings the outputs back. With C = M no choice is dropped
and the result is that of the reference. A smaller C, the capacity factor of GShard and the Switch Transformer,
drops the choices that come after a full buffer; Jamba's paper does not say whether its training drops any. The
gradients are left to jax.grad.
"""

import math

import jax
import jax.numpy as jnp


def config(vocab=13, d_model=8, n_blocks=1, block_len=4, moe_every=2, n_experts=4, top_k=2, d_ff=12, n_heads=2,
           n_kv=1, d_state=4, d_conv=3, expand=2, dt_rank=2, aux=0.01, eps=1e-6, capacity=None):  # fmt: skip
    """The config of the NumPy reference, and capacity: the slots of each expert's buffer as a multiple of the even
    share M K / n, or None for M slots, which drops nothing."""
    return {"vocab": vocab, "d_model": d_model, "n_blocks": n_blocks, "block_len": block_len, "moe_every": moe_every,
            "n_experts": n_experts, "top_k": top_k, "d_ff": d_ff, "n_heads": n_heads, "n_kv": n_kv,
            "head_dim": d_model // n_heads, "d_state": d_state, "d_conv": d_conv, "expand": expand,
            "d_inner": expand * d_model, "dt_rank": dt_rank, "n_layers": n_blocks * block_len, "aux": aux,
            "eps": eps, "capacity": capacity}  # fmt: skip


def layers(cfg):
    """eq-layout: attention at layer l // 2 of every block, a mixture of experts at every e-th layer from e - 1."""
    l, e = cfg["block_len"], cfg["moe_every"]
    return [
        ("attn" if i % l == l // 2 else "mamba", "moe" if i % e == e - 1 else "mlp") for i in range(cfg["n_layers"])
    ]


def rmsnorm(x, w, eps):
    return x * jax.lax.rsqrt((x * x).mean(-1, keepdims=True) + eps) * w


def attention(p, cfg, u, q_):
    """eq-attn: causal grouped-query self-attention with no positional encoding."""
    B, T, _ = u.shape
    G, A = cfg["n_kv"], cfg["head_dim"]
    q = (u @ p[q_ + "wq"]).reshape(B, T, G, -1, A)
    k, v = (u @ p[q_ + "wk"]).reshape(B, T, G, A), (u @ p[q_ + "wv"]).reshape(B, T, G, A)
    s = jnp.einsum("btgqd,bsgd->bgqts", q, k) / math.sqrt(A)
    a = jax.nn.softmax(jnp.where(jnp.tril(jnp.ones((T, T), bool)), s, -jnp.inf), -1)
    return jnp.einsum("bgqts,bsgd->btgqd", a, v).reshape(B, T, -1) @ p[q_ + "wo"]


def combine(left, right):
    """(a1, b1) then (a2, b2) is h -> a2 (a1 h + b1) + b2."""
    return left[0] * right[0], right[0] * left[1] + right[1]


def mamba(p, cfg, u, q_):
    """eq-mamba, with RMSNorm on Delta's low-rank input, B and C, and the scan as an associative scan."""
    I, N, R, K, eps = (cfg[k] for k in ("d_inner", "d_state", "dt_rank", "d_conv", "eps"))
    T = u.shape[1]
    xz = u @ p[q_ + "in_proj"]
    x, z = xz[..., :I], xz[..., I:]
    xp = jnp.pad(x, ((0, 0), (K - 1, 0), (0, 0)))
    xa = jax.nn.silu(p[q_ + "conv_b"] + sum(p[q_ + "conv_w"][:, k] * xp[:, k : k + T] for k in range(K)))
    dbl = xa @ p[q_ + "x_proj"]
    dt = rmsnorm(dbl[..., :R], p[q_ + "dt_norm"], eps)
    Bm, C = rmsnorm(dbl[..., R : R + N], p[q_ + "B_norm"], eps), rmsnorm(dbl[..., R + N :], p[q_ + "C_norm"], eps)
    dt = jax.nn.softplus(dt @ p[q_ + "dt_w"] + p[q_ + "dt_b"])
    dA = jnp.exp(dt[..., None] * -jnp.exp(p[q_ + "A_log"]))
    _, hs = jax.lax.associative_scan(combine, (dA, dt[..., None] * Bm[:, :, None] * xa[..., None]), axis=1)
    y = jnp.einsum("btin,btn->bti", hs, C) + p[q_ + "D"] * xa
    return (y * jax.nn.silu(z)) @ p[q_ + "out_proj"]


def swiglu(x, wg, wu, wd):
    return (jax.nn.silu(x @ wg) * (x @ wu)) @ wd


def dispatch(top, n, C):
    """The one-hot (M, K, n, C) tensor that puts choice k of token m in slot c of expert j, the choices taken in the
    order of the tokens, then of k; a choice past the C-th of its expert has no slot and is dropped."""
    M, K = top.shape
    hot = jax.nn.one_hot(top.reshape(-1), n, dtype=jnp.int32)  # (M K, n)
    slot = (jnp.cumsum(hot, 0) - 1) * hot  # the place of each choice in its expert's queue
    return (hot[..., None] * jax.nn.one_hot(slot, C, dtype=jnp.int32)).reshape(M, K, n, C)


def moe(p, cfg, v, q_):
    """eq-moe and eq-aux, by dispatch to fixed-size buffers. Returns the output, the balance term, and the number of
    choices dropped."""
    n, K = cfg["n_experts"], cfg["top_k"]
    x = v.reshape(-1, v.shape[-1])
    M = x.shape[0]
    C = M if cfg["capacity"] is None else max(1, math.ceil(cfg["capacity"] * M * K / n))
    probs = jax.nn.softmax(x @ p[q_ + "router"], -1)
    w, top = jax.lax.top_k(probs, K)
    d = dispatch(top, n, C).astype(x.dtype)
    buf = jnp.einsum("mkjc,md->jcd", d, x)  # (n, C, D)
    h = jax.nn.silu(jnp.einsum("jcd,jdf->jcf", buf, p[q_ + "e_gate"])) * jnp.einsum("jcd,jdf->jcf", buf, p[q_ + "e_up"])
    y = jnp.einsum("jcf,jfd->jcd", h, p[q_ + "e_down"])
    out = jnp.einsum("mkjc,mk,jcd->md", d, w, y)
    f = jax.nn.one_hot(top.reshape(-1), n).mean(0)
    dropped = M * K - d.sum().astype(jnp.int32)
    return out.reshape(v.shape), cfg["aux"] * n * (f * probs.mean(0)).sum(), dropped


def forward(p, cfg, tokens, kinds=None):
    """The (B, T, V) scores, the sum of the balance terms, and the number of dropped choices. kinds replaces the
    mixers of the layout by any list of "attn" and "mamba"."""
    lay = layers(cfg) if kinds is None else [(m, f) for m, (_, f) in zip(kinds, layers(cfg), strict=True)]
    r, aux, dropped, eps = p["embed"][tokens], 0.0, 0, cfg["eps"]
    for i, (m, f) in enumerate(lay):
        q_ = f"l{i}_"
        r = r + (attention if m == "attn" else mamba)(p, cfg, rmsnorm(r, p[q_ + "norm1"], eps), q_)
        v = rmsnorm(r, p[q_ + "norm2"], eps)
        if f == "moe":
            out, a, d = moe(p, cfg, v, q_)
            aux, dropped = aux + a, dropped + d
        else:
            out = swiglu(v, p[q_ + "w_gate"], p[q_ + "w_up"], p[q_ + "w_down"])
        r = r + out
    return rmsnorm(r, p["norm_f"], eps) @ p["head"], aux, dropped


def nll(logits, targets):
    logp = jax.nn.log_softmax(logits, -1)
    return -jnp.take_along_axis(logp, targets[..., None], -1).mean()


def loss(p, cfg, tokens):
    """eq-loss: the cross-entropy of tokens[:, 1:] given tokens[:, :-1], plus the balance terms."""
    logits, aux, _ = forward(p, cfg, tokens[:, :-1])
    return nll(logits, tokens[:, 1:]) + aux
