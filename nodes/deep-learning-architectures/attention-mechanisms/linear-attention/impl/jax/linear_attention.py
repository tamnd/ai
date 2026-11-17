# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/linear-attention/ (ai://NEW-linear-attention).
"""Linear attention in JAX, as pure functions of the parameters.

The parameters have the names and the layout of the NumPy reference, so the shared fixture loads as it is. Causal
attention comes three ways: causal_scan carries the state (S, z) of eq-rnn through jax.lax.scan, causal_chunked
splits the sequence into chunks of length T and adds, inside each chunk, the masked T x T products to the state left
by the chunks before it (eq-chunk on the page), and causal_quadratic is the masked N x N form. The gradients are left
to jax.grad.
"""

import jax
import jax.numpy as jnp


def elu_plus_one(x):
    return jax.nn.elu(x) + 1.0


def noncausal(fQ, fK, V):
    return (fQ @ (jnp.swapaxes(fK, -1, -2) @ V)) / (fQ @ fK.sum(-2)[..., None])


def causal_quadratic(fQ, fK, V):
    A = jnp.tril(fQ @ jnp.swapaxes(fK, -1, -2))
    return (A @ V) / A.sum(-1, keepdims=True)


def causal_scan(fQ, fK, V):
    """eq-rnn over the positions; fQ, fK (..., N, C) and V (..., N, M)."""
    S0 = jnp.zeros(fK.shape[:-2] + (fK.shape[-1], V.shape[-1]), V.dtype)
    z0 = jnp.zeros(fK.shape[:-2] + (fK.shape[-1],), V.dtype)

    def f(carry, qkv):
        S, z = carry
        q, k, v = qkv
        S, z = S + k[..., :, None] * v[..., None, :], z + k
        return (S, z), jnp.einsum("...c,...cm->...m", q, S) / jnp.einsum("...c,...c->...", q, z)[..., None]

    move = lambda a: jnp.moveaxis(a, -2, 0)  # noqa: E731
    _, out = jax.lax.scan(f, (S0, z0), (move(fQ), move(fK), move(V)))
    return jnp.moveaxis(out, 0, -2)


def causal_chunked(fQ, fK, V, T):
    """eq-chunk: N / T steps, each of T^2 (C + M) work inside the chunk plus T C M against the carried state."""
    *lead, N, C = fQ.shape
    Mv = V.shape[-1]
    if N % T:
        raise ValueError(f"N = {N} is not a multiple of the chunk length {T}")
    r = lambda a: jnp.moveaxis(a.reshape(*lead, N // T, T, a.shape[-1]), -3, 0)  # noqa: E731
    mask = jnp.tril(jnp.ones((T, T), bool))

    def f(carry, qkv):
        S, z = carry
        q, k, v = qkv
        A = jnp.where(mask, q @ jnp.swapaxes(k, -1, -2), 0.0)
        num = q @ S + A @ v
        den = q @ z[..., None] + A.sum(-1, keepdims=True)
        S = S + jnp.swapaxes(k, -1, -2) @ v
        z = z + k.sum(-2)
        return (S, z), num / den

    init = (jnp.zeros((*lead, C, Mv), V.dtype), jnp.zeros((*lead, C), V.dtype))
    _, out = jax.lax.scan(f, init, (r(fQ), r(fK), r(V)))
    return jnp.moveaxis(out, 0, -3).reshape(*lead, N, Mv)


def split(x, H):
    B, N, HD = x.shape
    return jnp.swapaxes(x.reshape(B, N, H, HD // H), 1, 2)


def forward(p, x, n_heads=2, causal=True, impl="scan", chunk=2):
    q, k, v = (split(x @ p[w], n_heads) for w in ("W_q", "W_k", "W_v"))
    fQ, fK = elu_plus_one(q), elu_plus_one(k)
    if not causal:
        a = noncausal(fQ, fK, v)
    elif impl == "scan":
        a = causal_scan(fQ, fK, v)
    elif impl == "chunked":
        a = causal_chunked(fQ, fK, v, chunk)
    else:
        a = causal_quadratic(fQ, fK, v)
    B, H, N, Mv = a.shape
    return jnp.swapaxes(a, 1, 2).reshape(B, N, H * Mv) @ p["W_o"]


def loss(p, x, target, **kw):
    return 0.5 * jnp.mean((forward(p, x, **kw) - target) ** 2)
