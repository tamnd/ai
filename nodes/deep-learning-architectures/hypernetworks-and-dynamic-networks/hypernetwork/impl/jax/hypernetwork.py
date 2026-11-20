# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/hypernetwork/ (ai://NEW-hypernetwork).
"""Hypernetworks in JAX, as pure functions of the parameters.

The parameters have the names and the layout of the NumPy reference, so the shared fixture loads as it is. forward
is the static hypernetwork with the network of the shape contract; hyper_rnn is eq-scaled run by jax.lax.scan, with
the hyper cell's new state giving the embeddings; hyper_lstm is the HyperLSTM of the paper's eqs. 10 to 13 without
the dropout. The gradients are left to jax.grad.
"""

import jax
import jax.numpy as jnp


def generate(p, z):
    """eq-static: z (E, N_z) to basic kernels (E, N_out, N_in, f, f)."""
    a = jnp.einsum("idz,ez->eid", p["W_in"], z) + p["B_in"]
    K = jnp.einsum("uwd,eid->eiuw", p["W_out"], a) + p["B_out"]
    E, Nin, f, w = K.shape
    return K.reshape(E, Nin, f, w // f, f).transpose(0, 3, 1, 2, 4)


def tile(basic, m):
    E, N, _, f, _ = basic.shape
    return basic.reshape(m, m, N, N, f, f).transpose(1, 2, 0, 3, 4, 5).reshape(m * N, m * N, f, f)


def conv(x, w, b):
    pad = w.shape[2] // 2
    y = jax.lax.conv_general_dilated(x, w, (1, 1), [(pad, pad), (pad, pad)], dimension_numbers=("NCHW", "OIHW", "NCHW"))
    return y + b[None, :, None, None]


def forward(p, x, mult=2, depth=2):
    basic = generate(p, p["z"])
    h = jax.nn.relu(conv(x, p["stem_w"], p["stem_b"]))
    for j in range(depth):
        w = tile(basic[j * mult * mult : (j + 1) * mult * mult], mult)
        h = jax.nn.relu(conv(h, w, p[f"conv{j}_b"]))
    return h.mean((2, 3)) @ p["head_w"] + p["head_b"]


def loss(p, x, y, **kw):
    logits = forward(p, x, **kw)
    return -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(logits), y[:, None], 1))


def hyper_rnn(p, x):
    """eq-scaled over x (B, T, N_x); returns the states (B, T, N_h)."""
    B = x.shape[0]
    Nh, Nhat = p["W_h"].shape[0], p["W_hhat"].shape[0]

    def step(carry, xt):
        h, hhat = carry
        xhat = jnp.concatenate([h, xt], -1)
        hhat = jnp.tanh(hhat @ p["W_hhat"].T + xhat @ p["W_xhat"].T + p["b_hat"])
        z_h, z_x, z_b = hhat @ p["W_hh"].T + p["b_hh"], hhat @ p["W_hx"].T + p["b_hx"], hhat @ p["W_hb"].T
        d_h, d_x = z_h @ p["W_hz"].T, z_x @ p["W_xz"].T
        h = jnp.tanh(d_h * (h @ p["W_h"].T) + d_x * (xt @ p["W_x"].T) + z_b @ p["W_bz"].T + p["b_0"])
        return (h, hhat), h

    init = (jnp.zeros((B, Nh), x.dtype), jnp.zeros((B, Nhat), x.dtype))
    _, hs = jax.lax.scan(step, init, jnp.swapaxes(x, 0, 1))
    return jnp.swapaxes(hs, 0, 1)


def rnn_loss(p, x, target):
    return 0.5 * jnp.mean((hyper_rnn(p, x) - target) ** 2)


def layer_norm(a, eps=1e-5):
    mu = a.mean(-1, keepdims=True)
    return (a - mu) / jnp.sqrt(((a - mu) ** 2).mean(-1, keepdims=True) + eps)


def hyper_lstm(p, x, ln=True):
    norm = layer_norm if ln else (lambda a: a)
    B = x.shape[0]
    Nh, Nhat = p["W_h"].shape[1], p["W_hhat"].shape[1]
    sig = jax.nn.sigmoid

    def step(carry, xt):
        h, c, hhat, chat = carry
        xhat = jnp.concatenate([h, xt], -1)
        gh = norm(jnp.einsum("ykj,bj->ybk", p["W_hhat"], hhat) + jnp.einsum("ykj,bj->ybk", p["W_xhat"], xhat)
                  + p["b_hat"][:, None])  # fmt: skip
        chat = sig(gh[2]) * chat + sig(gh[0]) * jnp.tanh(gh[1])
        hhat = sig(gh[3]) * jnp.tanh(norm(chat))
        z_h = jnp.einsum("yzk,bk->ybz", p["W_hh"], hhat) + p["b_hh"][:, None]
        z_x = jnp.einsum("yzk,bk->ybz", p["W_hx"], hhat) + p["b_hx"][:, None]
        z_b = jnp.einsum("yzk,bk->ybz", p["W_hb"], hhat)
        d_h = jnp.einsum("ynz,ybz->ybn", p["W_hz"], z_h)
        d_x = jnp.einsum("ynz,ybz->ybn", p["W_xz"], z_x)
        bias = jnp.einsum("ynz,ybz->ybn", p["W_bz"], z_b) + p["b_0"][:, None]
        g = norm(d_h * jnp.einsum("ynj,bj->ybn", p["W_h"], h) + d_x * jnp.einsum("ynj,bj->ybn", p["W_x"], xt) + bias)
        c = sig(g[2]) * c + sig(g[0]) * jnp.tanh(g[1])
        h = sig(g[3]) * jnp.tanh(norm(c))
        return (h, c, hhat, chat), h

    z = lambda n: jnp.zeros((B, n), x.dtype)  # noqa: E731
    _, hs = jax.lax.scan(step, (z(Nh), z(Nh), z(Nhat), z(Nhat)), jnp.swapaxes(x, 0, 1))
    return jnp.swapaxes(hs, 0, 1)
