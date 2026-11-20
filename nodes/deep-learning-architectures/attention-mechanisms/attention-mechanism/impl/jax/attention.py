# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/attention-mechanism/ (ai://B95BYJ).
"""RNNsearch of Bahdanau, Cho and Bengio (2015) in JAX, as pure functions of the parameters.

The parameters have the names and the layout of the NumPy reference (rows are vectors, gate matrices side by side in
the order z, r, candidate, no biases), so the shared fixture loads as it is. Both encoder GRUs and the decoder run as
jax.lax.scan; a padded source position carries the state through. energies() is the alignment model on a whole
(T, S) grid at once, from the decoder states, which is how the energies of a trained model are inspected. The
gradients are left to jax.grad.
"""

import jax
import jax.numpy as jnp


def config(src_vocab=7, tgt_vocab=6, embed=3, hidden=4, align=5, maxout=3, attention=True):
    return {"src_vocab": src_vocab, "tgt_vocab": tgt_vocab, "embed": embed, "hidden": hidden, "align": align,
            "maxout": maxout, "attention": attention}  # fmt: skip


def shapes(cfg):
    Kx, Ky, m, n, a, l = (cfg[k] for k in ("src_vocab", "tgt_vocab", "embed", "hidden", "align", "maxout"))
    return {
        "E_x": (Kx, m), "fwd_W": (m, 3 * n), "fwd_U": (n, 3 * n), "bwd_W": (m, 3 * n), "bwd_U": (n, 3 * n),
        "E_y": (Ky, m), "dec_W": (m, 3 * n), "dec_U": (n, 3 * n), "dec_C": (2 * n, 3 * n), "W_s": (n, n),
        "W_a": (n, a), "U_a": (2 * n, a), "v_a": (a,),
        "U_o": (n, 2 * l), "V_o": (m, 2 * l), "C_o": (2 * n, 2 * l), "W_o": (l, Ky),
    }  # fmt: skip


def init(key, cfg, scale=0.01):
    keys = jax.random.split(key, len(shapes(cfg)))
    return {k: scale * jax.random.normal(kk, s) for kk, (k, s) in zip(keys, shapes(cfg).items())}


def gru_step(x, h, W, U, cc=0.0):
    """cc is the context term c C of the decoder, 0 in the encoder."""
    n = h.shape[-1]
    a = x @ W + cc
    zr = jax.nn.sigmoid(a[:, : 2 * n] + h @ U[:, : 2 * n])
    z, r = zr[:, :n], zr[:, n:]
    return (1 - z) * h + z * jnp.tanh(a[:, 2 * n :] + (r * h) @ U[:, 2 * n :])


def encode(p, src, mask):
    """eq-annot: (B, S, 2n) annotations; scan over time-major inputs, the backward GRU on the reversed sequence."""
    ex = jnp.swapaxes(p["E_x"][src], 0, 1)  # (S, B, m)
    m = jnp.swapaxes(mask, 0, 1)[..., None].astype(ex.dtype)
    h0 = jnp.zeros((src.shape[0], p["fwd_U"].shape[0]), ex.dtype)

    def run(W, U, xs, ms):
        def f(h, xm):
            x, mm = xm
            h = mm * gru_step(x, h, W, U) + (1 - mm) * h
            return h, h

        return jax.lax.scan(f, h0, (xs, ms))[1]

    hf = run(p["fwd_W"], p["fwd_U"], ex, m)
    hb = run(p["bwd_W"], p["bwd_U"], ex[::-1], m[::-1])[::-1]
    return jnp.swapaxes(jnp.concatenate([hf, hb], -1), 0, 1)


def align(p, s, H, HU, mask):
    """eq-score, eq-weights and eq-context for one decoder step: c_i (B, 2n) and alpha_i (B, S)."""
    e = jnp.tanh((s @ p["W_a"])[:, None] + HU) @ p["v_a"]
    alpha = jax.nn.softmax(jnp.where(mask, e, -jnp.inf), -1)
    return jnp.einsum("bs,bsk->bk", alpha, H), alpha


def forward(p, cfg, src, tgt, src_mask=None):
    """Teacher-forced logits (B, T, K_y), the alignments (B, T, S) and the decoder states s_0..s_{T-1} (B, T, n)."""
    src_mask = jnp.ones(src.shape, bool) if src_mask is None else src_mask
    n = p["W_s"].shape[0]
    H = encode(p, src, src_mask)
    HU = H @ p["U_a"]
    fixed = jnp.concatenate([H[:, -1, :n], H[:, 0, n:]], -1)
    ey = jnp.concatenate([jnp.zeros((tgt.shape[0], 1, p["E_y"].shape[1])), p["E_y"][tgt[:, :-1]]], 1)

    def f(s, e):
        if cfg["attention"]:
            c, alpha = align(p, s, H, HU, src_mask)
        else:
            c, alpha = fixed, jnp.zeros(src.shape)
        tt = s @ p["U_o"] + e @ p["V_o"] + c @ p["C_o"]
        logits = jnp.maximum(tt[:, 0::2], tt[:, 1::2]) @ p["W_o"]
        return gru_step(e, s, p["dec_W"], p["dec_U"], c @ p["dec_C"]), (logits, alpha, s)

    s0 = jnp.tanh(H[:, 0, n:] @ p["W_s"])
    _, (logits, alpha, states) = jax.lax.scan(f, s0, jnp.swapaxes(ey, 0, 1))
    return tuple(jnp.swapaxes(a, 0, 1) for a in (logits, alpha, states))


def energies(p, states, H):
    """e_ij = v_a . tanh(s_{i-1} W_a + h_j U_a) for all i and j at once, (B, T, S), from the states of forward."""
    return jnp.tanh((states @ p["W_a"])[:, :, None] + (H @ p["U_a"])[:, None]) @ p["v_a"]


def loss(p, cfg, src, tgt, src_mask=None, tgt_mask=None):
    """eq-loss: the mean negative log-likelihood of the real target tokens."""
    logits = forward(p, cfg, src, tgt, src_mask)[0]
    lp = jnp.take_along_axis(jax.nn.log_softmax(logits, -1), tgt[..., None], -1)[..., 0]
    w = jnp.ones(tgt.shape) if tgt_mask is None else tgt_mask.astype(lp.dtype)
    return -(lp * w).sum() / w.sum()
