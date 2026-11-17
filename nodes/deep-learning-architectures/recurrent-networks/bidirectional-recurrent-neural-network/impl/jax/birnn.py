# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/bidirectional-recurrent-neural-network/ (ai://NEW-bidirectional-recurrent-neural-network).
"""A deep bidirectional recurrent network for sequence labelling in JAX, as pure functions of the parameters.

The parameters have the names and the layout of the NumPy reference, so the shared fixture loads as it is. Each
direction is a jax.lax.scan, the backward one with reverse=True. Padding is handled differently from the reference,
which reverses the real steps of each sequence: here the backward scan runs over all T steps from the end and holds
its state at zero while it is inside the padding, so that it starts afresh at the last real step. The gradient of
eq-loss is left to jax.grad.
"""

import jax
import jax.numpy as jnp

CELLS = {"lstm": 4, "tanh": 1}


def config(input_size=3, hidden_size=4, n_layers=2, n_classes=5, cell="lstm", bidirectional=True):
    """Return the config of the network."""
    if min(input_size, hidden_size, n_layers) < 1:
        raise ValueError("the sizes and the number of layers must be at least 1")
    if n_classes < 2:
        raise ValueError("a classifier needs at least two classes")
    if cell not in CELLS:
        raise ValueError(f"cell must be one of {sorted(CELLS)}")
    return {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "n_layers": n_layers,
        "n_classes": n_classes,
        "cell": cell,
        "bidirectional": bool(bidirectional),
    }


def directions(cfg):
    return ("fwd", "bwd") if cfg["bidirectional"] else ("fwd",)


def param_count(cfg):
    I, H, L, K = cfg["input_size"], cfg["hidden_size"], cfg["n_layers"], cfg["n_classes"]
    G, D = CELLS[cfg["cell"]], len(directions(cfg))
    return D * (G * H * (I + H) + G * H) + (L - 1) * D * (G * H * (D * H + H) + G * H) + D * H * K + K


def init(key, cfg, scale=0.1):
    """All parameters uniform in [-scale, scale]."""
    I, H, K = cfg["input_size"], cfg["hidden_size"], cfg["n_classes"]
    G, D = CELLS[cfg["cell"]], len(directions(cfg))
    shapes = {}
    for l in range(cfg["n_layers"]):
        for d in directions(cfg):
            n_in = I if l == 0 else D * H
            shapes |= {f"l{l}_{d}_W": (n_in, G * H), f"l{l}_{d}_U": (H, G * H), f"l{l}_{d}_b": (G * H,)}
    shapes |= {"out_W": (D * H, K), "out_b": (K,)}
    keys = jax.random.split(key, len(shapes))
    return {n: jax.random.uniform(k, s, minval=-scale, maxval=scale) for k, (n, s) in zip(keys, shapes.items())}


def run_direction(cell, W, U, b, x, mask, reverse):
    """One direction over x (B, T, n_in) with mask (B, T, 1): the state of every step, (B, T, H).

    The state is multiplied by the mask after every step, so it is zero at the padding; the backward scan therefore
    reaches the last real step of each sequence from a zero state.
    """
    B, H = x.shape[0], U.shape[0]

    def step(carry, inp):
        h, c = carry
        x_t, m_t = inp
        z = x_t @ W + h @ U + b
        if cell == "tanh":
            h = jnp.tanh(z)
        else:
            i, f, g, o = jnp.split(z, 4, axis=-1)
            c = jax.nn.sigmoid(f) * c + jax.nn.sigmoid(i) * jnp.tanh(g)
            h = jax.nn.sigmoid(o) * jnp.tanh(c)
        h, c = h * m_t, c * m_t
        return (h, c), h

    zeros = jnp.zeros((B, H), x.dtype)
    _, hs = jax.lax.scan(step, (zeros, zeros), (x.swapaxes(0, 1), mask.swapaxes(0, 1)), reverse=reverse)
    return hs.swapaxes(0, 1)


def mask_of(lengths, T, dtype):
    return (jnp.arange(T)[None, :] < lengths[:, None])[..., None].astype(dtype)


def forward(params, cfg, x, lengths=None):
    """The class scores (B, T, K) at every step of the inputs x (B, T, I)."""
    B, T, _ = x.shape
    lengths = jnp.full((B,), T) if lengths is None else jnp.asarray(lengths)
    mask = mask_of(lengths, T, x.dtype)
    inp = x
    for l in range(cfg["n_layers"]):
        outs = [
            run_direction(cfg["cell"], *(params[f"l{l}_{d}_{n}"] for n in "WUb"), inp, mask, reverse=(d == "bwd"))
            for d in directions(cfg)
        ]
        inp = jnp.concatenate(outs, axis=-1)
    return inp @ params["out_W"] + params["out_b"]


def nll(logits, labels, lengths=None):
    """eq-loss: the cross-entropy of the labels, summed over the real steps of each sequence, averaged over the batch."""
    B, T = labels.shape
    lengths = jnp.full((B,), T) if lengths is None else jnp.asarray(lengths)
    logp = jnp.take_along_axis(jax.nn.log_softmax(logits), labels[..., None], axis=-1)[..., 0]
    return -(logp * mask_of(lengths, T, logits.dtype)[..., 0]).sum() / B


def loss(params, cfg, x, labels, lengths=None):
    return nll(forward(params, cfg, x, lengths), labels, lengths)
