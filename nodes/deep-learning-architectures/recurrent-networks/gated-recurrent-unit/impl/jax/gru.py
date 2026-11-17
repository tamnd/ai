# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/gated-recurrent-unit/ (ai://NEW-gated-recurrent-unit).
"""One GRU layer in JAX, the JAX reference.

It computes eq-gates, eq-candidate and eq-update of the page as a pure function of the parameters, with the loop over
time as jax.lax.scan, and leaves the gradient to jax.grad. The parameters W, U, b_i and b_h have the names and the
layout of the NumPy reference, with the blocks in the order reset, update, candidate, so the shared fixture loads
as it is. reset="after" is the placement of torch.nn.GRU and reset="before" that of eq. (8) of Cho et al.
"""

import jax
import jax.numpy as jnp

RESETS = ("after", "before")


def config(input_size=3, hidden_size=4):
    """Return the config of the layer."""
    if min(input_size, hidden_size) < 1:
        raise ValueError("the sizes must be at least 1")
    return {"input_size": input_size, "hidden_size": hidden_size}


def init(key, cfg):
    """Return the parameters W (I, 3H), U (H, 3H), b_i (3H,) and b_h (3H,), uniform in [-1/sqrt(H), 1/sqrt(H)]."""
    n_in, H = cfg["input_size"], cfg["hidden_size"]
    k = H**-0.5
    kw, ku, kb, kc = jax.random.split(key, 4)
    return {
        "W": jax.random.uniform(kw, (n_in, 3 * H), minval=-k, maxval=k),  # (I, 3 * H)
        "U": jax.random.uniform(ku, (H, 3 * H), minval=-k, maxval=k),  # (H, 3 * H)
        "b_i": jax.random.uniform(kb, (3 * H,), minval=-k, maxval=k),  # (3 * H,)
        "b_h": jax.random.uniform(kc, (3 * H,), minval=-k, maxval=k),  # (3 * H,)
    }


def cell(params, x, h, reset="after"):
    """One step: return (h_new, gates) for x (B, I) and h (B, H)."""
    if reset not in RESETS:
        raise ValueError(f"reset must be one of {RESETS}")
    H = h.shape[-1]
    W, U, b_i, b_h = params["W"], params["U"], params["b_i"], params["b_h"]
    a = x @ W + b_i  # (B, 3 * H)
    m = h @ U[:, : 2 * H] + b_h[: 2 * H]  # (B, 2 * H)
    r = jax.nn.sigmoid(a[..., :H] + m[..., :H])  # (B, H)
    z = jax.nn.sigmoid(a[..., H : 2 * H] + m[..., H:])  # (B, H)
    if reset == "after":
        n = jnp.tanh(a[..., 2 * H :] + r * (h @ U[:, 2 * H :] + b_h[2 * H :]))  # (B, H): eq-candidate
    else:
        n = jnp.tanh(a[..., 2 * H :] + (r * h) @ U[:, 2 * H :] + b_h[2 * H :])  # (B, H): eq-candidate-cho
    h_new = (1.0 - z) * n + z * h  # (B, H): eq-update
    return h_new, {"r": r, "z": z, "n": n}


def forward(params, x, h0=None, reset="after"):
    """Return (h, cache) for x (B, T, I): h (B, T, H) is the state of every step, cache the gates and h_T."""
    B, _, _ = x.shape
    H = params["U"].shape[0]
    h0 = jnp.zeros((B, H), x.dtype) if h0 is None else h0

    def step(h, x_t):
        h_new, gates = cell(params, x_t, h, reset)
        return h_new, (h_new, gates)

    h_T, (hs, gates) = jax.lax.scan(step, h0, jnp.swapaxes(x, 0, 1))  # scan over time: (T, B, ...)
    cache = {"h_T": h_T} | {name: jnp.swapaxes(v, 0, 1) for name, v in gates.items()}  # (B, T, H) each
    return jnp.swapaxes(hs, 0, 1), cache  # (B, T, H)


def param_count(cfg):
    """The number of parameters, from the sizes alone."""
    n_in, H = cfg["input_size"], cfg["hidden_size"]
    return 3 * H * n_in + 3 * H * H + 6 * H
