# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/recurrent-neural-network/ (ai://NEW-recurrent-neural-network).
"""One simple (Elman) recurrent layer in JAX, the JAX reference.

It computes eq-step of the page as a pure function of the parameters, with the loop over time as jax.lax.scan, and
leaves backpropagation through time to jax.grad. The parameters W, U, b_i and b_h have the names and the layout of
the NumPy reference, so the shared fixture loads as it is.
"""

import jax
import jax.numpy as jnp

NONLINEARITIES = ("tanh", "relu")


def config(input_size=3, hidden_size=4):
    """Return the config of the layer."""
    if min(input_size, hidden_size) < 1:
        raise ValueError("the sizes must be at least 1")
    return {"input_size": input_size, "hidden_size": hidden_size}


def init(key, cfg):
    """Return the parameters W (I, H), U (H, H), b_i (H,) and b_h (H,), uniform in [-1/sqrt(H), 1/sqrt(H)]."""
    n_in, H = cfg["input_size"], cfg["hidden_size"]
    k = H**-0.5
    kw, ku, kb, kc = jax.random.split(key, 4)
    return {
        "W": jax.random.uniform(kw, (n_in, H), minval=-k, maxval=k),  # (I, H)
        "U": jax.random.uniform(ku, (H, H), minval=-k, maxval=k),  # (H, H)
        "b_i": jax.random.uniform(kb, (H,), minval=-k, maxval=k),  # (H,)
        "b_h": jax.random.uniform(kc, (H,), minval=-k, maxval=k),  # (H,)
    }


def cell(params, x, h, nonlinearity="tanh"):
    """One step of eq-step: return (h_new, a) for x (B, I) and h (B, H)."""
    if nonlinearity not in NONLINEARITIES:
        raise ValueError(f"nonlinearity must be one of {NONLINEARITIES}")
    a = x @ params["W"] + params["b_i"] + h @ params["U"] + params["b_h"]  # (B, H)
    return (jnp.tanh(a) if nonlinearity == "tanh" else jax.nn.relu(a)), a


def forward(params, x, h0=None, nonlinearity="tanh"):
    """Return (h, cache) for x (B, T, I): h (B, T, H) is the state of every step, cache holds h_T and a."""
    B, _, _ = x.shape
    H = params["U"].shape[0]
    h0 = jnp.zeros((B, H), x.dtype) if h0 is None else h0

    def step(h, x_t):
        h_new, a = cell(params, x_t, h, nonlinearity)
        return h_new, (h_new, a)

    h_T, (hs, As) = jax.lax.scan(step, h0, jnp.swapaxes(x, 0, 1))  # scan over time: (T, B, H)
    return jnp.swapaxes(hs, 0, 1), {"h_T": h_T, "a": jnp.swapaxes(As, 0, 1)}  # (B, T, H)


def param_count(cfg):
    """The number of parameters, from the sizes alone."""
    n_in, H = cfg["input_size"], cfg["hidden_size"]
    return H * n_in + H * H + 2 * H
