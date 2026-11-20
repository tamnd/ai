# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/normalizing-flow/ (ai://NEW-normalizing-flow).
"""The normalizing flow of the page in JAX: pure functions of a parameter dictionary, the affine coupling flow of
eq-coupling with alternating masks, its exact log-likelihood (eq-change-of-variables), jax.grad of the mean negative
log-likelihood, the planar flow with the fix of eq-planar, and one masked autoregressive layer whose sampling is a
jax.lax.fori_loop of d_data passes. The parameters have the names and shapes of the NumPy reference.
"""

import math

import jax
import jax.numpy as jnp

PARTS = ("W1", "b1", "Ws", "bs", "Wt", "bt")


def config(d_data=3, d_hidden=5, n_layers=3):
    if min(d_data, d_hidden, n_layers) < 1 or d_data < 2:
        raise ValueError("d_data must be at least 2 and every size positive")
    return {"d_data": int(d_data), "d_hidden": int(d_hidden), "n_layers": int(n_layers)}


def masks(cfg):
    i = jnp.arange(cfg["d_data"])
    return jnp.stack([((i + k) % 2 == 0).astype(jnp.float64) for k in range(cfg["n_layers"])])


def param_shapes(cfg):
    D, H = cfg["d_data"], cfg["d_hidden"]
    one = {"W1": (D, H), "b1": (H,), "Ws": (H, D), "bs": (D,), "Wt": (H, D), "bt": (D,)}
    return {f"c{k}_{n}": s for k in range(cfg["n_layers"]) for n, s in one.items()}


def param_count(cfg):
    return sum(math.prod(s) for s in param_shapes(cfg).values())


def flops_forward(cfg, batch):
    return 6 * batch * cfg["n_layers"] * cfg["d_data"] * cfg["d_hidden"]


def init(key, cfg):
    """W1 random, everything else 0: every layer starts as the identity."""
    p = {k: jnp.zeros(s) for k, s in param_shapes(cfg).items()}
    for k in range(cfg["n_layers"]):
        key, sub = jax.random.split(key)
        p[f"c{k}_W1"] = jax.random.normal(sub, (cfg["d_data"], cfg["d_hidden"])) / math.sqrt(cfg["d_data"])
    return p


def _scale_shift(p, k, b, x):
    h = jnp.tanh((x * b) @ p[f"c{k}_W1"] + p[f"c{k}_b1"])
    return (h @ p[f"c{k}_Ws"] + p[f"c{k}_bs"]) * (1 - b), (h @ p[f"c{k}_Wt"] + p[f"c{k}_bt"]) * (1 - b)


def forward(p, cfg, x):
    """f: x -> z and log |det df/dx| per row."""
    ld = jnp.zeros(x.shape[0])
    for k, b in enumerate(masks(cfg)):
        s, t = _scale_shift(p, k, b, x)
        x = x * jnp.exp(s) + t
        ld = ld + s.sum(-1)
    return x, ld


def inverse(p, cfg, z):
    for k, b in reversed(list(enumerate(masks(cfg)))):
        s, t = _scale_shift(p, k, b, z)
        z = (z - t) * jnp.exp(-s)
    return z


def log_prob(p, cfg, x):
    z, ld = forward(p, cfg, x)
    return -0.5 * jnp.sum(z**2, -1) - 0.5 * z.shape[-1] * math.log(2 * math.pi) + ld


def nll(p, cfg, x):
    return -jnp.mean(log_prob(p, cfg, x))


def u_hat(w, u):
    wu = w @ u
    return u + (jax.nn.softplus(wu) - 1 - wu) * w / (w @ w)


def planar(z, w, u, b):
    a = jnp.tanh(z @ w + b)
    return z + a[:, None] * u, jnp.log(jnp.abs(1 + (1 - a**2) * (w @ u)))


def ar_params(x, q):
    m = jnp.triu(jnp.ones((x.shape[-1], x.shape[-1])), 1)
    return x @ (q["Wm"] * m) + q["bm"], x @ (q["Wa"] * m) + q["ba"]


def ar_density(q, x):
    mu, al = ar_params(x, q)
    return (x - mu) * jnp.exp(-al), -al.sum(-1)


def ar_sample(q, u, passes=None):
    def body(_, x):
        mu, al = ar_params(x, q)
        return u * jnp.exp(al) + mu

    return jax.lax.fori_loop(0, u.shape[-1] if passes is None else passes, body, jnp.zeros_like(u))
