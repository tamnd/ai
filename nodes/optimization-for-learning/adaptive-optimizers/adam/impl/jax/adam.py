# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/adam/ (ai://NEW-adam).
"""Adam, AdamW and AMSGrad as pure functions on pytrees, the JAX reference.

init returns the state and update returns the new parameters and state, so a training step can be jitted. The update
is Algorithm 1 of Kingma and Ba (eq-moments, eq-bias-correction, eq-update) with the denominator sqrt(v_hat) + eps.
weight_decay is an L2 term added to the gradient, or, with decoupled=True, a factor 1 - lr * weight_decay on the
parameters before the step (eq-adamw), as in torch.optim.Adam and torch.optim.AdamW. The tests compare it with the
numbers of the NumPy reference, which the PyTorch tests check against torch.optim.
"""

import jax
import jax.numpy as jnp


def config(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, decoupled=False, amsgrad=False):
    """Return the configuration, with the defaults of torch.optim.Adam and of Algorithm 1 of Kingma and Ba."""
    b1, b2 = (float(b) for b in betas)
    if lr < 0:
        raise ValueError(f"lr must be non-negative, got {lr}")
    if not (0.0 <= b1 < 1.0 and 0.0 <= b2 < 1.0):
        raise ValueError(f"betas must be in [0, 1), got {betas}")
    if eps < 0:
        raise ValueError(f"eps must be non-negative, got {eps}")
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
    return {"lr": float(lr), "beta1": b1, "beta2": b2, "eps": float(eps), "weight_decay": float(weight_decay),
            "decoupled": bool(decoupled), "amsgrad": bool(amsgrad)}


def init(params):
    """Return the state: t = 0, and m, v and the running maximum of v at zero."""
    zeros = lambda: jax.tree_util.tree_map(jnp.zeros_like, params)  # noqa: E731
    return {"t": jnp.zeros((), jnp.int32), "m": zeros(), "v": zeros(), "v_max": zeros()}


def update(cfg, grads, state, params):
    """Return (params, state) after one step. cfg is static: jit with static_argnums=0 or close over it."""
    lr, b1, b2, eps, wd = cfg["lr"], cfg["beta1"], cfg["beta2"], cfg["eps"], cfg["weight_decay"]
    t = state["t"] + 1
    tf = t.astype(jnp.result_type(float))
    tm = jax.tree_util.tree_map
    if wd and cfg["decoupled"]:
        params = tm(lambda p: p * (1.0 - lr * wd), params)
    elif wd:
        grads = tm(lambda g, p: g + wd * p, grads, params)
    m = tm(lambda m, g: b1 * m + (1.0 - b1) * g, state["m"], grads)
    v = tm(lambda v, g: b2 * v + (1.0 - b2) * g * g, state["v"], grads)
    v_max = tm(jnp.maximum, state["v_max"], v) if cfg["amsgrad"] else state["v_max"]
    denom = v_max if cfg["amsgrad"] else v
    c1, c2 = 1.0 - b1**tf, 1.0 - b2**tf
    new = tm(lambda p, m, d: p - lr * (m / c1) / (jnp.sqrt(d / c2) + eps), params, m, denom)
    return new, {"t": t, "m": m, "v": v, "v_max": v_max}
