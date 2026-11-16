# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/adam/ (ai://NEW-adam).
"""Adam, AdamW and AMSGrad, the NumPy reference.

The parameters and the gradients are dicts of arrays. step is Algorithm 1 of Kingma and Ba (eq-moments,
eq-bias-correction, eq-update), written as PyTorch writes it: the denominator is sqrt(v_hat) + eps. With weight_decay
and decoupled=False the decay is an L2 term added to the gradient, as in torch.optim.Adam; with decoupled=True the
parameters are first multiplied by 1 - lr * weight_decay, as in torch.optim.AdamW (eq-adamw). With amsgrad=True the
second moment in the denominator is the running maximum of v_t, bias-corrected, as in PyTorch.
"""

import numpy as np


def config(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, decoupled=False, amsgrad=False):
    """Return the configuration. The defaults are those of torch.optim.Adam and of Algorithm 1 of Kingma and Ba;
    torch.optim.AdamW is decoupled=True with weight_decay=1e-2."""
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


def init_state(params):
    """Return the state: the step count t = 0 and the moments m_0 = v_0 = 0 (and the running maximum of v)."""
    zeros = {k: np.zeros_like(np.asarray(p, dtype=np.float64)) for k, p in params.items()}
    return {"t": 0, "m": zeros, "v": {k: z.copy() for k, z in zeros.items()},
            "v_max": {k: z.copy() for k, z in zeros.items()}}


def step(cfg, params, grads, state):
    """Return (params, state) after one step of the optimizer with the gradients grads."""
    lr, b1, b2, eps, wd = cfg["lr"], cfg["beta1"], cfg["beta2"], cfg["eps"], cfg["weight_decay"]
    t = state["t"] + 1
    new_params, m_new, v_new, vmax_new = {}, {}, {}, {}
    for k, p in params.items():
        p = np.asarray(p, dtype=np.float64)
        g = np.asarray(grads[k], dtype=np.float64)
        if wd and cfg["decoupled"]:
            p = p * (1.0 - lr * wd)  # decoupled weight decay, applied before the Adam step
        elif wd:
            g = g + wd * p  # the L2 penalty wd/2 |p|^2 folded into the gradient
        m = b1 * state["m"][k] + (1.0 - b1) * g
        v = b2 * state["v"][k] + (1.0 - b2) * g * g
        vmax = np.maximum(state["v_max"][k], v) if cfg["amsgrad"] else state["v_max"][k]
        m_hat = m / (1.0 - b1**t)
        v_hat = (vmax if cfg["amsgrad"] else v) / (1.0 - b2**t)
        new_params[k] = p - lr * m_hat / (np.sqrt(v_hat) + eps)
        m_new[k], v_new[k], vmax_new[k] = m, v, vmax
    return new_params, {"t": t, "m": m_new, "v": v_new, "v_max": vmax_new}


def step_uncorrected(cfg, params, grads, state):
    """One step without the bias correction: the update lr * m / (sqrt(v) + eps), as in RMSProp with momentum. It
    exists to show the large first steps that the correction removes."""
    lr, b1, b2, eps = cfg["lr"], cfg["beta1"], cfg["beta2"], cfg["eps"]
    new_params, m_new, v_new = {}, {}, {}
    for k, p in params.items():
        g = np.asarray(grads[k], dtype=np.float64)
        m = b1 * state["m"][k] + (1.0 - b1) * g
        v = b2 * state["v"][k] + (1.0 - b2) * g * g
        new_params[k] = np.asarray(p, dtype=np.float64) - lr * m / (np.sqrt(v) + eps)
        m_new[k], v_new[k] = m, v
    return new_params, {"t": state["t"] + 1, "m": m_new, "v": v_new, "v_max": state["v_max"]}


def step_eps_hat(cfg, params, grads, state):
    """One step of the efficient form of Section 2 of Kingma and Ba: lr_t = lr sqrt(1 - beta2^t) / (1 - beta1^t) and
    the denominator sqrt(v_t) + eps, so that eps is the "epsilon hat" of the paper, as in Keras."""
    lr, b1, b2, eps = cfg["lr"], cfg["beta1"], cfg["beta2"], cfg["eps"]
    t = state["t"] + 1
    lr_t = lr * np.sqrt(1.0 - b2**t) / (1.0 - b1**t)
    new_params, m_new, v_new = {}, {}, {}
    for k, p in params.items():
        g = np.asarray(grads[k], dtype=np.float64)
        m = b1 * state["m"][k] + (1.0 - b1) * g
        v = b2 * state["v"][k] + (1.0 - b2) * g * g
        new_params[k] = np.asarray(p, dtype=np.float64) - lr_t * m / (np.sqrt(v) + eps)
        m_new[k], v_new[k] = m, v
    return new_params, {"t": t, "m": m_new, "v": v_new, "v_max": state["v_max"]}


def ratio_bound(beta1, beta2, t):
    """Return the largest |m_hat_t| / sqrt(v_hat_t) over all gradient sequences of length t, with eps = 0
    (derivations/step-bound.md). It is inf when beta1^2 >= beta2 and t is infinite."""
    gamma = beta1**2 / beta2
    s = float(t) if gamma == 1.0 else (1.0 - gamma**t) / (1.0 - gamma)
    return (1 - beta1) / np.sqrt(1 - beta2) * np.sqrt(1 - beta2**t) / (1 - beta1**t) * np.sqrt(s)
