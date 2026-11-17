# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/learning-rate-warmup/ (ai://MA8ACN).
"""Learning rate warmup, the NumPy reference.

gradual_warmup is the ramp of Goyal et al. (eq-gradual), linear_warmup the ramp from zero (eq-linear), and noam the
warmup and inverse-square-root schedule of the Transformer (eq-noam). radam_rho and radam_rectification are eq-rho and
eq-rect of Liu et al., and radam_step is one step of RAdam with the threshold on rho_t as a parameter: 4 in Algorithm 2
of the paper, 5 in torch.optim.RAdam.
"""

import math

import numpy as np


def gradual_warmup(t, warmup_steps, start, peak):
    """The learning rate at iteration t (from 0) of a ramp from start to peak over warmup_steps iterations, then peak."""
    if warmup_steps <= 0:
        raise ValueError(f"warmup_steps must be positive, got {warmup_steps}")
    if t < 0:
        raise ValueError(f"t must be non-negative, got {t}")
    return start + (peak - start) * min(t, warmup_steps) / warmup_steps


def linear_warmup(t, warmup_steps, peak):
    """peak * min(1, t / warmup_steps): the ramp from zero of Xiong et al., without the decay that follows."""
    return gradual_warmup(t, warmup_steps, 0.0, peak)


def noam(step, d_model, warmup_steps):
    """d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5) for step >= 1 (Vaswani et al.)."""
    if step < 1:
        raise ValueError(f"step must be at least 1, got {step}")
    return d_model**-0.5 * min(step**-0.5, step * warmup_steps**-1.5)


def noam_peak(d_model, warmup_steps):
    """The largest rate of noam, reached at step = warmup_steps: (d_model * warmup_steps)^-0.5."""
    return (d_model * warmup_steps) ** -0.5


def radam_rho_inf(beta2):
    """rho_inf = 2 / (1 - beta2) - 1, the largest length of the approximated simple moving average."""
    if not 0.0 < beta2 < 1.0:
        raise ValueError(f"beta2 must be in (0, 1), got {beta2}")
    return 2.0 / (1.0 - beta2) - 1.0


def radam_rho(t, beta2):
    """rho_t = rho_inf - 2 t beta2^t / (1 - beta2^t) for t >= 1 (eq-rho)."""
    if t < 1:
        raise ValueError(f"t must be at least 1, got {t}")
    b = beta2**t
    return radam_rho_inf(beta2) - 2.0 * t * b / (1.0 - b)


def radam_rectification(t, beta2, threshold=4.0):
    """r_t of eq-rect when rho_t > threshold, and None when RAdam takes an un-adapted momentum step instead."""
    rho, rho_inf = radam_rho(t, beta2), radam_rho_inf(beta2)
    if rho <= threshold:
        return None
    return math.sqrt((rho - 4.0) * (rho - 2.0) * rho_inf / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho))


def first_adaptive_step(beta2, threshold=4.0, limit=10**6):
    """The first t with rho_t > threshold, or None if there is none up to limit (as when rho_inf <= threshold)."""
    if radam_rho_inf(beta2) <= threshold:
        return None
    for t in range(1, limit + 1):
        if radam_rho(t, beta2) > threshold:
            return t
    return None


def variance_first_order(rho, sigma=1.0):
    """Liu et al.'s first-order approximation rho / [2 (rho - 2)(rho - 4) sigma^2] of Var[psi], for rho > 4."""
    if rho <= 4:
        raise ValueError(f"rho must be above 4, got {rho}")
    return rho / (2.0 * (rho - 2.0) * (rho - 4.0) * sigma**2)


def radam_init(param):
    """The state of RAdam for one array: the step count and the two moving averages."""
    p = np.asarray(param, dtype=np.float64)
    return {"step": 0, "m": np.zeros_like(p), "v": np.zeros_like(p)}


def radam_step(param, grad, state, lr, beta1=0.9, beta2=0.999, eps=1e-8, threshold=4.0):
    """Return (param, state) after one RAdam step, without weight decay. eps is added to sqrt(v) as in PyTorch."""
    p, g = np.asarray(param, dtype=np.float64), np.asarray(grad, dtype=np.float64)
    t = state["step"] + 1
    m = beta1 * state["m"] + (1.0 - beta1) * g
    v = beta2 * state["v"] + (1.0 - beta2) * g * g
    m_hat = m / (1.0 - beta1**t)
    r = radam_rectification(t, beta2, threshold)
    if r is None:
        p = p - lr * m_hat
    else:
        p = p - lr * r * m_hat * math.sqrt(1.0 - beta2**t) / (np.sqrt(v) + eps)
    return p, {"step": t, "m": m, "v": v}
