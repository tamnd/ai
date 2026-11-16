# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/weight-decay/ (ai://NEW-weight-decay).
"""Weight decay, the NumPy reference.

The parameters and the gradients are dicts of arrays. sgd_step is torch.optim.SGD: with decoupled=False the
weight_decay is an L2 term added to the gradient before the momentum buffer (eq-l2-step); with decoupled=True the
parameters are multiplied by 1 - lr * weight_decay beside the gradient step (eq-decay-step), the SGDW of Loshchilov
and Hutter written as PyTorch writes AdamW. adam_step is torch.optim.Adam (L2) or torch.optim.AdamW (decoupled), kept
here only to compare the two forms; the Adam page has the full reference. The other functions are the conversions,
the MAP estimate and the scale-invariance formulas of the page.
"""

import numpy as np


def _arr(x):
    return np.asarray(x, dtype=np.float64)


def l2_penalty(params, weight_decay, mask=None):
    """(weight_decay / 2) times the sum of the squares of the parameters selected by mask (all by default)."""
    return 0.5 * weight_decay * sum(float(np.sum(_arr(p) ** 2)) for k, p in params.items() if mask is None or mask[k])


def l2_grad(params, weight_decay, mask=None):
    """The gradient of l2_penalty: weight_decay * p for the selected parameters, zero for the others."""
    return {k: (weight_decay * _arr(p) if mask is None or mask[k] else np.zeros_like(_arr(p))) for k, p in params.items()}


def decay_mask(params):
    """True for the parameters to decay: matrices and convolution kernels (ndim >= 2). Biases and the scales and shifts
    of normalization layers (ndim 1) are not decayed, as in Goyal et al. and the no-bias-decay of He et al."""
    return {k: _arr(p).ndim >= 2 for k, p in params.items()}


def sgd_config(lr, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False, decoupled=False):
    """Return the configuration of SGD. decoupled=False is torch.optim.SGD; decoupled=True decays by lr * weight_decay."""
    if lr < 0:
        raise ValueError(f"lr must be non-negative, got {lr}")
    if not 0.0 <= momentum < 1.0:
        raise ValueError(f"momentum must be in [0, 1), got {momentum}")
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
    if nesterov and (momentum <= 0 or dampening != 0):
        raise ValueError("nesterov needs a positive momentum and zero dampening")
    return {"lr": float(lr), "momentum": float(momentum), "dampening": float(dampening),
            "weight_decay": float(weight_decay), "nesterov": bool(nesterov), "decoupled": bool(decoupled)}


def sgd_init(params):
    """Return the state: no momentum buffers yet."""
    return {"buf": {k: None for k in params}}


def sgd_step(cfg, params, grads, state, mask=None):
    """Return (params, state) after one step. mask selects the parameters that are decayed (all by default)."""
    lr, mu, damp, wd = cfg["lr"], cfg["momentum"], cfg["dampening"], cfg["weight_decay"]
    new_p, new_buf = {}, {}
    for k, p in params.items():
        p, g = _arr(p), _arr(grads[k])
        decay = wd if mask is None or mask[k] else 0.0
        if decay and not cfg["decoupled"]:
            g = g + decay * p
        buf = state["buf"][k]
        if mu:
            buf = g.copy() if buf is None else mu * buf + (1 - damp) * g
            g = g + mu * buf if cfg["nesterov"] else buf
        if decay and cfg["decoupled"]:
            p = p * (1 - lr * decay)
        new_p[k] = p - lr * g
        new_buf[k] = buf
    return new_p, {"buf": new_buf}


def adam_config(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, decoupled=False):
    """torch.optim.Adam (decoupled=False, weight_decay as L2) or torch.optim.AdamW (decoupled=True)."""
    return {"lr": float(lr), "beta1": float(betas[0]), "beta2": float(betas[1]), "eps": float(eps),
            "weight_decay": float(weight_decay), "decoupled": bool(decoupled)}


def adam_init(params):
    return {"t": 0, "m": {k: np.zeros_like(_arr(p)) for k, p in params.items()},
            "v": {k: np.zeros_like(_arr(p)) for k, p in params.items()}}


def adam_step(cfg, params, grads, state):
    lr, b1, b2, eps, wd = cfg["lr"], cfg["beta1"], cfg["beta2"], cfg["eps"], cfg["weight_decay"]
    t = state["t"] + 1
    new_p, m, v = {}, {}, {}
    for k, p in params.items():
        p, g = _arr(p), _arr(grads[k])
        if cfg["decoupled"]:
            p = p * (1 - lr * wd)
        else:
            g = g + wd * p
        m[k] = b1 * state["m"][k] + (1 - b1) * g
        v[k] = b2 * state["v"][k] + (1 - b2) * g * g
        mhat, vhat = m[k] / (1 - b1**t), v[k] / (1 - b2**t)
        new_p[k] = p - lr * mhat / (np.sqrt(vhat) + eps)
    return new_p, {"t": t, "m": m, "v": v}


def decay_per_step(lr, weight_decay):
    """The decay delta = lr * weight_decay per step that an L2 coefficient gives under plain SGD (eq-decay-step), and
    the decay per step of torch.optim.AdamW. This delta is the lambda of Loshchilov and Hutter."""
    return lr * weight_decay


def l2_from_decay(lr, delta):
    """The L2 coefficient that gives the decay delta per step under plain SGD: delta / lr (Proposition 1 of Loshchilov
    and Hutter)."""
    return delta / lr


def matched_decay(lr, weight_decay, momentum=0.0):
    """The decoupled decay per step whose fixed point equals that of L2 with weight_decay under heavy-ball momentum:
    lr * weight_decay / (1 - momentum) (derivation sgd-equivalence, item 4)."""
    return lr * weight_decay / (1 - momentum)


def map_l2_coefficient(prior_var, n=None):
    """The L2 coefficient of the Gaussian prior N(0, prior_var I): 1 / prior_var on the summed negative log likelihood,
    or 1 / (n prior_var) on the mean over n examples (eq-map)."""
    return 1.0 / prior_var if n is None else 1.0 / (n * prior_var)


def map_ridge(X, y, noise_var, prior_var):
    """The MAP estimate (and posterior mean) of a linear model with Gaussian noise and the prior N(0, prior_var I):
    ridge regression with lambda = noise_var / prior_var."""
    X, y = _arr(X), _arr(y)
    lam = noise_var / prior_var
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def effective_lr(lr, w):
    """The effective learning rate lr / |w|^2 of the direction of scale-invariant weights w (eq-effective-lr)."""
    return lr / float(np.sum(_arr(w) ** 2))


def exp_lr_schedule(lr, weight_decay, t):
    """The learning rate at step t (from 0) of undecayed SGD that matches SGD with decay on a scale-invariant loss:
    lr (1 - lr weight_decay)^(-2t - 1), Theorem 1.1 of Li and Arora with momentum 0."""
    return lr * (1 - lr * weight_decay) ** (-2 * t - 1)


def equilibrium_norm_sq(lr, weight_decay, G):
    """The squared weight norm at equilibrium when |grad| = G / |w|: sqrt(lr G^2 / (wd (2 - lr wd))) (eq-equilibrium)."""
    return float(np.sqrt(lr * G**2 / (weight_decay * (2 - lr * weight_decay))))


def equilibrium_angle(lr, weight_decay):
    """The angle in radians that the direction turns per step at equilibrium; about sqrt(2 lr wd)."""
    d = lr * weight_decay
    return float(np.arctan(np.sqrt(d * (2 - d)) / (1 - d)))
