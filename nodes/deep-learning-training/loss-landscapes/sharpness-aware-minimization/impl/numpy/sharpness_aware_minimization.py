# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/sharpness-aware-minimization/ (ai://8BNWST).
"""Sharpness-aware minimization (SAM) in NumPy.

sam_perturbation is eq-eps-hat, the dual-norm solution of the linearized inner maximization, over one array or over a
list of arrays taken as one vector; sam_gradient is eq-sam-grad, the gradient at the perturbed point; sam_step is one
step of Algorithm 1 of Foret et al. around a base SGD step with optional momentum and weight decay; m_sharpness_gradient
averages SAM gradients computed on disjoint groups of m examples; filter_normalized_direction is the random direction
of Li et al. with each filter rescaled to the norm of the matching filter of the weights; loss_along is the 1-D slice
L(theta + alpha d) used to plot a loss surface."""

import numpy as np

EPS = 1e-12


def _as_list(x):
    if isinstance(x, (list, tuple)):
        return [np.asarray(v, dtype=np.float64) for v in x], True
    return [np.asarray(x, dtype=np.float64)], False


def sam_perturbation(grad, rho, p=2.0):
    """The perturbation rho sign(g) |g|^(q-1) / (||g||_q^q)^(1/p), 1/p + 1/q = 1, of p-norm rho (eq-eps-hat).

    grad is an array or a list of arrays treated as one vector; the result has the same shape. p = 2 gives
    rho g / ||g||_2 and p = inf gives rho sign(g). A zero gradient gives a zero perturbation."""
    gs, is_list = _as_list(grad)
    rho = float(rho)
    if rho < 0:
        raise ValueError("rho must be non-negative")
    p = float(p)
    if p <= 1.0:
        raise ValueError("p must be greater than 1")
    if np.isinf(p):
        out = [rho * np.sign(g) for g in gs]
        return out if is_list else out[0]
    scale = max(float(np.max(np.abs(g))) if g.size else 0.0 for g in gs)
    if scale == 0.0:
        out = [np.zeros_like(g) for g in gs]
        return out if is_list else out[0]
    q = p / (p - 1.0)
    # the formula is unchanged by g -> g / c for c > 0; dividing by the largest entry keeps |g|^q finite
    hs = [g / scale for g in gs]
    total = sum(float(np.sum(np.abs(h) ** q)) for h in hs)
    denom = total ** (1.0 / p)
    out = [rho * np.sign(h) * np.abs(h) ** (q - 1.0) / denom for h in hs]
    return out if is_list else out[0]


def _add(a, b, c=1.0):
    return [x + c * y for x, y in zip(a, b)]


def sam_gradient(w, grad_fn, rho, p=2.0):
    """The SAM gradient grad L(w + eps_hat(w)) (eq-sam-grad), with the second-order terms dropped.

    grad_fn maps parameters (an array, or a list of arrays) to the gradient of the loss in the same shape. Returns the
    gradient at the perturbed point and the perturbation."""
    ws, is_list = _as_list(w)
    g = grad_fn(ws if is_list else ws[0])
    e, _ = _as_list(sam_perturbation(g, rho, p))
    g_adv = grad_fn(_add(ws, e) if is_list else ws[0] + e[0])
    return g_adv, (e if is_list else e[0])


def sam_step(w, grad_fn, lr, rho, p=2.0, momentum=0.0, buf=None, weight_decay=0.0):
    """One step of SAM around SGD: perturb, take the gradient there, undo the perturbation, step from the original w.

    Weight decay adds weight_decay * w to the SAM gradient, as the lambda ||w||^2 term of eq-sam-objective (up to the
    factor 2) does. With momentum > 0 the heavy-ball buffer is buf <- momentum buf + g, and the step uses buf.
    Returns (new w, new buf)."""
    ws, is_list = _as_list(w)
    g_adv, _ = sam_gradient(ws if is_list else ws[0], grad_fn, rho, p)
    gs, _ = _as_list(g_adv)
    if weight_decay:
        gs = _add(gs, ws, weight_decay)
    if momentum:
        if buf is None:
            bs = [g.copy() for g in gs]
        else:
            bs, _ = _as_list(buf)
            bs = [momentum * b + g for b, g in zip(bs, gs)]
        gs = bs
    else:
        bs = None
    new = _add(ws, gs, -lr)
    if is_list:
        return new, bs
    return new[0], (None if bs is None else bs[0])


def m_sharpness_gradient(w, grad_fn, n, m, rho, p=2.0):
    """The average of SAM gradients computed independently on disjoint groups of m of the n examples.

    grad_fn(w, idx) returns the gradient of the mean loss over the examples idx. Each group gets its own perturbation
    from its own gradient, as when each accelerator computes SAM on its share of a batch. n must be a multiple of m;
    m = n is ordinary SAM on the whole batch."""
    if m <= 0 or n % m:
        raise ValueError("n must be a positive multiple of m")
    total = None
    for start in range(0, n, m):
        idx = np.arange(start, start + m)
        g_adv, _ = sam_gradient(w, lambda v, idx=idx: grad_fn(v, idx), rho, p)
        gs, is_list = _as_list(g_adv)
        total = gs if total is None else _add(total, gs)
    k = n // m
    out = [t / k for t in total]
    return out if is_list else out[0]


def filter_normalized_direction(params, rng, ignore_1d=True):
    """A random Gaussian direction with each filter rescaled to the norm of the matching filter of params (Li et al.).

    params is a list of weight arrays. A filter is a slice along the first axis: an output channel of a convolution or
    a row (one output unit) of a fully connected layer. Arrays with one dimension, biases and batch-normalization
    parameters, get a zero direction when ignore_1d is true; otherwise each entry is a filter. A filter of zero norm
    gets a zero direction."""
    out = []
    for theta in params:
        theta = np.asarray(theta, dtype=np.float64)
        d = rng.standard_normal(theta.shape)
        if theta.ndim <= 1:
            if ignore_1d:
                out.append(np.zeros_like(theta))
                continue
            t = theta.reshape(-1, 1)
            dd = d.reshape(-1, 1)
        else:
            t = theta.reshape(theta.shape[0], -1)
            dd = d.reshape(theta.shape[0], -1)
        tn = np.linalg.norm(t, axis=1, keepdims=True)
        dn = np.linalg.norm(dd, axis=1, keepdims=True)
        dd = dd / (dn + EPS) * tn
        out.append(dd.reshape(theta.shape))
    return out


def loss_along(loss_fn, params, direction, alphas):
    """The values L(theta + alpha d) for each alpha, the 1-D slice of a loss-surface plot."""
    return np.array([loss_fn(_add(params, direction, float(a))) for a in alphas])
