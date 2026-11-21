# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/classifier-free-guidance/ (ai://NEW-classifier-free-guidance).
"""Classifier-free guidance in NumPy, every gradient written by hand.

- The schedule and the exact models: the linear variance schedule, forward noising, and for data drawn from a mixture
  of Gaussians N(means[k], s2 I), one per class, the exact conditional and unconditional noise predictors, the exact
  classifier log p(c | z_t) and its gradient, and the log density of the noisy marginal.
- Guidance: the guided noise of eq-cfg with the scale s (s = 1 + w of Ho and Salimans), classifier guidance (eq-cg),
  the perturbed Gaussian mean of Dhariwal and Nichol, the rescale of Lin and colleagues (eq-rescale), static and
  dynamic thresholding of the x-prediction (eq-dyn), and the guidance interval of Kynkaanniemi and colleagues.
- Gaussians in closed form: the tilted density p1^s p0^(1-s) of two Gaussians (eq-tilt) and the end variance of the
  guided probability-flow ODE (eq-ode-var).
- Sampling: deterministic DDIM with a guided noise predictor, the interval, the rescale and a threshold, counting the
  network evaluations.
- The denoiser: a one-hidden-layer network on the noisy sample with a sinusoidal time embedding and a learned class
  embedding added before the tanh; class C is the null class. It predicts the noise as the noisy sample plus a
  correction. Its loss gradient is written by hand and it is trained with Adam, dropping the class to the null class
  with probability p_uncond (Algorithm 1 of Ho and Salimans).

A batch of samples is a (B, D) array; times are integer steps of the schedule.
"""

import math

import numpy as np

# ---------------------------------------------------------------------------------------------------------------------
# The schedule and the exact models of a Gaussian mixture


def alpha_bars(T, b0=1e-4, b1=2e-2):
    return np.cumprod(1.0 - np.linspace(b0, b1, T))


def q_sample(x0, abar, eps):
    a = np.reshape(abar, (-1,) + (1,) * (x0.ndim - 1))
    return np.sqrt(a) * x0 + np.sqrt(1.0 - a) * eps


def edm_sigma(abar):
    """The noise level sigma = sqrt((1 - abar) / abar) of x_t / sqrt(abar), the variable of the guidance interval."""
    return np.sqrt((1.0 - np.asarray(abar)) / np.asarray(abar))


def _resp(zt, abar, means, s2, weights):
    a = np.reshape(abar, (-1, 1))
    v = a * s2 + 1.0 - a
    r = zt[:, None, :] - np.sqrt(a)[:, None] * means[None]
    logw = np.log(weights)[None] - 0.5 * (r**2).sum(-1) / v
    lse = logw.max(1, keepdims=True) + np.log(np.exp(logw - logw.max(1, keepdims=True)).sum(1, keepdims=True))
    return a, v, r, logw - lse, lse[:, 0]


def exact_eps(zt, abar, means, s2, weights):
    """E[eps | z_t] when x0 is a mixture of N(means[k], s2 I) with the given weights: the best noise predictor."""
    a, v, r, logpost, _ = _resp(zt, abar, means, s2, weights)
    w = np.exp(logpost)
    return np.sqrt(1.0 - a) * np.einsum("bk,bkd->bd", w, r) / v


def exact_eps_cond(zt, abar, means, s2, c):
    """The exact noise predictor of class c alone, x0 ~ N(means[c], s2 I)."""
    return exact_eps(zt, abar, means[np.atleast_1d(c)], s2, np.ones(1))


def log_marginal(zt, abar, means, s2, weights):
    """log p(z_t) of the noisy mixture, one value per row."""
    D = zt.shape[1]
    a, v, _, _, lse = _resp(zt, abar, means, s2, weights)
    return lse - 0.5 * D * np.log(2.0 * math.pi * v[:, 0])


def log_classifier(zt, abar, means, s2, weights):
    """log p(c | z_t) for every class, a (B, C) array: the exact noisy classifier, by Bayes' rule."""
    return _resp(zt, abar, means, s2, weights)[3]


def classifier_grad(zt, abar, means, s2, weights, c):
    """The gradient of log p(c | z_t) in z_t: grad log p(z_t | c) - grad log p(z_t)."""
    a, v, r, logpost, _ = _resp(zt, abar, means, s2, weights)
    w = np.exp(logpost)
    return (-r[np.arange(len(zt)), c] + np.einsum("bk,bkd->bd", w, r)) / v


# ---------------------------------------------------------------------------------------------------------------------
# Guidance


def guide(e_cond, e_null, s):
    """eq-cfg: e_null + s (e_cond - e_null). s = 1 is the conditional model, s = 0 the unconditional one."""
    return e_null + s * (e_cond - e_null)


def guide_ho(e_cond, e_null, w):
    """Ho and Salimans' form (1 + w) e_cond - w e_null, the same as guide with s = 1 + w."""
    return (1.0 + w) * e_cond - w * e_null


def classifier_guided_eps(e, grad_log_classifier, abar, s):
    """eq-cg, Dhariwal and Nichol's eq. 14 with the scale s: e - s sqrt(1 - abar) grad log p(c | z_t)."""
    a = np.reshape(abar, (-1,) + (1,) * (e.ndim - 1))
    return e - s * np.sqrt(1.0 - a) * grad_log_classifier


def perturbed_mean(mu, Sigma, g, s=1.0):
    """The mean of Dhariwal and Nichol's Algorithm 1: mu + s Sigma g, with g the classifier gradient at mu."""
    return mu + s * g @ np.asarray(Sigma).T


def eps_to_x0(zt, e, abar):
    a = np.reshape(abar, (-1,) + (1,) * (zt.ndim - 1))
    return (zt - np.sqrt(1.0 - a) * e) / np.sqrt(a)


def x0_to_eps(zt, x0, abar):
    a = np.reshape(abar, (-1,) + (1,) * (zt.ndim - 1))
    return (zt - np.sqrt(a) * x0) / np.sqrt(1.0 - a)


def eps_to_v(zt, e, abar):
    a = np.reshape(abar, (-1,) + (1,) * (zt.ndim - 1))
    return np.sqrt(a) * e - np.sqrt(1.0 - a) * eps_to_x0(zt, e, abar)


def rescale(out_cond, out_cfg, phi):
    """eq-rescale: blend the guided output with its copy rescaled to the standard deviation of the conditional one."""
    axes = tuple(range(1, out_cfg.ndim))
    factor = out_cond.std(axis=axes, keepdims=True) / out_cfg.std(axis=axes, keepdims=True)
    return out_cfg * (phi * factor + (1.0 - phi))


def static_threshold(x0):
    return np.clip(x0, -1.0, 1.0)


def dynamic_threshold(x0, p):
    """eq-dyn: s is the p-th percentile of |x0| per sample, at least 1; clip to [-s, s] and divide by s."""
    axes = tuple(range(1, x0.ndim))
    s = np.maximum(np.percentile(np.abs(x0), p, axis=axes, keepdims=True), 1.0)
    return np.clip(x0, -s, s) / s


def interval_scale(sigma, s, lo, hi):
    """The guidance scale of the interval method: s for sigma in (lo, hi], 1 (no guidance) elsewhere."""
    return np.where((np.asarray(sigma) > lo) & (np.asarray(sigma) <= hi), s, 1.0)


# ---------------------------------------------------------------------------------------------------------------------
# Gaussians in closed form


def tilted_gaussian(m1, v1, m0, v0, s):
    """eq-tilt: p1^s p0^(1-s) for p1 = N(m1, v1), p0 = N(m0, v0) per coordinate; None if it does not normalize."""
    prec = s / v1 + (1.0 - s) / v0
    if np.any(prec <= 0):
        return None
    v = 1.0 / prec
    return v * (s * np.asarray(m1) / v1 + (1.0 - s) * np.asarray(m0) / v0), v


def guided_ode_variance(v1, v0, s):
    """eq-ode-var: the variance at sigma = 0 of the guided probability-flow ODE of two Gaussians, v1^s v0^(1-s)."""
    return v1**s * v0 ** (1.0 - s)


def gaussian_eps(zt, abar, m, v):
    """The exact noise predictor of x0 ~ N(m, v) per coordinate."""
    a = np.reshape(abar, (-1,) + (1,) * (zt.ndim - 1))
    return np.sqrt(1.0 - a) * (zt - np.sqrt(a) * m) / (a * v + 1.0 - a)


# ---------------------------------------------------------------------------------------------------------------------
# Sampling


def ddim_sample(eps_fn, zT, abar, steps, threshold=None):
    """Deterministic DDIM from z_T at the times steps[0] > ... > steps[-1], ending at abar = 1. A threshold, if given,
    is applied to each x-prediction and the noise is recomputed from it, as in Imagen."""
    z = zT
    for i, t in enumerate(steps):
        a = abar[t]
        a_prev = abar[steps[i + 1]] if i + 1 < len(steps) else 1.0
        e = eps_fn(z, t)
        x0 = (z - math.sqrt(1.0 - a) * e) / math.sqrt(a)
        if threshold is not None:
            x0 = threshold(x0)
            e = (z - math.sqrt(a) * x0) / math.sqrt(1.0 - a)
        z = math.sqrt(a_prev) * x0 + math.sqrt(1.0 - a_prev) * e
    return z


def guided_sampler(eps_cond, eps_null, s, abar, interval=None, phi=0.0):
    """A guided noise predictor for ddim_sample, and a counter of network evaluations. eps_cond(z, t) and
    eps_null(z, t) are the two models; interval = (lo, hi) limits guidance to sigma in (lo, hi]; phi is the rescale."""
    count = {"nfe": 0}

    def eps(z, t):
        sc = s if interval is None else float(interval_scale(edm_sigma(abar[t]), s, *interval))
        e_c = eps_cond(z, t)
        count["nfe"] += 1
        if sc == 1.0:
            return e_c
        e_u = eps_null(z, t)
        count["nfe"] += 1
        out = guide(e_c, e_u, sc)
        return rescale(e_c, out, phi) if phi else out

    return eps, count


# ---------------------------------------------------------------------------------------------------------------------
# The denoiser


def config(D, H, E, C):
    return {"dim": D, "hidden": H, "temb": E, "classes": C}


def param_shapes(cfg):
    D, H, E, C = cfg["dim"], cfg["hidden"], cfg["temb"], cfg["classes"]
    return {"W1": (D, H), "Wt": (E, H), "emb": (C + 1, H), "b1": (H,), "W2": (H, D), "b2": (D,)}


def param_count(cfg):
    return sum(math.prod(s) for s in param_shapes(cfg).values())


def flops_forward(cfg, B):
    """Multiply-adds of the three matrix products, counted as 2; the embedding lookup, biases and tanh are left out."""
    return 2 * B * (cfg["dim"] * cfg["hidden"] + cfg["temb"] * cfg["hidden"] + cfg["hidden"] * cfg["dim"])


def init(rng, cfg):
    out = {}
    for k, s in param_shapes(cfg).items():
        out[k] = np.zeros(s) if k.startswith("b") else rng.standard_normal(s) / math.sqrt(s[0] if k != "emb" else 1)
    return out


def time_embedding(t, E):
    """(len(t), E): sines at (E + 1) // 2 frequencies, then cosines at the first E // 2."""
    n = (E + 1) // 2
    a = np.asarray(t, float)[:, None] * np.exp(-math.log(10000.0) * np.arange(n) / n)
    return np.concatenate([np.sin(a), np.cos(a[:, : E // 2])], -1)


def _forward(p, cfg, z, t, y):
    te = time_embedding(np.broadcast_to(t, (len(z),)), cfg["temb"])
    h = np.tanh(z @ p["W1"] + te @ p["Wt"] + p["emb"][y] + p["b1"])
    return z + h @ p["W2"] + p["b2"], (te, h)


def forward(p, cfg, z, t, y):
    """The predicted noise for samples z at integer times t and classes y; y = C is the null class."""
    return _forward(p, cfg, z, t, np.broadcast_to(y, (len(z),)))[0]


def eps_loss(p, cfg, zt, t, y, eps):
    return float(((forward(p, cfg, zt, t, y) - eps) ** 2).sum(1).mean())


def eps_loss_grad(p, cfg, zt, t, y, eps):
    y = np.broadcast_to(y, (len(zt),))
    out, (te, h) = _forward(p, cfg, zt, t, y)
    go = 2.0 * (out - eps) / len(eps)
    da = (go @ p["W2"].T) * (1.0 - h**2)
    g = {"W2": h.T @ go, "b2": go.sum(0), "W1": zt.T @ da, "Wt": te.T @ da, "b1": da.sum(0)}
    g["emb"] = np.zeros_like(p["emb"])
    np.add.at(g["emb"], y, da)
    return g


def adam_init(p):
    return {"m": {k: np.zeros_like(v) for k, v in p.items()}, "v": {k: np.zeros_like(v) for k, v in p.items()}, "t": 0}


def adam_step(p, g, st, lr, b1=0.9, b2=0.999, eps=1e-8):
    st["t"] += 1
    for k in p:
        st["m"][k] = b1 * st["m"][k] + (1 - b1) * g[k]
        st["v"][k] = b2 * st["v"][k] + (1 - b2) * g[k] ** 2
        mh, vh = st["m"][k] / (1 - b1 ** st["t"]), st["v"][k] / (1 - b2 ** st["t"])
        p[k] = p[k] - lr * mh / (np.sqrt(vh) + eps)
    return p


def drop_condition(y, u, p_uncond, C):
    """Algorithm 1, line 3: replace the class by the null class C where the uniform draw u is below p_uncond."""
    return np.where(u < p_uncond, C, y)


def train(p, cfg, x0, y, abar, steps, rng, batch=256, lr=3e-3, p_uncond=0.1):
    """Joint training of the conditional and unconditional model (Algorithm 1 of Ho and Salimans) with Adam."""
    st, trace = adam_init(p), []
    for _ in range(steps):
        i = rng.integers(0, len(x0), batch)
        t = rng.integers(0, len(abar), batch)
        eps = rng.standard_normal(x0[i].shape)
        yi = drop_condition(y[i], rng.random(batch), p_uncond, cfg["classes"])
        zt = q_sample(x0[i], abar[t], eps)
        trace.append(eps_loss(p, cfg, zt, t, yi, eps))
        p = adam_step(p, eps_loss_grad(p, cfg, zt, t, yi, eps), st, lr)
    return p, np.array(trace)
