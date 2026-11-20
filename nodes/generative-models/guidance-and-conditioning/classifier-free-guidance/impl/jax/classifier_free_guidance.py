# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/classifier-free-guidance/ (ai://NEW-classifier-free-guidance).
"""Classifier-free guidance in JAX, as pure functions.

- The exact models of a Gaussian mixture, one Gaussian per class, with the classifier gradient by jax.grad.
- Guidance: eq-cfg, classifier guidance (eq-cg), the rescale of eq-rescale, static and dynamic thresholding (eq-dyn)
  with jnp.percentile, and the guidance interval.
- Deterministic DDIM as a jax.lax.scan over the step pairs; the guided noise predictor uses jax.lax.cond, so steps
  outside the guidance interval run the conditional model only, and the number of evaluations is counted in the scan.
- The denoiser as a parameter dictionary, its loss gradient by jax.grad, and Adam with the dropped condition of
  Algorithm 1 of Ho and Salimans as a jax.lax.scan with jax.random keys.
"""

import math

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def alpha_bars(T, b0=1e-4, b1=2e-2):
    return jnp.cumprod(1.0 - jnp.linspace(b0, b1, T))


def q_sample(x0, abar, eps):
    a = jnp.reshape(abar, (-1,) + (1,) * (x0.ndim - 1))
    return jnp.sqrt(a) * x0 + jnp.sqrt(1.0 - a) * eps


def edm_sigma(abar):
    return jnp.sqrt((1.0 - abar) / abar)


def _resp(zt, abar, means, s2, weights):
    a = jnp.reshape(abar, (-1, 1))
    v = a * s2 + 1.0 - a
    r = zt[:, None, :] - jnp.sqrt(a)[:, None] * means[None]
    logw = jnp.log(weights)[None] - 0.5 * (r**2).sum(-1) / v
    return a, v, r, jax.nn.log_softmax(logw, 1), jax.nn.logsumexp(logw, 1)


def exact_eps(zt, abar, means, s2, weights):
    a, v, r, logpost, _ = _resp(zt, abar, means, s2, weights)
    return jnp.sqrt(1.0 - a) * jnp.einsum("bk,bkd->bd", jnp.exp(logpost), r) / v


def exact_eps_cond(zt, abar, means, s2, c):
    return exact_eps(zt, abar, means[c : c + 1], s2, jnp.ones(1))


def log_marginal(zt, abar, means, s2, weights):
    a, v, _, _, lse = _resp(zt, abar, means, s2, weights)
    return lse - 0.5 * zt.shape[1] * jnp.log(2.0 * math.pi * v[:, 0])


def log_classifier(zt, abar, means, s2, weights):
    return _resp(zt, abar, means, s2, weights)[3]


def classifier_grad(zt, abar, means, s2, weights, c):
    """grad log p(c | z_t) by jax.grad, one row at a time."""

    def one(z, a, k):
        return log_classifier(z[None], a[None], means, s2, weights)[0, k]

    return jax.vmap(jax.grad(one))(zt, abar, c)


def guide(e_cond, e_null, s):
    return e_null + s * (e_cond - e_null)


def guide_ho(e_cond, e_null, w):
    return (1.0 + w) * e_cond - w * e_null


def classifier_guided_eps(e, grad_log_classifier, abar, s):
    a = jnp.reshape(abar, (-1,) + (1,) * (e.ndim - 1))
    return e - s * jnp.sqrt(1.0 - a) * grad_log_classifier


def rescale(out_cond, out_cfg, phi):
    axes = tuple(range(1, out_cfg.ndim))
    factor = out_cond.std(axis=axes, keepdims=True) / out_cfg.std(axis=axes, keepdims=True)
    return out_cfg * (phi * factor + (1.0 - phi))


def static_threshold(x0):
    return jnp.clip(x0, -1.0, 1.0)


def dynamic_threshold(x0, p):
    """Imagen's listing with the elementwise maximum: s = max(percentile(|x0|, p), 1); clip to [-s, s]; divide by s."""
    s = jnp.maximum(jnp.percentile(jnp.abs(x0), p, axis=tuple(range(1, x0.ndim)), keepdims=True), 1.0)
    return jnp.clip(x0, -s, s) / s


def interval_scale(sigma, s, lo, hi):
    return jnp.where((sigma > lo) & (sigma <= hi), s, 1.0)


def gaussian_eps(zt, abar, m, v):
    a = jnp.reshape(abar, (-1,) + (1,) * (zt.ndim - 1))
    return jnp.sqrt(1.0 - a) * (zt - jnp.sqrt(a) * m) / (a * v + 1.0 - a)


def ddim_sample(eps_fn, zT, abar, steps, threshold=None):
    """Deterministic DDIM as a jax.lax.scan; eps_fn(z, t) returns (noise, evaluations). Returns (sample, evaluations)."""
    steps = jnp.asarray(steps)
    ab = jnp.concatenate([abar, jnp.ones(1)])
    prev = jnp.concatenate([steps[1:], jnp.array([len(abar)])])

    def step(carry, tp):
        z, n = carry
        t, t_prev = tp
        a, a_prev = ab[t], ab[t_prev]
        e, k = eps_fn(z, t)
        x0 = (z - jnp.sqrt(1.0 - a) * e) / jnp.sqrt(a)
        if threshold is not None:
            x0 = threshold(x0)
            e = (z - jnp.sqrt(a) * x0) / jnp.sqrt(1.0 - a)
        return (jnp.sqrt(a_prev) * x0 + jnp.sqrt(1.0 - a_prev) * e, n + k), None

    (z, n), _ = jax.lax.scan(step, (zT, jnp.zeros((), int)), (steps, prev))
    return z, n


def guided_eps(eps_cond, eps_null, s, abar, interval=None, phi=0.0):
    """A guided noise predictor for ddim_sample that skips the null model outside the interval with jax.lax.cond."""

    def eps(z, t):
        sc = s if interval is None else interval_scale(edm_sigma(abar[t]), s, *interval)
        e_c = eps_cond(z, t)

        def with_null(e_c):
            out = guide(e_c, eps_null(z, t), sc)
            return (rescale(e_c, out, phi) if phi else out), 2

        return jax.lax.cond(sc == 1.0, lambda e_c: (e_c, 1), with_null, e_c)

    return eps


def plain_eps(eps_fn):
    return lambda z, t: (eps_fn(z, t), 1)


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


def init(key, cfg):
    shapes = param_shapes(cfg)
    out = {}
    for k, (name, s) in zip(jax.random.split(key, len(shapes)), shapes.items(), strict=True):
        if name.startswith("b"):
            out[name] = jnp.zeros(s)
        else:
            out[name] = jax.random.normal(k, s) / (math.sqrt(s[0]) if name != "emb" else 1.0)
    return out


def time_embedding(t, E):
    """(len(t), E): sines at (E + 1) // 2 frequencies, then cosines at the first E // 2."""
    n = (E + 1) // 2
    a = jnp.asarray(t, jnp.float64)[:, None] * jnp.exp(-math.log(10000.0) * jnp.arange(n) / n)
    return jnp.concatenate([jnp.sin(a), jnp.cos(a[:, : E // 2])], -1)


def forward(p, cfg, z, t, y):
    B = z.shape[0]
    te = time_embedding(jnp.broadcast_to(t, (B,)), cfg["temb"])
    h = jnp.tanh(z @ p["W1"] + te @ p["Wt"] + p["emb"][jnp.broadcast_to(y, (B,))] + p["b1"])
    return z + h @ p["W2"] + p["b2"]


def eps_loss(p, cfg, zt, t, y, eps):
    return ((forward(p, cfg, zt, t, y) - eps) ** 2).sum(1).mean()


eps_loss_grad = jax.grad(eps_loss)


def drop_condition(y, u, p_uncond, C):
    return jnp.where(u < p_uncond, C, y)


def train(p, cfg, x0, y, abar, steps, key, batch=256, lr=3e-3, p_uncond=0.1, b1=0.9, b2=0.999, ad_eps=1e-8):
    def step(carry, k):
        q, m, v, n = carry
        ki, kt, ke, kd = jax.random.split(k, 4)
        i = jax.random.randint(ki, (batch,), 0, len(x0))
        t = jax.random.randint(kt, (batch,), 0, len(abar))
        eps = jax.random.normal(ke, x0[i].shape)
        yi = drop_condition(y[i], jax.random.uniform(kd, (batch,)), p_uncond, cfg["classes"])
        loss, g = jax.value_and_grad(eps_loss)(q, cfg, q_sample(x0[i], abar[t], eps), t, yi, eps)
        n = n + 1
        m = jax.tree.map(lambda a, b: b1 * a + (1 - b1) * b, m, g)
        v = jax.tree.map(lambda a, b: b2 * a + (1 - b2) * b * b, v, g)
        q = jax.tree.map(lambda w, mm, vv: w - lr * (mm / (1 - b1**n)) / (jnp.sqrt(vv / (1 - b2**n)) + ad_eps), q, m, v)
        return (q, m, v, n), loss

    zeros = jax.tree.map(jnp.zeros_like, p)
    (p, *_), trace = jax.lax.scan(step, (p, zeros, zeros, jnp.zeros(())), jax.random.split(key, steps))
    return p, trace
