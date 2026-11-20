# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/diffusion-model/ (ai://NEW-diffusion-model).
"""The denoising diffusion model of the page in JAX: pure functions of a parameter dictionary, the linear schedule,
the forward marginal (eq-forward-marginal), the posterior (eq-posterior), the noise-prediction network, jax.grad of
the simple loss (eq-simple-loss), the ancestral step (eq-ddpm-step), the DDIM step (eq-ddim-step), a deterministic
DDIM sampler as a jax.lax.fori_loop, and classifier-free guidance (eq-guidance). The parameters have the names and
shapes of the NumPy reference; steps are integer arrays t = 1..T, and alpha_bar_0 = 1.
"""

import math

import jax
import jax.numpy as jnp


def config(d_data=3, d_hidden=5, d_time=4, n_steps=10, beta_1=0.02, beta_T=0.5):
    if min(d_data, d_hidden, d_time, n_steps) < 1:
        raise ValueError("every size must be positive")
    if not 0 < beta_1 <= beta_T < 1:
        raise ValueError("the variances must satisfy 0 < beta_1 <= beta_T < 1")
    return {"d_data": int(d_data), "d_hidden": int(d_hidden), "d_time": int(d_time), "n_steps": int(n_steps),
            "beta_1": float(beta_1), "beta_T": float(beta_T)}  # fmt: skip


def param_shapes(cfg):
    D, H, E = cfg["d_data"], cfg["d_hidden"], cfg["d_time"]
    return {"W1": (D + E, H), "b1": (H,), "W2": (H, D), "b2": (D,)}


def param_count(cfg):
    return sum(math.prod(s) for s in param_shapes(cfg).values())


def flops_forward(cfg, batch):
    D, H, E = cfg["d_data"], cfg["d_hidden"], cfg["d_time"]
    return 2 * batch * H * (2 * D + E)


def init(key, cfg):
    p = {k: jnp.zeros(s) for k, s in param_shapes(cfg).items()}
    p["W1"] = jax.random.normal(key, p["W1"].shape) / math.sqrt(p["W1"].shape[0])
    return p


def schedule(cfg):
    beta = jnp.linspace(cfg["beta_1"], cfg["beta_T"], cfg["n_steps"])
    alpha = 1.0 - beta
    abar = jnp.cumprod(alpha)
    abar0 = jnp.concatenate([jnp.ones(1), abar])
    return {"beta": beta, "alpha": alpha, "abar": abar, "abar0": abar0,
            "beta_tilde": (1.0 - abar0[:-1]) / (1.0 - abar) * beta}  # fmt: skip


def time_embedding(t, d_time):
    """eq-time-embedding: entry 2i is sin(t w_i), entry 2i + 1 is cos(t w_i), w_i = 10000^(-2i / d_time)."""
    i = jnp.arange(d_time)
    w = 10000.0 ** (-2.0 * (i // 2) / d_time)
    a = jnp.asarray(t, jnp.float64)[:, None] * w
    return jnp.where(i % 2 == 0, jnp.sin(a), jnp.cos(a))


def eps_model(p, cfg, x, t):
    inp = jnp.concatenate([x, time_embedding(t, cfg["d_time"])], 1)
    return jnp.tanh(inp @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]


def _col(v, t):
    return v[jnp.asarray(t) - 1][:, None]


def q_sample(cfg, x0, t, eps):
    ab = _col(schedule(cfg)["abar"], t)
    return jnp.sqrt(ab) * x0 + jnp.sqrt(1.0 - ab) * eps


def posterior(cfg, xt, x0, t):
    s = schedule(cfg)
    b, a, ab, abp = (
        _col(s["beta"], t),
        _col(s["alpha"], t),
        _col(s["abar"], t),
        _col(s["abar0"][:-1], t),
    )
    mean = jnp.sqrt(abp) * b / (1.0 - ab) * x0 + jnp.sqrt(a) * (1.0 - abp) / (1.0 - ab) * xt
    return mean, _col(s["beta_tilde"], t)


def reverse_mean(cfg, xt, t, eps):
    s = schedule(cfg)
    b, a, ab = _col(s["beta"], t), _col(s["alpha"], t), _col(s["abar"], t)
    return (xt - b / jnp.sqrt(1.0 - ab) * eps) / jnp.sqrt(a)


def ddpm_step(cfg, xt, t, eps, z):
    """eq-ddpm-step with sigma_t^2 = beta_t."""
    return reverse_mean(cfg, xt, t, eps) + jnp.sqrt(_col(schedule(cfg)["beta"], t)) * z


def ddim_step(cfg, xt, t, s, eps, z, eta=0.0):
    ab0 = schedule(cfg)["abar0"]
    at, as_ = ab0[jnp.asarray(t)][:, None], ab0[jnp.asarray(s)][:, None]
    sig = eta * jnp.sqrt((1.0 - as_) / (1.0 - at) * (1.0 - at / as_))
    x0 = (xt - jnp.sqrt(1.0 - at) * eps) / jnp.sqrt(at)
    return jnp.sqrt(as_) * x0 + jnp.sqrt(1.0 - as_ - sig**2) * eps + sig * z


def simple_loss(p, cfg, x0, t, eps):
    """eq-simple-loss: the mean over rows of ||eps - eps_theta(x_t, t)||^2."""
    return jnp.mean(jnp.sum((eps_model(p, cfg, q_sample(cfg, x0, t, eps), t) - eps) ** 2, 1))


def guided_eps(eps_cond, eps_uncond, w):
    """eq-guidance."""
    return (1.0 + w) * eps_cond - w * eps_uncond


def ddim_sample(p, cfg, x_T):
    """The deterministic sampler (eta = 0) through all T steps, as a jax.lax.fori_loop that can be jitted."""
    T = cfg["n_steps"]

    def body(i, x):
        t = jnp.full((x.shape[0],), T - i)
        return ddim_step(cfg, x, t, t - 1, eps_model(p, cfg, x, t), jnp.zeros_like(x))

    return jax.lax.fori_loop(0, T, body, x_T)
