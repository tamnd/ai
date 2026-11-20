# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/latent-diffusion-model/ (ai://NEW-latent-diffusion-model).
"""A latent diffusion model in JAX, as pure functions of a parameter dictionary.

- The first stage: the principal component autoencoder that downsamples by f (eq-autoencoder), the KL penalty and the
  latent scale of eq-scale.
- Vector quantization (eq-vq): the straight-through estimator as z_e + stop_gradient(z_q - z_e), the codebook and
  commitment terms of eq-vq-loss with jax.lax.stop_gradient, their gradients by jax.grad, the k-means step and the
  EMA codebook of eq-ema with segment sums.
- The second stage: the linear schedule, forward noising, the exact noise predictor of a Gaussian mixture, DDIM as a
  jax.lax.scan over the step pairs, and classifier-free guidance.
- The denoiser conditioned through the cross-attention of eq-cross-attn, its loss gradient by jax.grad and Adam on
  eq-ldm as a jax.lax.scan with jax.random keys.
"""

import math

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------------------------------------------------
# The first stage


def patches(x, f):
    return x.reshape(x.shape[0], -1, f)


def fit_autoencoder(x, f, c):
    P = patches(x, f).reshape(-1, f)
    mean = P.mean(0)
    _, vecs = jnp.linalg.eigh((P - mean).T @ (P - mean) / P.shape[0])
    We = vecs[:, ::-1][:, :c]
    return {"We": We, "be": -mean @ We, "Wd": We.T, "bd": mean}


def encode(ae, x, f):
    return patches(x, f) @ ae["We"] + ae["be"]


def decode(ae, z):
    return (z @ ae["Wd"] + ae["bd"]).reshape(z.shape[0], -1)


def kl_to_standard_normal(mu, logvar):
    return 0.5 * (mu**2 + jnp.exp(logvar) - 1.0 - logvar).reshape(mu.shape[0], -1).sum(1)


def latent_scale(z):
    return float(jnp.std(z))


# ---------------------------------------------------------------------------------------------------------------------
# Vector quantization


def quantize(z, codebook):
    d = (z**2).sum(-1, keepdims=True) - 2.0 * z @ codebook.T + (codebook**2).sum(-1)
    idx = d.argmin(-1)
    return idx, codebook[idx]


def straight_through(z_e, z_q):
    return z_e + jax.lax.stop_gradient(z_q - z_e)


def vq_terms(z_e, codebook, idx, beta):
    sg = jax.lax.stop_gradient
    e = codebook[idx]
    return ((sg(z_e) - e) ** 2).sum(-1).mean(), beta * ((z_e - sg(e)) ** 2).sum(-1).mean()


def vq_grads(z_e, codebook, idx, g_zq, beta):
    """The gradients of <g_zq, z_q> through the straight-through estimator plus the two terms, by jax.grad."""

    def loss(ze, cb):
        sq, commit = vq_terms(ze, cb, idx, beta)
        return (g_zq * straight_through(ze, cb[idx])).sum() + sq + commit

    return jax.grad(loss, argnums=(0, 1))(z_e, codebook)


def kmeans_step(z, codebook):
    z = z.reshape(-1, codebook.shape[1])
    idx, _ = quantize(z, codebook)
    n = jax.ops.segment_sum(jnp.ones(len(z)), idx, len(codebook))
    s = jax.ops.segment_sum(z, idx, len(codebook))
    return jnp.where(n[:, None] > 0, s / jnp.maximum(n, 1.0)[:, None], codebook)


def ema_init(codebook):
    return {"N": jnp.zeros(len(codebook)), "m": jnp.zeros_like(codebook), "e": codebook}


def ema_update(state, z_e, idx, gamma):
    K = len(state["e"])
    z, idx = z_e.reshape(-1, state["e"].shape[1]), idx.reshape(-1)
    N = gamma * state["N"] + (1 - gamma) * jax.ops.segment_sum(jnp.ones(len(z)), idx, K)
    m = gamma * state["m"] + (1 - gamma) * jax.ops.segment_sum(z, idx, K)
    e = jnp.where(N[:, None] > 0, m / jnp.where(N > 0, N, 1.0)[:, None], state["e"])
    return {"N": N, "m": m, "e": e}


# ---------------------------------------------------------------------------------------------------------------------
# Diffusion in the latent space


def alpha_bars(T, b0=1e-4, b1=2e-2):
    return jnp.cumprod(1.0 - jnp.linspace(b0, b1, T))


def q_sample(z0, abar, eps):
    a = jnp.reshape(abar, (-1,) + (1,) * (z0.ndim - 1))
    return jnp.sqrt(a) * z0 + jnp.sqrt(1.0 - a) * eps


def snr(abar, var=1.0):
    return abar * var / (1.0 - abar)


def exact_eps(zt, abar, means, s2, weights):
    a = jnp.reshape(abar, (-1, 1))
    v = a * s2 + 1.0 - a
    r = zt[:, None, :] - jnp.sqrt(a)[:, None] * means[None]
    w = jax.nn.softmax(jnp.log(weights)[None] - 0.5 * (r**2).sum(-1) / v, 1)
    z0 = means[None] + (jnp.sqrt(a) * s2 / v)[:, None] * r
    return (zt - jnp.sqrt(a) * jnp.einsum("bk,bkd->bd", w, z0)) / jnp.sqrt(1.0 - a)


def ddim_sample(eps_fn, zT, abar, steps):
    """Deterministic DDIM as a jax.lax.scan over the pairs (t, t_prev); the last step goes to alpha_bar = 1."""
    steps = jnp.asarray(steps)
    ab = jnp.concatenate([abar, jnp.ones(1)])
    prev = jnp.concatenate([steps[1:], jnp.array([len(abar)])])

    def step(z, tp):
        t, t_prev = tp
        a, a_prev = ab[t], ab[t_prev]
        e = eps_fn(z, t)
        z0 = (z - jnp.sqrt(1.0 - a) * e) / jnp.sqrt(a)
        return jnp.sqrt(a_prev) * z0 + jnp.sqrt(1.0 - a_prev) * e, None

    z, _ = jax.lax.scan(step, zT, (steps, prev))
    return z


def guided(e_cond, e_null, s):
    return e_null + s * (e_cond - e_null)


# ---------------------------------------------------------------------------------------------------------------------
# The denoiser


def config(N, c, H, E, M, d, C):
    return {"positions": N, "channels": c, "hidden": H, "temb": E, "ctx_tokens": M, "attn": d, "classes": C}


def param_shapes(cfg):
    D, H, E, M, d, C = (cfg["positions"] * cfg["channels"], cfg["hidden"], cfg["temb"], cfg["ctx_tokens"],
                        cfg["attn"], cfg["classes"])  # fmt: skip
    return {"W1": (D, H), "b1": (H,), "Wt": (E, H), "ctx": (C + 1, M, d), "Wq": (H, d), "Wk": (d, d),
            "Wv": (d, H), "W2": (H, D), "b2": (D,)}  # fmt: skip


def param_count(cfg):
    return sum(math.prod(s) for s in param_shapes(cfg).values())


def flops_forward(cfg, B):
    D, H, E, M, d = cfg["positions"] * cfg["channels"], cfg["hidden"], cfg["temb"], cfg["ctx_tokens"], cfg["attn"]
    return 2 * B * (D * H + E * H + H * d + M * d * d + M * d * H + M * d + M * H + H * D)


def init(key, cfg):
    shapes = param_shapes(cfg)
    keys = jax.random.split(key, len(shapes))
    out = {}
    for k, (name, s) in zip(keys, shapes.items(), strict=True):
        if name.startswith("b"):
            out[name] = jnp.zeros(s)
        else:
            out[name] = jax.random.normal(k, s) / (math.sqrt(s[-2]) if name != "ctx" else 1.0)
    return out


def time_embedding(t, E):
    w = jnp.exp(-math.log(10000.0) * jnp.arange(E // 2) / (E // 2))
    a = jnp.asarray(t, jnp.float64)[:, None] * w
    return jnp.concatenate([jnp.sin(a), jnp.cos(a)], -1)


def forward(p, cfg, z, t, y):
    B = z.shape[0]
    zf = z.reshape(B, -1)
    h = jnp.tanh(zf @ p["W1"] + time_embedding(jnp.broadcast_to(t, (B,)), cfg["temb"]) @ p["Wt"] + p["b1"])
    q, Cx = h @ p["Wq"], p["ctx"][y]
    w = jax.nn.softmax(jnp.einsum("bd,bmd->bm", q, Cx @ p["Wk"]) / math.sqrt(cfg["attn"]), 1)
    return zf + jnp.tanh(h + jnp.einsum("bm,bmh->bh", w, Cx @ p["Wv"])) @ p["W2"] + p["b2"]


def eps_loss(p, cfg, zt, t, y, eps):
    return ((forward(p, cfg, zt, t, y) - eps.reshape(len(eps), -1)) ** 2).sum(1).mean()


eps_loss_grad = jax.grad(eps_loss)


def train(p, cfg, z0, y, abar, steps, key, batch=256, lr=3e-3, p_null=0.1, b1=0.9, b2=0.999, ad_eps=1e-8):
    """Adam on eq-ldm as a jax.lax.scan, one jax.random key per step; returns the parameters and the loss trace."""

    def step(carry, k):
        q, m, v, n = carry
        ki, kt, ke, kd = jax.random.split(k, 4)
        i = jax.random.randint(ki, (batch,), 0, len(z0))
        t = jax.random.randint(kt, (batch,), 0, len(abar))
        eps = jax.random.normal(ke, z0[i].shape)
        yi = jnp.where(jax.random.uniform(kd, (batch,)) < p_null, cfg["classes"], y[i])
        loss, g = jax.value_and_grad(eps_loss)(q, cfg, q_sample(z0[i], abar[t], eps), t, yi, eps)
        n = n + 1
        m = jax.tree.map(lambda a, b: b1 * a + (1 - b1) * b, m, g)
        v = jax.tree.map(lambda a, b: b2 * a + (1 - b2) * b * b, v, g)
        q = jax.tree.map(lambda w, mm, vv: w - lr * (mm / (1 - b1**n)) / (jnp.sqrt(vv / (1 - b2**n)) + ad_eps), q, m, v)
        return (q, m, v, n), loss

    zeros = jax.tree.map(jnp.zeros_like, p)
    (p, *_), trace = jax.lax.scan(step, (p, zeros, zeros, jnp.zeros(())), jax.random.split(key, steps))
    return p, trace
