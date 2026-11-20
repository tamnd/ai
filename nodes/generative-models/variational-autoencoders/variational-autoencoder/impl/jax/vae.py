# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/variational-autoencoder/ (ai://NEW-variational-autoencoder).
"""The variational autoencoder of the page in JAX: pure init and apply for eq-encoder and eq-decoder, the
reparameterization of eq-reparam, the closed-form KL of eq-kl, and the objectives "elbo", "elbo_mc" and "iwae" of the
NumPy reference; jax.grad gives the gradients and jax.random the noise.

The parameter dict is that of the NumPy reference, with W of shape (n_in, n_out) and a layer computing h @ W + b. The
noise eps of shape (S, B, J) is an argument, so a fixture can fix it.
"""

import math

import jax
import jax.numpy as jnp

DECODERS = ("bernoulli", "gaussian")
OBJECTIVES = ("elbo", "elbo_mc", "iwae")
LOG2PI = math.log(2.0 * math.pi)


def config(d_in=6, d_hidden=5, d_latent=2, decoder="bernoulli"):
    if decoder not in DECODERS or min(d_in, d_hidden, d_latent) < 1:
        raise ValueError("positive sizes, and a decoder among DECODERS")
    return {"d_in": int(d_in), "d_hidden": int(d_hidden), "d_latent": int(d_latent), "decoder": decoder}


def param_shapes(cfg):
    D, H, J = cfg["d_in"], cfg["d_hidden"], cfg["d_latent"]
    s = {"enc_W": (D, H), "enc_b": (H,), "mu_W": (H, J), "mu_b": (J,), "lv_W": (H, J), "lv_b": (J,),
         "dec_W": (J, H), "dec_b": (H,), "out_W": (H, D), "out_b": (D,)}  # fmt: skip
    if cfg["decoder"] == "gaussian":
        s |= {"olv_W": (H, D), "olv_b": (D,)}
    return s


def param_count(cfg):
    return sum(math.prod(s) for s in param_shapes(cfg).values())


def init(key, cfg, scale=1.0):
    p = {}
    for k, s in param_shapes(cfg).items():
        key, sub = jax.random.split(key)
        p[k] = jax.random.normal(sub, s) * scale / math.sqrt(s[0]) if len(s) == 2 else jnp.zeros(s)
    return p


def encode(p, x):
    h = jnp.tanh(x @ p["enc_W"] + p["enc_b"])
    return h @ p["mu_W"] + p["mu_b"], h @ p["lv_W"] + p["lv_b"]


def decode(p, cfg, z):
    g = jnp.tanh(z @ p["dec_W"] + p["dec_b"])
    out = {"a": g @ p["out_W"] + p["out_b"]}
    if cfg["decoder"] == "gaussian":
        out["olv"] = g @ p["olv_W"] + p["olv_b"]
    return out


def forward(p, cfg, x, eps):
    mu, lv = encode(p, x)
    z = mu + jnp.exp(0.5 * lv) * eps  # eq-reparam
    return {"mu": mu, "lv": lv, "z": z} | decode(p, cfg, z)


def kl(mu, lv):
    return 0.5 * jnp.sum(mu**2 + jnp.exp(lv) - 1.0 - lv, -1)


def nll(cfg, x, out):
    a = out["a"]
    if cfg["decoder"] == "bernoulli":
        return jnp.sum(jax.nn.softplus(a) - x * a, -1)
    return 0.5 * jnp.sum(LOG2PI + out["olv"] + (x - a) ** 2 * jnp.exp(-out["olv"]), -1)


def log_weights(cfg, x, out, eps):
    log_prior = -0.5 * jnp.sum(LOG2PI + out["z"] ** 2, -1)
    log_q = -0.5 * jnp.sum(LOG2PI + out["lv"] + eps**2, -1)
    return -nll(cfg, x, out) + log_prior - log_q


def objective(p, cfg, x, eps, kind="elbo", beta=1.0):
    out = forward(p, cfg, x, eps)
    if kind == "elbo":
        return jnp.mean(nll(cfg, x, out)) + beta * jnp.mean(kl(out["mu"], out["lv"]))
    lw = log_weights(cfg, x, out, eps)
    if kind == "elbo_mc":
        return -jnp.mean(lw)
    if kind == "iwae":
        return -jnp.mean(jax.nn.logsumexp(lw, 0) - math.log(lw.shape[0]))
    raise ValueError(kind)


def loss(p, cfg, x, key, S=1, kind="elbo", beta=1.0):
    """The objective with fresh noise from a key: S draws per example."""
    eps = jax.random.normal(key, (S, x.shape[0], cfg["d_latent"]))
    return objective(p, cfg, x, eps, kind, beta)


def sample(p, cfg, key, n):
    out = decode(p, cfg, jax.random.normal(key, (n, cfg["d_latent"])))
    return jax.nn.sigmoid(out["a"]) if cfg["decoder"] == "bernoulli" else out["a"]
