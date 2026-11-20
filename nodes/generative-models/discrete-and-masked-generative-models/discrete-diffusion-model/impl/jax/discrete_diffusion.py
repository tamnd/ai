# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/discrete-diffusion-model/ (ai://NEW-discrete-diffusion-model).
"""Discrete diffusion in JAX, as pure functions of a parameter dictionary.

- D3PM: the uniform and absorbing transition matrices of eq-d3pm-forward, their products, and the posterior of
  eq-d3pm-posterior, which with a predicted distribution in place of x0 is the reverse step of eq-x0-param.
- Masked diffusion: the linear and cosine schedules, forward masking with explicit uniforms, the SUBS denoiser of
  eq-subs, and the bound of eq-mdlm as a one-sample weighted cross-entropy whose gradient is jax.grad. The exact bound
  of one sequence is a jax.vmap over every masked subset, weighted by mask_coefficient.
- Sampling: the ancestral sampler of eq-subs as a jax.lax.fori_loop over T steps with jax.random keys.
- Score entropy (eq-score-entropy) and its denoising form, whose jax.grad vanishes at the true ratios.

The denoiser is a one-hidden-layer network on the one-hot of the partly masked sequence, as in the NumPy reference.
Tokens are integers 0, ..., K - 1 and the mask is K.
"""

import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------------------------------------------------
# D3PM


def uniform_q(beta, K):
    return (1.0 - beta) * jnp.eye(K) + beta / K * jnp.ones((K, K))


def absorbing_q(beta, K):
    return (1.0 - beta) * jnp.eye(K + 1) + beta * jnp.zeros((K + 1, K + 1)).at[:, K].set(1.0)


def uniform_qbar(alpha_bar, K):
    return uniform_q(1.0 - alpha_bar, K)


def absorbing_qbar(alpha_bar, K):
    return absorbing_q(1.0 - alpha_bar, K)


def cumulative(Qs):
    out = [jnp.eye(Qs[0].shape[0])]
    for Q in Qs:
        out.append(out[-1] @ Q)
    return out


def one_hot(z, n):
    return jax.nn.one_hot(z, n, dtype=jnp.float64)


def posterior(xt, x0, Qt, Qbar_prev):
    num = (xt @ Qt.T) * (x0 @ Qbar_prev)
    return num / num.sum(-1, keepdims=True)


def multinomial_posterior(xt, x0, alpha_t, alpha_bar_prev, K):
    th = (alpha_t * xt + (1.0 - alpha_t) / K) * (alpha_bar_prev * x0 + (1.0 - alpha_bar_prev) / K)
    return th / th.sum(-1, keepdims=True)


# ---------------------------------------------------------------------------------------------------------------------
# Masked diffusion


def alpha(t, kind="linear"):
    if kind == "linear":
        return 1.0 - t
    if kind == "cosine":
        return 1.0 - jnp.sin(0.5 * jnp.pi * t)  # = 1 - cos(pi (1 - t) / 2), exact at t = 0 and 1
    raise ValueError(kind)


def weight(t, kind="linear"):
    """-alpha_t' / (1 - alpha_t), the weight of eq-mdlm, with the derivative from jax.grad."""
    da = jax.vmap(jax.grad(lambda s: alpha(s, kind)))(jnp.atleast_1d(jnp.asarray(t, jnp.float64)))
    return (-da / (1.0 - alpha(jnp.atleast_1d(t), kind))).reshape(jnp.shape(t))


def mask_tokens(x, t, u, K, kind="linear"):
    return jnp.where(u < (1.0 - alpha(t, kind))[:, None], K, x)


def subs_probs(logits, z, K):
    keep = one_hot(jnp.where(z == K, 0, z), K)
    return jnp.where((z == K)[..., None], jax.nn.softmax(logits, -1), keep)


def mdlm_estimate(logp, z, x, t, K, kind="linear"):
    nll = -jnp.take_along_axis(logp, x[..., None], -1)[..., 0]
    return weight(t, kind) * jnp.where(z == K, nll, 0.0).sum(-1)


def mask_coefficient(m, L):
    return math.factorial(L - m) * math.factorial(m - 1) / math.factorial(L)


def all_masks(L):
    return np.array([m for m in itertools.product([False, True], repeat=L) if any(m)])


def mask_weights(L):
    return jnp.array([mask_coefficient(int(m.sum()), L) for m in all_masks(L)])


# ---------------------------------------------------------------------------------------------------------------------
# The denoiser


def config(L, K, H):
    return {"seq_len": L, "vocab": K, "hidden": H}


def param_shapes(cfg):
    L, K, H = cfg["seq_len"], cfg["vocab"], cfg["hidden"]
    return {"W1": (L * (K + 1), H), "b1": (H,), "W2": (H, L * K), "b2": (L * K,)}


def param_count(cfg):
    return sum(math.prod(s) for s in param_shapes(cfg).values())


def flops_forward(cfg, B):
    L, K, H = cfg["seq_len"], cfg["vocab"], cfg["hidden"]
    return 2 * B * L * (K + 1) * H + 2 * B * H * L * K


def init(key, cfg, scale=1.0):
    L, H = cfg["seq_len"], cfg["hidden"]
    k1, k2 = jax.random.split(key)
    s = param_shapes(cfg)
    return {"W1": scale / math.sqrt(L) * jax.random.normal(k1, s["W1"]), "b1": jnp.zeros(s["b1"]),
            "W2": scale / math.sqrt(H) * jax.random.normal(k2, s["W2"]), "b2": jnp.zeros(s["b2"])}  # fmt: skip


def forward(p, z, cfg):
    L, K = cfg["seq_len"], cfg["vocab"]
    x = one_hot(z, K + 1).reshape(z.shape[0], L * (K + 1))
    h = jnp.tanh(x @ p["W1"] + p["b1"])
    return (h @ p["W2"] + p["b2"]).reshape(z.shape[0], L, K)


def weighted_ce(p, cfg, Z, X, Wt):
    lp = jax.nn.log_softmax(forward(p, Z, cfg), -1)
    return (Wt * -jnp.take_along_axis(lp, X[..., None], -1)[..., 0]).sum()


weighted_ce_grad = jax.grad(weighted_ce)


def mc_batch(x, t, u, K, kind="linear"):
    z = mask_tokens(x, t, u, K, kind)
    return z, x, jnp.where(z == K, weight(t, kind)[:, None], 0.0) / x.shape[0]


def exact_nelbo(p, cfg, x):
    """eq-mdlm for one sequence exactly: the masked subsets in one batch, weighted by mask_coefficient."""
    K, L = cfg["vocab"], cfg["seq_len"]
    M = jnp.asarray(all_masks(L))
    Z = jnp.where(M, K, x[None])
    Wt = jnp.where(M, mask_weights(L)[:, None], 0.0)
    return weighted_ce(p, cfg, Z, jnp.broadcast_to(x, Z.shape), Wt)


def exact_batch(data_p, K):
    L = data_p.ndim
    seqs = np.array(list(itertools.product(range(K), repeat=L)))
    M = all_masks(L)
    w = np.array([mask_coefficient(int(m.sum()), L) for m in M])
    Z = np.where(M[None], K, seqs[:, None]).reshape(-1, L)
    X = np.repeat(seqs, len(M), 0)
    px = np.asarray(data_p).reshape(-1)
    Wt = (px[:, None, None] * np.where(M, w[:, None], 0.0)[None]).reshape(-1, L)
    return jnp.asarray(Z), jnp.asarray(X), jnp.asarray(Wt)


def adam_init(p):
    z = jax.tree.map(jnp.zeros_like, p)
    return {"m": z, "v": z, "t": jnp.zeros(())}


def train_exact(p, cfg, data_p, steps, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
    """Adam on the exact bound for steps steps, as a jax.lax.scan; returns the parameters and the loss trace."""
    Z, X, Wt = exact_batch(data_p, cfg["vocab"])

    def step(carry, _):
        q, s = carry
        loss, g = jax.value_and_grad(weighted_ce)(q, cfg, Z, X, Wt)
        t = s["t"] + 1
        m = jax.tree.map(lambda a, b: b1 * a + (1 - b1) * b, s["m"], g)
        v = jax.tree.map(lambda a, b: b2 * a + (1 - b2) * b * b, s["v"], g)
        q = jax.tree.map(lambda w, mm, vv: w - lr * (mm / (1 - b1**t)) / (jnp.sqrt(vv / (1 - b2**t)) + eps), q, m, v)
        return (q, {"m": m, "v": v, "t": t}), loss

    (p, _), trace = jax.lax.scan(step, (p, adam_init(p)), None, length=steps)
    return p, trace


# ---------------------------------------------------------------------------------------------------------------------
# Sampling


def sample(p, cfg, key, B, T, kind="linear"):
    """Ancestral sampling of eq-subs from the all-mask sequence over T steps, as a jax.lax.fori_loop."""
    L, K = cfg["seq_len"], cfg["vocab"]

    def body(j, carry):
        z, key = carry
        i = T - j
        a_s, a_t = alpha((i - 1) / T, kind), alpha(i / T, kind)
        key, k1, k2 = jax.random.split(key, 3)
        tok = jax.random.categorical(k1, forward(p, z, cfg), -1)
        unmask = (z == K) & (jax.random.uniform(k2, z.shape) < (a_s - a_t) / (1.0 - a_t))
        return jnp.where(unmask, tok, z), key

    z, _ = jax.lax.fori_loop(0, T, body, (jnp.full((B, L), K), key))
    return z


def masked_marginal(p, z, a, K):
    idx = tuple(slice(None) if int(zi) == K else int(zi) for zi in z)
    m = int(sum(int(zi) == K for zi in z))
    return float(jnp.sum(p[idx])) * a ** (len(z) - m) * (1.0 - a) ** m


# ---------------------------------------------------------------------------------------------------------------------
# Score entropy


def concrete_score(p):
    return p[None, :] / p[:, None]


def score_entropy(s, p, w):
    r = concrete_score(p)
    off = ~jnp.eye(p.shape[0], dtype=bool)
    terms = w * (s - r * jnp.log(s) + r * (jnp.log(r) - 1.0))
    return (p[:, None] * jnp.where(off, terms, 0.0)).sum()


def denoising_score_entropy(s, p0, kern, w):
    off = ~jnp.eye(kern.shape[1], dtype=bool)
    r = kern[:, None, :] / kern[:, :, None]
    inner = jnp.where(off, w * (s - r * jnp.log(s)), 0.0)
    return (p0[:, None, None] * kern[:, :, None] * inner).sum()


def markov_data(L, K, stay=0.1, step=0.8):
    if K < 3:
        raise ValueError("markov_data needs K >= 3, so that a third token takes the remaining mass")
    T = np.full((K, K), (1.0 - stay - step) / max(K - 2, 1))
    for a in range(K):
        T[a, a] = stay
        T[a, (a + 1) % K] = step
    p = np.zeros((K,) * L)
    for x in itertools.product(range(K), repeat=L):
        p[x] = math.prod(T[x[i], x[i + 1]] for i in range(L - 1)) / K
    return jnp.asarray(p)


def entropy(p):
    q = np.asarray(p).ravel()
    q = q[q > 0]
    return float(-(q * np.log(q)).sum())
