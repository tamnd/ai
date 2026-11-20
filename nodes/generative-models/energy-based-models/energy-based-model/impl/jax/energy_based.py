# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/energy-based-model/ (ai://NEW-energy-based-model).
"""Energy-based models in JAX, as pure functions of a parameter dictionary.

- A restricted Boltzmann machine: the energy of eq-rbm-energy, the free energy of eq-free-energy, the conditionals of
  eq-rbm-conditionals, block Gibbs sampling with explicit uniforms and as a jax.lax.fori_loop, and log Z by
  enumeration. The exact gradient of eq-ml-grad is jax.grad of the mean log-likelihood; the CD and PCD estimates of
  eq-cd are jax.grad of the mean free energy of the negative samples minus that of the data, with the samples behind
  jax.lax.stop_gradient.
- A continuous energy E(x) = tanh(x W1 + b1) . w2 + gamma |x|^2 / 2. The score is jax.grad of the energy of one point
  in that point, mapped over the batch by jax.vmap; the Laplacian is the trace of jax.hessian; the score-matching
  objective of eq-score-matching is built from both and differentiated again by jax.grad. Langevin dynamics
  (eq-langevin) is a jax.lax.fori_loop, and one step of training against Langevin samples a jitted function.
- Noise-contrastive estimation (eq-nce) fitted by Newton's method with jax.grad and jax.hessian, and the energy of a
  classifier's logits (eq-jem).

Vectors are rows, and W has the shape (D, H), as in the NumPy reference.
"""

import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def config(d_visible=4, d_hidden=3):
    if d_visible < 1 or d_hidden < 1:
        raise ValueError("an RBM needs at least one visible and one hidden unit")
    return {"d_visible": int(d_visible), "d_hidden": int(d_hidden)}


def param_shapes(cfg):
    D, H = cfg["d_visible"], cfg["d_hidden"]
    return {"W": (D, H), "b": (D,), "c": (H,)}


def param_count(cfg):
    return sum(math.prod(s) for s in param_shapes(cfg).values())


def flops_forward(cfg, batch):
    """eq-flops: one full Gibbs step v -> h -> v on B vectors, the products v W and h W^T, a multiply-add as 2."""
    return 4 * batch * cfg["d_visible"] * cfg["d_hidden"]


def init(key, cfg, scale=0.01):
    p = {k: jnp.zeros(s) for k, s in param_shapes(cfg).items()}
    p["W"] = scale * jax.random.normal(key, p["W"].shape)
    return p


def all_binary(D):
    return jnp.array(list(itertools.product([0.0, 1.0], repeat=D)))


def energy(p, v, h):
    """eq-rbm-energy for rows v (B, D) and h (B, H)."""
    return -jnp.sum((v @ p["W"]) * h, 1) - v @ p["b"] - h @ p["c"]


def free_energy(p, v):
    """eq-free-energy: F(v) = -b.v - sum_j softplus(c_j + (v W)_j)."""
    return -v @ p["b"] - jnp.sum(jax.nn.softplus(p["c"] + v @ p["W"]), 1)


def p_h(p, v):
    return jax.nn.sigmoid(p["c"] + v @ p["W"])


def p_v(p, h):
    return jax.nn.sigmoid(p["b"] + h @ p["W"].T)


def log_partition(p):
    return jax.nn.logsumexp(-free_energy(p, all_binary(p["W"].shape[0])))


def log_prob(p, v):
    return -free_energy(p, v) - log_partition(p)


def mean_log_lik(p, data):
    """Its jax.grad is eq-ml-grad, exact because log Z is enumerated."""
    return jnp.mean(log_prob(p, data))


def cd_objective(p, pos, neg):
    """mean F(neg) - mean F(pos) with neg held fixed: its jax.grad is eq-ml-grad with the model expectation replaced by
    an average over neg, the CD-k estimate of eq-cd when neg are k Gibbs steps from pos."""
    return jnp.mean(free_energy(p, jax.lax.stop_gradient(neg))) - jnp.mean(free_energy(p, pos))


def gibbs_step(p, v, u_h, u_v):
    h = (u_h < p_h(p, v)).astype(v.dtype)
    return h, (u_v < p_v(p, h)).astype(v.dtype)


def gibbs_chain(p, v, k, key):
    """k block Gibbs steps as a fori_loop, the uniforms of step s drawn from fold_in(key, s)."""
    H = p["W"].shape[1]

    def body(s, v):
        kh, kv = jax.random.split(jax.random.fold_in(key, s))
        _, v = gibbs_step(p, v, jax.random.uniform(kh, (v.shape[0], H)), jax.random.uniform(kv, v.shape))
        return v

    return jax.lax.fori_loop(0, k, body, v)


def make_rbm_step(k=1, persistent=False, batch=100, lr=0.05):
    """One jitted update of gradient ascent on cd_objective: CD-k from the batch, or PCD from the chains."""

    @jax.jit
    def step(p, chains, data, key):
        kb, kc = jax.random.split(key)
        pos = data[jax.random.randint(kb, (batch,), 0, data.shape[0])]
        neg = gibbs_chain(p, chains if persistent else pos, k, kc)
        g = jax.grad(cd_objective)(p, pos, neg)
        p = jax.tree.map(lambda w, d: w + lr * d, p, g)
        return p, (neg if persistent else chains)

    return step


# A continuous energy: E(x) = tanh(x W1 + b1) . w2 + gamma |x|^2 / 2.


def mlp_shapes(d, H):
    return {"W1": (d, H), "b1": (H,), "w2": (H,)}


def mlp_init(key, d, H, scale=1.0):
    k1, k2 = jax.random.split(key)
    return {"W1": scale / math.sqrt(d) * jax.random.normal(k1, (d, H)), "b1": 0.5 * jax.random.normal(k2, (H,)),
            "w2": jnp.zeros(H)}  # fmt: skip


def mlp_energy1(q, x, gamma=0.1):
    """The energy of one point x of shape (d,)."""
    return jnp.tanh(x @ q["W1"] + q["b1"]) @ q["w2"] + 0.5 * gamma * jnp.sum(x * x)


def mlp_energy(q, x, gamma=0.1):
    return jax.vmap(mlp_energy1, (None, 0, None))(q, x, gamma)


def grad_x(q, x, gamma=0.1):
    """dE/dx row by row: jax.grad of the energy of one point, vmapped."""
    return jax.vmap(jax.grad(mlp_energy1, 1), (None, 0, None))(q, x, gamma)


def laplacian(q, x, gamma=0.1):
    """The trace of the Hessian in x of the energy of each row."""
    return jax.vmap(lambda u: jnp.trace(jax.hessian(mlp_energy1, 1)(q, u, gamma)))(x)


def sm_objective(q, x, gamma=0.1):
    """eq-score-matching with psi = -dE/dx: the mean of -Laplacian E + |dE/dx|^2 / 2; jax.grad of it in q is the
    gradient of score matching, a derivative of second derivatives."""
    g = grad_x(q, x, gamma)
    return jnp.mean(-laplacian(q, x, gamma) + 0.5 * jnp.sum(g * g, 1))


def langevin_step(q, x, step, z, noise=None, gamma=0.1):
    """eq-langevin with the standard normal draws z: x - (step/2) dE/dx + noise z, noise = sqrt(step) unless given."""
    s = math.sqrt(step) if noise is None else noise
    return x - 0.5 * step * grad_x(q, x, gamma) + s * z


def langevin(q, x, n_steps, step, key, noise=None, gamma=0.1):
    def body(i, x):
        return langevin_step(q, x, step, jax.random.normal(jax.random.fold_in(key, i), x.shape), noise, gamma)

    return jax.lax.fori_loop(0, n_steps, body, x)


def adam_init(q):
    return {"m": jax.tree.map(jnp.zeros_like, q), "v": jax.tree.map(jnp.zeros_like, q), "t": 0}


def make_ebm_step(gamma=0.1, batch=128, k=20, step=0.02, noise=0.05, alpha=0.1, lr=0.01, box=3.0, reinit=0.05,
                  b1=0.9, b2=0.999, eps=1e-8):  # fmt: skip
    """One jitted iteration of Algorithm 1 of Du and Mordatch: chains from the replay buffer (fresh uniform noise with
    probability reinit), k Langevin steps, and an Adam step on mean E(x+) - mean E(x-) + alpha (mean E(x+)^2 +
    mean E(x-)^2) with the samples fixed. Returns the parameters, the optimizer state, the buffer and
    mean E(x+) - mean E(x-)."""

    def loss(q, pos, neg):
        ep, en = mlp_energy(q, pos, gamma), mlp_energy(q, neg, gamma)
        return jnp.mean(ep) - jnp.mean(en) + alpha * (jnp.mean(ep**2) + jnp.mean(en**2)), jnp.mean(ep) - jnp.mean(en)

    @jax.jit
    def update(q, st, buf, data, key):
        kd, ki, kr, ku, kl = jax.random.split(key, 5)
        pos = data[jax.random.randint(kd, (batch,), 0, data.shape[0])]
        idx = jax.random.randint(ki, (batch,), 0, buf.shape[0])
        fresh = jax.random.uniform(kr, (batch, 1)) < reinit
        x0 = jnp.where(fresh, jax.random.uniform(ku, (batch, buf.shape[1]), minval=-box, maxval=box), buf[idx])
        neg = jnp.clip(langevin(q, x0, k, step, kl, noise, gamma), -box, box)
        buf = buf.at[idx].set(neg)
        g, gap = jax.grad(loss, has_aux=True)(q, pos, neg)
        t = st["t"] + 1
        m = jax.tree.map(lambda a, b: b1 * a + (1 - b1) * b, st["m"], g)
        v = jax.tree.map(lambda a, b: b2 * a + (1 - b2) * b**2, st["v"], g)
        q = jax.tree.map(lambda w, mi, vi: w - lr * (mi / (1 - b1**t)) / (jnp.sqrt(vi / (1 - b2**t)) + eps), q, m, v)
        return q, {"m": m, "v": v, "t": t}, buf, gap

    return update


def gaussian_energy(mu, M, x):
    r = x - mu
    return 0.5 * jnp.sum((r @ M) * r, 1)


def gaussian_sm_objective(mu, M, x):
    """eq-score-matching for the Gaussian energy, with the score and its divergence from jax.grad and jax.jacfwd."""

    def e1(u):
        return 0.5 * (u - mu) @ M @ (u - mu)

    g = jax.vmap(jax.grad(e1))(x)
    lap = jax.vmap(lambda u: jnp.trace(jax.jacfwd(jax.grad(e1))(u)))(x)
    return jnp.mean(-lap + 0.5 * jnp.sum(g * g, 1))


# Noise-contrastive estimation of log p_m(x) = theta . (-x^2/2, x, 1), with Gaussian noise (eq-nce, nu = 1).


def nce_features(x):
    return jnp.stack([-0.5 * x * x, x, jnp.ones_like(x)], 1)


def gauss_logpdf(x, mean, std):
    return -0.5 * ((x - mean) / std) ** 2 - math.log(std) - 0.5 * math.log(2 * math.pi)


def nce_objective(theta, x, y, noise_mean, noise_std):
    gx = nce_features(x) @ theta - gauss_logpdf(x, noise_mean, noise_std)
    gy = nce_features(y) @ theta - gauss_logpdf(y, noise_mean, noise_std)
    return jnp.mean(jax.nn.log_sigmoid(gx)) + jnp.mean(jax.nn.log_sigmoid(-gy))


def nce_fit(x, y, noise_mean, noise_std, iters=50):
    """Newton's method on the concave J with jax.grad and jax.hessian, from theta = (1, 0, 0), as a fori_loop."""
    grad = jax.grad(nce_objective)
    hess = jax.hessian(nce_objective)

    def body(_, theta):
        return theta - jnp.linalg.solve(
            hess(theta, x, y, noise_mean, noise_std), grad(theta, x, y, noise_mean, noise_std)
        )

    return jax.lax.fori_loop(0, iters, body, jnp.array([1.0, 0.0, 0.0]))


def gaussian_natural(mean, std):
    return np.array([1 / std**2, mean / std**2, -0.5 * mean**2 / std**2 - math.log(std) - 0.5 * math.log(2 * math.pi)])


def jem_energy(logits):
    """E(x) = -logsumexp_y f(x)[y] for the (B, K) logits of a classifier (eq-jem)."""
    return -jax.nn.logsumexp(logits, 1)


def jem_class_probs(logits):
    return jax.nn.softmax(logits, 1)
