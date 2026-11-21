# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/energy-based-model/ (ai://NEW-energy-based-model).
"""Energy-based models in NumPy, every gradient written by hand.

- A restricted Boltzmann machine over binary vectors: the energy of eq-rbm-energy, the free energy of eq-free-energy,
  the factorized conditionals of eq-rbm-conditionals, block Gibbs sampling, the exact log-partition function by
  enumeration for small models, the exact log-likelihood gradient of eq-ml-grad, and the contrastive divergence (CD-k)
  and persistent contrastive divergence (PCD) estimates of it (eq-cd).
- A continuous energy given by a one-hidden-layer network plus a fixed quadratic term: its energy, its gradient in the
  input and in the parameters, Langevin dynamics (eq-langevin), and training by contrasting data with Langevin samples
  started from a replay buffer.
- Score matching (eq-score-matching) for a Gaussian energy, where it has a closed form, and its objective for the
  network energy, whose Laplacian is written out.
- Noise-contrastive estimation (eq-nce) of a one-dimensional unnormalized Gaussian with its log-normalizer as a
  parameter, fitted by Newton's method.
- The energy of a classifier's logits (eq-jem).

Vectors are rows: a batch of B binary vectors is a (B, D) array, and W has the shape (D, H).
"""

import itertools
import math

import numpy as np


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def softplus(z):
    return np.logaddexp(0.0, z)


def logsumexp(a, axis=-1):
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(a - m), axis=axis))


# ---------------------------------------------------------------------------------------------------------------------
# Restricted Boltzmann machine: E(v, h) = -v W h - b.v - c.h over v in {0,1}^D, h in {0,1}^H.


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


def init(rng, cfg, scale=0.01):
    """Small random weights and zero biases, the usual start."""
    p = {k: np.zeros(s) for k, s in param_shapes(cfg).items()}
    p["W"] = rng.normal(0.0, scale, p["W"].shape)
    return p


def energy(p, v, h):
    """eq-rbm-energy for rows v (B, D) and h (B, H)."""
    return -np.sum((v @ p["W"]) * h, 1) - v @ p["b"] - h @ p["c"]


def free_energy(p, v):
    """eq-free-energy: F(v) = -b.v - sum_j softplus(c_j + (v W)_j), so p(v) = exp(-F(v)) / Z."""
    return -v @ p["b"] - np.sum(softplus(p["c"] + v @ p["W"]), 1)


def p_h(p, v):
    """eq-rbm-conditionals: p(h_j = 1 | v) = sigmoid(c_j + (v W)_j), independent over j."""
    return sigmoid(p["c"] + v @ p["W"])


def p_v(p, h):
    """eq-rbm-conditionals: p(v_i = 1 | h) = sigmoid(b_i + (W h)_i), independent over i."""
    return sigmoid(p["b"] + h @ p["W"].T)


def all_binary(D):
    return np.array(list(itertools.product([0.0, 1.0], repeat=D)))


def log_partition(p):
    """log Z = log sum_v exp(-F(v)), by enumerating the 2^D visible vectors."""
    return float(logsumexp(-free_energy(p, all_binary(p["W"].shape[0])), 0))


def log_prob(p, v):
    return -free_energy(p, v) - log_partition(p)


def gibbs_step(p, v, u_h, u_v):
    """One block Gibbs step v -> h -> v' with the uniforms u_h (B, H) and u_v (B, D): h_j = 1[u < p(h_j = 1 | v)]."""
    h = (u_h < p_h(p, v)).astype(float)
    return h, (u_v < p_v(p, h)).astype(float)


def gibbs_chain(p, v, k, rng):
    for _ in range(k):
        _, v = gibbs_step(p, v, rng.random((len(v), p["W"].shape[1])), rng.random(v.shape))
    return v


def stats(p, v):
    """The mean over the rows of v of -dF/dtheta: v^T p(h | v), v and p(h | v)."""
    ph = p_h(p, v)
    return {"W": v.T @ ph / len(v), "b": v.mean(0), "c": ph.mean(0)}


def grad_from_samples(p, pos, neg):
    """eq-ml-grad with the model expectation replaced by an average over the samples neg: the ascent direction of the
    mean log-likelihood of pos. With neg drawn exactly from the model it is unbiased."""
    sp, sn = stats(p, pos), stats(p, neg)
    return {k: sp[k] - sn[k] for k in sp}


def model_probs(p):
    """p(v) for every v of all_binary(D), in that order."""
    return np.exp(-free_energy(p, all_binary(p["W"].shape[0])) - log_partition(p))


def exact_grad(p, data):
    """eq-ml-grad exactly: the data statistics minus the statistics weighted by p(v) over all 2^D vectors."""
    X = all_binary(p["W"].shape[0])
    w = model_probs(p)
    ph = p_h(p, X)
    sp = stats(p, data)
    return {"W": sp["W"] - (X * w[:, None]).T @ ph, "b": sp["b"] - w @ X, "c": sp["c"] - w @ ph}


def gibbs_transition(p):
    """T[a, b] = P(v' = X[b] | v = X[a]) for one block Gibbs step, over all 2^D visible vectors X and 2^H hidden ones."""
    X, Hs = all_binary(p["W"].shape[0]), all_binary(p["W"].shape[1])
    ph, pv = p_h(p, X), p_v(p, Hs)
    P_h = np.prod(np.where(Hs[None] == 1, ph[:, None], 1 - ph[:, None]), -1)  # (2^D, 2^H)
    P_v = np.prod(np.where(X[None] == 1, pv[:, None], 1 - pv[:, None]), -1)  # (2^H, 2^D)
    return P_h @ P_v


def train_rbm(p, data, steps, rng, k=1, persistent=False, n_chains=100, batch=100, lr=0.05):
    """Gradient ascent on the log-likelihood with the CD-k estimate (chains started at the batch) or the PCD estimate
    (n_chains persistent chains advanced k steps per update). Returns the parameters and the trace of the exact mean
    log-likelihood when D is small enough to enumerate, else an empty list."""
    p = {key: v.copy() for key, v in p.items()}
    chains = data[rng.integers(0, len(data), n_chains)].copy()
    trace = []
    for _ in range(steps):
        pos = data[rng.integers(0, len(data), batch)]
        if persistent:
            chains = gibbs_chain(p, chains, k, rng)
            neg = chains
        else:
            neg = gibbs_chain(p, pos, k, rng)
        g = grad_from_samples(p, pos, neg)
        for key in p:
            p[key] += lr * g[key]
        if p["W"].shape[0] <= 12:
            trace.append(float(log_prob(p, data).mean()))
    return p, trace


# ---------------------------------------------------------------------------------------------------------------------
# A continuous energy: E(x) = tanh(x W1 + b1) . w2 + gamma |x|^2 / 2, for x in R^d.


def mlp_shapes(d, H):
    return {"W1": (d, H), "b1": (H,), "w2": (H,)}


def mlp_init(rng, d, H, scale=1.0):
    return {"W1": rng.normal(0.0, scale / math.sqrt(d), (d, H)), "b1": rng.normal(0.0, 0.5, H), "w2": np.zeros(H)}


def mlp_energy(q, x, gamma=0.1):
    t = np.tanh(x @ q["W1"] + q["b1"])
    return t @ q["w2"] + 0.5 * gamma * np.sum(x * x, 1)


def mlp_grad_x(q, x, gamma=0.1):
    """dE/dx, row by row: ((1 - t^2) * w2) W1^T + gamma x."""
    t = np.tanh(x @ q["W1"] + q["b1"])
    return ((1 - t * t) * q["w2"]) @ q["W1"].T + gamma * x


def mlp_grad_params(q, x):
    """The mean over the rows of x of dE/dtheta; the quadratic term has no parameter."""
    t = np.tanh(x @ q["W1"] + q["b1"])
    s = (1 - t * t) * q["w2"]
    return {"W1": x.T @ s / len(x), "b1": s.mean(0), "w2": t.mean(0)}


def mlp_laplacian(q, x, gamma=0.1):
    """sum_i d^2E/dx_i^2 = sum_j w2_j tanh''(z_j) |W1[:, j]|^2 + gamma d, with tanh'' = -2 t (1 - t^2)."""
    t = np.tanh(x @ q["W1"] + q["b1"])
    return (-2 * t * (1 - t * t) * q["w2"]) @ np.sum(q["W1"] ** 2, 0) + gamma * x.shape[1]


def langevin_step(q, x, step, z, noise=None, gamma=0.1):
    """eq-langevin with the standard normal draws z: x - (step/2) dE/dx + noise z, noise = sqrt(step) unless given."""
    s = math.sqrt(step) if noise is None else noise
    return x - 0.5 * step * mlp_grad_x(q, x, gamma) + s * z


def langevin(q, x, n_steps, step, rng, noise=None, gamma=0.1):
    for _ in range(n_steps):
        x = langevin_step(q, x, step, rng.standard_normal(x.shape), noise, gamma)
    return x


def train_ebm(q, data, steps, rng, gamma=0.1, batch=128, k=20, step=0.02, noise=0.05, alpha=0.1, lr=0.01, box=3.0,
              buffer_size=4096, reinit=0.05):  # fmt: skip
    """Contrast data with Langevin samples (Algorithm 1 of Du and Mordatch): start each chain from the replay buffer,
    or from uniform noise on [-box, box]^d with probability reinit, run k steps of eq-langevin, and descend
    mean E(x+) - mean E(x-) + alpha (mean E(x+)^2 + mean E(x-)^2), the samples held fixed, with Adam.
    Returns the parameters, the buffer and the trace of mean E(x+) - mean E(x-)."""
    q = {key: v.copy() for key, v in q.items()}
    d = data.shape[1]
    buf = rng.uniform(-box, box, (buffer_size, d))
    m = {key: np.zeros_like(v) for key, v in q.items()}
    s = {key: np.zeros_like(v) for key, v in q.items()}
    trace = []
    for it in range(1, steps + 1):
        pos = data[rng.integers(0, len(data), batch)]
        idx = rng.integers(0, buffer_size, batch)
        x0 = buf[idx].copy()
        fresh = rng.random(batch) < reinit
        x0[fresh] = rng.uniform(-box, box, (int(fresh.sum()), d))
        neg = np.clip(langevin(q, x0, k, step, rng, noise, gamma), -box, box)
        buf[idx] = neg
        ep, en = mlp_energy(q, pos, gamma), mlp_energy(q, neg, gamma)
        gp, gn = mlp_grad_params(q, pos), mlp_grad_params(q, neg)
        # d/dtheta of alpha * mean E^2 is alpha * mean 2 E dE/dtheta; mlp_grad_params gives means, so weight the rows
        t_pos = np.tanh(pos @ q["W1"] + q["b1"])
        t_neg = np.tanh(neg @ q["W1"] + q["b1"])
        g2 = {}
        for e, xx, t in ((ep, pos, t_pos), (en, neg, t_neg)):
            sw = (1 - t * t) * q["w2"] * (2 * e)[:, None]
            part = {"W1": xx.T @ sw / batch, "b1": sw.mean(0), "w2": (t * (2 * e)[:, None]).mean(0)}
            for key in part:
                g2[key] = g2.get(key, 0.0) + part[key]
        for key in q:
            g = gp[key] - gn[key] + alpha * g2[key]
            m[key] = 0.9 * m[key] + 0.1 * g
            s[key] = 0.999 * s[key] + 0.001 * g * g
            q[key] = q[key] - lr * (m[key] / (1 - 0.9**it)) / (np.sqrt(s[key] / (1 - 0.999**it)) + 1e-8)
        trace.append(float(ep.mean() - en.mean()))
    return q, buf, trace


# ---------------------------------------------------------------------------------------------------------------------
# Score matching: J(theta) = E_data[ sum_i d psi_i / dx_i + psi_i^2 / 2 ] with psi = -dE/dx (eq-score-matching).


def gaussian_energy(mu, M, x):
    """E(x) = (x - mu) M (x - mu)^T / 2 for a symmetric positive-definite M."""
    r = x - mu
    return 0.5 * np.sum((r @ M) * r, 1)


def gaussian_sm_objective(mu, M, x):
    """eq-score-matching for the Gaussian energy: psi = -(x - mu) M, d psi_i / dx_i = -M_ii."""
    r = (x - mu) @ M
    return float(np.mean(-np.trace(M) + 0.5 * np.sum(r * r, 1)))


def gaussian_sm_fit(x):
    """The minimizer of gaussian_sm_objective: the sample mean and the inverse of the sample covariance (1/T)."""
    mu = x.mean(0)
    r = x - mu
    return mu, np.linalg.inv(r.T @ r / len(x))


def mlp_sm_objective(q, x, gamma=0.1):
    """eq-score-matching for the network energy: mean of -Laplacian E + |dE/dx|^2 / 2."""
    g = mlp_grad_x(q, x, gamma)
    return float(np.mean(-mlp_laplacian(q, x, gamma) + 0.5 * np.sum(g * g, 1)))


# ---------------------------------------------------------------------------------------------------------------------
# Noise-contrastive estimation of log p_m(x) = theta . (-x^2/2, x, 1), with Gaussian noise (eq-nce, nu = 1).


def nce_features(x):
    return np.stack([-0.5 * x * x, x, np.ones_like(x)], 1)


def gauss_logpdf(x, mean, std):
    return -0.5 * ((x - mean) / std) ** 2 - math.log(std) - 0.5 * math.log(2 * math.pi)


def nce_objective(theta, x, y, noise_mean, noise_std):
    """J = mean log sigmoid(G(x)) + mean log(1 - sigmoid(G(y))), G(u) = log p_m(u) - log p_n(u), with the data x and
    as many noise samples y."""
    gx = nce_features(x) @ theta - gauss_logpdf(x, noise_mean, noise_std)
    gy = nce_features(y) @ theta - gauss_logpdf(y, noise_mean, noise_std)
    return float(np.mean(-softplus(-gx)) + np.mean(-softplus(gy)))


def nce_grad(theta, x, y, noise_mean, noise_std):
    fx, fy = nce_features(x), nce_features(y)
    gx = fx @ theta - gauss_logpdf(x, noise_mean, noise_std)
    gy = fy @ theta - gauss_logpdf(y, noise_mean, noise_std)
    return (1 - sigmoid(gx)) @ fx / len(x) - sigmoid(gy) @ fy / len(y)


def nce_fit(x, y, noise_mean, noise_std, iters=50):
    """Newton's method on the concave J; the start theta = (1, 0, 0) is a standard normal without its normalizer."""
    theta = np.array([1.0, 0.0, 0.0])
    fx, fy = nce_features(x), nce_features(y)
    for _ in range(iters):
        g = nce_grad(theta, x, y, noise_mean, noise_std)
        sx = sigmoid(fx @ theta - gauss_logpdf(x, noise_mean, noise_std))
        sy = sigmoid(fy @ theta - gauss_logpdf(y, noise_mean, noise_std))
        Hm = -(fx * (sx * (1 - sx))[:, None]).T @ fx / len(x) - (fy * (sy * (1 - sy))[:, None]).T @ fy / len(y)
        theta = theta - np.linalg.solve(Hm, g)
    return theta


def gaussian_natural(mean, std):
    """The theta of nce_features for N(mean, std^2), with its log-normalizer in the third entry."""
    return np.array([1 / std**2, mean / std**2, -0.5 * mean**2 / std**2 - math.log(std) - 0.5 * math.log(2 * math.pi)])


# ---------------------------------------------------------------------------------------------------------------------
# A classifier as an energy-based model (eq-jem).


def jem_energy(logits):
    """E(x) = -logsumexp_y f(x)[y] for the (B, K) logits of a classifier."""
    return -logsumexp(logits, 1)


def jem_class_probs(logits):
    """p(y | x) = softmax(f(x)), the same for logits shifted by any constant."""
    return np.exp(logits - logsumexp(logits, 1)[:, None])
