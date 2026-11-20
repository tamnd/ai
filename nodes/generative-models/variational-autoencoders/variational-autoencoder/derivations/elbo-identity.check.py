# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/variational-autoencoder/ (ai://NEW-variational-autoencoder).
"""Numeric check of the derivation elbo-identity on linear Gaussian models, where every term has a closed form."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import vae  # noqa: E402


def log_normal(x, mean, cov):
    d = x - mean
    _, logdet = np.linalg.slogdet(cov)
    return -0.5 * (len(x) * np.log(2 * np.pi) + logdet + d @ np.linalg.solve(cov, d))


def kl_gauss(m0, C0, m1, C1):
    """KL(N(m0, C0) || N(m1, C1))."""
    J = len(m0)
    d = m1 - m0
    _, l0 = np.linalg.slogdet(C0)
    _, l1 = np.linalg.slogdet(C1)
    return 0.5 * (np.trace(np.linalg.solve(C1, C0)) + d @ np.linalg.solve(C1, d) - J + l1 - l0)


def recon(x, W, b, s2, m, C):
    """E_q log p(x|z) for q = N(m, C) and p(x|z) = N(W z + b, s2 I), step 3."""
    D = len(x)
    r = x - W @ m - b
    return -0.5 * D * np.log(2 * np.pi * s2) - 0.5 * (r @ r + np.trace(W @ C @ W.T)) / s2


rng = np.random.default_rng(0)
for _ in range(50):
    D, J = (int(v) for v in rng.integers([1, 1], [7, 5]))
    W, b, s2 = rng.normal(0, 1, (D, J)), rng.normal(0, 1, D), float(rng.uniform(0.1, 2.0))
    x = rng.normal(0, 2, D)
    M = np.eye(J) + W.T @ W / s2
    post_C = np.linalg.inv(M)
    post_m = post_C @ W.T @ (x - b) / s2
    log_px = log_normal(x, b, W @ W.T + s2 * np.eye(D))
    # the posterior from Bayes' theorem at a random z, statement 3
    z = rng.normal(0, 1, J)
    lhs = log_normal(z, np.zeros(J), np.eye(J)) + log_normal(x, W @ z + b, s2 * np.eye(D)) - log_px
    assert abs(lhs - log_normal(z, post_m, post_C)) < 1e-9
    for diag in (True, False):
        m = rng.normal(0, 1, J)
        A = rng.normal(0, 0.5, (J, J))
        C = np.diag(rng.uniform(0.1, 2.0, J)) if diag else A @ A.T + 0.1 * np.eye(J)
        elbo = recon(x, W, b, s2, m, C) - kl_gauss(m, C, np.zeros(J), np.eye(J))  # statement 2
        gap = kl_gauss(m, C, post_m, post_C)
        assert abs(elbo + gap - log_px) < 1e-9 and gap >= 0 and elbo <= log_px  # statement 1
    assert abs(recon(x, W, b, s2, post_m, post_C) - kl_gauss(post_m, post_C, np.zeros(J), np.eye(J)) - log_px) < 1e-9

# the closed-form reconstruction term against sampling
D, J = 4, 2
W, b, s2, x = rng.normal(0, 1, (D, J)), rng.normal(0, 1, D), 0.5, rng.normal(0, 1, D)
m, C = rng.normal(0, 1, J), np.diag([0.3, 1.2])
z = m + rng.standard_normal((400_000, J)) * np.sqrt(np.diag(C))
mc = np.mean(-0.5 * D * np.log(2 * np.pi * s2) - 0.5 * np.sum((x - z @ W.T - b) ** 2, 1) / s2)
assert abs(mc - recon(x, W, b, s2, m, C)) < 0.02

# the reference's estimator B on a decoder whose hidden layer is the identity: a linear Gaussian model. The decoder
# of the reference is tanh(z V + c) U + e, so we compare on a tiny V where tanh is linear up to O(|zV|^3) and rescale.
cfg = vae.config(d_in=D, d_hidden=J, d_latent=J, decoder="gaussian")
p = vae.init(rng, cfg)
k = 1e-4
p["dec_W"], p["dec_b"] = k * np.eye(J), np.zeros(J)
p["out_W"], p["out_b"] = W.T / k, b
p["olv_W"], p["olv_b"] = np.zeros((J, D)), np.full(D, np.log(s2))
p["enc_W"], p["enc_b"] = np.zeros((D, cfg["d_hidden"])), np.zeros(cfg["d_hidden"])
p["mu_W"], p["mu_b"] = np.zeros((J, J)), m
p["lv_W"], p["lv_b"] = np.zeros((J, J)), np.log(np.diag(C))
eps = rng.standard_normal((200_000, 1, J))
value = vae.objective(p, cfg, x[None], eps, "elbo")[0]
exact = recon(x, W, b, s2, m, C) - kl_gauss(m, C, np.zeros(J), np.eye(J))
assert abs(-value - exact) < 0.02, (value, exact)
print("elbo-identity: ok")
