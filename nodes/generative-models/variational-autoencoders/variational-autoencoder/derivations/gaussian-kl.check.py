# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/variational-autoencoder/ (ai://NEW-variational-autoencoder).
"""Numeric check of the derivation gaussian-kl."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import vae  # noqa: E402

rng = np.random.default_rng(0)

# statement 1 in one dimension against the integral of q log(q / p) on a fine grid
for mu, s in [(0.0, 1.0), (1.0, 1.0), (0.0, 0.5), (-1.3, 2.0), (0.7, 0.1)]:
    z = np.linspace(mu - 12 * s, mu + 12 * s, 400_001)
    q = np.exp(-0.5 * ((z - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    f = q * (-0.5 * ((z - mu) / s) ** 2 - np.log(s) + 0.5 * z**2)
    integral = np.sum((f[1:] + f[:-1]) / 2) * (z[1] - z[0])
    closed = vae.kl(np.array([[mu]]), np.log(np.array([[s * s]])))[0]
    assert abs(integral - closed) < 1e-8, (mu, s, integral, closed)
assert vae.kl(np.zeros((1, 3)), np.zeros((1, 3)))[0] == 0.0

# statement 1 against Monte Carlo, and statement 3 at a diagonal covariance
for _ in range(5):
    J = int(rng.integers(1, 6))
    mu, lv = rng.normal(0, 1, J), rng.normal(0, 0.8, J)
    eps = rng.standard_normal((400_000, J))
    z = mu + np.exp(0.5 * lv) * eps
    mc = np.mean(vae.log_q(lv[None], eps) - vae.log_prior(z))
    closed = vae.kl(mu[None], lv[None])[0]
    assert abs(mc - closed) < 0.01 * max(1.0, closed), (mc, closed)
    C = np.diag(np.exp(lv))
    full = 0.5 * (np.trace(C) - np.linalg.slogdet(C)[1] + mu @ mu - J)
    assert abs(full - closed) < 1e-12


# statement 3 against the general KL(N(m0, C0) || N(m1, C1)) with m1 = 0, C1 = I
def kl_gauss(m0, C0, m1, C1):
    d = m1 - m0
    return 0.5 * (np.trace(np.linalg.solve(C1, C0)) + d @ np.linalg.solve(C1, d) - len(m0)
                  + np.linalg.slogdet(C1)[1] - np.linalg.slogdet(C0)[1])  # fmt: skip


for _ in range(20):
    J = int(rng.integers(1, 6))
    A = rng.normal(0, 1, (J, J))
    C, mu = A @ A.T + 0.1 * np.eye(J), rng.normal(0, 1, J)
    full = 0.5 * (np.trace(C) - np.linalg.slogdet(C)[1] + mu @ mu - J)
    assert abs(full - kl_gauss(mu, C, np.zeros(J), np.eye(J))) < 1e-10

# statement 2 against finite differences
mu, lv = rng.normal(0, 1, (3, 4)), rng.normal(0, 1, (3, 4))
h = 1e-6
for i in np.ndindex(mu.shape):
    e = np.zeros_like(mu)
    e[i] = h
    dmu = (vae.kl(mu + e, lv) - vae.kl(mu - e, lv))[i[0]] / (2 * h)
    dlv = (vae.kl(mu, lv + e) - vae.kl(mu, lv - e))[i[0]] / (2 * h)
    assert abs(dmu - mu[i]) < 1e-7 and abs(dlv - 0.5 * (np.exp(lv[i]) - 1)) < 1e-7
assert abs(0.5 * (np.exp(-50.0) - 1) + 0.5) < 1e-12  # finite as sigma -> 0
print("gaussian-kl: ok")
