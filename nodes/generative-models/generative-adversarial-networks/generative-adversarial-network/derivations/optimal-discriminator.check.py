# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/generative-adversarial-network/ (ai://NEW-generative-adversarial-network).
"""Numeric check of the derivation optimal-discriminator."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import gan  # noqa: E402


def npdf(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))


rng = np.random.default_rng(0)
y = np.linspace(1e-6, 1 - 1e-6, 200001)
for _ in range(50):  # steps 1 and 2: the integrand's maximum is a / (a + b)
    a, b = rng.uniform(0.0, 3.0, 2)
    best = y[np.argmax(a * np.log(y) + b * np.log1p(-y))]
    assert abs(best - a / (a + b)) < 1e-5, (a, b, best)

x = np.linspace(-8.0, 9.0, 20001)
dx = x[1] - x[0]
pd, pg = npdf(x, 0.0, 1.0), npdf(x, 1.5, 0.7)
ds = gan.optimal_discriminator(pd, pg)
top = gan.value_at(pd, pg, ds, dx)
for _ in range(20):  # D* beats every perturbation of it
    other = np.clip(
        ds + rng.uniform(0.01, 0.1) * np.sin(rng.uniform(0.3, 4.0) * x + rng.uniform(0, 6)), 1e-12, 1 - 1e-12
    )
    assert gan.value_at(pd, pg, other, dx) < top

# steps 3 and 4: logistic regression on the mixture, with the features 1, x, x^2, learns the log-ratio, which is
# quadratic for two Gaussians
n = 200000
xs = np.concatenate([rng.normal(0.0, 1.0, n), rng.normal(1.5, 0.7, n)])
lab = np.concatenate([np.ones(n), np.zeros(n)])
F = np.stack([np.ones_like(xs), xs, xs**2], 1)
w = np.zeros(3)
for _ in range(50):  # Newton's method on the mean log-likelihood
    pr = gan.sigmoid(F @ w)
    H = (F * (pr * (1 - pr))[:, None]).T @ F / len(xs)
    w += np.linalg.solve(H, F.T @ (lab - pr) / len(xs))
grid = np.linspace(-1.5, 2.5, 9)
logit = np.stack([np.ones_like(grid), grid, grid**2], 1) @ w
assert np.max(np.abs(logit - np.log(npdf(grid, 0.0, 1.0) / npdf(grid, 1.5, 0.7)))) < 0.1
print("optimal-discriminator: ok")
