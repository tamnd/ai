# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/classifier-free-guidance/ (ai://NEW-classifier-free-guidance).
"""Numeric check of the derivation perturbed-gaussian-mean."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import classifier_free_guidance as cg  # noqa: E402

rng = np.random.default_rng(3)
d = 4
A = rng.standard_normal((d, d))
Sigma = 0.1 * (A @ A.T + d * np.eye(d))
mu, g = rng.standard_normal(d), rng.standard_normal(d)
P = np.linalg.inv(Sigma)


def lognormal(x, m):
    r = x - m
    return -0.5 * np.einsum("bi,ij,bj->b", r, P, r)


x = rng.standard_normal((200, d))
for s in (0.5, 1.0, 4.0):
    m = cg.perturbed_mean(mu[None], Sigma, g[None], s)[0]
    assert np.allclose(m, mu + s * Sigma @ g)
    diff = lognormal(x, mu) + s * (x - mu) @ g - lognormal(x, m)
    assert np.allclose(diff, 0.5 * s**2 * g @ Sigma @ g)
    if s > 1.0:
        continue  # the importance weights below degenerate for large s
    # importance weights p(y | x)^s with a linear log-classifier move the mean of N(mu, Sigma) samples
    L = np.linalg.cholesky(Sigma)
    xs = mu + rng.standard_normal((400000, d)) @ L.T
    lw = s * (xs - mu) @ g
    wts = np.exp(lw - lw.max())
    wts /= wts.sum()
    assert np.allclose(wts @ xs, m, atol=0.03)
