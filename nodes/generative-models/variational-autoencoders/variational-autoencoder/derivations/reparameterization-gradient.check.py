# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/variational-autoencoder/ (ai://NEW-variational-autoencoder).
"""Numeric check of the derivation reparameterization-gradient."""

import numpy as np

rng = np.random.default_rng(0)
n = 2_000_000
for mu, s in [(0.0, 1.0), (1.0, 1.0), (1.0, 0.3), (-2.0, 0.5), (0.5, 2.0)]:
    eps = rng.standard_normal(n)
    z = mu + s * eps
    rep = 2 * z  # statement 1 with f = z^2
    score = z**2 * (z - mu) / s**2  # statement 2
    for est in (rep, score):
        assert abs(est.mean() - 2 * mu) < 6 * est.std() / np.sqrt(n)
    assert abs(rep.var() / (4 * s * s) - 1) < 0.01
    want = mu**4 / s**2 + 14 * mu**2 + 15 * s**2  # statement 3
    assert abs(score.var() / want - 1) < 0.03, (mu, s, score.var(), want)
assert 0**4 / 1 + 0 + 15 * 1 == 15

# statement 1 for f = sin: E sin(mu + s eps) = sin(mu) exp(-s^2 / 2)
for mu, s in [(0.3, 0.7), (-1.0, 1.5)]:
    eps = rng.standard_normal(n)
    dmu, ds = np.cos(mu + s * eps), eps * np.cos(mu + s * eps)
    assert abs(dmu.mean() - np.cos(mu) * np.exp(-s * s / 2)) < 6 * dmu.std() / np.sqrt(n)
    assert abs(ds.mean() - (-s * np.sin(mu) * np.exp(-s * s / 2))) < 6 * ds.std() / np.sqrt(n)

# as sigma -> 0 with mu = 1 the score-function variance grows like 1 / sigma^2
for s in (0.1, 0.01):
    assert 1 / s**2 + 14 + 15 * s * s > 100 * 4 * s * s
print("reparameterization-gradient: ok")
