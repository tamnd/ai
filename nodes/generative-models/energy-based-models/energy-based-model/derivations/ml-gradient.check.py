# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/energy-based-model/ (ai://NEW-energy-based-model).
"""Numeric check of the derivation ml-gradient."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import energy_based as eb  # noqa: E402

rng = np.random.default_rng(1)
for D, H in ((3, 2), (4, 3), (5, 2)):
    p = {k: rng.standard_normal(s) for k, s in eb.param_shapes(eb.config(D, H)).items()}
    X = eb.all_binary(D)
    data = X[rng.integers(0, len(X), 9)]
    g = eb.exact_grad(p, data)
    for k in p:
        for idx in np.ndindex(p[k].shape):
            a = {key: v.copy() for key, v in p.items()}
            b = {key: v.copy() for key, v in p.items()}
            a[k][idx] += 1e-6
            b[k][idx] -= 1e-6
            fd = (eb.log_prob(a, data).mean() - eb.log_prob(b, data).mean()) / 2e-6
            assert abs(g[k][idx] - fd) < 1e-8, (k, idx)
    # the gradient is the difference of the data statistics and the model statistics
    pm = eb.model_probs(p)
    sd = eb.stats(p, data)
    ph = eb.p_h(p, X)
    np.testing.assert_allclose(g["W"], sd["W"] - (X * pm[:, None]).T @ ph, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(g["b"], sd["b"] - pm @ X, rtol=1e-12, atol=1e-14)

# KL form: the mean log-likelihood gradient is minus the gradient of KL(q || p), q the empirical distribution
p = {k: rng.standard_normal(s) for k, s in eb.param_shapes(eb.config(4, 3)).items()}
X = eb.all_binary(4)
q = rng.dirichlet(np.ones(16))


def kl(par):
    return float(np.sum(q * (np.log(q) - eb.log_prob(par, X))))


a, b = {k: v.copy() for k, v in p.items()}, {k: v.copy() for k, v in p.items()}
a["c"][1] += 1e-6
b["c"][1] -= 1e-6
gq = float(np.sum(q * (eb.log_prob(a, X) - eb.log_prob(b, X))) / 2e-6)
assert abs(-(kl(a) - kl(b)) / 2e-6 - gq) < 1e-8

# exponential family: a fully visible model with E = -theta . T(x), T = (x, x_i x_j); gradient ascent matches moments
iu = np.triu_indices(4, 1)
T = np.concatenate([X, (X[:, :, None] * X[:, None, :])[:, iu[0], iu[1]]], 1)
theta = np.zeros(T.shape[1])
for _ in range(4000):
    pm = np.exp(T @ theta)
    pm /= pm.sum()
    theta += 0.5 * (q @ T - pm @ T)
pm = np.exp(T @ theta)
pm /= pm.sum()
np.testing.assert_allclose(pm @ T, q @ T, atol=1e-9)
