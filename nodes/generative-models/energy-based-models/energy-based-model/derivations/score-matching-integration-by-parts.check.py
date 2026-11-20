# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/energy-based-model/ (ai://NEW-energy-based-model).
"""Numeric check of the derivation score-matching-integration-by-parts."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import energy_based as eb  # noqa: E402

# 1-D: the data density q is a two-component Gaussian mixture, the models E(x) = a x^4/4 + b x^2/2 + c x
x = np.linspace(-12.0, 12.0, 240001)
dx = x[1] - x[0]
n1 = np.exp(-0.5 * ((x + 1.0) / 0.6) ** 2) / (0.6 * np.sqrt(2 * np.pi))
n2 = np.exp(-0.5 * ((x - 1.5) / 0.9) ** 2) / (0.9 * np.sqrt(2 * np.pi))
q = 0.4 * n1 + 0.6 * n2
dq = 0.4 * n1 * (-(x + 1.0) / 0.36) + 0.6 * n2 * (-(x - 1.5) / 0.81)
psi_q = dq / q
consts = []
rng = np.random.default_rng(4)
for _ in range(6):
    a, b, c = rng.uniform(0.05, 0.5), rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)
    psi = -(a * x**3 + b * x + c)
    dpsi = -(3 * a * x**2 + b)
    explicit = 0.5 * np.sum(q * (psi - psi_q) ** 2) * dx
    implicit = np.sum(q * (dpsi + 0.5 * psi**2)) * dx
    consts.append(explicit - implicit)
assert np.ptp(consts) < 1e-8
assert abs(consts[0] - 0.5 * np.sum(q * psi_q**2) * dx) < 1e-8

# the Gaussian energy: score matching = maximum likelihood, for any sample
for T in (7, 50, 3000):
    xs = rng.standard_normal((T, 3)) @ rng.standard_normal((3, 3)) + rng.standard_normal(3)
    mu, M = eb.gaussian_sm_fit(xs)
    r = xs - xs.mean(0)
    S = r.T @ r / T
    np.testing.assert_allclose(mu, xs.mean(0), rtol=1e-12)
    np.testing.assert_allclose(M, np.linalg.inv(S), rtol=1e-9)
    base = eb.gaussian_sm_objective(mu, M, xs)
    for i in range(3):
        for j in range(i, 3):
            d = np.zeros((3, 3))
            d[i, j] = d[j, i] = 1e-5
            g = (eb.gaussian_sm_objective(mu, M + d, xs) - eb.gaussian_sm_objective(mu, M - d, xs)) / 2e-5
            assert abs(g) < 1e-6
            assert eb.gaussian_sm_objective(mu, M + 100 * d, xs) > base
        e = np.zeros(3)
        e[i] = 1e-5
        assert abs(eb.gaussian_sm_objective(mu + e, M, xs) - eb.gaussian_sm_objective(mu - e, M, xs)) < 1e-12

# the network energy: the objective is the mean of -Laplacian + |grad|^2 / 2
qn = {k: rng.standard_normal(s) for k, s in eb.mlp_shapes(3, 6).items()}
xs = rng.standard_normal((5, 3))
h = 1e-4
I3 = np.eye(3)
lap = np.zeros(5)
for i in range(3):
    lap += (eb.mlp_energy(qn, xs + h * I3[i]) - 2 * eb.mlp_energy(qn, xs) + eb.mlp_energy(qn, xs - h * I3[i])) / h**2
grad = np.stack([(eb.mlp_energy(qn, xs + 1e-6 * I3[i]) - eb.mlp_energy(qn, xs - 1e-6 * I3[i])) / 2e-6 for i in range(3)], 1)
assert abs(eb.mlp_sm_objective(qn, xs) - np.mean(-lap + 0.5 * np.sum(grad**2, 1))) < 1e-5
