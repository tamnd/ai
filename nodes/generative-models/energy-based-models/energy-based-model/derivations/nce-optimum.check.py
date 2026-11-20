# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/energy-based-model/ (ai://NEW-energy-based-model).
"""Numeric check of the derivation nce-optimum."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import energy_based as eb  # noqa: E402

x = np.linspace(-15.0, 15.0, 150001)
dx = x[1] - x[0]
q = 0.3 * np.exp(eb.gauss_logpdf(x, -1.0, 0.5)) + 0.7 * np.exp(eb.gauss_logpdf(x, 1.0, 0.8))
log_pn = eb.gauss_logpdf(x, 0.0, 2.5)
pn = np.exp(log_pn)


def J(f):
    g = f - log_pn
    return float(np.sum(q * -eb.softplus(-g) + pn * -eb.softplus(g)) * dx)


best = J(np.log(q))
rng = np.random.default_rng(5)
for _ in range(10):
    bump = rng.uniform(-0.5, 0.5) * np.exp(-0.5 * ((x - rng.uniform(-2, 2)) / rng.uniform(0.2, 1.0)) ** 2)
    assert J(np.log(q) + bump) < best
for c in (-0.3, -0.05, 0.05, 0.3):
    assert J(np.log(q) + c) < best  # the optimum is normalized: no constant shift helps

# a model of the right shape with a free constant: f = -E + c, E(x) = (x - 1)^2 / (2 * 0.8^2), data N(1, 0.8^2)
q1 = np.exp(eb.gauss_logpdf(x, 1.0, 0.8))
E = (x - 1.0) ** 2 / (2 * 0.64)
logZ = np.log(np.sum(np.exp(-E)) * dx)
cs = np.linspace(-2.0, 0.0, 2001)


def J1(c):
    g = -E + c - log_pn
    return float(np.sum(q1 * -eb.softplus(-g) + pn * -eb.softplus(g)) * dx)


c_star = cs[int(np.argmax([J1(c) for c in cs]))]
assert abs(c_star + logZ) < 2e-3

# finite samples: Newton's method finds the stationary point, close to the true parameters and normalizer
xs, ys = rng.normal(-0.5, 1.2, 60000), rng.normal(0.0, 2.5, 60000)
theta = eb.nce_fit(xs, ys, 0.0, 2.5)
assert np.abs(eb.nce_grad(theta, xs, ys, 0.0, 2.5)).max() < 1e-10
np.testing.assert_allclose(theta, eb.gaussian_natural(-0.5, 1.2), atol=0.03)
