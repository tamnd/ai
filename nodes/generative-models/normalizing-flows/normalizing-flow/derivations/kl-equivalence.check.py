# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/normalizing-flow/ (ai://NEW-normalizing-flow).
"""Numeric check of the derivation kl-equivalence."""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import flow  # noqa: E402


def kl_gauss(m1, s1, m2, s2):
    """KL(N(m1, s1^2) || N(m2, s2^2))."""
    return math.log(s2 / s1) + (s1**2 + (m1 - m2) ** 2) / (2 * s2**2) - 0.5


rng = np.random.default_rng(3)
for _ in range(20):  # statement 2 with f(x) = (x - c) / a: model N(c, a^2), data N(m, s^2), pushed data N((m - c) / a, (s / a)^2)
    m, s, c, a = rng.normal(), rng.uniform(0.3, 2), rng.normal(), rng.uniform(0.3, 2)
    assert abs(kl_gauss(m, s, c, a) - kl_gauss((m - c) / a, s / a, 0.0, 1.0)) < 1e-12
    nll = 0.5 * math.log(2 * math.pi * a**2) + (s**2 + (m - c) ** 2) / (2 * a**2)  # statement 1
    assert abs(nll - (0.5 * math.log(2 * math.pi * math.e * s**2) + kl_gauss(m, s, c, a))) < 1e-12

cfg = flow.config(2, 6, 3)  # step 2, pointwise, for a coupling flow and a two-component mixture
p = {k: rng.normal(0.0, 0.5, sh) for k, sh in flow.param_shapes(cfg).items()}
mus, sd = np.array([[-1.5, 0.0], [1.5, 1.0]]), 0.6


def log_pi_x(x):
    comp = [-0.5 * np.sum((x - m) ** 2, 1) / sd**2 - math.log(2 * math.pi * sd**2) for m in mus]
    return np.logaddexp(*comp) - math.log(2)


x = mus[rng.integers(0, 2, 5000)] + sd * rng.standard_normal((5000, 2))
z, ld, _ = flow.forward(p, cfg, x)
log_pi_z = log_pi_x(x) - ld  # the density of the pushed-forward data at z = f(x)
log_p_z = flow.log_normal(z)
assert np.allclose(log_pi_z - log_p_z, log_pi_x(x) - flow.log_prob(p, cfg, x), atol=1e-12)
assert np.mean(log_pi_x(x) - flow.log_prob(p, cfg, x)) > 0  # a KL estimate of an untrained flow is positive
print("kl-equivalence: ok")
