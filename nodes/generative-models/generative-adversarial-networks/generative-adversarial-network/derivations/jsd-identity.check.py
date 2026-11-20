# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/generative-adversarial-network/ (ai://NEW-generative-adversarial-network).
"""Numeric check of the derivation jsd-identity."""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import gan  # noqa: E402


def npdf(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))


x = np.linspace(-15.0, 15.0, 60001)
dx = x[1] - x[0]
rng = np.random.default_rng(0)
for _ in range(30):
    m1, m2, m3 = rng.uniform(-3, 3, 3)
    s1, s2, s3 = rng.uniform(0.3, 2.0, 3)
    w = rng.uniform(0.1, 0.9)
    pd = npdf(x, m1, s1)
    pg = w * npdf(x, m2, s2) + (1 - w) * npdf(x, m3, s3)
    c = gan.value_at(pd, pg, gan.optimal_discriminator(pd, pg), dx)
    j = gan.jsd(pd, pg, dx)
    assert abs(c - (-math.log(4) + 2 * j)) < 1e-8, (c, j)
    assert 0.0 < j <= math.log(2)
    assert abs(gan.value_at(pd, pd, gan.optimal_discriminator(pd, pd), dx) + math.log(4)) < 1e-8

# disjoint supports: JSD = log 2 and C = 0
u1 = np.where((x > -2) & (x < -1), 1.0, 0.0)
u2 = np.where((x > 1) & (x < 2), 1.0, 0.0)
assert abs(gan.jsd(u1, u2, dx) - math.log(2)) < 1e-3
assert abs(gan.value_at(u1, u2, gan.optimal_discriminator(u1, u2 + 1e-300), dx)) < 1e-3
print("jsd-identity: ok")
