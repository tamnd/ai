# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/energy-based-model/ (ai://NEW-energy-based-model).
"""Numeric check of the derivation langevin-stationary-variance."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import energy_based as eb  # noqa: E402

for s2 in (0.25, 1.0, 3.0):
    for frac in (0.01, 0.1, 0.5, 0.9):
        lam = frac * 4 * s2
        a = 1 - lam / (2 * s2)
        v = 0.0
        for _ in range(20000):
            v = a * a * v + lam
        want = s2 / (1 - lam / (4 * s2))
        assert abs(v / want - 1) < 1e-9 and want > s2
        assert abs(want - (s2 + lam / 4)) <= lam * lam / (4 * s2) * 1.01 or frac > 0.4

zero = {"W1": np.zeros((3, 1)), "b1": np.zeros(1), "w2": np.zeros(1)}
rng = np.random.default_rng(6)
s2 = 0.8
for lam in (0.05, 0.4, 1.6):
    xs = eb.langevin(zero, rng.standard_normal((40000, 3)), 400, lam, rng, None, 1 / s2)
    assert abs(xs.var() / (s2 / (1 - lam / (4 * s2))) - 1) < 0.02
lam = 4 * s2 * 1.05
xs = eb.langevin(zero, 0.1 * rng.standard_normal((100, 3)), 200, lam, rng, None, 1 / s2)
assert xs.var() > 1e3
