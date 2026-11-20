# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/generative-adversarial-network/ (ai://NEW-generative-adversarial-network).
"""Numeric check of the derivation wasserstein-parallel-lines."""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import gan  # noqa: E402

rng = np.random.default_rng(0)
n = 2000
for theta in (-1.5, -0.2, 0.01, 0.7, 3.0):
    z0, z1 = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    p0 = np.stack([np.zeros(n), np.sort(z0)], 1)
    p1 = np.stack([np.full(n, theta), np.sort(z1)], 1)
    upper = np.mean(np.linalg.norm(p0 - p1, axis=1))  # the sorted coupling, close to the horizontal one
    assert abs(theta) <= upper < abs(theta) + 0.05
    for _ in range(5):  # any coupling costs at least |theta|
        assert np.mean(np.linalg.norm(p0 - p1[rng.permutation(n)], axis=1)) >= abs(theta)
    lower = np.mean(-np.sign(theta) * p0[:, 0]) - np.mean(-np.sign(theta) * p1[:, 0])  # a 1-Lipschitz critic
    assert abs(lower - abs(theta)) < 1e-12

# JS and total variation of segments thickened to width w, on a grid of the first coordinate; the second coordinate
# is the same uniform for both and drops out
x = np.linspace(-1.0, 2.0, 300001)
dx = x[1] - x[0]
w = 0.01


def seg(c):
    return np.where(np.abs(x - c) < w / 2, 1.0 / w, 0.0)


for theta in (0.02, 0.1, 1.0):
    p, q = seg(0.0), seg(theta)
    assert abs(gan.jsd(p, q, dx) - math.log(2)) < 1e-3
    assert abs(0.5 * np.sum(np.abs(p - q)) * dx - 1.0) < 1e-3
    assert np.any((p > 0) & (q == 0))  # so KL(P_0 || P_theta) = +inf
assert gan.jsd(seg(0.0), seg(0.0), dx) < 1e-12
print("wasserstein-parallel-lines: ok")
