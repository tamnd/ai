# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/consistency-model/ (ai://NEW-consistency-model).
"""Numeric check of the derivation score-estimator."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import consistency_model as cm  # noqa: E402

means, s2, w = np.array([[-1.0], [0.3], [2.0]]), 0.05, np.array([0.2, 0.5, 0.3])
grid = np.linspace(-6, 8, 200_001)
dx = grid[1] - grid[0]
pdata = sum(
    wk * np.exp(-0.5 * (grid - m[0]) ** 2 / s2) / np.sqrt(2 * np.pi * s2) for wk, m in zip(w, means, strict=True)
)
for t in (0.05, 0.4, 2.0, 10.0):
    for y in (-1.5, 0.0, 0.7, 2.4):
        lik = np.exp(-0.5 * (y - grid) ** 2 / t**2)
        post = pdata * lik
        est = (post * (-(y - grid) / t**2)).sum() / post.sum()
        exact = cm.score(np.array([[y]]), t, means, s2, w)[0, 0]
        assert abs(est - exact) < 1e-6 * max(1.0, abs(exact)), (t, y, est, exact)
assert abs(pdata.sum() * dx - 1.0) < 1e-9

rng = np.random.default_rng(3)
x, z = rng.standard_normal((6, 3)), rng.standard_normal((6, 3))
t_lo = np.array([0.002, 0.01, 0.3, 1.1, 7.0, 50.0])
t_hi = t_lo * np.exp(rng.random(6))


def one_sample_score(xt, t):
    return -(xt - x) / cm._col(t, len(xt)) ** 2


x_hi, x_lo = cm.cd_pair(x, z, t_lo, t_hi, one_sample_score, "euler")
c_hi, c_lo = cm.ct_pair(x, z, t_lo, t_hi)
assert np.allclose(x_hi, c_hi, rtol=0, atol=0) and np.allclose(x_lo, c_lo, rtol=1e-13, atol=1e-13)
print("score-estimator: ok")
