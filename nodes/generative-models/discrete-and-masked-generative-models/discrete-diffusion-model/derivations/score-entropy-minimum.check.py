# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/discrete-diffusion-model/ (ai://NEW-discrete-diffusion-model).
"""Numeric check of the derivation score-entropy-minimum."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import discrete_diffusion as dd  # noqa: E402

r = np.linspace(0.01, 10.0, 2001)
f = r - np.log(r) - 1.0
assert f.min() >= 0.0 and abs(f[np.argmin(np.abs(r - 1.0))]) < 1e-4
rng = np.random.default_rng(6)
a, s = rng.uniform(0.1, 5, 100), rng.uniform(0.1, 5, 100)
np.testing.assert_allclose(s - a * np.log(s) + dd.k_const(a), a * (s / a - np.log(s / a) - 1.0), rtol=1e-12, atol=1e-14)

N = 6
for _ in range(5):
    w = rng.uniform(0.5, 2.0, (N, N))
    p0 = rng.dirichlet(np.ones(N))
    kern = rng.dirichlet(np.ones(N), size=N)
    p = p0 @ kern
    ratio = dd.concrete_score(p)
    assert abs(dd.score_entropy(ratio, p, w)) < 1e-13
    off = ~np.eye(N, dtype=bool)
    const = float(np.sum(p[:, None] * np.where(off, w * dd.k_const(ratio), 0.0)))
    for _ in range(10):
        s = np.exp(rng.standard_normal((N, N)))
        assert dd.score_entropy(s, p, w) > 0
        diff = dd.score_entropy(s, p, w) - dd.denoising_score_entropy(s, p0, kern, w)
        assert abs(diff - const) < 1e-11
    h = 1e-6
    for x, y in zip(*np.nonzero(off), strict=True):
        e = np.zeros((N, N))
        e[x, y] = h
        g = (dd.denoising_score_entropy(ratio + e, p0, kern, w) - dd.denoising_score_entropy(ratio - e, p0, kern, w)) / (
            2 * h
        )
        assert abs(g) < 1e-7, (x, y, g)
