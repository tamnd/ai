# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/mamba/ (ai://NEW-mamba).
"""Numeric check of the derivation discretization: the zero-order hold against the rule of the reference code."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import mamba  # noqa: E402

rng = np.random.default_rng(1)
A = -np.exp(rng.normal(size=(3, 5)))
Bm = rng.normal(size=(1, 1, 5))
for d in (1e-1, 1e-2, 1e-3):
    dA_s, dB_s = mamba.discretize(np.full((1, 1, 3), d), A, Bm, "simple")
    dA_z, dB_z = mamba.discretize(np.full((1, 1, 3), d), A, Bm, "zoh")
    assert np.array_equal(dA_s, dA_z)
    x = d * A
    ratio = dB_z[0, 0] / dB_s[0, 0]
    assert np.allclose(ratio, np.expm1(x) / x, rtol=1e-12)
    assert np.allclose(ratio, 1 + x / 2 + x * x / 6, atol=np.abs(x).max() ** 3 / 24 + 1e-15)
    assert (ratio > 0).all() and (ratio < 1).all()

for d in (10.0, 100.0, 1000.0):
    _, dB_z = mamba.discretize(np.full((1, 1, 3), d), A, Bm, "zoh")
    _, dB_s = mamba.discretize(np.full((1, 1, 3), d), A, Bm, "simple")
    if d >= 100:
        assert np.allclose(dB_z[0, 0], -Bm[0, 0] / A, rtol=1e-12)
    assert np.allclose(dB_s[0, 0], d * Bm[0, 0])

z = rng.normal(0, 3, size=(1, 40))
g = mamba.sigmoid(z)
dt = mamba.softplus(z)
assert np.allclose(np.exp(-dt), 1 - g) and np.allclose(dt, -np.log1p(-g)) and (dt >= g).all()
x = rng.normal(size=(1, 40, 1))
ones = np.ones((1, 40, 1))
for disc, weight in (("zoh", g), ("simple", dt)):
    dA, dB = mamba.discretize(dt[..., None], -np.ones((1, 1)), ones, disc)
    y, _ = mamba.selective_scan(dA, dB, x, ones)
    h, want = 0.0, []
    for t in range(40):
        h = (1 - g[0, t]) * h + weight[0, t] * x[0, t, 0]
        want.append(h)
    assert np.allclose(y[0, :, 0], want, atol=1e-12), disc
print("discretization: ok")
