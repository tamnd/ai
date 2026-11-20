# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/classifier-free-guidance/ (ai://NEW-classifier-free-guidance).
"""Numeric check of the derivation tilted-gaussian."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import classifier_free_guidance as cg  # noqa: E402


def logn(x, m, v):
    return -0.5 * (x - m) ** 2 / v - 0.5 * np.log(2 * np.pi * v)


x = np.linspace(-30, 30, 600001)
dx = x[1] - x[0]
for m1, v1, m0, v0 in ((1.0, 0.25, 0.0, 1.25), (-0.5, 0.6, 1.5, 2.0), (2.0, 0.3, 0.0, 0.3)):
    for s in (0.0, 0.5, 1.0, 2.0, 3.0, 7.5):
        out = cg.tilted_gaussian(m1, v1, m0, v0, s)
        assert out is not None
        m, v = out
        lp = s * logn(x, m1, v1) + (1 - s) * logn(x, m0, v0)
        p = np.exp(lp - lp.max())
        p /= p.sum() * dx
        mean = (x * p).sum() * dx
        var = ((x - mean) ** 2 * p).sum() * dx
        assert abs(mean - m) < 1e-8 and abs(var - v) < 1e-8
        assert np.isclose(m, m1 + (1 - s) * v / v0 * (m0 - m1))
        if s > 1:
            assert v <= v1 + 1e-15 and (m - m1) * (m1 - m0) >= 0
# the narrow model guided by a wider one never fails; the reverse fails from s = v1 / (v1 - v0)
v1, v0 = 1.25, 0.25
thr = v1 / (v1 - v0)
assert np.isclose(thr, 1.25)
assert cg.tilted_gaussian(0.0, v1, 1.0, v0, thr - 1e-6) is not None
assert cg.tilted_gaussian(0.0, v1, 1.0, v0, thr) is None
assert cg.tilted_gaussian(0.0, v1, 1.0, v0, 3.0) is None
assert all(cg.tilted_gaussian(1.0, 0.25, 0.0, 1.25, s) is not None for s in np.linspace(0, 100, 101))
m, v = cg.tilted_gaussian(1.0, 0.25, 0.0, 1.25, 2.0)
assert (round(m, 3), round(v, 4)) == (1.111, 0.1389)
m, v = cg.tilted_gaussian(1.0, 0.25, 0.0, 1.25, 3.0)
assert (round(m, 3), round(v, 4)) == (1.154, 0.0962)
