# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/classifier-free-guidance/ (ai://NEW-classifier-free-guidance).
"""Numeric check of the derivation thresholding."""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import classifier_free_guidance as cg  # noqa: E402

rng = np.random.default_rng(7)
# static thresholding is the nearest point of the cube: no random point of the cube is closer
x = 2.0 * rng.standard_normal((50, 6))
st = cg.static_threshold(x)
for i in range(len(x)):
    cand = rng.uniform(-1, 1, (4000, 6))
    assert np.all(((cand - x[i]) ** 2).sum(1) >= ((st[i] - x[i]) ** 2).sum() - 1e-12)
for n in (5, 40, 1000):
    x = 1.7 * rng.standard_normal((20, n))
    for p in (0.0, 50.0, 90.0, 99.5, 100.0):
        d = cg.dynamic_threshold(x, p)
        assert np.all(np.abs(d) <= 1.0 + 1e-15)
        s = np.maximum(np.percentile(np.abs(x), p, axis=1, keepdims=True), 1.0)
        assert np.allclose(d, np.clip(x / s, -1, 1))
        kept = (np.abs(x) <= s).sum(1)
        assert np.all(kept >= math.floor(p / 100 * (n - 1)) + 1)
        small = (np.percentile(np.abs(x), p, axis=1) <= 1.0)[:, None]
        assert np.allclose(np.where(small, d, 0.0), np.where(small, cg.static_threshold(x), 0.0))
        unclipped = np.abs(x) <= s
        assert np.allclose(d[unclipped], (x / s)[unclipped])
    assert np.allclose(cg.dynamic_threshold(x, 0.0), cg.static_threshold(x))
# a prediction inside the range is left alone by both
x = rng.uniform(-0.9, 0.9, (3, 30))
assert np.allclose(cg.dynamic_threshold(x, 99.5), x) and np.allclose(cg.static_threshold(x), x)
# the listing's np.max(s, 1.0) reads 1.0 as an axis
try:
    np.max(np.array([[0.5, 2.0]]), 1.0)  # type: ignore[call-overload]
    raise AssertionError("np.max(s, 1.0) did not fail")
except TypeError:
    pass
