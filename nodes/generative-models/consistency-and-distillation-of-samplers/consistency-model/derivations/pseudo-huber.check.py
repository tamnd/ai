# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/consistency-model/ (ai://NEW-consistency-model).
"""Numeric check of the derivation pseudo-huber."""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import consistency_model as cm  # noqa: E402

rng = np.random.default_rng(5)
for c in (1e-3, 0.03, 1.0):
    r = np.geomspace(1e-6, 1e3, 400)
    a = np.stack([r, np.zeros_like(r)], 1)
    b = np.zeros_like(a)
    # sqrt(r^2 + c^2) - c loses digits to cancellation when r << c; the conjugate form is exact to rounding
    d = r**2 / (np.sqrt(r**2 + c**2) + c)
    assert np.allclose(cm.metric(a, b, c), d, rtol=1e-9, atol=4e-16 * c)
    assert np.all(r**2 / (2 * np.sqrt(r**2 + c**2)) <= d * (1 + 1e-12))
    assert np.all(d <= np.minimum(r**2 / (2 * c), r) * (1 + 1e-12))
    small, large = r < 1e-3 * c, r > 1e3 * c
    assert np.allclose(d[small], r[small] ** 2 / (2 * c), rtol=1e-5)
    assert np.allclose(d[large], r[large] - c, rtol=1e-3)
    x, y = rng.standard_normal((6, 4)), rng.standard_normal((6, 4))
    g = cm.metric_grad(x, y, c)
    assert np.all(np.linalg.norm(g, axis=1) < 1)
    h = 1e-6
    for i in range(4):
        e = np.zeros(4)
        e[i] = h
        assert np.allclose((cm.metric(x + e, y, c) - cm.metric(x - e, y, c)) / (2 * h), g[:, i], rtol=1e-6, atol=1e-9)
    # second derivative at x = y: (1 / c) I, finite where the plain distance has a kink
    y0 = rng.standard_normal((1, 4))
    for i in range(4):
        e = np.zeros((1, 4))
        e[0, i] = 1e-4 * c
        second = (cm.metric(y0 + e, y0, c) - 2 * cm.metric(y0, y0, c) + cm.metric(y0 - e, y0, c)) / (1e-4 * c) ** 2
        assert np.isclose(second[0], 1 / c, rtol=1e-4)
assert np.allclose(cm.metric_grad(x, y), 2 * (x - y))
assert abs(0.00054 * math.sqrt(3 * 32 * 32) - 0.0299) < 1e-4
assert abs(0.00054 * math.sqrt(3 * 64 * 64) - 0.0599) < 1e-4
print("pseudo-huber: ok")
