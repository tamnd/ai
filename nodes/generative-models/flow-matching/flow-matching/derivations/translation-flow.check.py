# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/flow-matching/ (ai://NEW-flow-matching).
"""Numeric check of the derivation translation-flow."""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import flow_matching as fm  # noqa: E402

means, s, w = np.array([[2.0]]), 1.0, np.array([1.0])


def v(z, t):
    return fm.mixture_field(z, t, means, s, w)


x0 = np.array([[-1.0], [0.0], [0.5], [1.5]])
for t in (0.0, 0.25, 0.5, 0.8, 1.0):
    g = math.sqrt((1 - t) ** 2 + t**2)
    xt = 2 * t + x0 * g
    # the claimed path satisfies dx/dt = u_t(x), by central differences
    h = 1e-6
    dx = (2 * (t + h) + x0 * math.sqrt((1 - t - h) ** 2 + (t + h) ** 2) - 2 * (t - h)
          - x0 * math.sqrt((1 - t + h) ** 2 + (t - h) ** 2)) / (2 * h)  # fmt: skip
    assert np.allclose(dx, v(xt, np.full(4, t)), atol=1e-6)
    k = (2 * t - 1) / g**2
    assert np.allclose(v(xt, np.full(4, t)), 2 + k * (xt - 2 * t), atol=1e-12)
assert np.allclose(fm.midpoint_sample(v, x0, 400), x0 + 2, atol=1e-5)
pair = np.array([[0.0], [1.0]])
for n, gap in ((1, 0.0), (2, 0.5), (4, 0.72)):
    out = fm.euler_sample(v, pair, n)
    assert abs((out[1] - out[0])[0] - gap) < 1e-12, (n, out)
assert np.allclose(fm.midpoint_sample(v, x0, 1), x0 + 2, atol=1e-12)
print("translation-flow: ok")
