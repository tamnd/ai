# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/flow-matching/ (ai://NEW-flow-matching).
"""Numeric check of the derivation reflow-transport-cost."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import flow_matching as fm  # noqa: E402

rng = np.random.default_rng(7)
means, s, w = np.array([[-2.0, 0.0], [2.0, 0.0]]), 0.3, np.array([0.5, 0.5])


def v(z, t):
    return fm.mixture_field(z, t, means, s, w)


n = 4000
x0 = rng.standard_normal((n, 2))
x1 = means[rng.choice(2, n, p=w)] + s * rng.standard_normal((n, 2))
z1 = fm.midpoint_sample(v, x0, 200)
zh = fm.midpoint_sample(v, x0, 100, 0.0, 0.5)
# marginal preservation: Z_t has the moments of X_t
for zt, t in ((zh, 0.5), (z1, 1.0)):
    xt = fm.path(fm.config(d_data=2), x0, x1, np.full(n, t))
    assert np.allclose(zt.mean(0), xt.mean(0), atol=0.08), (t, zt.mean(0), xt.mean(0))
    assert np.allclose(np.cov(zt.T), np.cov(xt.T), atol=0.15), (t, np.cov(zt.T), np.cov(xt.T))
# the rectified coupling is cheaper for convex costs
for cost in (lambda d: np.sum(d**2, 1), lambda d: np.sqrt(np.sum(d**2, 1)), lambda d: np.abs(d[:, 0]) ** 3):
    assert cost(z1 - x0).mean() < cost(x1 - x0).mean()
# Jensen's step: the cost of the straight displacement is at most the time average of the cost of the velocity
zs, acc, steps = x0.copy(), 0.0, 200
for i in range(steps):
    acc += np.mean(np.sum(v(zs, np.full(n, (i + 0.5) / steps)) ** 2, 1)) / steps
    zs = fm.midpoint_sample(v, zs, 1, i / steps, (i + 1) / steps)
assert np.mean(np.sum((z1 - x0) ** 2, 1)) <= acc + 1e-9
assert acc <= np.mean(np.sum((x1 - x0) ** 2, 1)) + 0.15
print("reflow-transport-cost: ok")
