# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/consistency-model/ (ai://NEW-consistency-model).
"""Numeric check of the derivation gaussian-consistency-function."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import consistency_model as cm  # noqa: E402

rng = np.random.default_rng(2)
mu, s2 = np.array([[0.4, -1.2, 2.0]]), 0.3


def sf(x, t):
    return cm.score(x, t, mu, s2, np.ones(1))


# the exact score of one Gaussian
x = rng.standard_normal((5, 3))
assert np.allclose(sf(x, 2.0), -(x - mu) / (s2 + 4.0))
# eq-gauss-cf against the solved ODE
for t in (0.05, 1.0, 10.0, 80.0):
    xt = mu + np.sqrt(s2 + t**2) * rng.standard_normal((6, 3))
    assert np.allclose(
        cm.solve_ode(xt, t, cm.EPS, sf, 400), cm.gaussian_consistency(xt, t, mu, s2), rtol=1e-4, atol=1e-4
    )
# one path: the same value at every point
ts = cm.karras_steps(400)[::-1]
xs = [mu + 80.0 * rng.standard_normal((4, 3))]
for a, b in zip(ts[:-1], ts[1:], strict=True):
    xs.append(cm.teacher_step(xs[-1], a, b, sf))
ends = np.stack([cm.gaussian_consistency(xi, ti, mu, s2) for xi, ti in zip(xs, ts, strict=True)])
assert np.abs(ends - xs[-1]).max() < 1e-4
assert np.allclose(cm.gaussian_consistency(xs[-1], cm.EPS, mu, s2), xs[-1], rtol=1e-15, atol=1e-15)
# the total derivative along the ODE is zero
h = 1e-6
for t in (0.1, 3.0, 50.0):
    x = mu + rng.standard_normal((4, 3))
    dx = -t * sf(x, t)
    fp = cm.gaussian_consistency(x + h * dx, t + h, mu, s2)
    fm = cm.gaussian_consistency(x - h * dx, t - h, mu, s2)
    assert np.abs((fp - fm) / (2 * h)).max() < 1e-7
# f(., T) carries N(mu, (s2 + T^2) I) to N(mu, (s2 + eps^2) I)
xT = mu + np.sqrt(s2 + cm.T_MAX**2) * rng.standard_normal((200_000, 3))
out = cm.gaussian_consistency(xT, cm.T_MAX, mu, s2)
assert np.allclose(out.mean(0), mu[0], atol=0.01)
assert np.allclose(np.cov(out.T), (s2 + cm.EPS**2) * np.eye(3), atol=0.004)
print("gaussian-consistency-function: ok")
