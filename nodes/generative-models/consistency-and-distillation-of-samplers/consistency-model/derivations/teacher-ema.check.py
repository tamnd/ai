# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/consistency-model/ (ai://NEW-consistency-model).
"""Numeric check of the derivation teacher-ema."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import consistency_model as cm  # noqa: E402

rng = np.random.default_rng(4)
xi = 0.7
for N in (101, 4001):
    ts, z = np.linspace(0.1, 1.0, N), rng.standard_normal((1, 20))
    smin, lo, hi = ts[0], ts[:-1], ts[1:]
    for theta in (-0.4, 1.3, 2.0):
        for theta_minus in (0.0, 0.5, theta):
            loss, grad = cm.toy_ct_loss(theta, theta_minus, xi, ts, z)
            delta = smin * xi * (1 / hi - 1 / lo) + (1 - smin / hi) * theta - (1 - smin / lo) * theta_minus
            assert np.isclose(loss, (delta**2).mean(), rtol=1e-10, atol=1e-30)  # z has dropped out
            assert np.isclose(grad, (2 * delta * (1 - smin / hi)).mean(), rtol=1e-10, atol=1e-30)
            if N == 4001 and theta_minus != theta:
                assert np.isclose(loss, ((1 - smin / lo) ** 2).mean() * (theta - theta_minus) ** 2, rtol=1e-2)
            if theta_minus == theta:
                limit = (2 * smin / lo**2 * (1 - smin / lo)).mean() * (theta - xi) * (ts[1] - ts[0])
                assert np.isclose(grad, limit, rtol=0.05 if N == 101 else 2e-3)
# the stop-gradient teacher drives theta to xi; an EMA teacher frozen elsewhere drives it to the teacher
ts, z = np.linspace(0.1, 1.0, 4001), rng.standard_normal((1, 20))
theta = -1.0
for _ in range(400):
    theta -= 2000.0 * cm.toy_ct_loss(theta, theta, xi, ts, z)[1]
assert abs(theta - xi) < 1e-6
theta = -1.0
for _ in range(400):
    theta -= 0.5 * cm.toy_ct_loss(theta, 0.2, xi, ts, z)[1]
assert abs(theta - 0.2) < 1e-3 * abs(0.2 - xi)
print("teacher-ema: ok")
