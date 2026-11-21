# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/consistency-model/ (ai://NEW-consistency-model).
"""Numeric check of the derivation progressive-distillation-target."""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import consistency_model as cm  # noqa: E402


def cosine(t):
    return math.cos(0.5 * math.pi * t), math.sin(0.5 * math.pi * t)


def teacher(z, t):
    """an x-prediction that depends on z and t nonlinearly"""
    return np.tanh(z) * (1.0 - 0.4 * t) + 0.1 * np.sin(3.0 * z)


rng = np.random.default_rng(11)
z = rng.standard_normal((8, 3))
for N in (1, 2, 4, 16, 64):
    for t in (1.0, 0.75, 2.0 / N if N > 2 else 1.0):
        t1, t2 = t - 0.5 / N, t - 1.0 / N
        (a, s), (a1, s1), (a2, s2) = cosine(t), cosine(t1), cosine(t2)
        z1 = cm.ddim_step(z, teacher(z, t), a, s, a1, s1)
        z2 = cm.ddim_step(z1, teacher(z1, t1), a1, s1, a2, s2)
        xt = cm.pd_target(z, z2, a, s, a2, s2)
        assert np.allclose(cm.ddim_step(z, xt, a, s, a2, s2), z2, rtol=1e-12, atol=1e-12)
        den = a2 - s2 / s * a
        assert den > 0 and math.isclose(den, math.sin(0.5 * math.pi * (t - t2)) / s, rel_tol=1e-12)
        x_const = np.tanh(z[:1])  # the same x-prediction at both teacher steps
        w1 = cm.ddim_step(z, x_const, a, s, a1, s1)
        w2 = cm.ddim_step(w1, x_const, a1, s1, a2, s2)
        assert np.allclose(cm.pd_target(z, w2, a, s, a2, s2), np.broadcast_to(x_const, z.shape), atol=1e-12)
# at t = 1 (alpha = 0) the denominator is alpha_{t''}
a2, s2 = cosine(1.0 - 1.0 / 4)
assert math.isclose(a2 - s2 / 1.0 * 0.0, a2)

# error amplification of the epsilon- and v-links as alpha -> 0
x = rng.standard_normal(1000)
e = rng.standard_normal(1000)
dlt = 1e-3 * rng.standard_normal(1000)
prev = 0.0
for t in (0.5, 0.9, 0.99, 0.999, 0.9999):
    a, s = cosine(t)
    zt = a * x + s * e
    v = a * e - s * x
    assert np.allclose(a * zt - s * v, x, atol=1e-12)
    err_eps = np.abs((zt - s * (e + dlt)) / a - x).max()
    err_v = np.abs(a * zt - s * (v + dlt) - x).max()
    assert math.isclose(err_eps, s / a * np.abs(dlt).max(), rel_tol=1e-6)
    assert err_v <= np.abs(dlt).max() * (1 + 1e-9)
    assert err_eps > prev
    prev = err_eps
assert prev > 5.0  # 1e-3 noise error becomes an x error above 5 at t = 0.9999
print("progressive-distillation-target: ok")
