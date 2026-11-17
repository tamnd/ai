# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/edge-of-stability/ (ai://17Q1PD).
# Check of derivations/catapult-update.md: for f = v.u / sqrt(n) and L = f^2 / 2, one gradient step gives exactly
# f' = (1 - eta lam + eta^2 f^2 / n) f and lam' = lam + eta f^2 / n (eta lam - 4); and the lazy, catapult and divergent
# phases at eta below 2 / lam0, between 2 / lam0 and 4 / lam0, and above 4 / lam0.
import numpy as np


def fl(u, v):
    n = len(u)
    return v @ u / np.sqrt(n), (u @ u + v @ v) / n


def step(u, v, eta):
    n = len(u)
    f, _ = fl(u, v)
    return u - eta * f * v / np.sqrt(n), v - eta * f * u / np.sqrt(n)


rng = np.random.default_rng(0)
for n in (1, 5, 100, 3000):
    for eta in (0.1, 1.0, 1.9, 3.0, 5.0):
        u, v = rng.normal(size=n), rng.normal(size=n)
        f, lam = fl(u, v)
        f2, lam2 = fl(*step(u, v, eta))
        assert np.isclose(f2, (1 - eta * lam + eta**2 * f**2 / n) * f, rtol=1e-10, atol=1e-12)
        assert np.isclose(lam2, lam + eta * f**2 / n * (eta * lam - 4), rtol=1e-12)
        assert np.sign(lam2 - lam) == np.sign(eta * lam - 4)

n = 1000
u0, v0 = rng.normal(size=n), rng.normal(size=n)
_, lam0 = fl(u0, v0)
results = {}
with np.errstate(over="ignore", invalid="ignore"):
    for c in (1.0, 1.9, 2.5, 3.5, 4.5):
        u, v = u0.copy(), v0.copy()
        peak = 0.0
        for _ in range(400):
            u, v = step(u, v, c / lam0)
            f, lam = fl(u, v)
            peak = max(peak, abs(f)) if np.isfinite(f) else np.inf
        results[c] = (f, lam, peak)
for c in (1.0, 1.9):  # lazy: lambda stays put, f converges
    f, lam, _ = results[c]
    assert abs(f) < 1e-6 and abs(lam / lam0 - 1) < 0.01
for c in (2.5, 3.5):  # catapult: f grows first, lambda ends below 2 / eta, f converges
    f, lam, peak = results[c]
    assert abs(f) < 1e-6 and lam < 2 / (c / lam0) and peak > 5
assert not np.isfinite(results[4.5][0]) or abs(results[4.5][0]) > 1e6  # divergent
print("catapult-update: ok")
