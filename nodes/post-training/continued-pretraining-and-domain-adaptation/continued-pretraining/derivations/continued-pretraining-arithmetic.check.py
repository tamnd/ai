# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/continued-pretraining/ (ai://NEW-continued-pretraining).
"""Checks every step of continued-pretraining-arithmetic.md numerically."""

import itertools
import math

import numpy as np

rng = np.random.default_rng(0)

# 1. Compute.
N = 7e9
assert math.isclose(6 * N * 300e9 / (6 * N * (300e9 + 300e9)), 0.5)


# 2. The cosine schedule.
def cosine(s, hi, lo):
    return lo + (hi - lo) * (1 + np.cos(np.pi * s)) / 2


hi, lo = 4.5e-5, 4.5e-7
assert math.isclose(cosine(0, hi, lo), hi) and math.isclose(cosine(1, hi, lo), lo)
assert math.isclose(cosine(0.5, hi, lo), (hi + lo) / 2)
s = np.linspace(0, 1, 100001)
eta = cosine(s, hi, lo)
assert np.all(np.diff(eta) <= 0)
slope = np.abs(np.diff(eta))
assert slope.argmax() in range(49000, 51001) and slope[0] < slope[50000] / 100 and slope[-1] < slope[50000] / 100
frac = (9e-6 - lo) / (hi - lo)
assert math.isclose(frac, 0.19 / 0.99) and abs(frac - 0.1919) < 1e-4
cross = math.acos(2 * frac - 1) / math.pi
assert abs(2 * frac - 1 + 0.6162) < 1e-4 and abs(cross - 0.711) < 1e-3
assert math.isclose(cosine(cross, hi, lo), 9e-6) and math.isclose(9e-6, hi / 5)
assert abs((1 - cross) - 0.29) < 0.01

# 3. Replay.
for r, want in ((0.05, 1.053), (0.25, 4 / 3), (0.5, 2.0)):
    assert abs(1 / (1 - r) - want) < 1e-3
assert math.isclose(0.05 * 100, 5) and math.isclose(0.95 * 100, 95)
# Mixed batches: the mean loss over many draws approaches r L_old + (1 - r) L_new, and so does the gradient.
theta = 0.3
old, new = rng.normal(1.0, 1.0, 400000), rng.normal(-2.0, 1.0, 400000)


def loss(x, t):
    return 0.5 * (t - x) ** 2


r = 0.25
pick = rng.random(400000) < r
mixed = np.where(pick, old, new)
want = r * loss(old, theta).mean() + (1 - r) * loss(new, theta).mean()
assert abs(loss(mixed, theta).mean() - want) < 0.03
grad_want = r * (theta - old).mean() + (1 - r) * (theta - new).mean()
assert abs((theta - mixed).mean() - grad_want) < 0.01

# 4. EWC.
for _ in range(200):
    a, lf = rng.uniform(0.1, 5), rng.uniform(0, 5)
    tb, ta = rng.normal(), rng.normal()
    star = (a * tb + lf * ta) / (a + lf)
    assert abs(a * (star - tb) + lf * (star - ta)) < 1e-12
    grid = np.linspace(star - 1, star + 1, 2001)
    obj = 0.5 * a * (grid - tb) ** 2 + 0.5 * lf * (grid - ta) ** 2
    assert abs(grid[obj.argmin()] - star) < 1e-3
assert math.isclose((2.0 * 1.0 + 0 * 5.0) / 2.0, 1.0)
assert abs((2.0 * 1.0 + 1e9 * 5.0) / (2.0 + 1e9) - 5.0) < 1e-6

# 5. Vocabulary extension.
k = 49953 - 32000
assert k == 17953 and 2 * k * 4096 == 147070976
for _ in range(200):
    V, kk, H = 50, int(rng.integers(1, 40)), 8
    W, h = rng.normal(size=(V, H)), rng.normal(size=H) * rng.uniform(0.1, 3)
    z = W @ h
    p = np.exp(z - z.max())
    p /= p.sum()
    W2 = np.vstack([W, np.tile(W.mean(0), (kk, 1))])
    z2 = W2 @ h
    assert np.allclose(z2[V:], z.mean())
    q = np.exp(z2 - z2.max())
    q /= q.sum()
    assert q[V:].sum() <= kk / (V + kk) + 1e-12
    assert np.allclose(q[:V] / q[0], p / p[0])
    kl = float(np.sum(p * np.log(p / q[:V])))
    assert kl <= math.log(1 + kk / V) + 1e-12
assert abs(17953 / 49953 - 0.36) < 0.005 and abs(math.log(1 + 17953 / 32000) - 0.45) < 0.005
# Random new rows have no bound: one aligned row takes almost all the mass.
W, h = rng.normal(size=(50, 8)), rng.normal(size=8)
z = np.append(W @ h, 100 * h @ h)
q = np.exp(z - z.max())
assert q[-1] / q.sum() > 0.99


# 6. RoPE wavelengths.
def wavelength(b, i, d=128):
    return 2 * math.pi * b ** (2 * i / d)


assert math.isclose(wavelength(1e4, 0), 2 * math.pi) and math.isclose(wavelength(1e6, 0), 2 * math.pi)
assert abs(wavelength(1e4, 63) - 54410) < 1 and abs(wavelength(1e6, 63) - 5063256) < 1
assert abs(wavelength(1e6, 63) / wavelength(1e4, 63) - 93.06) < 0.01
assert abs(wavelength(5e5, 63) - 2.56e6) < 0.01e6 and abs(50 ** (126 / 128) - 47) < 0.1
ratios = [wavelength(1e6, i) / wavelength(1e4, i) for i in range(64)]
assert all(x < y for x, y in itertools.pairwise(ratios))
# The angle at position m is m times the frequency.
m, i = 1000, 10
assert math.isclose(m * 1e4 ** (-2 * i / 128) * wavelength(1e4, i) / (2 * math.pi), m)

# 7. Token counts.
assert math.isclose(0.85 * 500, 425) and abs(425 / 80 - 5.3) < 0.02
assert math.isclose(0.95 * 200, 190) and abs(190 / 55 - 3.45) < 0.01
assert math.isclose(0.56 * 500, 280) and abs(280 / 120 - 2.33) < 0.01
print("ok")
