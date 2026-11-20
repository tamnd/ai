# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/fine-tuning/ (ai://NEW-fine-tuning).
"""Checks the statements of fine-tuning-arithmetic.md numerically."""

import math

import numpy as np

rng = np.random.default_rng(0)

# 1: gradients of B lie in the row space of X; a perfect probe is a fixed point.
n, d, k = 5, 8, 3
X, Y = rng.normal(size=(n, d)), rng.normal(size=n)
B, v = rng.normal(size=(k, d)), rng.normal(size=k)
r = X @ B.T @ v - Y
gB = np.outer(v, r @ X) / n
null = np.linalg.svd(X)[2][n:]
assert np.allclose(gB @ null.T, 0)
X2, Y2 = X[:2], X[:2] @ B.T @ rng.normal(size=k)  # realizable, so the probe fits exactly
v_lp = np.linalg.lstsq(X2 @ B.T, Y2, rcond=None)[0]
r2 = X2 @ B.T @ v_lp - Y2
assert np.allclose(r2, 0) and np.allclose(B @ X2.T @ r2, 0) and np.allclose(np.outer(v_lp, r2 @ X2), 0)

# 2: BB^T - vv^T is conserved by gradient flow (small-step gradient descent, checked to the step error).
B, v = rng.normal(size=(k, d)), np.zeros(k)
c0 = B @ B.T - np.outer(v, v)
lr = 1e-4
for _ in range(20000):
    r = X @ B.T @ v - Y
    B, v = B - lr * np.outer(v, r @ X) / n, v - lr * B @ X.T @ r / n
assert np.abs(B @ B.T - np.outer(v, v) - c0).max() < 1e-2

# 3: the golden-ratio example.
phi = (1 + math.sqrt(5)) / 2
b = math.sqrt(phi)
assert abs(b**4 - b**2 - 1) < 1e-12 and abs(b * (1 / b) - 1) < 1e-12 and abs(b**2 - (1 / b) ** 2 - 1) < 1e-12
B, v, x = np.array([1.0, 1.0]), 0.0, np.array([1.0, 0.0])
for _ in range(200000):
    r = v * (B @ x) - 1.0
    B, v = B - 1e-3 * v * r * x, v - 1e-3 * (B @ x) * r
assert B[1] == 1.0 and abs(B[0] - b) < 1e-3 and abs(v - 1 / b) < 1e-3
assert round(1 / b, 3) == 0.786 and round(b, 3) == 1.272


# 4: schedules.
def stlr(t, T, cut_frac=0.1, ratio=32, eta=0.01):
    cut = math.floor(T * cut_frac)
    p = t / cut if t < cut else 1 - (t - cut) / (cut * (1 / cut_frac - 1))
    return eta * (1 + p * (ratio - 1)) / ratio


T = 1000
assert math.isclose(stlr(0, T), 0.01 / 32) and math.isclose(stlr(100, T), 0.01) and math.isclose(stlr(T, T), 0.01 / 32)
assert math.isclose(0.01 / 32, 0.0003125)
assert math.isclose(2.6**3, 17.576) and math.isclose(2.6**4, 45.6976)
w, peak = 0.1, 1.0
ts = np.linspace(0, T, 1_000_001)
sched = np.where(ts < w * T, peak * ts / (w * T), peak * (T - ts) / (T - w * T))
assert abs(sched.mean() - peak / 2) < 1e-5 and abs(ts[np.argmax(sched)] - w * T) < 1e-2


# 5: Adam with and without bias correction under a constant gradient.
def ratio(t, b1=0.9, b2=0.999):
    return (1 - b1**t) / math.sqrt(1 - b2**t)


g, m, s = 0.37, 0.0, 0.0
for t in range(1, 101):
    m, s = 0.9 * m + 0.1 * g, 0.999 * s + 0.001 * g * g
    corrected = (m / (1 - 0.9**t)) / math.sqrt(s / (1 - 0.999**t))
    assert math.isclose(corrected, 1.0) and math.isclose((m / math.sqrt(s)) / corrected, ratio(t))
assert math.isclose(ratio(1), math.sqrt(10)) and round(ratio(1), 3) == 3.162
best = max(range(1, 5000), key=ratio)
assert best == 12 and round(ratio(12), 3) == 6.569
assert round(ratio(100), 3) == 3.241 and round(ratio(1000), 3) == 1.258 and abs(ratio(100000) - 1) < 1e-6

# 6: the EWC optimum is the precision-weighted average.
for a, bb, th, lf in [(2.0, 1.0, -1.0, 3.0), (0.5, 4.0, 0.0, 0.1)]:
    grid = np.linspace(-5, 5, 2_000_001)
    obj = a / 2 * (grid - bb) ** 2 + lf / 2 * (grid - th) ** 2
    assert abs(grid[np.argmin(obj)] - (a * bb + lf * th) / (a + lf)) < 1e-5
assert abs((2 * 1 + 1e-9 * 5) / (2 + 1e-9) - 1) < 1e-8 and abs((2 * 1 + 1e9 * 5) / (2 + 1e9) - 5) < 1e-8

print("fine-tuning-arithmetic: ok")
