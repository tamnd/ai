# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/cosine-learning-rate-schedule/ (ai://NEW-cosine-learning-rate-schedule).
# Check of derivations/mean-rate.md: the mean of one cosine cycle is (eta_max + eta_min) / 2 by integration and by the
# sum over the T + 1 points t = 0..T; over the T points t = 0..T-1 it is larger by (eta_max - eta_min) / (2 T).
import numpy as np


def eta(t, T, hi, lo):
    return lo + 0.5 * (hi - lo) * (1 + np.cos(np.pi * t / T))


for hi, lo, T in [(0.1, 0.0, 100), (0.05, 0.001, 37), (3e-4, 3e-5, 1000), (1.0, 0.5, 1)]:
    mid = (hi + lo) / 2
    # integration: trapezoid and Simpson on a fine grid
    n = 20_000
    x = np.linspace(0.0, T, n + 1)
    y = eta(x, T, hi, lo)
    trap = float(np.sum((y[1:] + y[:-1]) / 2) * (T / n)) / T
    w = np.ones(n + 1)
    w[1:-1:2], w[2:-1:2] = 4, 2
    simpson = float(np.sum(w * y) * (T / n) / 3) / T
    assert abs(trap - mid) < 1e-9 * max(hi, 1) and abs(simpson - mid) < 1e-12 * max(hi, 1), (trap, simpson, mid)
    # discrete: T + 1 points give the midpoint exactly, T points give the midpoint plus (hi - lo) / (2 T)
    t = np.arange(T + 1)
    assert abs(eta(t, T, hi, lo).mean() - mid) < 1e-12
    assert abs(eta(t[:-1], T, hi, lo).mean() - (mid + (hi - lo) / (2 * T))) < 1e-12

# the cosine sum identity behind step 4, for every T up to 1000
for T in range(1, 1001):
    s = np.cos(np.pi * np.arange(T) / T).sum()
    assert abs(s - 1.0) < 1e-9, (T, s)

# the same total learning rate as a constant rate at the midpoint and as a linear decay with the same ends
T, hi, lo = 200, 0.1, 0.01
t = np.arange(T + 1)
lin = hi - (hi - lo) * t / T
assert abs(eta(t, T, hi, lo).sum() - lin.sum()) < 1e-12
assert abs(eta(t, T, hi, lo).sum() - (T + 1) * (hi + lo) / 2) < 1e-12
print("ok")
