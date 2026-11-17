# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/mode-connectivity/ (ai://MDXMTA).
# Check of derivations/loss-barrier.md: both barriers are non-negative; for a convex loss B_lin = 0 and
# B_mean = |L0 - L1| / 2; and |B_mean - B_lin| <= |L0 - L1| / 2, with equal gaps at t = 1/2.
import numpy as np

rng = np.random.default_rng(0)
ts = np.linspace(0, 1, 1001)


def barriers(L):
    b_mean = L.max() - 0.5 * (L[0] + L[-1])
    b_lin = (L - (1 - ts) * L[0] - ts * L[-1]).max()
    return b_mean, b_lin


for _ in range(50):
    A = rng.normal(size=(5, 5))
    H = A @ A.T
    g, c = rng.normal(size=5), rng.normal()
    w1, w2 = rng.normal(size=(2, 5))
    W = (1 - ts)[:, None] * w1 + ts[:, None] * w2
    L = 0.5 * np.einsum("ti,ij,tj->t", W, H, W) + W @ g + c
    b_mean, b_lin = barriers(L)
    assert abs(b_lin) < 1e-12
    assert abs(b_mean - 0.5 * abs(L[0] - L[-1])) < 1e-9

for _ in range(200):
    L0, L1, h, mu = rng.normal(), rng.normal(), rng.uniform(0, 3), rng.uniform(0.1, 0.9)
    L = (1 - ts) * L0 + ts * L1 + h * np.exp(-((ts - mu) ** 2) / 0.01) + 0.1 * rng.normal() * np.sin(7 * ts)
    L0, L1 = L[0], L[-1]
    b_mean, b_lin = barriers(L)
    assert b_lin >= 0 and b_mean >= 0.5 * abs(L0 - L1) - 1e-12
    assert abs(b_mean - b_lin) <= 0.5 * abs(L0 - L1) + 1e-12
    k = 500
    assert abs((L[k] - 0.5 * (L0 + L1)) - (L[k] - 0.5 * L0 - 0.5 * L1)) < 1e-12
    L_eq = L - ts * (L[-1] - L[0])
    assert np.isclose(*barriers(L_eq))
print("ok")
