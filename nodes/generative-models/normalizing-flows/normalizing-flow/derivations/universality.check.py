# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/normalizing-flow/ (ai://NEW-normalizing-flow).
"""Numeric check of the derivation universality."""

import math

import numpy as np

erf = np.vectorize(math.erf)


def Phi(v):
    return 0.5 * (1 + erf(np.asarray(v) / math.sqrt(2)))


def Phi_inv(p, lo=-40.0, hi=40.0):
    p = np.asarray(p, float)
    a, b = np.full_like(p, lo), np.full_like(p, hi)
    for _ in range(100):
        m = (a + b) / 2
        below = Phi(m) < p
        a, b = np.where(below, m, a), np.where(below, b, m)
    return (a + b) / 2


def log_p(x1, x2):  # x2 ~ N(0, 4), x1 | x2 ~ N(x2^2 / 4, 1)
    return -0.5 * (x2 / 2) ** 2 - math.log(2 * math.sqrt(2 * math.pi)) - 0.5 * (x1 - x2**2 / 4) ** 2 - 0.5 * math.log(2 * math.pi)


def F(x):  # the order (x2, x1)
    return np.stack([Phi(x[..., 1] / 2), Phi(x[..., 0] - x[..., 1] ** 2 / 4)], -1)


rng = np.random.default_rng(5)
for x in rng.standard_normal((10, 2)) * [1.5, 2.0] + [1.0, 0.0]:
    eps = 1e-6
    J = np.stack([(F(x + eps * e) - F(x - eps * e)) / (2 * eps) for e in np.eye(2)], 1)
    assert abs(J[0, 0]) < 1e-12  # z1 depends on x2 only: triangular in the order (x2, x1), so |det| = p
    assert abs(abs(np.linalg.det(J)) - math.exp(log_p(*x))) < 1e-7 * max(1.0, math.exp(log_p(*x)))

x2 = 2 * rng.standard_normal(200000)
x = np.stack([x2**2 / 4 + rng.standard_normal(200000), x2], 1)
z = F(x)
counts, _, _ = np.histogram2d(z[:, 0], z[:, 1], bins=10, range=[[0, 1], [0, 1]])
assert np.max(np.abs(counts / len(z) - 0.01)) < 0.0015  # uniform on the unit square

u = rng.standard_normal((100000, 2))  # T = F^-1 o G with G = Phi applied to each coordinate
g = Phi(u)
y2 = 2 * Phi_inv(g[:, 0])
y1 = y2**2 / 4 + Phi_inv(g[:, 1])
assert abs(np.mean(y2)) < 0.03 and abs(np.var(y2) - 4) < 0.08 and abs(np.mean(y1) - 1) < 0.03
print("universality: ok")
