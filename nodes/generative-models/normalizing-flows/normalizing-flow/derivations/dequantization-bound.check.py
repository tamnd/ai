# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/normalizing-flow/ (ai://NEW-normalizing-flow).
"""Numeric check of the derivation dequantization-bound."""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import flow  # noqa: E402

rng = np.random.default_rng(4)
L = 4


def q1(y):  # a density on [0, 1): 0.5 + y, with cumulative 0.5 y + y^2 / 2
    return 0.5 + y


def Q1(y):
    return 0.5 * y + 0.5 * y**2


for k in range(L):  # one dimension, bin by bin
    u = rng.random(400000)
    lhs = np.mean(np.log(q1((k + u) / L))) - math.log(L)
    rhs = math.log(Q1((k + 1) / L) - Q1(k / L))
    assert lhs < rhs and rhs - lhs < 0.02, (k, lhs, rhs)

P = np.array([Q1((k + 1) / L) - Q1(k / L) for k in range(L)])
assert abs(P.sum() - 1) < 1e-12

for k1 in range(L):  # two dimensions, a product density q(y) = q1(y1) q1(y2)
    for k2 in range(L):
        u = rng.random((200000, 2))
        y = (np.array([k1, k2]) + u) / L
        lhs = np.mean(np.log(q1(y[:, 0]) * q1(y[:, 1]))) - 2 * math.log(L)
        assert lhs < math.log(P[k1] * P[k2])

step = rng.dirichlet(np.ones(L)) * L  # a density constant on the bins: equality
u = rng.random(1000)
for k in range(L):
    assert abs(np.mean(np.log(np.full_like(u, step[k]))) - math.log(L) - math.log(step[k] / L)) < 1e-12

assert abs(flow.bits_per_dim(0.0, 3072) - 8.0) < 1e-12
pix = rng.integers(0, 256, (1000, 5))
yq = flow.dequantize(pix, rng)
assert np.all((yq >= pix / 256) & (yq < (pix + 1) / 256))
print("dequantization-bound: ok")
