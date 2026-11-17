# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/resnet/ (ai://NEW-resnet).
"""Numeric check of the derivation residual-gradient: the unrolled sum, the Jacobian product, the paths and the scaled shortcut."""

import itertools
import math

import numpy as np

rng = np.random.default_rng(20151210)
d, n = 5, 6
W1 = [rng.normal(scale=0.4, size=(d, d)) for _ in range(n)]
W2 = [rng.normal(scale=0.4, size=(d, d)) for _ in range(n)]


def F(i, x):
    return W2[i] @ np.tanh(W1[i] @ x)


def J(i, x):
    return W2[i] @ np.diag(1 - np.tanh(W1[i] @ x) ** 2) @ W1[i]


def run(x0, lam=1.0):
    xs = [x0]
    for i in range(n):
        xs.append(lam * xs[-1] + F(i, xs[-1]))
    return xs


x0 = rng.normal(size=d)
xs = run(x0)
assert np.allclose(xs[-1], x0 + sum(F(i, xs[i]) for i in range(n)))  # eq-residual-unroll

# The Jacobian dx_n/dx_0 by finite differences against the product of (I + J_i).
h = 1e-6
fd = np.stack([(run(x0 + h * e)[-1] - run(x0 - h * e)[-1]) / (2 * h) for e in np.eye(d)], axis=1)
prod = np.eye(d)
for i in range(n):
    prod = (np.eye(d) + J(i, xs[i])) @ prod
assert np.allclose(fd, prod, atol=1e-6)
g = rng.normal(size=d)  # dE/dx_L for E = g . x_L
assert np.allclose(g @ prod, g + g @ (prod - np.eye(d)))  # eq-residual-grad: the direct term is g itself

# Linear units: the product is the sum of 2^n ordered subset products, C(n, k) of length k.
A = [rng.normal(scale=0.3, size=(d, d)) for _ in range(n)]
full = np.eye(d)
for a in A:
    full = (np.eye(d) + a) @ full
paths, lengths = np.zeros((d, d)), [0] * (n + 1)
for S in itertools.product([0, 1], repeat=n):
    term = np.eye(d)
    for i, pick in enumerate(S):
        if pick:
            term = A[i] @ term
    paths += term
    lengths[sum(S)] += 1
assert np.allclose(full, paths)
assert lengths == [math.comb(n, k) for k in range(n + 1)] and sum(lengths) == 2**n

# The scaled shortcut: in the expansion of prod (lam I + A_i), the term without any A_i is lam^n I.
for lam in (0.9, 1.1):
    full = np.eye(d)
    for a in A:
        full = (lam * np.eye(d) + a) @ full
    rest = np.zeros((d, d))
    for S in itertools.product([0, 1], repeat=n):
        if any(S):
            term = np.eye(d)
            for i, pick in enumerate(S):
                term = (A[i] if pick else lam * np.eye(d)) @ term
            rest += term
    assert np.allclose(full - rest, lam**n * np.eye(d))
assert 0.9**50 < 0.01 and 1.1**50 > 100  # over 50 units the direct term vanishes or explodes
print(f"residual-gradient: unroll and Jacobian product hold (n={n}); {2**n} paths with lengths {lengths}; 0.9^50={0.9**50:.4f}, 1.1^50={1.1**50:.1f}")
