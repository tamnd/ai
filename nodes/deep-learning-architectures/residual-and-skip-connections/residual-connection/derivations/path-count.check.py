# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/residual-connection/ (ai://F5K1AF).
"""Numeric check of the derivation path-count: 2^n paths, C(n, k) of length k, the shares for n = 54, and the dense connections."""

import itertools
import math

import numpy as np

rng = np.random.default_rng(20160520)
for n in range(1, 9):
    A = [rng.normal(scale=0.4, size=(3, 3)) for _ in range(n)]
    full = np.eye(3)
    for a in A:
        full = (np.eye(3) + a) @ full
    total, lengths = np.zeros((3, 3)), [0] * (n + 1)
    for S in itertools.product([0, 1], repeat=n):
        term = np.eye(3)
        for a, pick in zip(A, S):
            if pick:
                term = a @ term
        total += term
        lengths[sum(S)] += 1
    assert np.allclose(full, total)  # eq-paths
    assert lengths == [math.comb(n, k) for k in range(n + 1)] and sum(lengths) == 2**n

n = 54
mid = sum(math.comb(n, k) for k in range(19, 36)) / 2**n
short = sum(math.comb(n, k) for k in range(5, 18)) / 2**n
assert round(mid, 3) == 0.980 and mid > 0.95
assert round(short, 4) == 0.0045


def dense_connections(L):
    return sum(len(range(ell)) for ell in range(1, L + 1))  # layer ell takes x_0 .. x_{ell-1}


assert all(dense_connections(L) == L * (L + 1) // 2 for L in range(1, 50))
print(f"path-count: expansions hold for n = 1..8; n=54: {mid:.3f} of paths through 19..35 units, {short:.4f} through 5..17; dense L(L+1)/2 holds")
