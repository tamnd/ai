# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/speculative-decoding/ (ai://NEW-speculative-decoding).
# Check of derivations/expected-tokens.md: under i.i.d. acceptance the tokens per run are a geometric count capped at
# gamma + 1 with mean (1 - alpha^(gamma+1)) / (1 - alpha), bounded by gamma + 1 and 1 / (1 - alpha); the walltime and
# arithmetic factors reproduce Tables 1 and 4 of Leviathan et al.
from fractions import Fraction as F

import numpy as np


def mean(a, g):
    return g + 1 if a == 1 else (1 - a ** (g + 1)) / (1 - a)


for a in (F(0), F(1, 4), F(1, 2), F(3, 4), F(1)):
    for g in range(9):
        pmf = [a ** (n - 1) * (1 - a) for n in range(1, g + 1)] + [a**g]
        assert sum(pmf) == 1
        assert sum(n * w for n, w in enumerate(pmf, start=1)) == mean(a, g) == sum(a**k for k in range(g + 1))
assert mean(F(3, 4), 3) == F(175, 64)

rng = np.random.default_rng(0)
for a, g in ((0.3, 2), (0.75, 3), (0.9, 10)):
    acc = rng.random((200_000, g)) < a
    n = 1 + np.cumprod(acc, axis=1).sum(axis=1)
    assert abs(n.mean() - mean(a, g)) < 5 * n.std() / np.sqrt(n.size)
    assert n.min() >= 1 and n.max() <= g + 1

grid = np.linspace(0, 0.99, 100)
for g in range(0, 12):
    m = np.array([mean(a, g) for a in grid])
    assert np.all(m >= 1) and np.all(m <= g + 1 + 1e-12) and np.all(m <= 1 / (1 - grid) + 1e-12)
    assert np.all(np.diff(m) >= -1e-12)
    assert np.all(np.array([mean(a, g + 1) for a in grid]) >= m)


def speed(a, g, c):
    return mean(a, g) / (g * c + 1)


def ops(a, g, ch):
    return (g * ch + g + 1) / mean(a, g)


table1 = [(0.6, 2, 1.53, 1.96), (0.7, 3, 1.58, 2.53), (0.8, 2, 1.23, 2.44),
          (0.8, 5, 1.63, 3.69), (0.9, 2, 1.11, 2.71), (0.9, 10, 1.60, 6.86)]
for a, g, o, s in table1:
    assert round(speed(a, g, 0), 2) == s and round(ops(a, g, 0), 2) == o
assert round(speed(0.75, 7, 0.02), 1) == 3.2  # EnDe, T5-small, temp 0 (Table 4)
assert round(speed(0.65, 5, 0.02), 1) == 2.4  # CNNDM, T5-small, temp 0 (Table 4)
for a in grid:
    for c in (0.0, 0.05, 0.3):
        assert abs(speed(a, 1, c) - (1 + a) / (1 + c)) < 1e-12
        assert (speed(a, 1, c) > 1) == (a > c)
print("expected-tokens: ok")
