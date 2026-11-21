# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/chain-of-thought-faithfulness/ (ai://NEW-chain-of-thought-faithfulness).
"""Checks every step of faithfulness-metrics.md numerically."""

import math

import numpy as np

rng = np.random.default_rng(0)


def alpha(p, q, n):
    return 1 - q / ((n - 2) * p)


# 1. Hint-test normalization.
assert math.isclose(alpha(0.30, 0.04, 4), 0.9333, abs_tol=5e-5)
assert math.isclose(0.20 / alpha(0.30, 0.04, 4), 0.2143, abs_tol=5e-5)
assert math.isclose(alpha(0.10, 0.12, 4), 0.4, abs_tol=1e-12)
assert math.isclose(0.10 / alpha(0.10, 0.12, 4), 0.25, abs_tol=1e-12)
assert alpha(0.05, 0.10, 4) <= 0

# Simulation: noise moves are uniform and never verbalized; caused moves are verbalized with f.
n, trials, f, use, noise = 4, 400_000, 0.3, 0.2, 0.12
au = rng.integers(1, n, trials)  # the unhinted answer, never the hint 0
caused = rng.random(trials) < use
ah = au.copy()
moved = (~caused) & (rng.random(trials) < noise)
others = np.array([[o for o in range(n) if o != a] for a in range(n)])
ah[moved] = others[au[moved], rng.integers(0, n - 1, moved.sum())]
ah[caused] = 0
verbal = caused & (rng.random(trials) < f)
p = np.mean(ah == 0)
q = np.mean((ah != 0) & (ah != au))
keep = ah == 0
raw = verbal[keep].mean()
assert abs(raw / alpha(p, q, n) - f) < 0.01, raw / alpha(p, q, n)
assert raw < f


# 2. Early-answering AOC.
def aoc(s):
    s = np.asarray(s, float)
    L = len(s) - 1
    return 1 - np.sum((s[1:] + s[:-1]) / 2) / L


assert math.isclose(aoc([0.3, 0.4, 0.6, 0.9, 1.0]), 0.3625, abs_tol=1e-12)
assert aoc([1, 1, 1, 1, 1]) == 0
for m in (2, 4, 5):
    for L in (3, 4, 6):
        assert math.isclose(aoc([1 / m] * L + [1]), (1 - 1 / m) * (1 - 1 / (2 * L)), abs_tol=1e-12)
assert math.isclose(aoc([0.2] * 4 + [1]), 0.7, abs_tol=1e-12)


# 3. Detection over K episodes.
def caught(v, K):
    return 1 - (1 - v) ** K


assert math.isclose(caught(0.02, 1), 0.02, abs_tol=1e-12)
assert math.isclose(caught(0.02, 10), 0.1829, abs_tol=5e-5)
assert math.isclose(caught(0.02, 100), 0.8674, abs_tol=5e-5)
assert caught(0.02, 148) < 0.95 < caught(0.02, 149)
sim = (rng.random((100_000, 100)) < 0.02).any(axis=1).mean()
assert abs(sim - caught(0.02, 100)) < 0.005

# 4. Pair contradictions.
n1, n2 = 0.99, 0.63
assert math.isclose(n1 * n2, 0.6237, abs_tol=1e-12)
assert math.isclose(n1 * n2 + (1 - n1) * (1 - n2), 0.6274, abs_tol=1e-12)
a1 = rng.random(200_000) < n1
a2 = rng.random(200_000) < n2
assert abs(np.mean(a1 == a2) - 0.6274) < 0.005

print("ok")
