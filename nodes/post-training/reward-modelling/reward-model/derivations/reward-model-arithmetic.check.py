# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/reward-model/ (ai://NEW-reward-model).
"""Checks the statements of reward-model-arithmetic.md numerically."""

import itertools
import math

import numpy as np


def sig(t):
    return 1.0 / (1.0 + np.exp(-t))


def loss(d):
    return np.log1p(np.exp(-d))


rng = np.random.default_rng(0)

# 1: the Bradley-Terry probability is the logistic function of the difference.
rw, rl = rng.normal(size=1000) * 3, rng.normal(size=1000) * 3
p = np.exp(rw) / (np.exp(rw) + np.exp(rl))
assert np.allclose(p, sig(rw - rl))
assert np.allclose(-np.log(p), loss(rw - rl))

# 2: the gradients, against central differences.
h = 1e-6
for a, b in zip(rw[:50], rl[:50], strict=True):
    gw = (loss(a + h - b) - loss(a - h - b)) / (2 * h)
    gl = (loss(a - (b + h)) - loss(a - (b - h))) / (2 * h)
    assert abs(gw + sig(b - a)) < 1e-7 and abs(gl - sig(b - a)) < 1e-7
    assert abs(gw + gl) < 1e-7

# 3: shift invariance, and the effect of scaling.
c = rng.normal()
assert np.allclose(loss((rw + c) - (rl + c)), loss(rw - rl))
d = rw - rl
assert np.all(np.abs(sig(2 * d[d != 0]) - sig(d[d != 0])) > 0)

# 4: forward passes and comparisons for ranked responses.
K = 9
assert math.comb(K, 2) == 36 and 2 * math.comb(K, 2) == K * (K - 1) == 72
assert 64 * math.comb(K, 2) == 2304
assert max(64 * math.comb(k, 2) for k in range(4, 10)) == 2304

# 5: label noise.
eps = 0.1


def noisy(d):
    return -np.log(eps / 2 + (1 - eps) * sig(d))


grid = np.linspace(-40, 40, 8001)
# Strictly below on a range where float64 can resolve the gap; at -40 the gap is below machine precision.
assert np.all(noisy(np.linspace(-20, 40, 6001)) < -math.log(eps / 2))
assert np.all(noisy(grid) <= -math.log(eps / 2))
assert abs(-math.log(0.05) - 2.995732) < 1e-6
gnoisy = (noisy(grid + h) - noisy(grid - h)) / (2 * h)
assert abs(gnoisy[0]) < 1e-12 + 1e-9
assert abs(loss(-3.0) - 3.048587) < 1e-6 and abs(noisy(-3.0) - 2.378567) < 1e-6
assert abs(sig(3.0) - 0.952574) < 1e-6
g3 = (noisy(-3 + h) - noisy(-3 - h)) / (2 * h)
assert abs(abs(g3) - 0.438687) < 1e-6

# 6: the margin loss.
assert abs(loss(0.0) - math.log(2)) < 1e-12
assert abs(loss(1.0 - 1.0) - 0.693147) < 1e-6 and abs(loss(1.0) - 0.313262) < 1e-6
assert np.all(np.diff(loss(grid - 1.0)) < 0)

# 7: the accuracy ceiling under Bradley-Terry labels.
for dd, want in [(1, 0.731059), (2, 0.880797), (3, 0.952574)]:
    assert abs(sig(dd) - want) < 1e-6
    labels = rng.random(200_000) < sig(dd)  # True: labelled in the order of r*
    assert abs(labels.mean() - sig(dd)) < 4 * math.sqrt(sig(dd) * (1 - sig(dd)) / 200_000)
    assert max(labels.mean(), 1 - labels.mean()) <= sig(dd) + 0.01

# 8: ensembles.
R = np.array([1.0, 1.2, 3.0])
m, v = R.mean(), R.var()
assert abs(m - 1.733333) < 1e-6 and R.min() == 1.0 and abs(v - 0.808889) < 1e-6
assert abs(m - 0.5 * v - 1.328889) < 1e-6
for _ in range(200):
    Rs = rng.normal(size=int(rng.integers(2, 8)))
    assert Rs.min() <= Rs.mean() and Rs.mean() - 0.7 * Rs.var() <= Rs.mean()


# 9: the KL of best-of-n.
def bon_kl(n):
    return math.log(n) - (n - 1) / n


for n, want in [(4, 0.636294), (16, 1.835089), (64, 3.174508)]:
    assert abs(bon_kl(n) - want) < 1e-6
    # Monte Carlo on a continuous reward: the log density ratio is log n + (n - 1) log F(R) at the best sample.
    u = rng.random((40_000, n)).max(axis=1)
    mc = np.mean(np.log(n) + (n - 1) * np.log(u))
    assert abs(mc - bon_kl(n)) < 0.02, (n, mc)
    # Exact on discrete policies with distinct rewards: never above the bound, and close to it with many outputs.
    for size in (3, 50, 20_000):
        q = rng.dirichlet(np.ones(size))  # base probabilities, in increasing order of reward
        F = np.cumsum(q)
        Fm = np.concatenate([[0.0], F[:-1]])
        pb = F**n - Fm**n
        kl = float(np.sum(pb * np.log(pb / q)))
        assert kl <= bon_kl(n) + 1e-9
        if size == 20_000:
            assert bon_kl(n) - kl < 0.01


# 10: the cleaning robot.
def hackable(true, proxy):
    subsets = list(itertools.product([0, 1], repeat=3))
    J = {s: float(np.dot(s, true)) for s in subsets}
    Jp = {s: float(np.dot(s, proxy)) for s in subsets}
    return any(Jp[a] > Jp[b] and J[a] < J[b] for a in subsets for b in subsets)


assert hackable([1, 1, 1], [1, 0, 0])
assert not hackable([1, 1, 1], [1, 1, 0])
assert not hackable([1, 1.5, 2], [1, 1, 1])
assert hackable([1, 1.5, 3], [1, 1, 1])
S, Sp = (1, 1, 0), (0, 0, 1)
assert np.dot(S, [1, 1, 1]) == 2 and np.dot(Sp, [1, 1, 1]) == 1
assert np.dot(S, [1, 1.5, 3]) == 2.5 and np.dot(Sp, [1, 1.5, 3]) == 3

# 11: the worked example.
r = np.array([2.0, 0.5, 0.0])
pairs = [(0, 1), (0, 2), (1, 2)]
pl = [loss(r[i] - r[j]) for i, j in pairs]
assert np.allclose(pl, [0.201413, 0.126928, 0.474077], atol=1e-6)
assert abs(np.mean(pl) - 0.267473) < 1e-6
grad = np.zeros(3)
for i, j in pairs:
    grad[i] -= sig(r[j] - r[i]) / 3
    grad[j] += sig(r[j] - r[i]) / 3
assert np.allclose(grad, [-0.100543, -0.065038, 0.165581], atol=1e-6)
num = np.zeros(3)
for k in range(3):
    e = np.zeros(3)
    e[k] = h
    num[k] = (
        np.mean([loss((r + e)[i] - (r + e)[j]) for i, j in pairs])
        - np.mean([loss((r - e)[i] - (r - e)[j]) for i, j in pairs])
    ) / (2 * h)
assert np.allclose(num, grad, atol=1e-8)

print("reward-model-arithmetic: ok")
