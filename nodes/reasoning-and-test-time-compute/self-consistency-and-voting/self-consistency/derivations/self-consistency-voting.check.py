# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/self-consistency/ (ai://NEW-self-consistency).
"""Checks every step of self-consistency-voting.md numerically."""

import math
from fractions import Fraction as F

import numpy as np

rng = np.random.default_rng(0)


def binom_major(p, m):
    return sum(math.comb(m, j) * p**j * (1 - p) ** (m - j) for j in range(m // 2 + 1, m + 1))


# 1. The vote converges to the mode of the marginal, not the most likely path.
paths = {("r1", "A"): 0.3, ("r2", "B"): 0.25, ("r3", "B"): 0.25, ("r4", "C"): 0.2}
marg = {}
for (r, a), pr in paths.items():
    marg[a] = marg.get(a, 0) + pr
assert max(paths, key=paths.get)[1] == "A" and max(marg, key=marg.get) == "B"
keys = list(paths)
draws = rng.choice(len(keys), size=20000, p=list(paths.values()))
ans = np.array([keys[i][1] for i in draws])
counts = {a: int((ans == a).sum()) for a in marg}
assert max(counts, key=counts.get) == "B"
for a, pa in marg.items():
    assert abs(counts[a] / 20000 - pa) < 0.02

# 2. Weighted votes.
a_s = np.array([0, 1, 1, 0, 2])
assert np.argmax(np.bincount(a_s, weights=np.ones(5))) == np.argmax(np.bincount(a_s))
q = 0.9
for K in (5, 10, 40):
    assert q ** (K + 1) < q**K and math.isclose((q**K) ** (1 / K), q)
short, long_ = 0.6**5, 0.9**40  # a 5-token path of weak tokens beats a 40-token path of strong ones
assert short > long_ and math.isclose(short ** (1 / 5), 0.6) and math.isclose(long_ ** (1 / 40), 0.9)

# 3. Binomial vote.
assert [round(binom_major(0.6, m), 3) for m in (1, 5, 11, 41)] == [0.6, 0.683, 0.753, 0.903]
assert [round(binom_major(0.4, m), 3) for m in (1, 5, 11, 41)] == [0.4, 0.317, 0.247, 0.097]
for m in (1, 5, 11, 41, 101):
    assert math.isclose(binom_major(0.4, m), 1 - binom_major(0.6, m))
assert binom_major(F(3, 5), 401) > F(999, 1000) and binom_major(F(2, 5), 401) < F(1, 1000)

# 4. Hoeffding bound.
assert math.exp(-40 * 0.09 / 2) < 0.17 and math.isclose(math.exp(-1.8), 0.165, abs_tol=5e-4)
assert math.exp(-100 * 0.09 / 2) < 0.012 and math.isclose(math.exp(-4.5), 0.0111, abs_tol=5e-5)
# exact error of a three-way vote stays under the bound
for m in (10, 40, 100):
    p = np.array([0.5, 0.2, 0.3])  # delta = 0.2 against the nearest competitor
    s = rng.multinomial(m, p, size=20000)
    wrong = (s[:, 0] <= s[:, 1:].max(1)).mean()
    assert wrong <= 2 * math.exp(-m * 0.04 / 2) + 0.01


# 5. Mixture.
def mix(m):
    return 0.6 * binom_major(0.6, m) + 0.4 * binom_major(0.45, m)


assert math.isclose(mix(1), 0.54)
ms = range(1, 402, 2)
best = max(ms, key=lambda m: mix(m))
assert best == 71 and round(mix(71), 3) == 0.653 and round(mix(401), 3) == 0.609
assert mix(401) > 0.6


# 6. Beta stopping rule.
def beta_conf(v1, v2):
    n = v1 + v2 + 1
    return 1 - F(sum(math.comb(n, j) for j in range(v1 + 1, n + 1)), 2**n)


def beta_conf_num(v1, v2, n=2_000_000):
    x = (np.arange(n) + 0.5) / n
    dens = x**v1 * (1 - x) ** v2
    return dens[x > 0.5].sum() / dens.sum()


for v in [(9, 1), (4, 0), (3, 0), (6, 2), (5, 3)]:
    assert abs(float(beta_conf(*v)) - beta_conf_num(*v)) < 1e-6
assert beta_conf(9, 1) == 1 - F(12, 2048) and round(float(beta_conf(9, 1)), 3) == 0.994
assert beta_conf(4, 0) == 1 - F(1, 32) and beta_conf(4, 0) > F(95, 100)
assert beta_conf(3, 0) == F(15, 16) and beta_conf(3, 0) < F(95, 100)
assert round(float(beta_conf(6, 2)), 3) == 0.910 and beta_conf(6, 2) < F(95, 100)

# 7. Windows.
for w in (2, 3, 5, 8):
    for _ in range(2000):
        P = rng.dirichlet(np.ones(rng.integers(2, 6)))
        P.sort()
        if P[-1] - P[-2] < 1e-9:
            continue
        assert (P[:-1] ** w).sum() <= 2.0 ** -(w - 1) + 1e-15
assert F(1, 2**7) == F(1, 128)
P = np.array([0.9, 0.08, 0.02])
qq = (P**5).sum()
assert round(qq, 3) == 0.590 and round(5 / qq, 1) == 8.5
# simulate
tot = 0
for _ in range(20000):
    n = 0
    while True:
        win = rng.choice(3, size=5, p=P)
        n += 5
        if (win == win[0]).all():
            break
    tot += n
assert abs(tot / 20000 - 5 / qq) < 0.15

# 8. Coverage.
assert round(1 - 0.8**10, 3) == 0.893
assert 1 - 0.8**200 > 1 - 1e-15
P = np.array([0.2, 0.3, 0.5 / 3, 0.5 / 3, 0.5 / 3])  # answer 0 is correct, answer 1 is the mode
assert np.argmax(P) == 1
for n, c, k in [(10, 3, 1), (10, 3, 4), (20, 0, 5), (20, 20, 5), (7, 2, 7)]:
    est = 1 - F(math.comb(n - c, k), math.comb(n, k))
    assert 0 <= est <= 1
# unbiasedness, exactly: sum over c of Bin(n, p)(c) * estimate = 1 - (1 - p)^k
for n in (5, 10):
    for k in range(1, n + 1):
        p = F(1, 5)
        e = sum(
            math.comb(n, c) * p**c * (1 - p) ** (n - c) * (1 - F(math.comb(n - c, k), math.comb(n, k)))
            for c in range(n + 1)
        )
        assert e == 1 - (1 - p) ** k
print("ok")
