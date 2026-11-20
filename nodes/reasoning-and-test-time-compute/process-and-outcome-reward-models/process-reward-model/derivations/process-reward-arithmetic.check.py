# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/process-reward-model/ (ai://NEW-process-reward-model).
"""Checks every step of process-reward-arithmetic.md numerically."""

import itertools
import math

import numpy as np

rng = np.random.default_rng(0)

# 1. Cross-entropy is minimized at the label mean.
r = np.linspace(1e-4, 1 - 1e-4, 99999)
for mu in (0.1, 0.37, 0.5, 0.9):
    loss = -(mu * np.log(r) + (1 - mu) * np.log(1 - r))
    assert abs(r[loss.argmin()] - mu) < 1e-4

# 2. Soft and hard labels.
N = 8
for q in (0.05, 0.1, 0.5, 0.9):
    c = rng.binomial(N, q, size=400000)
    assert abs((c / N).mean() - q) < 3e-3
    assert abs((c / N).var() - q * (1 - q) / N) < 2e-3
    assert abs((c >= 1).mean() - (1 - (1 - q) ** N)) < 3e-3
    assert 1 - (1 - q) ** N > q
assert round(1 - 0.9**8, 3) == 0.570
assert math.isclose(max(q * (1 - q) / 8 for q in np.linspace(0, 1, 1001)), 1 / 32)
assert round(math.sqrt(1 / 32), 3) == 0.177


# 3. Binary search for the first error.
def first_error(passes, M):
    lo, hi, checks = 1, M + 1, 0  # answer in [lo, hi]
    while lo < hi:
        mid = (lo + hi) // 2
        checks += 1
        if passes(mid):
            lo = mid + 1
        else:
            hi = mid
    return lo, checks


for M in (1, 2, 5, 20, 33):
    worst = 0
    for f in range(1, M + 2):
        got, checks = first_error(lambda t, f=f: t < f, M)
        assert got == f
        worst = max(worst, checks)
    assert worst == math.ceil(math.log2(M + 1))
assert math.ceil(math.log2(21)) == 5

# 4. Implicit rewards on a toy autoregressive model with 3 steps over 3 symbols.
V, T, beta = 3, 3, 0.5


def table(seed):
    g = np.random.default_rng(seed)
    out = {}
    for t in range(T):
        for prefix in itertools.product(range(V), repeat=t):
            p = g.random(V) + 0.1
            out[prefix] = p / p.sum()
    return out


pt, pr = table(1), table(2)


def logp(tab, y):
    return sum(math.log(tab[tuple(y[:i])][y[i]]) for i in range(len(y)))


def q(y, t):
    return beta * sum(math.log(pt[tuple(y[:i])][y[i]] / pr[tuple(y[:i])][y[i]]) for i in range(t))


for y in itertools.product(range(V), repeat=T):
    rew = beta * (logp(pt, y) - logp(pr, y))
    steps = [q(y, t) - q(y, t - 1) for t in range(1, T + 1)]
    assert math.isclose(sum(steps), rew, abs_tol=1e-12)
    for t in range(T + 1):
        pre = y[:t]
        e = 0.0
        for rest in itertools.product(range(V), repeat=T - t):
            full = pre + rest
            pref = math.exp(logp(pr, full) - logp(pr, pre))
            e += pref * math.exp(beta * (logp(pt, full) - logp(pr, full)) / beta)
        assert math.isclose(q(y, t), beta * math.log(e), abs_tol=1e-10)

# 5. Aggregation.
for _ in range(2000):
    p = rng.uniform(1e-3, 1, size=rng.integers(1, 12))
    assert np.prod(p) <= p.min() + 1e-15 and p.min() <= p[-1]
    assert np.prod(np.append(p, rng.uniform(1e-3, 1))) <= np.prod(p)
assert round(0.95**10, 3) == 0.599 and 0.95**10 < 0.7


# 6. F1.
def f1(x, y):
    return 0.0 if x + y == 0 else 2 * x * y / (x + y)


assert f1(1.0, 0.0) == 0 and f1(0.0, 1.0) == 0
assert round(f1(0.9, 0.5), 3) == 0.643 and f1(0.9, 0.5) < 0.7
print("ok")
