# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/test-time-scaling/ (ai://NEW-test-time-scaling).
"""Checks every step of coverage-and-voting.md numerically."""

import itertools
import math

import numpy as np

rng = np.random.default_rng(0)

# 1. Coverage of one problem: geometric failure, concave increasing gains.
for p in (0.0, 0.05, 0.3, 0.9):
    c = [1 - (1 - p) ** k for k in range(13)]
    gains = np.diff(c)
    assert np.allclose(gains, [p * (1 - p) ** k for k in range(12)])
    if 0 < p < 1:
        assert (gains > 0).all() and (np.diff(gains) < 0).all()
    else:
        assert p != 0 or max(c) == 0
# Monte Carlo agreement.
p, k = 0.13, 7
hits = (rng.random((200_000, k)) < p).any(1).mean()
assert abs(hits - (1 - (1 - p) ** k)) < 4e-3


# 2. The unbiased estimator, exactly by summing over the binomial number of correct samples.
def est(n, c, k):
    return 1 - math.comb(n - c, k) / math.comb(n, k)


for n, p in ((10, 0.2), (20, 0.05), (7, 0.6)):
    for k in range(1, n + 1):
        mean = sum(math.comb(n, c) * p**c * (1 - p) ** (n - c) * est(n, c, k) for c in range(n + 1))
        assert abs(mean - (1 - (1 - p) ** k)) < 1e-12
        plug = sum(math.comb(n, c) * p**c * (1 - p) ** (n - c) * (1 - (1 - c / n) ** k) for c in range(n + 1))
        if k >= 2:
            assert plug < mean
# The estimator is the probability that a uniform k-subset contains a correct sample.
lab = [1, 0, 0, 1, 0, 0, 0]
for k in range(1, 8):
    subs = list(itertools.combinations(range(7), k))
    frac = sum(any(lab[i] for i in s) for s in subs) / len(subs)
    assert abs(frac - est(7, 2, k)) < 1e-12

# 3. Beta mixture: exact failure rate, power-law tail, exponentiated power law.
al, be = 0.5, 2.0


def fail(k):
    return math.exp(math.lgamma(al + be) + math.lgamma(be + k) - math.lgamma(be) - math.lgamma(al + be + k))


A = math.exp(math.lgamma(al + be) - math.lgamma(be))
assert abs(A - 1.329) < 1e-3
cov = {k: 1 - fail(k) for k in (1, 10, 100, 10_000)}
assert abs(cov[1] - 0.20) < 1e-9 and abs(cov[10] - 0.612) < 5e-4 and abs(cov[100] - 0.868) < 5e-4
assert abs(cov[10_000] - 0.9867) < 5e-5 and abs(A * 10_000**-al - 0.0133) < 5e-5
assert abs(fail(10_000) / (A * 10_000**-al) - 1) < 1e-3
# Monte Carlo of the mixture.
ps = rng.beta(al, be, size=400_000)
assert abs((1 - (1 - ps) ** 10).mean() - cov[10]) < 3e-3
# Fit log(-log c) against log k at large k: slope b close to -alpha, intercept close to log A.
ks = np.logspace(3, 6, 20)
y = np.log(-np.log(1 - np.array([fail(k) for k in ks])))
b, loga = np.polyfit(np.log(ks), y, 1)
assert abs(b + al) < 0.01 and abs(math.exp(loga) - A) < 0.06
# Fraction of problems below eps scales as eps^alpha.
f1, f2 = (ps < 1e-3).mean(), (ps < 1e-5).mean()
assert abs(math.log(f1 / f2) / math.log(100) - al) < 0.05
# Unsolvable problems cap coverage.
pi0 = 0.1
assert all(pi0 * 0 + (1 - pi0) * (1 - fail(k)) <= 1 - pi0 for k in (1, 100, 10**6))

# 4. Majority voting.
assert abs(math.exp(-100 * 0.1**2 / 2) - 0.6065) < 1e-4
assert abs(math.exp(-1000 * 0.1**2 / 2) - 0.0067) < 1e-4


def vote_acc(q, n, trials=20_000):
    ans = rng.choice(len(q), size=(trials, n), p=q)
    counts = np.stack([(ans == j).sum(1) for j in range(len(q))], 1)
    top = counts.max(1, keepdims=True)
    win = (counts[:, 0] == top[:, 0]) & ((counts == top).sum(1) == 1)
    return win.mean()


q = np.array([0.4, 0.3, 0.3])  # answer 0 is correct and the unique mode
accs = [vote_acc(q, n) for n in (11, 101, 1001)]
assert accs[0] < accs[1] < accs[2] and accs[2] > 0.99
for n, acc in zip((101, 1001), accs[1:]):
    assert 1 - acc <= 2 * math.exp(-n * 0.1**2 / 2) + 1e-2
q2 = np.array([0.35, 0.45, 0.2])  # the correct answer is not the mode
assert vote_acc(q2, 1001) < 0.01
# Benchmark limit: fraction of problems whose correct answer is the mode; coverage limit: fraction with q* > 0.
probs = [np.array(x) for x in ([0.6, 0.4], [0.3, 0.7], [0.0, 1.0], [0.5, 0.2, 0.3])]
vote_lim = np.mean([x[0] > x[1:].max() for x in probs])
cov_lim = np.mean([x[0] > 0 for x in probs])
assert vote_lim == 0.5 and cov_lim == 0.75


# 5. Best-of-n with a verifier that can be fooled.
def bon(p, eps, n):
    return (1 - eps) ** n - (1 - eps - p) ** n


p, eps = 0.3, 0.001
a, b = 1 - eps, 1 - eps - p
nstar = math.log(math.log(b) / math.log(a)) / math.log(a / b)
assert abs(nstar - 16.5) < 0.05
assert max(range(1, 5000), key=lambda n: bon(p, eps, n)) == 16
assert abs(bon(p, eps, 1) - 0.30) < 1e-9 and abs(bon(p, eps, 16) - 0.981) < 5e-4
assert abs(bon(p, eps, 100) - 0.905) < 5e-4 and abs(bon(p, eps, 1000) - 0.368) < 5e-4
# Simulation of the ranking model.
n, trials = 16, 100_000
u = rng.random((trials, n))
kind = np.where(u < eps, 2, np.where(u < eps + p, 1, 0))  # 2 fools, 1 correct, 0 plain wrong
score = kind.astype(float) * 10 * np.where(kind == 2, 2, 1) + rng.random((trials, n))
pick = kind[np.arange(trials), score.argmax(1)]
assert abs((pick == 1).mean() - bon(p, eps, n)) < 3e-3
# Rarer fooling answers move the peak later.
peaks = [math.log(math.log(1 - e - p) / math.log(1 - e)) / math.log((1 - e) / (1 - e - p)) for e in (1e-2, 1e-3, 1e-4)]
assert peaks[0] < peaks[1] < peaks[2]


# 6. Greedy allocation is optimal for separable concave coverage.
def total(ps, ks):
    return sum(1 - (1 - p) ** k for p, k in zip(ps, ks))


def greedy(ps, K):
    ks = [0] * len(ps)
    for _ in range(K):
        i = max(range(len(ps)), key=lambda j: ps[j] * (1 - ps[j]) ** ks[j])
        ks[i] += 1
    return ks


for _ in range(30):
    ps = list(rng.choice([0.0, 0.05, 0.2, 0.5, 0.9], size=4))
    K = int(rng.integers(1, 12))
    best = max(total(ps, ks) for ks in itertools.product(range(K + 1), repeat=4) if sum(ks) == K)
    assert abs(total(ps, greedy(ps, K)) - best) < 1e-12
assert abs(0.9 * 0.1 - 0.09) < 1e-12 and abs(0.9 * 0.01 - 0.009) < 1e-12
assert abs(0.05 * 0.95**9 - 0.0315) < 5e-5
ks = greedy([0.9, 0.05, 0.0], 30)
assert ks[2] == 0 and ks[1] > ks[0]

# 7. FLOPs matching.
for M, ratio, want in ((14, 1, 53), (14, 0.1, 17.9), (14, 10, 404)):  # ratio = D_pretrain / D_inference
    m = M + 3 * ratio * (M - 1)
    assert abs(m - want) < 1e-9
    N, Dinf = 1.0, 7.0
    Dpre = ratio * Dinf
    assert abs(6 * M * N * Dpre + 2 * M * N * Dinf - (6 * N * Dpre + 2 * N * m * Dinf)) < 1e-9

# 8. Train-test slope.
assert abs(10**1.2 - 15.85) < 0.01
for base in (10, math.e, 2):
    train = 1e15
    t1 = -1.2 * math.log(train, base)
    t2 = -1.2 * math.log(10 * train, base)
    assert abs(base ** (t1 - t2) - 10**1.2) < 1e-6

print("ok")
