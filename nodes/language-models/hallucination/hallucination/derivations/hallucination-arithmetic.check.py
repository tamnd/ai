# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/hallucination/ (ai://NEW-hallucination).
# Check of derivations/hallucination-arithmetic.md: the expected monofact fraction exceeds the expected missing
# mass by sum p^2 (1 - p)^(n - 1), between 0 and 1/n; a generator that spreads the monofact mass over unseen true
# facts and false factoids hallucinates at MF M / (M + U), about 36.7% in the uniform example; and under a
# confidence target t answering pays (q - t) / (1 - t), so binary grading (t = 0) never favours abstaining.
import math
import random
from collections import Counter

rng = random.Random(7)


def gap_terms(p, n):
    mf = sum(pi * (1 - pi) ** (n - 1) for pi in p)
    miss = sum(pi * (1 - pi) ** n for pi in p)
    gap = sum(pi * pi * (1 - pi) ** (n - 1) for pi in p)
    return mf, miss, gap


# item 1: the gap identity and its bounds, exactly, on random distributions
for _ in range(400):
    k = rng.randint(1, 60)
    w = [rng.random() ** rng.choice([1, 3, 8]) for _ in range(k)]
    s = sum(w)
    p = [x / s for x in w]
    n = rng.randint(1, 400)
    mf, miss, gap = gap_terms(p, n)
    assert math.isclose(mf - miss, gap, rel_tol=1e-9, abs_tol=1e-15)
    assert -1e-15 <= gap <= 1 / n + 1e-15
    # the bound comes from x (1 - x)^(n - 1) <= 1/n, reached near x = 1/n
    xs = [i / 2000 for i in range(2001)]
    assert max(x * (1 - x) ** (n - 1) for x in xs) <= 1 / n + 1e-12

# item 1 by simulation: monofact fraction and missing mass for a Zipf law
N, n = 2000, 2000
n_zipf = n
z = [1 / i for i in range(1, N + 1)]
h = sum(z)
pz = [x / h for x in z]
mf_e, miss_e, gap_e = gap_terms(pz, n)
mfs, misses = [], []
for _ in range(300):
    c = Counter(rng.choices(range(N), weights=pz, k=n))
    mfs.append(sum(1 for v in c.values() if v == 1) / n)
    misses.append(sum(pz[i] for i in range(N) if i not in c))
assert abs(sum(mfs) / len(mfs) - mf_e) < 0.004
assert abs(sum(misses) / len(misses) - miss_e) < 0.004
assert 0 <= gap_e <= 1 / n

# item 2: the uniform example, N equally likely true facts, n = N draws, M = 100 N false factoids
N = 100_000
n = N
M = 100 * N
miss = (1 - 1 / N) ** n
mf = (1 - 1 / N) ** (n - 1)
assert abs(miss - 1 / math.e) < 1e-5 and math.isclose(mf, 0.36788, abs_tol=1e-5)
U = N * miss
rate = mf * M / (M + U)
assert round(rate, 4) == 0.3665, rate
gap_u = N * (1 / N) ** 2 * (1 - 1 / N) ** (n - 1)
assert math.isclose(gap_u, mf / N) and gap_u <= 1 / n
# simulate the generator: draw the training facts, then its hallucination rate is MF M / (M + U)
rates = []
for _ in range(5):
    c = Counter(rng.randrange(N) for _ in range(n))
    mf_s = sum(1 for v in c.values() if v == 1) / n
    u_s = N - len(c)
    rates.append(mf_s * M / (M + u_s))
    # sample generations: observed facts at observed frequency with mass 1 - MF, the rest spread evenly
    false = 0
    draws = 20_000
    for _ in range(draws):
        if rng.random() < mf_s:
            false += rng.random() < M / (M + u_s)
    assert abs(false / draws - rates[-1]) < 0.012
assert all(abs(r - rate) < 0.006 for r in rates), rates
# with few false factoids the rate falls well below MF
assert mf * 1000 / (1000 + U) < 0.01

# item 3: confidence targets
for t in [0.0, 0.25, 0.5, 2 / 3, 0.75, 0.9]:
    pen = t / (1 - t)
    for i in range(101):
        qv = i / 100
        ev = qv * 1 - (1 - qv) * pen
        assert math.isclose(ev, (qv - t) / (1 - t), abs_tol=1e-12)
        if qv > t + 1e-12:
            assert ev > 0
        if qv < t - 1e-12:
            assert ev < 0
    if t == 0.0:
        assert all(i / 100 >= 0 for i in range(101))
assert [round(t / (1 - t), 9) for t in (0.5, 0.75, 0.9, 2 / 3)] == [1.0, 3.0, 9.0, 2.0]

# item 4: Model B against Model A under binary grading
for _ in range(500):
    qs = [rng.random() for _ in range(rng.randint(1, 50))]
    a_score, b_score = 0.0, sum(qs)
    a_err, b_err = 0.0, sum(1 - x for x in qs)
    assert b_score - a_score == sum(qs) and b_score >= a_score
    assert math.isclose(b_err - a_err, len(qs) - sum(qs))
    # a simulated grading run agrees with the expectation
    trials = 400
    tot = sum(sum(rng.random() < x for x in qs) for _ in range(trials)) / trials
    assert abs(tot - sum(qs)) < 4 * math.sqrt(len(qs) / 4 / trials) + 1e-9

print(
    f"Zipf N = n = 2000: MF {mf_e:.4f}, missing mass {miss_e:.4f}, gap {gap_e:.2e} <= 1/n = {1 / n_zipf:.2e}"
)
print(
    f"uniform N = n = 100000, M = 100 N: missing mass {miss:.5f}, hallucination rate {rate:.4f}"
)
print("hallucination-arithmetic: ok")
