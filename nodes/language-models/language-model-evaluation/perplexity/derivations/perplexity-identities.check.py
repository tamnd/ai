# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/perplexity/ (ai://NEW-perplexity).
# Check of derivations/perplexity-identities.md: the geometric-mean form, base invariance, the uniform model, the
# bounds by the harmonic and arithmetic means, the unit-weighted corpus value, and the expected log-loss identity.
import numpy as np

rng = np.random.default_rng(20261121)
for _ in range(2000):
    t = int(rng.integers(1, 60))
    p = rng.uniform(1e-6, 1.0, size=t)
    ppl = float(np.exp(-np.log(p).mean()))
    assert np.isclose(ppl, float(np.prod((1 / p) ** (1 / t))), rtol=1e-9)  # item 1
    for b in (2.0, 10.0, np.e):
        h = -np.log(p).mean() / np.log(b)
        assert np.isclose(b**h, ppl, rtol=1e-9)  # item 2
    assert 1 <= 1 / p.max() <= ppl * (1 + 1e-12) and ppl <= (1 / p.min()) * (
        1 + 1e-12
    )  # item 4
    assert 1 / p.mean() <= ppl * (1 + 1e-12) <= (1 / p).mean() * (1 + 1e-9)

for v in (2, 7, 50257, 10**6):
    assert np.isclose(np.exp(-np.log(np.full(5, 1 / v)).mean()), v, rtol=1e-9)  # item 3

for _ in range(1000):
    m = int(rng.integers(1, 10))
    s = rng.uniform(0, 50, size=m)
    n = rng.integers(1, 40, size=m).astype(float)
    per = np.exp(s / n)
    corpus = np.exp(s.sum() / n.sum())
    assert np.isclose(
        np.log(corpus), (n / n.sum() * np.log(per)).sum(), rtol=1e-9, atol=1e-12
    )  # item 5
    assert per.min() * (1 - 1e-12) <= corpus <= per.max() * (1 + 1e-12)
    eq = np.exp(s.sum() / (m * 5.0))
    per_eq = np.exp(s / 5.0)
    assert np.isclose(
        eq, np.exp(np.log(per_eq).mean()), rtol=1e-9
    ) and eq <= per_eq.mean() * (1 + 1e-12)

for _ in range(1000):
    k = int(rng.integers(2, 30))
    P, M = rng.dirichlet(np.ones(k)), rng.dirichlet(np.ones(k))
    H = -(P * np.log(P)).sum()
    kl = (P * np.log(P / M)).sum()
    assert (
        np.isclose(-(P * np.log(M)).sum(), H + kl, rtol=1e-9, atol=1e-12)
        and kl >= -1e-12
    )  # item 6
    assert np.isclose(-(P * np.log(P)).sum(), H)
    # sampled: the empirical log-loss of P on draws from P is below that of M, on average
    x = rng.choice(k, size=4000, p=P)
    assert np.mean(-np.log(P[x])) - np.mean(-np.log(M[x])) < 0.2

print("perplexity-identities: ok")
