# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-probabilistic-language-model/ (ai://NEW-neural-probabilistic-language-model).
"""Check of nce-gradient: the gradient of the exact noise-contrastive objective for one context, its limit P_d - P as
the number of noise samples k grows, the 1/k rate, and the stationary point at P = P_d for every k."""

import itertools
import math
import random


def log_sigmoid(z):
    return -math.log1p(math.exp(-z)) if z > 0 else z - math.log1p(math.exp(z))


def objective(s, pd, pn, k):
    total = 0.0
    for w in range(len(s)):
        delta = s[w] - math.log(k * pn[w])
        total += pd[w] * log_sigmoid(delta) + k * pn[w] * log_sigmoid(-delta)
    return total


def closed_form(s, pd, pn, k):
    return [k * pn[w] / (math.exp(s[w]) + k * pn[w]) * (pd[w] - math.exp(s[w])) for w in range(len(s))]


def simplex(rng, n):
    x = [rng.expovariate(1.0) for _ in range(n)]
    return [v / sum(x) for v in x]


def main():
    rng = random.Random(12)
    worst = 0.0
    for _ in range(50):
        V = rng.randint(2, 12)
        pd, pn = simplex(rng, V), simplex(rng, V)
        s = [rng.gauss(math.log(1 / V), 1) for _ in range(V)]
        k = rng.choice([1, 2, 5, 25, 100])
        g = closed_form(s, pd, pn, k)
        for w in range(V):
            hi, lo = list(s), list(s)
            hi[w] += 1e-6
            lo[w] -= 1e-6
            num = (objective(hi, pd, pn, k) - objective(lo, pd, pn, k)) / 2e-6
            worst = max(worst, abs(num - g[w]))
        # The data distribution is stationary for every k.
        assert max(abs(v) for v in closed_form([math.log(p) for p in pd], pd, pn, k)) < 1e-15
    assert worst < 1e-7
    V = 8
    pd, pn = simplex(rng, V), simplex(rng, V)
    s = [rng.gauss(-2, 1) for _ in range(V)]
    limit = [pd[w] - math.exp(s[w]) for w in range(V)]
    gaps = []
    for k in (1, 10, 100, 1000, 10_000):
        g = closed_form(s, pd, pn, k)
        gap = max(abs(a - b) for a, b in zip(g, limit, strict=True))
        bound = max(math.exp(s[w]) / (k * pn[w]) * abs(limit[w]) for w in range(V))
        assert gap <= bound + 1e-15
        gaps.append(gap)
    assert all(b < a for a, b in itertools.pairwise(gaps))
    assert 5 < gaps[3] / gaps[4] < 20
    print(
        f"nce-gradient: finite-difference error {worst:.1e}; gap to the limit for k = 1 .. 10^4: "
        + ", ".join(f"{g:.2e}" for g in gaps)
    )


if __name__ == "__main__":
    main()
