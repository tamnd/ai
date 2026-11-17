# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/speculative-decoding/ (ai://JKWQF4).
# Check of derivations/speculative-sampling.md: accepting x ~ q with probability min(1, p(x)/q(x)) and otherwise
# resampling from norm(max(0, p - q)) outputs exactly p, the acceptance probability is sum min(p, q) = 1 - TV(p, q),
# it is 1 iff p = q and 0 iff the supports are disjoint, and a one-hot p always outputs its token.
import itertools
from fractions import Fraction as F

import numpy as np


def simplex(v, d):
    for c in itertools.product(range(d + 1), repeat=v):
        if sum(c) == d:
            yield tuple(F(k, d) for k in c)


def law(p, q):
    acc = [qi * min(F(1), pi / qi) if qi > 0 else F(0) for pi, qi in zip(p, q)]
    beta = sum(acc)
    pos = [max(F(0), pi - qi) for pi, qi in zip(p, q)]
    if beta == 1:
        return acc, beta
    z = sum(pos)
    return [a + (1 - beta) * r / z for a, r in zip(acc, pos)], beta


pairs = 0
for v in (2, 3):
    for d in (4, 6):
        dists = list(simplex(v, d))
        for p in dists:
            for q in dists:
                out, beta = law(p, q)
                assert out == list(p)
                assert beta == sum(min(a, b) for a, b in zip(p, q))
                assert beta == 1 - sum(abs(a - b) for a, b in zip(p, q)) / 2
                assert (beta == 1) == (p == q)
                assert (beta == 0) == all(a == 0 or b == 0 for a, b in zip(p, q))
                pairs += 1
assert pairs > 1000

# The worked example of the page.
out, beta = law((F(1, 2), F(1, 3), F(1, 6)), (F(1, 4), F(1, 2), F(1, 4)))
assert out == [F(1, 2), F(1, 3), F(1, 6)] and beta == F(3, 4)


def run(p, q, n, rng):
    x = rng.choice(len(q), size=n, p=q)
    u = rng.random(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.minimum(1.0, p[x] / q[x])
    keep = u < a
    r = np.maximum(p - q, 0.0)
    y = x.copy()
    if (~keep).any():
        y[~keep] = rng.choice(len(p), size=int((~keep).sum()), p=r / r.sum())
    return y, keep


rng = np.random.default_rng(0)
n = 200_000
cases = [rng.dirichlet(np.ones(5), size=2) for _ in range(5)]
cases.append(np.array([[0.5, 0.5, 0, 0, 0], [0, 0, 0.2, 0.3, 0.5]]))
cases.append(np.array([[0, 0, 1.0, 0, 0], [0.2, 0.2, 0.2, 0.2, 0.2]]))
for p, q in cases:
    y, keep = run(p, q, n, rng)
    freq = np.bincount(y, minlength=len(p)) / n
    assert np.abs(freq - p).max() < 5 * np.sqrt(0.25 / n), (freq, p)
    assert abs(keep.mean() - np.minimum(p, q).sum()) < 5 * np.sqrt(0.25 / n)
    assert np.all(freq[p == 0] == 0)
greedy_y, greedy_keep = run(cases[-1][0], cases[-1][1], 1000, rng)
assert np.all(greedy_y == 2)
print("speculative-sampling: ok,", pairs, "exact pairs")
