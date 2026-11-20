# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/nucleus-sampling/ (ai://NEW-nucleus-sampling).
# Check of derivations/temperature.md: softmax(u / t) keeps the order of the logits, tends to the uniform distribution
# on the argmax as t -> 0 and on the vocabulary as t -> infinity, has entropy derivative Var(u) / t^3, and gives the
# top-m tokens a mass that falls with t, so the nucleus for a fixed p grows with t.
import itertools

import numpy as np


def tempered(u, t):
    z = u / t
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def entropy(pi):
    pi = pi[pi > 0]
    return float(-(pi * np.log(pi)).sum())


def nucleus_size(pi, order, p):
    cum = np.cumsum(pi[order])
    return min(int(np.searchsorted(cum, p - 1e-12)) + 1, len(pi))


rng = np.random.default_rng(1)
temps = np.geomspace(0.05, 20.0, 80)
for trial in range(200):
    n = int(rng.integers(2, 12))
    u = rng.normal(0.0, float(rng.choice([0.5, 2.0, 5.0])), n)
    if trial % 4 == 0:
        u[1] = u[0]
    order = np.lexsort((np.arange(n), -u))
    for t in (0.1, 0.7, 1.0, 3.0):
        pi = tempered(u, t)
        for i in range(n):
            for j in range(n):
                if u[i] > u[j]:
                    assert pi[i] > pi[j]
    a = np.isclose(u, u.max())
    gap = float(u.max() - u[~a].max()) if (~a).any() else 1.0
    assert np.allclose(tempered(u, gap / 50), a / a.sum(), atol=1e-9)
    assert np.allclose(tempered(u, 1e4), np.full(n, 1.0 / n), atol=1e-3)
    for t in (0.3, 1.0, 2.5):
        pi = tempered(u, t)
        var = float(pi @ u**2 - (pi @ u) ** 2)
        h = 1e-5
        num = (entropy(tempered(u, t + h)) - entropy(tempered(u, t - h))) / (2 * h)
        assert abs(num - var / t**3) < 1e-6 * max(1.0, abs(num)), (num, var / t**3)
    masses = np.array([np.cumsum(tempered(u, t)[order]) for t in temps])
    assert np.all(np.diff(masses, axis=0) <= 1e-12)
    for p in (0.5, 0.8, 0.95):
        sizes = [nucleus_size(tempered(u, t), order, p) for t in temps]
        assert all(a <= b for a, b in itertools.pairwise(sizes))

# worked example of the node
u = np.log(np.array([0.5, 0.2, 0.15, 0.1, 0.05]))
order = np.arange(5)
pi = tempered(u, 0.5)
assert np.allclose(pi, [0.769, 0.123, 0.069, 0.031, 0.008], atol=6e-4)
assert nucleus_size(tempered(u, 1.0), order, 0.8) == 3
assert nucleus_size(pi, order, 0.8) == 2
print("ok")
