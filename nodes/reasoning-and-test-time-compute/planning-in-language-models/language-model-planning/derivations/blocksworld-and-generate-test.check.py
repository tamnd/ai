# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/language-model-planning/ (ai://NEW-language-model-planning).
"""Checks every step of blocksworld-and-generate-test.md numerically."""

import math
from collections import deque

import numpy as np


def splits(items):
    """Every way to arrange items into a set of towers, each a bottom-to-top tuple."""
    if not items:
        yield []
        return
    first = items[0]
    for rest in splits(items[1:]):
        for i, tower in enumerate(rest):
            for j in range(len(tower) + 1):
                yield rest[:i] + [tower[:j] + (first,) + tower[j:]] + rest[i + 1 :]
        yield rest + [(first,)]


def all_states(n):
    out = set()
    for held in [None, *range(n)]:
        for towers in splits([b for b in range(n) if b != held]):
            out.add((frozenset(towers), held))
    return out


def successors(state):
    towers, held = state
    towers = list(towers)
    if held is None:
        for i, t in enumerate(towers):
            rest = towers[:i] + towers[i + 1 :] + ([t[:-1]] if len(t) > 1 else [])
            yield frozenset(rest), t[-1]
    else:
        yield frozenset([*towers, (held,)]), None
        for i, t in enumerate(towers):
            yield frozenset(towers[:i] + towers[i + 1 :] + [(*t, held)]), None


def distances(start):
    d = {start: 0}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in successors(u):
            if v not in d:
                d[v] = d[u] + 1
                q.append(v)
    return d


# 1. State counts: the Lah sum, the recurrence and enumeration agree.
def lah_sum(n):
    return sum(math.comb(n - 1, k - 1) * math.factorial(n) // math.factorial(k) for k in range(1, n + 1))


a = [1, 1]
for n in range(2, 21):
    a.append((2 * n - 1) * a[n - 1] - (n - 1) * (n - 2) * a[n - 2])
assert a[3:6] == [13, 73, 501]
assert all(a[n] == lah_sum(n) for n in range(1, 21))
assert math.isclose(a[20], 3.277e20, rel_tol=1e-3)
for n, empty, total in ((3, 13, 22), (4, 73, 125), (5, 501, 866)):
    S = all_states(n)
    hand_empty = [s for s in S if s[1] is None]
    assert len(hand_empty) == empty == a[n]
    assert len(S) == total == a[n] + n * a[n - 1]

# 2. Plan lengths: the 2u + 2g bound holds for every pair and the longest optimal plan is 4(n - 1).
for n, longest in ((3, 8), (4, 12), (5, 16)):
    hand_empty = [s for s in all_states(n) if s[1] is None]
    worst = 0
    for s in hand_empty:
        d = distances(s)
        assert len(d) == len(all_states(n))  # connected
        u = sum(len(t) - 1 for t in s[0])
        for g_state in hand_empty:
            g = sum(len(t) - 1 for t in g_state[0])
            assert d[g_state] <= 2 * u + 2 * g
            worst = max(worst, d[g_state])
    assert worst == longest == 4 * (n - 1)

# Table-to-stack needs exactly 2(n - 1) actions.
for n in (2, 3, 4, 5):
    start = (frozenset((b,) for b in range(n)), None)
    goal = (frozenset([tuple(range(n))]), None)
    assert distances(start)[goal] == 2 * (n - 1)
assert 2 * (4 - 1) == 6 and 2 * (20 - 1) == 38

# 3. Sound verifier: 1 - (1 - p)^K, and a simulation of independent attempts agrees.
assert math.isclose(1 - 0.6**15, 0.99953, abs_tol=5e-6)
rng = np.random.default_rng(0)
trials = 200_000
hits = (rng.random((trials, 15)) < 0.4).any(axis=1).mean()
assert abs(hits - (1 - 0.6**15)) < 1e-3

# 4. Unsound verifier: precision and the loop, checked by simulation.
p, t, f = 0.4, 0.982, 0.8445
acc_good, acc_bad = p * t, (1 - p) * f
prec = acc_good / (acc_good + acc_bad)
r = 1 - acc_good - acc_bad
assert math.isclose(acc_good, 0.3928, abs_tol=1e-9) and math.isclose(acc_bad, 0.5067, abs_tol=1e-9)
assert math.isclose(prec, 0.4367, abs_tol=5e-5) and math.isclose(r, 0.1005, abs_tol=1e-9)
for K in (2, 15):
    assert abs(prec * (1 - r**K) - 0.4367) < 5e-3
good = bad = 0
for _ in range(100_000):
    for _k in range(15):
        ok = rng.random() < p
        if rng.random() < (t if ok else f):
            good += ok
            bad += not ok
            break
assert abs(good / 100_000 - prec * (1 - r**15)) < 5e-3

# 5. Time ratios.
assert math.isclose(40.43 / 0.265, 152.6, abs_tol=0.05)
assert math.isclose(83.37 / 0.265, 314.6, abs_tol=0.05)
print("ok")
