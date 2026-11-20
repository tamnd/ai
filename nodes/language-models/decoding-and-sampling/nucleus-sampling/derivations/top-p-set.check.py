# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/nucleus-sampling/ (ai://NEW-nucleus-sampling).
# Check of derivations/top-p-set.md: the nucleus is the shortest sorted prefix with mass at least p, every smallest
# set with mass at least p has its size, the kept mass p' satisfies p <= p' < p + q_m, the total variation distance
# to the renormalized distribution is 1 - p', the two limits are greedy and pure sampling, and the nucleus grows with p.
import itertools

import numpy as np


def nucleus(probs, p):
    order = np.argsort(-probs, kind="stable")
    cum = np.cumsum(probs[order])
    m = int(np.searchsorted(cum, p - 1e-12)) + 1
    m = min(m, len(probs))
    return order[:m]


def tv(a, b):
    return 0.5 * np.abs(a - b).sum()


def renormalize(probs, keep):
    out = np.zeros_like(probs)
    out[keep] = probs[keep] / probs[keep].sum()
    return out


rng = np.random.default_rng(0)
for trial in range(400):
    n = int(rng.integers(2, 10))
    probs = rng.dirichlet(np.full(n, float(rng.choice([0.2, 1.0, 5.0]))))
    p = float(rng.uniform(0.05, 1.0))
    keep = nucleus(probs, p)
    m = len(keep)
    q = np.sort(probs)[::-1]
    sizes = [
        len(s)
        for r in range(1, n + 1)
        for s in itertools.combinations(range(n), r)
        if probs[list(s)].sum() >= p - 1e-12
    ]
    assert min(sizes) == m, (min(sizes), m)
    kept = probs[keep].sum()
    assert kept >= p - 1e-12 and kept < p + q[m - 1] + 1e-12
    assert abs(tv(probs, renormalize(probs, keep)) - (1 - kept)) < 1e-12
    assert tv(probs, renormalize(probs, keep)) <= 1 - p + 1e-12
    if q[0] >= p:
        assert m == 1
    grid = np.linspace(0.01, 1.0, 60)
    sets = [set(nucleus(probs, g).tolist()) for g in grid]
    for a, b in itertools.pairwise(sets):
        assert a <= b

# limit of pure sampling: p above 1 - q_s keeps every positive token
probs = np.array([0.4, 0.3, 0.2, 0.1, 0.0])
assert sorted(nucleus(probs, 0.95).tolist()) == [0, 1, 2, 3]

# worked example of the node
probs = np.array([0.5, 0.2, 0.15, 0.1, 0.05])
keep = nucleus(probs, 0.8)
assert sorted(keep.tolist()) == [0, 1, 2]
new = renormalize(probs, keep)
assert np.allclose(new, [10 / 17, 4 / 17, 3 / 17, 0, 0])
assert abs(tv(probs, new) - 0.15) < 1e-12
print("ok")
