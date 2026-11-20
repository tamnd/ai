# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/lost-in-the-middle/ (ai://NEW-lost-in-the-middle).
# Check of derivations/position-bias-models.md: under an additive model of document attention, subtracting the
# attention of a dummy document at the same position removes the position term and keeps every relevance gap
# above 4 delta in order; attention sorting under a bias that never falls with position never moves the most
# relevant document earlier, puts it last in one sort when its relevance margin beats the bias, and can stall;
# dividing RoPE positions by r turns the score at distance D into the unscaled score at distance D / r.
import itertools

import numpy as np

rng = np.random.default_rng(20240214)

# item 1: calibration under Attn(d, k) = rel(d) + bias(k) + eps(d, k), |eps| <= delta
for _ in range(2000):
    n_docs = int(rng.integers(2, 30))
    rel = rng.normal(size=n_docs)
    rel_dummy = float(rng.normal())
    bias = rng.normal(size=n_docs) * float(rng.uniform(0, 5))
    delta = float(rng.uniform(0, 0.2))

    def attn(r, k, delta=delta, bias=bias):
        return r + bias[k] + rng.uniform(-delta, delta)

    order = rng.permutation(n_docs)  # document order[k] sits at position k
    calibrated = np.empty(n_docs)
    for k, d in enumerate(order):
        calibrated[d] = attn(rel[d], k) - attn(rel_dummy, k)
    assert np.all(np.abs(calibrated - (rel - rel_dummy)) <= 2 * delta + 1e-12)
    for a, b in itertools.combinations(range(n_docs), 2):
        if rel[a] - rel[b] > 4 * delta:
            assert calibrated[a] > calibrated[b]
        if rel[b] - rel[a] > 4 * delta:
            assert calibrated[b] > calibrated[a]
    if delta == 0.0:
        assert np.allclose(calibrated, rel - rel_dummy)

# with no noise the calibrated value does not depend on the position at all
rel = rng.normal(size=10)
bias = np.abs(np.linspace(-1, 1, 10)) * 3.0  # a U-shaped bias
for d in range(10):
    vals = [(rel[d] + bias[k]) - (0.3 + bias[k]) for k in range(10)]
    assert np.ptp(vals) < 1e-12
# without calibration a less relevant document at the start beats the gold one in the middle
gold, distractor = 1.0, 0.5
assert gold + bias[5] < distractor + bias[0]


# item 2: attention sorting with a = f(r, k), f strictly increasing in r and non-decreasing in k
def sort_once(rels, f):
    att = [f(r, k) for k, r in enumerate(rels)]
    idx = sorted(
        range(len(rels)), key=lambda i: att[i]
    )  # stable, highest attention last
    return [rels[i] for i in idx]


def make_f(kind, beta, gamma):
    if kind == "additive":
        return lambda r, k: r + beta * k
    if kind == "multiplicative":
        return lambda r, k: (1.0 + r) * (1.0 + beta * k)
    return lambda r, k: r + beta * np.log1p(gamma * k)


for _ in range(3000):
    n_docs = int(rng.integers(2, 25))
    kind = ["additive", "multiplicative", "log"][int(rng.integers(3))]
    beta, gamma = float(rng.uniform(0, 0.3)), float(rng.uniform(0, 5))
    f = make_f(kind, beta, gamma)
    rels = list(rng.uniform(0, 1, size=n_docs - 1))
    gold_rel = 1.0 + float(rng.uniform(1e-6, 0.5))
    p = int(rng.integers(n_docs))
    rels.insert(p, gold_rel)
    before = set(rels[:p])
    pos = p
    one_step = all(f(gold_rel, p) > f(r, n_docs - 1) for r in rels if r != gold_rel)
    for step in range(n_docs + 2):
        new = sort_once(rels, f)
        new_pos = new.index(gold_rel)
        assert new_pos >= pos  # the gold document never moves earlier
        if step == 0:
            assert before <= set(new[:new_pos])  # all that preceded it still precede it
            if one_step:
                assert new_pos == n_docs - 1
        rels, pos = new, new_pos

# the sufficient condition for one sort, additive case: margin > beta (K - 1 - p)
f = make_f("additive", 0.1, 0.0)
rels = [1.0] + [0.5] * 4  # gold first, margin 0.5 > 0.1 * 4
assert sort_once(rels, f).index(1.0) == 4

# a stall: gold first with relevance 1.0, two distractors 0.9, bias 0.2 per position
f = make_f("additive", 0.2, 0.0)
rels = [1.0, 0.9, 0.9]
att = [f(r, k) for k, r in enumerate(rels)]
assert np.allclose(att, [1.0, 1.1, 1.3])
for _ in range(10):
    rels = sort_once(rels, f)
    assert rels.index(1.0) == 0


# item 3: RoPE with positions divided by r
def rope(x, pos, base=10000.0):
    d = x.shape[-1]
    theta = base ** (-np.arange(0, d, 2) / d)
    ang = pos * theta
    x1, x2 = x[0::2], x[1::2]
    out = np.empty_like(x)
    out[0::2] = x1 * np.cos(ang) - x2 * np.sin(ang)
    out[1::2] = x1 * np.sin(ang) + x2 * np.cos(ang)
    return out


for _ in range(500):
    d = 2 * int(rng.integers(1, 64))
    q, k = rng.normal(size=d), rng.normal(size=d)
    m, n = rng.uniform(0, 5000, size=2)
    r = float(rng.uniform(0.25, 4.0))
    scaled = rope(q, m / r) @ rope(k, n / r)
    shifted = rope(q, (m - n) / r) @ rope(k, 0.0)
    assert abs(scaled - shifted) < 1e-8 * (1 + abs(scaled))
    assert abs(rope(q, m) @ rope(k, n) - rope(q, m - n) @ rope(k, 0.0)) < 1e-8 * (
        1 + abs(scaled)
    )

q, k = rng.normal(size=128), rng.normal(size=128)
assert abs(rope(q, 4000 / 2) @ rope(k, 0.0) - rope(q, 2000.0) @ rope(k, 0.0)) < 1e-9
print("ratio 2: the score at distance 4000 equals the unscaled score at distance 2000")
print("position-bias-models: ok")
