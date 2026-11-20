# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/instruction-tuning/ (ai://NEW-instruction-tuning).
"""Checks the arithmetic of instruction tuning: capped mixing rates and passes, the worked example, the jump in
T0's size rule, the chain-rule identity behind the response-only loss, its gradient, token versus sequence
averaging, and the compute figures of Flan-PaLM 540B."""

import itertools

import numpy as np

rng = np.random.default_rng(0)


def rates(sizes, cap=None):
    e = np.asarray(sizes, dtype=float)
    w = e if cap is None else np.minimum(e, cap)
    return w / w.sum()


# 1 and 2: rates and passes on random sizes.
for _ in range(300):
    m = int(rng.integers(2, 12))
    e = rng.integers(1, 200_000, size=m).astype(float)
    cap = float(rng.integers(100, 50_000))
    r = rates(e, cap)
    assert abs(r.sum() - 1) < 1e-12
    big = e >= cap
    if big.sum() > 1:
        assert np.allclose(r[big], r[big][0])
    z = np.minimum(e, cap).sum()
    s = float(rng.integers(1_000, 1_000_000))
    passes = s * r / e
    assert np.allclose(passes[e <= cap], s / z)
    assert np.allclose(passes[e > cap], s * cap / (e[e > cap] * z))
    assert np.all(passes[e > cap] < s / z + 1e-12)

# Simulated draws agree with the expected counts.
e = np.array([1000.0, 3000.0, 100000.0])
r = rates(e, 3000)
draws = rng.choice(3, size=200_000, p=r)
freq = np.bincount(draws, minlength=3) / draws.size
assert np.all(np.abs(freq - r) < 4 * np.sqrt(r * (1 - r) / draws.size))

# 3: the worked example.
assert np.allclose(r, [1 / 7, 3 / 7, 3 / 7])
r0 = rates(e)
assert np.allclose(r0, [1000 / 104000, 3000 / 104000, 100000 / 104000])
assert np.allclose(np.round(r0, 4), [0.0096, 0.0288, 0.9615])
assert np.allclose(70_000 * r / e, [10, 10, 0.3])


# 4: T0's rule is not monotone.
def t0_size(n, templates):
    return n if n <= 500_000 else 500_000 / templates


assert t0_size(500_000, 10) == 500_000
assert t0_size(500_001, 10) == 50_000
assert t0_size(500_001, 1) > t0_size(500_000, 1) - 1

# 5: the chain rule on a random bigram model, by enumeration.
V, P, R = 4, 2, 3
start = rng.dirichlet(np.ones(V))
trans = rng.dirichlet(np.ones(V), size=V)


def token_nll(seq):
    out = [-np.log(start[seq[0]])]
    for a, b in itertools.pairwise(seq):
        out.append(-np.log(trans[a, b]))
    return np.array(out)


for x in itertools.product(range(V), repeat=P):
    px = np.exp(-token_nll(list(x)).sum())
    total = 0.0
    for y in itertools.product(range(V), repeat=R):
        nll = token_nll(list(x) + list(y))
        pxy = np.exp(-nll.sum())
        p_y_given_x = pxy / px
        total += p_y_given_x
        assert abs(nll[P:].sum() + np.log(p_y_given_x)) < 1e-10
        assert abs(nll.sum() - (-np.log(px) - np.log(p_y_given_x))) < 1e-10
    assert abs(total - 1) < 1e-10


# 6: gradient of the masked mean loss.
def masked_loss(z, y, m):
    zs = z - z.max(axis=-1, keepdims=True)
    logp = zs - np.log(np.exp(zs).sum(axis=-1, keepdims=True))
    nll = -logp[np.arange(len(y)), y]
    return (m * nll).sum() / m.sum()


for _ in range(20):
    T = int(rng.integers(3, 9))
    z = rng.normal(size=(T, V))
    y = rng.integers(0, V, size=T)
    m = (rng.random(T) < 0.6).astype(float)
    m[-1] = 1.0
    p = np.exp(z - z.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    g = m[:, None] * (p - np.eye(V)[y]) / m.sum()
    num = np.zeros_like(z)
    h = 1e-6
    for i in range(T):
        for j in range(V):
            zp, zm = z.copy(), z.copy()
            zp[i, j] += h
            zm[i, j] -= h
            num[i, j] = (masked_loss(zp, y, m) - masked_loss(zm, y, m)) / (2 * h)
    assert np.allclose(g, num, atol=1e-8)
    assert np.all(g[m == 0] == 0)

# 7: token versus sequence averaging.
lens = np.array([10, 90])
losses = [rng.random(n) for n in lens]
token_mean = np.concatenate(losses).sum() / lens.sum()
seq_means = np.array([x.mean() for x in losses])
assert abs(token_mean - (lens / lens.sum()) @ seq_means) < 1e-12
assert np.allclose(lens / lens.sum(), [0.1, 0.9])

# 8: compute.
pre = 6 * 540e9 * 780e9
fin = 6 * 540e9 * 1.4e9
assert abs(pre - 2.5272e24) < 1e20 and round(pre / 1e24, 1) == 2.5
assert abs(fin - 4.536e21) < 1e17
assert round(fin / 5.6e21, 2) == 0.81
assert round(1.4 / 780 * 100, 2) == 0.18
assert round(5.6e21 / 2.5e24 * 100, 2) == 0.22

# 9: the worked loss.
seq = [5, 6, 7] + [8, 9] + [1]
targets, mask = seq[1:], np.array([0, 0, 1, 1, 1])
assert targets == [6, 7, 8, 9, 1]
p_t = np.array([0.01, 0.02, 0.5, 0.25, 0.8])
nll = -np.log(p_t)
assert abs((mask * nll).sum() - np.log(10)) < 1e-12
assert round((mask * nll).sum(), 6) == 2.302585 and round((mask * nll).sum() / 3, 6) == 0.767528
assert round(nll.sum(), 6) == 10.819778 and round(nll.mean(), 6) == 2.163956

print("instruction-tuning-arithmetic: ok")
