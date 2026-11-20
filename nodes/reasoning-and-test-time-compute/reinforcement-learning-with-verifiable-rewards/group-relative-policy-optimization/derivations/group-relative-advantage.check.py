# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/group-relative-policy-optimization/ (ai://NEW-group-relative-policy-optimization).
"""Checks every step of group-relative-advantage.md numerically."""

import math

import numpy as np

rng = np.random.default_rng(0)


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


# 1. A fixed baseline does not change the expected gradient (exact, by enumeration).
K = 5
theta = rng.normal(size=K)
pi = softmax(theta)
reward = rng.normal(size=K)
grad_logp = np.eye(K) - pi  # row o: gradient of log pi(o) with respect to theta
exact = (pi[:, None] * reward[:, None] * grad_logp).sum(0)
for b in (-3.0, 0.0, 0.7, 10.0):
    with_b = (pi[:, None] * (reward - b)[:, None] * grad_logp).sum(0)
    assert np.allclose(with_b, exact)
    assert np.allclose((pi[:, None] * b * grad_logp).sum(0), 0)

# 2. Centred group advantage = (G - 1)/G times the leave-one-out advantage; unbiased up to that constant.
for G in (2, 4, 16):
    r = rng.normal(size=G)
    loo = np.array([r[i] - np.delete(r, i).mean() for i in range(G)])
    assert np.allclose(r - r.mean(), (G - 1) / G * loo)
G, trials = 4, 200000
o = np.stack([rng.choice(K, p=pi, size=trials) for _ in range(G)], 1)
r = reward[o]
adv = r - r.mean(1, keepdims=True)
est = (adv[..., None] * grad_logp[o]).sum(1).mean(0)
assert np.allclose(est, (G - 1) / G * G * exact, atol=0.02), (est, exact)

# 3. Advantages sum to zero; binary closed forms.
for G in (4, 8, 16):
    for c in range(1, G):
        r = np.array([1.0] * c + [0.0] * (G - c))
        a = (r - r.mean()) / r.std()
        p = c / G
        assert abs(a.sum()) < 1e-12
        assert np.allclose(a[:c], math.sqrt((1 - p) / p))
        assert np.allclose(a[c:], -math.sqrt(p / (1 - p)))
r = np.ones(8)
assert np.all((r - r.mean()) / (r.std() + 1e-6) == 0)


# 4. Probability that a group is informative.
def informative(q, G):
    return 1 - q**G - (1 - q) ** G


assert round(informative(0.9, 16), 3) == 0.815
assert round(informative(0.95, 8), 3) == 0.337
assert round(0.9**16, 3) == 0.185 and round(0.95**8, 3) == 0.663
sim = rng.random((200000, 16)) < 0.9
assert abs((sim.any(1) & ~sim.all(1)).mean() - informative(0.9, 16)) < 3e-3

# 5. Normalization reweights questions by 1/sqrt(p(1 - p)).
for G in (8, 16):
    for c in range(1, G):
        p = c / G
        r = np.array([1.0] * c + [0.0] * (G - c))
        assert math.isclose(np.abs(r - r.mean()).sum(), 2 * G * p * (1 - p))
        assert math.isclose(np.abs((r - r.mean()) / r.std()).sum(), 2 * G * math.sqrt(p * (1 - p)))
assert round(1 / math.sqrt(1 / 16 * 15 / 16), 2) == 4.13
assert math.isclose(16 / math.sqrt(15), 1 / math.sqrt(15 / 256))
assert 1 / math.sqrt(0.25) == 2
assert round(4.131 / 2, 2) == 2.07

# 6. Per-token weights of the three loss reductions.
lengths = np.array([100, 1000, 300, 50])
A = np.array([-1.0, -1.0, 0.5, 1.5])
G, Lmax = len(lengths), 1024
seq = A / (G * lengths)
tok = A / lengths.sum()
const = A / (G * Lmax)
assert math.isclose(seq[1] / seq[0], 0.1)
for w in (tok, const):
    assert math.isclose(w[1], w[0])
for i in range(G):  # total push per output under the sequence mean is A_i / G
    assert math.isclose(seq[i] * lengths[i], A[i] / G)

# 7. The k3 estimator is nonnegative and unbiased for KL(pi || ref).
x = np.linspace(1e-3, 50, 100001)
assert np.all(x - np.log(x) - 1 >= -1e-12)
ref = softmax(rng.normal(size=K))
kl = float((pi * np.log(pi / ref)).sum())
exact_k3 = float((pi * (ref / pi - np.log(ref / pi) - 1)).sum())
assert math.isclose(exact_k3, kl, rel_tol=1e-12)
assert math.isclose(float((pi * (ref / pi)).sum()), 1.0)
s = rng.choice(K, p=pi, size=400000)
xs = ref[s] / pi[s]
assert abs((xs - np.log(xs) - 1).mean() - kl) < 5e-3
assert (-np.log(xs)).min() < 0 <= (xs - np.log(xs) - 1).min()


# 8. Sharpening lowers pass@64.
def passk(p, k):
    return 1 - (1 - p) ** k


assert math.isclose((0.2 + 0.05) / 2, 0.125)
assert round((passk(0.2, 64) + passk(0.05, 64)) / 2, 3) == 0.981
assert round(0.95**64, 4) == 0.0375
assert math.isclose((0.6 + 0.0) / 2, 0.3)
assert round((passk(0.6, 64) + passk(0.0, 64)) / 2, 3) == 0.5
print("ok")
