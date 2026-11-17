# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/structured-state-space-model/ (ai://NEW-structured-state-space-model).
"""Numeric check of the derivation selective-scan: the gate form, the loss of linearity, the associative scan."""

import numpy as np

rng = np.random.default_rng(20231201)
softplus = lambda z: np.log1p(np.exp(z))  # noqa: E731
sigmoid = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731


def selective(u, delta, a, b, c):
    x, y = np.zeros(len(a)), []
    for k in range(len(u)):
        ab = np.exp(delta[k] * a)
        x = ab * x + (ab - 1) / a * b[k] * u[k]
        y.append(c[k] @ x)
    return np.array(y)


L = 32
u, z = rng.normal(size=L), rng.normal(size=L)
y = selective(u, softplus(z), np.array([-1.0]), np.ones((L, 1)), np.ones((L, 1)))
h, ref = 0.0, []
for k in range(L):
    h = (1 - sigmoid(z[k])) * h + sigmoid(z[k]) * u[k]
    ref.append(h)
assert np.allclose(y, ref)  # statement 1

N = 4
a = -np.exp(rng.normal(size=N))
B, C = rng.normal(size=(L, N)), rng.normal(size=(L, N))
W = 1.5
f = lambda u: selective(u, softplus(W * u), a, B, C)  # noqa: E731
u1, u2 = rng.normal(size=L), rng.normal(size=L)
assert not np.allclose(f(u1 + u2), f(u1) + f(u2))  # statement 2
assert not np.allclose(f(2 * u1), 2 * f(u1))

op = lambda p, q: (q[0] * p[0], q[0] * p[1] + q[1])  # noqa: E731
for _ in range(100):
    p, q, r = [(rng.normal(size=N), rng.normal(size=N)) for _ in range(3)]
    left, right = op(op(p, q), r), op(p, op(q, r))
    assert np.allclose(left[0], right[0]) and np.allclose(left[1], right[1])

# Statement 3: Hillis-Steele recursive doubling over the pairs gives the states in ceil(log2 L) rounds.
delta = softplus(rng.normal(size=L))
abar = np.exp(delta[:, None] * a)
bu = (abar - 1) / a * B * u1[:, None]
pa, pb = abar.copy(), bu.copy()
rounds, shift = 0, 1
while shift < L:
    na, nb = pa.copy(), pb.copy()
    na[shift:] = pa[shift:] * pa[:-shift]
    nb[shift:] = pa[shift:] * pb[:-shift] + pb[shift:]
    pa, pb, shift, rounds = na, nb, shift * 2, rounds + 1
x, states = np.zeros(N), []
for k in range(L):
    x = abar[k] * x + bu[k]
    states.append(x)
assert np.allclose(pb, np.array(states)) and rounds == 5
print(f"selective-scan: gate form holds over L={L}; additivity fails; the operator is associative; the prefix scan gives the states in {rounds} rounds")
