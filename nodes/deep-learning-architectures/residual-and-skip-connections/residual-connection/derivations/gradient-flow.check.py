# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/residual-connection/ (ai://F5K1AF).
"""Numeric check of the derivation gradient-flow: the unrolled sum, the Jacobian product, the bounds, the scaled shortcut, pre-LN and post-LN."""

import itertools

import numpy as np

rng = np.random.default_rng(20160316)
d, n = 6, 8
W1 = [rng.normal(scale=0.3, size=(d, d)) for _ in range(n)]
W2 = [rng.normal(scale=0.3, size=(d, d)) for _ in range(n)]
EYE = np.eye(d)


def F(i, x):
    return W2[i] @ np.tanh(W1[i] @ x)


def J(i, x):
    return W2[i] @ np.diag(1 - np.tanh(W1[i] @ x) ** 2) @ W1[i]


def LN(x):
    return (x - x.mean()) / np.sqrt(x.var() + 1e-5)


def J_LN(x):
    s = np.sqrt(x.var() + 1e-5)
    c = EYE - 1.0 / d
    u = c @ x
    return (c - np.outer(u, u) / (d * s**2)) / s


def run(x, unit):
    xs = [x]
    for i in range(n):
        xs.append(unit(i, xs[-1]))
    return xs


def fd(fn, x, h=1e-6):
    return np.stack([(fn(x + h * e) - fn(x - h * e)) / (2 * h) for e in EYE], axis=1)


x0 = rng.normal(size=d)
plain_res = lambda i, x: x + F(i, x)  # noqa: E731
xs = run(x0, plain_res)
assert np.allclose(xs[-1], x0 + sum(F(i, xs[i]) for i in range(n)))  # statement 1, eq-unroll

prod = EYE
for i in range(n):
    prod = (EYE + J(i, xs[i])) @ prod  # eq-jacobian
assert np.allclose(fd(lambda v: run(v, plain_res)[-1], x0), prod, atol=1e-7)
g = rng.normal(size=d)
assert np.allclose(g @ prod, g + g @ (prod - EYE))  # eq-grad: the direct term is g itself
gi = g.copy()
for i in reversed(range(n)):
    gi = gi + gi @ J(i, xs[i])  # eq-backward
assert np.allclose(gi, g @ prod)

# Statement 3: the bounds, with eps the largest spectral norm along the path.
eps = max(np.linalg.norm(J(i, xs[i]), 2) for i in range(n))
r = np.linalg.norm(g @ prod) / np.linalg.norm(g)
assert eps < 1 and (1 - eps) ** n <= r <= (1 + eps) ** n
ps = run(x0, lambda i, x: F(i, x))
pj = EYE
for i in range(n):
    pj = J(i, ps[i]) @ pj
bound = np.prod([np.linalg.norm(J(i, ps[i]), 2) for i in range(n)])
rp = np.linalg.norm(g @ pj) / np.linalg.norm(g)
assert rp <= bound * (1 + 1e-9)

# Statement 4: the scaled shortcut, for linear branches, expanded over the subsets.
A = [rng.normal(scale=0.3, size=(d, d)) for _ in range(n)]
for lam in (0.9, 1.1):
    full = EYE
    for a in A:
        full = (lam * EYE + a) @ full
    rest = np.zeros((d, d))
    for S in itertools.product([0, 1], repeat=n):
        if any(S):
            term = EYE
            for i, pick in enumerate(S):
                term = (A[i] if pick else lam * EYE) @ term
            rest += term
    assert np.allclose(full - rest, lam**n * EYE)  # eq-scaled

# Statement 5: pre-LN keeps the unrolled sum with branch F o LN; post-LN puts J_LN on the direct path.
pre = run(x0, lambda i, x: x + F(i, LN(x)))
assert np.allclose(pre[-1], x0 + sum(F(i, LN(pre[i])) for i in range(n)))
assert np.allclose(fd(LN, x0), J_LN(x0), atol=1e-7)
post = run(x0, lambda i, x: LN(x + F(i, x)))
pj = EYE
direct = EYE
for i in range(n):
    y = post[i] + F(i, post[i])
    pj = J_LN(y) @ (EYE + J(i, post[i])) @ pj
    direct = J_LN(y) @ direct
assert np.allclose(fd(lambda v: run(v, lambda i, x: LN(x + F(i, x)))[-1], x0), pj, atol=1e-6)
assert not np.allclose(direct, EYE)  # the post-LN direct term is not the identity
print(f"gradient-flow: n={n}, eps={eps:.3f}, residual ratio {r:.3f} in [{(1 - eps) ** n:.3f}, {(1 + eps) ** n:.3f}], plain ratio {rp:.2e} <= {bound:.2e}; scaled and LN checks hold")
