# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/graph-neural-network/ (ai://PS4TCX).
"""Numeric check of the derivation gcn-renormalization: the first-order filter, the two spectra and the limit."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import graph_neural_network as gnn  # noqa: E402

rng = np.random.default_rng(1609)


def random_graph(n, p):
    upper = np.triu(rng.random((n, n)) < p, 1)
    return (upper | upper.T).astype(float)


def connected(a):
    seen, todo = {0}, [0]
    while todo:
        for u in np.flatnonzero(a[todo.pop()]):
            if u not in seen:
                seen.add(int(u))
                todo.append(int(u))
    return len(seen) == a.shape[0]


checked_limit = 0
for _ in range(200):
    n = int(rng.integers(2, 12))
    a = random_graph(n, rng.uniform(0.1, 0.9))
    a_hat = gnn.normalized_adjacency(a)  # step 4
    lam = np.linalg.eigvalsh(a_hat)
    assert np.allclose(a_hat, a_hat.T) and lam.min() >= -1 - 1e-12 and lam.max() <= 1 + 1e-12
    s = np.sqrt((a + np.eye(n)).sum(axis=1))
    assert np.allclose(a_hat @ s, s)
    d = a.sum(axis=1)
    if d.min() > 0:  # steps 1 and 3 need every degree positive
        sym = a / np.sqrt(np.outer(d, d))
        lam = np.linalg.eigvalsh(np.eye(n) + sym)
        assert lam.min() >= -1e-12 and lam.max() <= 2 + 1e-12
        L = np.eye(n) - sym
        x = rng.normal(size=n)
        t0, t1 = rng.normal(size=2)
        L_tilde = 2 / 2.0 * L - np.eye(n)  # lambda_max = 2
        cheb = t0 * x + t1 * (L_tilde @ x)  # T_0 = 1, T_1(y) = y
        assert np.allclose(cheb, t0 * x - t1 * sym @ x)  # Eq. 6
        assert np.allclose(t0 * x - (-t0) * sym @ x, t0 * (np.eye(n) + sym) @ x)  # Eq. 7: theta_1 = -theta_0
    if connected(a):  # step 5
        x = rng.normal(size=n)
        z = np.linalg.matrix_power(a_hat, 2000) @ x
        assert np.allclose(z, s * (s @ x) / (s @ s), atol=1e-8)
        checked_limit += 1
assert checked_limit > 50

# Two components: the limit keeps one coefficient per component (Li, Han and Wu, Theorem 1), not one overall.
a = np.zeros((5, 5))
a[0, 1] = a[1, 0] = a[2, 3] = a[3, 2] = a[3, 4] = a[4, 3] = 1.0
z = np.linalg.matrix_power(gnn.normalized_adjacency(a), 2000) @ np.array([1.0, 0, 0, 0, 0])
assert np.allclose(z[2:], 0) and z[0] > 0 and np.isclose(z[0], z[1])
print("gcn-renormalization: ok")
