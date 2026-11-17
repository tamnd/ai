# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/mode-connectivity/ (ai://NEW-mode-connectivity).
# Check of derivations/permutation-symmetry.md: a hidden-unit permutation leaves a one-hidden-layer network
# unchanged; the segment between the |x| network and its swapped copy has barrier 1 at alpha = 1/2; and weight
# matching by one linear assignment on C finds the permutation of least distance, against brute force.
import itertools

import numpy as np

rng = np.random.default_rng(0)


def f(p, X, act):
    W1, b1, W2, b2 = p
    return act(X @ W1.T + b1) @ W2.T + b2


def perm(p, pi):
    W1, b1, W2, b2 = p
    return W1[pi], b1[pi], W2[:, pi], b2


relu = lambda z: np.maximum(z, 0.0)  # noqa: E731
for act in (relu, np.tanh):
    for _ in range(20):
        p = (rng.normal(size=(6, 3)), rng.normal(size=6), rng.normal(size=(2, 6)), rng.normal(size=2))
        X = rng.normal(size=(10, 3))
        assert np.allclose(f(p, X, act), f(perm(p, rng.permutation(6)), X, act))

a = (np.array([[1.0], [-1.0]]), np.zeros(2), np.array([[1.0, 1.0]]), np.zeros(1))
b = perm(a, np.array([1, 0]))
X, y = np.array([[-1.0], [1.0]]), np.array([1.0, 1.0])
xs = np.linspace(-2, 2, 9)[:, None]
assert np.allclose(f(a, xs, relu)[:, 0], np.abs(xs[:, 0])) and np.allclose(f(b, xs, relu), f(a, xs, relu))
alphas = np.linspace(0, 1, 101)
L = []
for al in alphas:
    m = tuple((1 - al) * u + al * v for u, v in zip(a, b))
    L.append(np.mean((f(m, X, relu)[:, 0] - y) ** 2))
L = np.array(L)
assert np.allclose(L, (1 - np.abs(1 - 2 * alphas)) ** 2)
assert L[0] == 0 and L[-1] == 0 and abs(L.max() - 1) < 1e-12 and alphas[L.argmax()] == 0.5


def flat(p):
    return np.concatenate([x.ravel() for x in p])


for _ in range(10):
    A = (rng.normal(size=(6, 3)), rng.normal(size=6), rng.normal(size=(2, 6)), rng.normal(size=2))
    B = (rng.normal(size=(6, 3)), rng.normal(size=6), rng.normal(size=(2, 6)), rng.normal(size=2))
    C = A[0] @ B[0].T + np.outer(A[1], B[1]) + A[2].T @ B[2]
    best_d = min(itertools.permutations(range(6)), key=lambda pi: np.sum((flat(A) - flat(perm(B, list(pi)))) ** 2))
    best_c = max(itertools.permutations(range(6)), key=lambda pi: sum(C[i, pi[i]] for i in range(6)))
    assert best_d == best_c
print("ok")
