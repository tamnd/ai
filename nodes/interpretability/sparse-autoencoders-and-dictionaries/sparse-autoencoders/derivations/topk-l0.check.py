# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/sparse-autoencoders/ (ai://ZTPNXN).
# Check of derivations/topk-l0.md: ReLU(TopK(a)) has at most k nonzeros, is a nearest nonnegative k-sparse vector to a,
# and for an orthonormal decoder W with encoder W^T it is the best nonnegative k-sparse code of x.
import itertools

import numpy as np


def topk_relu(a, k):
    idx = np.argsort(-a, kind="stable")[:k]
    z = np.zeros_like(a)
    z[idx] = np.maximum(a[idx], 0.0)
    return z


def best_k_sparse(a, k):
    best = np.inf
    for size in range(k + 1):
        for s in itertools.combinations(range(a.size), size):
            v = np.zeros_like(a)
            v[list(s)] = np.maximum(a[list(s)], 0.0)
            best = min(best, np.sum((a - v) ** 2))
    return best


rng = np.random.default_rng(5)
for _ in range(300):
    n = int(rng.integers(1, 8))
    k = int(rng.integers(1, n + 1))
    a = rng.normal(size=n) * 2.0
    z = topk_relu(a, k)
    assert np.count_nonzero(z) == min(k, int(np.sum(a > 0)))
    assert abs(np.sum((a - z) ** 2) - best_k_sparse(a, k)) < 1e-12
    kept = z > 0
    assert np.array_equal(z[kept], a[kept])

for _ in range(100):
    m = int(rng.integers(2, 7))
    n = int(rng.integers(1, m + 1))
    k = int(rng.integers(1, n + 1))
    Q, _ = np.linalg.qr(rng.normal(size=(m, m)))
    W = Q[:, :n]
    x = rng.normal(size=m)
    a = W.T @ x
    z = topk_relu(a, k)
    r = x - W @ a
    assert abs(np.sum((x - W @ z) ** 2) - (np.sum(r**2) + np.sum((a - z) ** 2))) < 1e-12
    # brute force over supports of the code itself
    best = np.inf
    for s in itertools.combinations(range(n), k):
        v = np.zeros(n)
        v[list(s)] = np.maximum(a[list(s)], 0.0)
        best = min(best, np.sum((x - W @ v) ** 2))
    assert np.sum((x - W @ z) ** 2) <= best + 1e-12
print("ok")
