# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/sparse-mixture-of-experts/ (ai://1H67K0).
# Check of derivations/top-k-gate.md: softmax(KeepTopK(h, k)) equals the top k entries of softmax(h) renormalized,
# zero elsewhere; the top-1 gate is 1 and every gate vector sums to 1.
import numpy as np


def softmax(v):
    e = np.exp(v - v[np.isfinite(v)].max())
    return e / e.sum()


rng = np.random.default_rng(7)
for _ in range(200):
    n = int(rng.integers(2, 9))
    h = rng.normal(size=n) * 3.0
    order = np.argsort(-h, kind="stable")
    for k in range(1, n + 1):
        s = order[:k]
        masked = np.full(n, -np.inf)
        masked[s] = h[s]
        left = softmax(masked)
        p = softmax(h)
        right = np.zeros(n)
        right[s] = p[s] / p[s].sum()
        assert np.allclose(left, right, rtol=1e-12, atol=1e-15)
        assert abs(left.sum() - 1.0) < 1e-12
        if k == 1:
            assert left[s[0]] == 1.0
print("ok")
