# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/linear-attention/ (ai://NEW-linear-attention).
"""Numeric check of the derivation cost: products counted on the NumPy functions, the crossovers and the exact
polynomial feature map."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import linear_attention as L  # noqa: E402

COUNT = [0]


class Counted(np.ndarray):
    def __matmul__(self, other):
        a, b = np.asarray(self), np.asarray(other)
        COUNT[0] += 2 * a.size * (b.shape[-1] if b.ndim > 1 else 1)
        return np.ndarray.__matmul__(a, b).view(Counted)

    def __rmatmul__(self, other):
        return Counted.__matmul__(np.asarray(other).view(Counted), self)


rng = np.random.default_rng(2)
for _ in range(30):
    N, D, M, C = (int(v) for v in rng.integers([1, 1, 1, 1], [40, 9, 9, 20]))
    Q, K, V = (rng.normal(size=s).view(Counted) for s in ((N, D), (N, D), (N, M)))
    COUNT[0] = 0
    L.softmax_attention(Q, K, V)
    assert COUNT[0] + 2 * N * N == L.flops(N, D, M, kind="softmax")  # the row sums are np.sum in the code
    fQ, fK = (np.abs(rng.normal(size=(N, C))).view(Counted) for _ in range(2))
    COUNT[0] = 0
    L.linear_attention(fQ, fK, V)
    assert COUNT[0] + 2 * N * C == L.flops(N, D, M, C)  # z = phi(K)^T 1 is np.sum in the code
    # item 2 for these sizes
    bound = 2 * C * (M + 1) / (D + M + 1)
    for n in range(1, 200):
        assert (L.flops(n, D, M, C) < L.flops(n, D, M, kind="softmax")) == (n > bound)
D = M = 64
assert min(n for n in range(1, 10**4) if L.flops(n, D, M) < L.flops(n, D, M, kind="softmax")) == 65
C = 1 + D + D * D
first = min(n for n in range(1, 10**5) if L.flops(n, D, M, C) < L.flops(n, D, M, kind="softmax"))
assert first == 4194 and D * D < first < D * D + 2 * D + 2
# item 3
for _ in range(20):
    d, c = int(rng.integers(1, 8)), float(rng.uniform(0.1, 3))
    q, k = rng.normal(size=d), rng.normal(size=d)
    assert np.isclose(L.poly2(q, c) @ L.poly2(k, c), (q @ k + c) ** 2) and L.poly2(q).size == 1 + d + d * d
print("cost: ok")
