# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/linear-attention/ (ai://NEW-linear-attention).
"""Numeric check of the derivation causal-gradient: Algorithm 1 against the gradient of the masked quadratic form
and finite differences, and the numbers each method keeps."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import linear_attention as L  # noqa: E402

rng = np.random.default_rng(1)
for _ in range(30):
    N, C, M = (int(v) for v in rng.integers([1, 1, 1], [12, 6, 6]))
    q, k, v = L.elu_plus_one(rng.normal(size=(N, C))), L.elu_plus_one(rng.normal(size=(N, C))), rng.normal(size=(N, M))
    G = rng.normal(size=(N, M))
    dq, dk, dv = L.causal_numerator_backward(q, k, v, G)
    T = np.tril(np.ones((N, N)))
    dA = T * (G @ v.T)
    assert np.allclose(dq, dA @ k) and np.allclose(dk, dA.T @ q) and np.allclose(dv, (T * (q @ k.T)).T @ G)

    def f(q, k, v):
        return (L.causal_linear_attention(q, k, v)[1] * G).sum()

    for arr, grad in ((q, dq), (k, dk), (v, dv)):
        i = tuple(rng.integers(0, s) for s in arr.shape)
        o = arr[i]
        arr[i] = o + 1e-6
        up = f(q, k, v)
        arr[i] = o - 1e-6
        down = f(q, k, v)
        arr[i] = o
        assert abs((up - down) / 2e-6 - grad[i]) < 1e-6
    # item 3: the stored numbers, inputs and G against all the states
    assert q.size + k.size + v.size + G.size == N * (2 * C + 2 * M)
    states = np.cumsum(k[:, :, None] * v[:, None, :], 0)
    assert states.size == N * C * M
print("causal-gradient: ok")
