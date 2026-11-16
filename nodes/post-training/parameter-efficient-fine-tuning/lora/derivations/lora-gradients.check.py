# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/lora/ (ai://NEW-lora).
"""Numeric check of the derivation lora-gradients: the gradients of B, A and X against central finite differences."""

import numpy as np

rng = np.random.default_rng(20261114)
n, k, d, r, s = 6, 7, 5, 3, 2.5
X = rng.normal(size=(n, k))
W0 = rng.normal(size=(d, k))
b0 = rng.normal(size=d)
A = rng.normal(size=(r, k))
B = rng.normal(size=(d, r))
G = rng.normal(size=(n, d))


def loss(a, b, x):
    y = x @ W0.T + b0 + s * x @ a.T @ b.T
    return float(np.sum(G * y))


def numeric(f, m, eps=1e-6):
    out = np.zeros_like(m)
    for idx in np.ndindex(m.shape):
        up, down = m.copy(), m.copy()
        up[idx] += eps
        down[idx] -= eps
        out[idx] = (f(up) - f(down)) / (2 * eps)
    return out


GW = G.T @ X
dB = s * G.T @ X @ A.T
dA = s * B.T @ G.T @ X
dX = G @ (W0 + s * B @ A)
np.testing.assert_allclose(dB, numeric(lambda m: loss(A, m, X), B), rtol=1e-6)
np.testing.assert_allclose(dA, numeric(lambda m: loss(m, B, X), A), rtol=1e-6)
np.testing.assert_allclose(dX, numeric(lambda m: loss(A, B, m), X), rtol=1e-6)
np.testing.assert_allclose(dB, s * GW @ A.T, rtol=1e-12)
np.testing.assert_allclose(dA, s * B.T @ GW, rtol=1e-12)

# Item 4: with B = 0 the gradient of A is zero and the gradient of B is s G_W A^T.
Z = np.zeros_like(B)
assert np.allclose(numeric(lambda m: loss(m, Z, X), A), 0.0, atol=1e-8)
np.testing.assert_allclose(numeric(lambda m: loss(A, m, X), Z), s * GW @ A.T, rtol=1e-6)
print("lora-gradients: the three gradients match finite differences to 1e-6, and the start with B = 0 moves only B")
