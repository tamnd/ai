# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/structured-state-space-model/ (ai://MDN3YW).
"""Numeric check of the derivation hippo-nplr: the HiPPO matrix of S4 is normal plus rank one, and its eigenvectors are ill-conditioned."""

import numpy as np


def hippo(n):
    k = np.arange(n)
    a = np.sqrt(np.outer(2 * k + 1, 2 * k + 1))
    return -(np.tril(a, -1) + np.diag(k + 1.0))


for n in range(1, 41):
    A = hippo(n)
    P = np.sqrt(np.arange(n) + 0.5)
    S = A + np.outer(P, P)
    J = S + 0.5 * np.eye(n)
    assert np.allclose(J, -J.T)  # step 3
    assert np.allclose(S @ S.T, S.T @ S)  # step 4
    w, U = np.linalg.eig(S)
    assert np.allclose(w.real, -0.5)
    assert np.allclose(U.conj().T @ U, np.eye(n), atol=1e-8)  # normal: orthonormal eigenvectors
    assert np.allclose(np.sort(np.diag(A)), -np.arange(n, 0, -1)) and np.allclose(np.triu(A, 1), 0)  # step 6
conds = {}
for n in (4, 8, 16):
    w, V = np.linalg.eig(hippo(n))
    conds[n] = np.linalg.cond(V)
assert 5e4 < conds[8] < 1.5e5 and 5e10 < conds[16] < 1.5e11
print("hippo-nplr: A = S - P P^T with S normal, Re eig(S) = -1/2, for N = 1..40; cond(V) = " + ", ".join(f"{k}: {v:.2e}" for k, v in conds.items()))
