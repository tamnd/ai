# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/deep-sets/ (ai://NEW-deep-sets).
"""Numeric check of the derivation equivariant-layers: the commutant of the permutations, the layers and the masks."""

import itertools
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import deepsets  # noqa: E402


def perm_matrix(p):
    return np.eye(len(p))[list(p)]


# Step 1 and 2: the linear maps vec(x) -> vec(T x) that commute with every permutation of the rows.
for M in range(1, 5):
    for D, E in [(1, 1), (2, 1), (2, 3)]:
        rows = []
        for p in itertools.permutations(range(M)):
            P = np.kron(perm_matrix(p), np.eye(1))
            # T acts on vec(x) for x of shape (M, D) in row-major order: an (M E) x (M D) matrix W, W vec(x) = vec(T x).
            # T(Px) = P T(x) is (P kron I_E) W = W (P kron I_D), linear in W.
            A, B = np.kron(P, np.eye(E)), np.kron(P, np.eye(D))
            rows.append(np.kron(A, np.eye(M * D)) - np.kron(np.eye(M * E), B.T))
        C = np.vstack(rows)
        null = C.shape[1] - np.linalg.matrix_rank(C)
        assert null == (2 * D * E if M >= 2 else D * E), (M, D, E, null)

rng = np.random.default_rng(0)

# Step 3: one channel, sigma(Theta x) with Theta = lambda I + gamma 1 1^T.
for M in range(1, 7):
    lam, gam = rng.normal(size=2)
    x = rng.normal(size=M)
    Theta = deepsets.equivariant_matrix(lam, gam, M)
    assert np.allclose(np.diag(Theta), lam + gam) and np.allclose(Theta[~np.eye(M, dtype=bool)], gam)
    assert np.allclose(np.tanh(Theta @ x), np.tanh(lam * x + gam * x.sum()))
    # the reference layer with the sum pool and Gamma = -gamma is the same layer
    z, _ = deepsets.equivariant(x[None, :, None], np.ones((1, M)), np.array([[lam]]), np.array([[-gam]]), np.zeros(1),
                                "sum")  # fmt: skip
    assert np.allclose(z[0, :, 0], Theta @ x)

# Step 4: the layers of the reference are equivariant for each pool, and the set head is invariant.
for pool in deepsets.POOLS:
    for interact in (True, False):
        cfg = deepsets.config(d_in=3, width=5, n_layers=2, pool=pool, readout=pool, interact=interact, head="element")
        params = deepsets.init(rng, cfg, 0.5)
        x = rng.normal(size=(2, 6, 3))
        p = rng.permutation(6)
        s, _ = deepsets.forward(params, cfg, x)
        sp, _ = deepsets.forward(params, cfg, x[:, p])
        assert np.allclose(sp, s[:, p], atol=1e-12)
        cfg = deepsets.config(d_in=3, width=5, n_layers=2, pool=pool, readout=pool, interact=interact)
        params = deepsets.init(rng, cfg, 0.5)
        assert np.allclose(deepsets.forward(params, cfg, x[:, p])[0], deepsets.forward(params, cfg, x)[0], atol=1e-12)

# Step 5: masks and padding.
for pool in deepsets.POOLS:
    cfg = deepsets.config(d_in=2, width=4, n_layers=2, pool=pool, readout=pool)
    params = deepsets.init(rng, cfg, 0.5)
    x = rng.normal(size=(1, 3, 2))
    pad = np.concatenate([x, np.zeros((1, 4, 2))], 1)
    mask = np.array([[1, 1, 1, 0, 0, 0, 0]])
    assert np.allclose(deepsets.forward(params, cfg, pad, mask)[0], deepsets.forward(params, cfg, x)[0], atol=1e-12)
cfg = deepsets.config(d_in=2, width=4, n_layers=1, pool="sum", readout="sum", interact=False)
params = deepsets.init(rng, cfg, 0.5)
x = rng.normal(size=(1, 3, 2))
pad = np.concatenate([x, np.zeros((1, 4, 2))], 1)
alone = deepsets.forward(params, cfg, x)[1]["s"]
unmasked = deepsets.forward(params, cfg, pad)[1]["s"]
phi0 = np.tanh(params["eq0_beta"])
assert np.allclose(unmasked - alone, 4 * phi0) and np.abs(phi0).max() > 0

print("equivariant-layers: ok")
