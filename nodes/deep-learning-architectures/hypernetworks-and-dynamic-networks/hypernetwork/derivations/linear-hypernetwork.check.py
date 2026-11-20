# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/hypernetwork/ (ai://NEW-hypernetwork).
"""Numeric check of the derivation linear-hypernetwork: affinity, the shared subspace of the slices, and the tiling."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import hypernetwork as H  # noqa: E402

rng = np.random.default_rng(0)
N, f, Nz, d = 4, 3, 5, 3
p = {"W_in": rng.normal(size=(N, d, Nz)), "B_in": rng.normal(size=(N, d)), "W_out": rng.normal(size=(f, N * f, d)),
     "B_out": rng.normal(size=(f, N * f))}  # fmt: skip


def K(z):
    return H.generate(p, z)[1]


# 1. affine: K(s u + (1 - s) v) = s K(u) + (1 - s) K(v) for any s, and the map is A z + c
for _ in range(50):
    u, v, s = rng.normal(size=(1, Nz)), rng.normal(size=(1, Nz)), rng.normal()
    assert np.allclose(K(s * u + (1 - s) * v), s * K(u) + (1 - s) * K(v))
c = K(np.zeros((1, Nz)))
A = np.stack([(K(np.eye(Nz)[k : k + 1]) - c).ravel() for k in range(Nz)], 1)
z = rng.normal(size=(1, Nz))
assert np.allclose(K(z).ravel(), A @ z[0] + c.ravel())

# 2. every slice of every kernel, minus B_out, lies in a subspace of dimension d
Z = rng.normal(size=(200, Nz))
slices = (K(Z) - p["B_out"]).reshape(-1, f * N * f)  # (200 N, f N f)
assert np.linalg.matrix_rank(slices) == d
basis = p["W_out"].reshape(f * N * f, d)
coef, *_ = np.linalg.lstsq(basis, slices.T, rcond=None)
assert np.allclose(basis @ coef, slices.T)

# 3. a one-layer map z -> K reaches min(N_z, f N f) dimensions for each slice
W1 = rng.normal(size=(N, f, N * f, Nz))
one = np.einsum("iuwz,ez->eiuw", W1, Z)
assert np.linalg.matrix_rank(one[:, 0].reshape(200, -1)) == min(Nz, f * N * f)
assert np.linalg.matrix_rank(one.reshape(-1, f * N * f)) > d

# 4. the tiling: block (r, c) of an m_in x m_out kernel depends only on embedding r m_out + c
m_in, m_out = 2, 3
Z = rng.normal(size=(m_in * m_out, Nz))
w = H.tile(H.to_filters(K(Z), f), m_in, m_out)
assert w.shape == (m_out * N, m_in * N, f, f)
for j in range(m_in * m_out):
    Z2 = Z.copy()
    Z2[j] += 1.0
    w2 = H.tile(H.to_filters(K(Z2), f), m_in, m_out)
    changed = np.abs(w2 - w).reshape(m_out, N, m_in, N, f, f).max((1, 3, 4, 5)) > 0
    r, col = divmod(j, m_out)
    assert changed[col, r] and changed.sum() == 1
assert np.array_equal(H.untile(w, m_in, m_out), H.to_filters(K(Z), f))

print("linear-hypernetwork: ok")
