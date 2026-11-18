# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/group-equivariant-convolutional-network/ (ai://NEW-group-equivariant-convolutional-network).
"""Numeric check of the derivation split-filter-bank."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import gcnn  # noqa: E402

rng = np.random.default_rng(1)
O, C, k = 2, 3, 3
W = rng.permutation(O * C * 4 * k * k).astype(float).reshape(O, C, 4, k, k)
bank = gcnn.gconv_bank(W).reshape(O, 4, C, 4, k, k)

# Statement 2: the bank entry (o, r, c, s) is the stored plane s - r turned r times.
for r in range(4):
    for s in range(4):
        np.testing.assert_array_equal(bank[:, r, :, s], gcnn.rotate(W[:, :, (s - r) % 4], r))

# Statement 3: each row is a permutation of a stored filter, and the bank holds each weight 4 times.
for o in range(O):
    for r in range(4):
        for c in range(C):
            np.testing.assert_array_equal(np.sort(bank[o, r, c].ravel()), np.sort(W[o, c].ravel()))
values, counts = np.unique(bank, return_counts=True)
assert len(values) == W.size and set(counts) == {4}
lift_bank = gcnn.lift_bank(W[:, :, 0])
assert set(np.unique(lift_bank, return_counts=True)[1]) == {4}

# Statement 1: one planar call with the bank equals the sum over the group.
f = rng.normal(size=(2, C, 4, 6, 6))
Wr = rng.normal(size=(O, C, 4, k, k))
for mode in ("zero", "circular"):
    np.testing.assert_allclose(gcnn.gconv(f, Wr, mode), gcnn.gconv_direct(f, Wr, mode), atol=1e-11)

# Statement 4: the gradient of a stored weight sums its copies; compare with finite differences.
dout = rng.normal(size=(2, O, 4, 6, 6))
_, dW = gcnn.gconv_backward(f, Wr, dout)
eps = 1e-6
for _ in range(10):
    idx = tuple(rng.integers(s) for s in Wr.shape)
    up, dn = Wr.copy(), Wr.copy()
    up[idx] += eps
    dn[idx] -= eps
    num = (np.sum(gcnn.gconv(f, up) * dout) - np.sum(gcnn.gconv(f, dn) * dout)) / (2 * eps)
    assert abs(num - dW[idx]) < 1e-6 * max(1.0, abs(num)), (idx, num, dW[idx])
print("split-filter-bank: ok")
