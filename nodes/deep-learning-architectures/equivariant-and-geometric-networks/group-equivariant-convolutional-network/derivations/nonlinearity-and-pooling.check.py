# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/group-equivariant-convolutional-network/ (ai://NEW-group-equivariant-convolutional-network).
"""Numeric check of the derivation nonlinearity-and-pooling."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import gcnn  # noqa: E402

rng = np.random.default_rng(2)
F = rng.normal(size=(2, 3, 4, 5, 5))

# Statement 1: ReLU and a shared bias commute with the action; a bias per plane does not.
relu = lambda a: np.maximum(a, 0.0)  # noqa: E731
np.testing.assert_array_equal(relu(gcnn.rotate_p4(F)), gcnn.rotate_p4(relu(F)))
b = rng.normal(size=(3, 1, 1, 1))
np.testing.assert_allclose(gcnn.rotate_p4(F) + b, gcnn.rotate_p4(F + b), atol=1e-15)
b_plane = rng.normal(size=(3, 4, 1, 1))
assert np.abs((gcnn.rotate_p4(F) + b_plane) - gcnn.rotate_p4(F + b_plane)).max() > 0.1

# Statement 2: coset pooling turns with the input.
for r in range(4):
    np.testing.assert_array_equal(gcnn.rotate_p4(F, r).max(axis=2), gcnn.rotate(F.max(axis=2), r))

# Statement 3: the model is invariant.
for mode, n in (("zero", 7), ("circular", 6)):
    cfg = gcnn.config(2, 3, 3, 3, 4, mode=mode)
    params = gcnn.init(rng, cfg)
    for k in ("b1", "b2", "bc"):
        params[k] = rng.normal(0.0, 0.3, params[k].shape)
    x = rng.normal(size=(3, 2, n, n))
    for r in range(1, 4):
        np.testing.assert_allclose(gcnn.forward(params, gcnn.rotate(x, r), cfg), gcnn.forward(params, x, cfg), atol=1e-12)
    if mode == "circular":
        np.testing.assert_allclose(gcnn.forward(params, np.roll(x, (3, 1), (2, 3)), cfg), gcnn.forward(params, x, cfg),
                                   atol=1e-12)

# Statement 4: stride 2 keeps equivariance on an odd grid only.
W = rng.normal(size=(2, 1, 3, 3))
for n, ok in ((7, True), (6, False)):
    x = rng.normal(size=(1, 1, n, n))
    sub = lambda a: a[..., ::2, ::2]  # noqa: E731
    gap = np.abs(sub(gcnn.lift(gcnn.rotate(x), W)) - gcnn.rotate_p4(sub(gcnn.lift(x, W)))).max()
    assert (gap < 1e-12) == ok, (n, gap)
print("nonlinearity-and-pooling: ok")
