# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/group-equivariant-convolutional-network/ (ai://NEW-group-equivariant-convolutional-network).
"""Numeric check of the derivation group-correlation-equivariance."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import gcnn  # noqa: E402

rng = np.random.default_rng(0)
for mode in ("zero", "circular"):
    for n in (5, 6, 7):
        x, w = rng.normal(size=(2, 3, n, n)), rng.normal(size=(4, 3, 3, 3))
        # Statement 1: translations commute (circular), rotations need the counter-rotated filter.
        if mode == "circular":
            np.testing.assert_allclose(gcnn.corr2d(np.roll(x, (1, 2), (2, 3)), w, mode),
                                       np.roll(gcnn.corr2d(x, w, mode), (1, 2), (2, 3)), atol=1e-12)
        np.testing.assert_allclose(gcnn.corr2d(gcnn.rotate(x), w, mode),
                                   gcnn.rotate(gcnn.corr2d(x, gcnn.rotate(w, -1), mode)), atol=1e-12)
        assert np.abs(gcnn.corr2d(gcnn.rotate(x), w, mode) - gcnn.rotate(gcnn.corr2d(x, w, mode))).max() > 0.1
        # Statement 2: both layers are equivariant for every rotation.
        W1, W2 = rng.normal(size=(2, 3, 3, 3)), rng.normal(size=(3, 2, 4, 3, 3))
        f = gcnn.lift(x, W1, mode)
        for r in range(4):
            np.testing.assert_allclose(gcnn.lift(gcnn.rotate(x, r), W1, mode), gcnn.rotate_p4(f, r), atol=1e-12)
            np.testing.assert_allclose(gcnn.gconv(gcnn.rotate_p4(f, r), W2, mode),
                                       gcnn.rotate_p4(gcnn.gconv(f, W2, mode), r), atol=1e-11)
        if mode == "circular":
            t = lambda a: np.roll(a, (2, -1), (-2, -1))  # noqa: E731
            np.testing.assert_allclose(gcnn.lift(t(x), W1, mode), t(f), atol=1e-12)
            np.testing.assert_allclose(gcnn.gconv(t(f), W2, mode), t(gcnn.gconv(f, W2, mode)), atol=1e-11)
        # Statement 3: rotation r of the lifting layer is the planar correlation with the filter turned r times,
        # and the group correlation equals its definition summed over the group.
        for r in range(4):
            np.testing.assert_allclose(f[:, :, r], gcnn.corr2d(x, gcnn.rotate(W1, r), mode), atol=1e-12)
        np.testing.assert_allclose(gcnn.gconv(f, W2, mode), gcnn.gconv_direct(f, W2, mode), atol=1e-11)
print("group-correlation-equivariance: ok")
