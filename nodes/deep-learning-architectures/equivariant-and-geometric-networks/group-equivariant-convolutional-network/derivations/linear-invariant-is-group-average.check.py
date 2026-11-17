# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/group-equivariant-convolutional-network/ (ai://NEW-group-equivariant-convolutional-network).
"""Numeric check of the derivation linear-invariant-is-group-average."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import gcnn  # noqa: E402

rng = np.random.default_rng(3)
n = 6
x = rng.normal(size=(n, n))
rotations = [lambda a, r=r: gcnn.rotate(a, r) for r in range(4)]
shifts = [lambda a, i=i, j=j: np.roll(a, (i, j), (0, 1)) for i in range(n) for j in range(n)]

for group in (rotations, shifts):
    w0 = rng.normal(size=(n, n))
    w = np.mean([g(w0) for g in group], axis=0)  # the averaged weight: f(x) = <w, x> is invariant
    for g in group:
        assert abs(np.sum(w * g(x)) - np.sum(w * x)) < 1e-12
    avg = np.mean([g(x) for g in group], axis=0)
    assert abs(np.sum(w * avg) - np.sum(w * x)) < 1e-12  # Statement 1

# Statement 2: under the cyclic shifts the average is the constant image at the mean.
np.testing.assert_allclose(np.mean([g(x) for g in shifts], axis=0), np.full((n, n), x.mean()), atol=1e-14)

# Statement 3: y = x - A x + A z has the same rotation average as z, yet the G-CNN separates x and y while staying
# invariant, so its invariant is not a function of the average.
avg_r = lambda a: np.mean([gcnn.rotate(a, r) for r in range(4)], axis=0)  # noqa: E731
z = rng.normal(size=(n, n))
y = x - avg_r(x) + avg_r(z)
np.testing.assert_allclose(avg_r(y), avg_r(z), atol=1e-14)
cfg = gcnn.config(1, 3, 3, 3, 2)
params = gcnn.init(rng, cfg)
out = lambda a: gcnn.forward(params, a[None, None], cfg)  # noqa: E731
assert np.abs(out(y) - out(z)).max() > 1e-3
np.testing.assert_allclose(out(gcnn.rotate(y)), out(y), atol=1e-12)
print("linear-invariant-is-group-average: ok")
