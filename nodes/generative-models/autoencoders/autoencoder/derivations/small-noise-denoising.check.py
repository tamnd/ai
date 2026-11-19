# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/autoencoder/ (ai://NEW-autoencoder).
"""Numeric check of the derivation small-noise-denoising."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import autoencoder as ae  # noqa: E402

nodes, weights = np.polynomial.hermite_e.hermegauss(40)  # for the weight exp(-t^2 / 2)
weights = weights / weights.sum()
T1, T2 = np.meshgrid(nodes, nodes, indexing="ij")
EPS = np.stack([T1.ravel(), T2.ravel()], 1)  # standard normal points in R^2
WTS = np.outer(weights, weights).ravel()


def expected_error(r, x, sigma):
    return WTS @ np.sum((x - r(x + sigma * EPS)) ** 2, 1)


rng = np.random.default_rng(1)
cfg = ae.from_widths([2, 3, 2])  # 2-3-2 with a sigmoid hidden layer, a linear code and a sigmoid output
p = ae.init(rng, cfg, gain=1.5)
p = ae.tree_map(lambda a: a + 0.3 * rng.normal(size=a.shape), p)


def r(x):
    return ae.reconstruct(p, cfg, np.atleast_2d(x))


x = np.array([0.3, 0.7])
e0 = x - r(x)[0]
J = ae.jacobian(p, cfg, x, upto="recon")
np.testing.assert_allclose(J, np.stack([(r(x + 1e-6 * e)[0] - r(x - 1e-6 * e)[0]) / 2e-6 for e in np.eye(2)], 1),
                           rtol=1e-6, atol=1e-10)  # fmt: skip
h = 1e-3
lap = sum(r(x + h * e)[0] - 2 * r(x)[0] + r(x - h * e)[0] for e in np.eye(2)) / h**2
first = np.sum(J**2) - e0 @ lap
assert abs(first) > 1e-3  # the second-order term is not trivially 0 here
gaps = []
for sigma in (0.08, 0.04, 0.02):
    gap = expected_error(r, x, sigma) - (e0 @ e0 + sigma**2 * first)
    gaps.append(abs(gap))
ratios = [gaps[i] / gaps[i + 1] for i in range(2)]
assert all(12 < q < 20 for q in ratios), ratios  # the remainder is O(sigma^4): halving sigma divides it by 16

# statement 2: affine reconstructions, exactly
A, c = rng.normal(size=(2, 2)), rng.normal(size=2)


def affine(z):
    return z @ A.T + c


for sigma in (0.1, 1.0, 3.0):
    np.testing.assert_allclose(expected_error(affine, x, sigma), np.sum((x - affine(x)) ** 2) + sigma**2 * np.sum(A**2),
                               rtol=1e-12)  # fmt: skip
    eps = sigma * rng.normal(size=(200000, 2))
    errs = np.sum((x - affine(x + eps)) ** 2, 1)
    want = np.sum((x - affine(x)) ** 2) + sigma**2 * np.sum(A**2)
    assert abs(errs.mean() - want) < 4 * errs.std() / np.sqrt(len(errs))
print("small-noise-denoising: ok")
