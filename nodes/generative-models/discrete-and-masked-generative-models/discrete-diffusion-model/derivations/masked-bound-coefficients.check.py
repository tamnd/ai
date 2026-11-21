# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/discrete-diffusion-model/ (ai://NEW-discrete-diffusion-model).
"""Numeric check of the derivation masked-bound-coefficients."""

import math

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import discrete_diffusion as dd  # noqa: E402

for L in range(1, 9):
    for m in range(1, L + 1):
        nodes, wts = np.polynomial.legendre.leggauss(40)
        a, w = 0.5 * (nodes + 1.0), 0.5 * wts
        beta = float(np.sum(w * a ** (L - m) * (1.0 - a) ** (m - 1)))
        c = dd.mask_coefficient(m, L)
        assert abs(beta - c) < 1e-14 * max(1.0, 1.0 / c), (L, m, beta, c)
        assert abs(math.comb(L, m) * c - 1.0 / m) < 1e-14
    # the coefficients over all nonempty masks sum to the harmonic number H_L
    tot = sum(dd.mask_coefficient(int(mm.sum()), L) for mm in dd.all_masks(L))
    assert abs(tot - sum(1.0 / m for m in range(1, L + 1))) < 1e-12

rng = np.random.default_rng(2)
for L, K in ((2, 3), (3, 3), (4, 2)):
    cfg = dd.config(L, K, 6)
    den = dd.denoise_fn(dd.init(rng, cfg), cfg)
    for _ in range(3):
        x = rng.integers(0, K, size=L)
        exact = dd.exact_nelbo(den, x, K)
        for kind in ("linear", "cosine"):
            quad = dd.nelbo_by_quadrature(den, x, K, kind)
            assert abs(quad - exact) < 1e-10 * exact, (L, kind, quad, exact)
