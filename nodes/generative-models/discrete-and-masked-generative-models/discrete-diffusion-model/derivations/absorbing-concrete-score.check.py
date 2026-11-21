# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/discrete-diffusion-model/ (ai://NEW-discrete-diffusion-model).
"""Numeric check of the derivation absorbing-concrete-score."""

import itertools

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import discrete_diffusion as dd  # noqa: E402

rng = np.random.default_rng(5)
for p in (rng.dirichlet(np.ones(27)).reshape(3, 3, 3), dd.markov_data(3, 3)):
    K, L = p.shape[0], p.ndim
    den = dd.conditional_denoiser(p, K)
    for a in (0.1, 0.45, 0.8):
        brute = {}
        for x in itertools.product(range(K), repeat=L):
            for m in itertools.product([False, True], repeat=L):
                z = tuple(K if mi else xi for xi, mi in zip(x, m, strict=True))
                k = sum(m)
                brute[z] = brute.get(z, 0.0) + p[x] * (1 - a) ** k * a ** (L - k)
        for z, pz in brute.items():
            assert abs(dd.masked_marginal(p, np.array(z), a, K) - pz) < 1e-14
        for z in brute:
            zz = np.array(z)
            pr = den(zz)
            for i in np.flatnonzero(zz == K):
                ratios = []
                for y in range(K):
                    zy = zz.copy()
                    zy[i] = y
                    r = dd.masked_marginal(p, zy, a, K) / dd.masked_marginal(p, zz, a, K)
                    assert abs(r - a / (1 - a) * pr[i, y]) < 1e-12 * max(1.0, r)
                    ratios.append(r)
                assert abs(sum(ratios) - a / (1 - a)) < 1e-12 * max(1.0, a / (1 - a))
