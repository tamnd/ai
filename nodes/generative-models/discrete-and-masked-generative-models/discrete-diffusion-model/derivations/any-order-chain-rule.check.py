# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/discrete-diffusion-model/ (ai://NEW-discrete-diffusion-model).
"""Numeric check of the derivation any-order-chain-rule."""

import itertools
import math

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import discrete_diffusion as dd  # noqa: E402

for L in range(1, 6):
    count = {}
    for sigma in itertools.permutations(range(L)):
        for k, ell in enumerate(sigma):
            M = frozenset(range(L)) - frozenset(sigma[:k])
            count[(M, ell)] = count.get((M, ell), 0) + 1
    for (M, ell), c in count.items():
        assert ell in M
        assert abs(c / math.factorial(L) - dd.mask_coefficient(len(M), L)) < 1e-15

rng = np.random.default_rng(3)
cases = [dd.markov_data(L, 3) for L in (2, 3, 4)]
cases += [rng.dirichlet(np.ones(K**L)).reshape((K,) * L) for L, K in ((2, 4), (3, 3), (4, 2))]
for p in cases:
    K, L = p.shape[0], p.ndim
    den = dd.conditional_denoiser(p, K)
    exp_bound = 0.0
    for x in itertools.product(range(K), repeat=L):
        b = dd.exact_nelbo(den, np.array(x), K)
        assert abs(b + math.log(p[x])) < 1e-12 * max(1.0, b), (x, b)
        exp_bound += p[x] * b
    # a perturbed denoiser: mix the true conditionals with the uniform distribution
    def worse(z, den=den, K=K):
        return np.where((np.asarray(z) == K)[:, None], 0.8 * den(z) + 0.2 / K, den(z))

    exp_worse = sum(p[x] * dd.exact_nelbo(worse, np.array(x), K) for x in itertools.product(range(K), repeat=L))
    assert exp_worse > exp_bound + 1e-6
