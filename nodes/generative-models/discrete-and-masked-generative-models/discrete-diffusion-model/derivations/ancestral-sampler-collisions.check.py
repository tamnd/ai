# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/discrete-diffusion-model/ (ai://NEW-discrete-diffusion-model).
"""Numeric check of the derivation ancestral-sampler-collisions."""

import itertools
import math

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import discrete_diffusion as dd  # noqa: E402


def p_no_collision(L, T, kind):
    ts = np.arange(T + 1) / T
    pi = np.diff(dd.alpha(ts, kind)) * -1.0  # pi_i = alpha(t_{i-1}) - alpha(t_i), i = 1..T
    assert abs(pi.sum() - 1.0) < 1e-12
    e = sum(math.prod(c) for c in itertools.combinations(pi, L))
    return math.factorial(L) * e


rng = np.random.default_rng(4)
cases = [dd.markov_data(3, 3), rng.dirichlet(np.ones(16)).reshape(2, 2, 2, 2), rng.dirichlet(np.ones(27)).reshape(3, 3, 3)]
for p in cases:
    K, L = p.shape[0], p.ndim
    den = dd.conditional_denoiser(p, K)
    marg = [p.sum(axis=tuple(j for j in range(L) if j != i)) for i in range(L)]
    prod = marg[0]
    for m in marg[1:]:
        prod = np.multiply.outer(prod, m)
    np.testing.assert_allclose(dd.sampler_distribution(den, L, K, 1), prod, atol=1e-14)
    for kind in ("linear", "cosine"):
        prev = None
        for T in (1, 2, 3, 4, 6, 8, 16, 32):
            pT = dd.sampler_distribution(den, L, K, T, kind)
            ok = p_no_collision(L, T, kind)
            if kind == "linear":
                closed = math.factorial(T) / (math.factorial(T - L) * T**L) if T >= L else 0.0
                assert abs(ok - closed) < 1e-12
            rest = pT - ok * p
            assert rest.min() > -1e-12
            assert abs(rest.sum() - (1.0 - ok)) < 1e-12
            tv = 0.5 * np.abs(pT - p).sum()
            assert tv <= 1.0 - ok + 1e-12, (kind, T, tv, 1 - ok)
            assert 1.0 - ok <= L * (L - 1) / (2 * T) + 1e-12 or kind == "cosine"
            if prev is not None:
                assert tv < prev + 1e-12
            prev = tv
        assert prev < 0.1
