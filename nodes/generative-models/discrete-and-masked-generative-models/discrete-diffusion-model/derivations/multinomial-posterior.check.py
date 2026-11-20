# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/discrete-diffusion-model/ (ai://NEW-discrete-diffusion-model).
"""Numeric check of the derivation multinomial-posterior."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import discrete_diffusion as dd  # noqa: E402

rng = np.random.default_rng(0)
for K in (2, 3, 5, 8):
    betas = rng.uniform(0.05, 0.6, size=6)
    Qs = [dd.uniform_q(b, K) for b in betas]
    Qbar = dd.cumulative(Qs)
    abar = np.cumprod(1.0 - betas)
    for t in range(1, len(betas) + 1):
        np.testing.assert_allclose(Qbar[t], dd.uniform_qbar(abar[t - 1], K), atol=1e-14)
    for t in range(2, len(betas) + 1):
        Qt, Qp = Qs[t - 1], Qbar[t - 1]
        for x0 in range(K):
            for xt in range(K):
                joint = Qp[x0, :] * Qt[:, xt]
                got = dd.posterior(dd.one_hot(xt, K), dd.one_hot(x0, K), Qt, Qp)
                np.testing.assert_allclose(got, joint / joint.sum(), atol=1e-14)
        for _ in range(5):
            p0 = rng.dirichlet(np.ones(K))
            xt = dd.one_hot(int(rng.integers(K)), K)
            np.testing.assert_allclose(dd.posterior(xt, p0, Qt, Qp),
                                       dd.multinomial_posterior(xt, p0, 1.0 - betas[t - 1], abar[t - 2], K),
                                       atol=1e-14)  # fmt: skip
