# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/diffusion-model/ (ai://NEW-diffusion-model).
"""Numeric check of the derivation tweedie-score."""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import diffusion as dm  # noqa: E402

rng = np.random.default_rng(2)
means, s, w = np.array([[-2.0, 0.0], [2.0, 1.0]]), 0.4, np.array([0.3, 0.7])
for abar in (0.95, 0.5, 0.05):
    k = rng.choice(2, 4, p=w)
    x = math.sqrt(abar) * (means[k] + s * rng.standard_normal((4, 2))) + math.sqrt(1 - abar) * rng.standard_normal(
        (4, 2)
    )
    eps = dm.mixture_eps(x, abar, means, s, w)
    num = np.stack([(dm.mixture_log_density(x + 1e-5 * e, abar, means, s, w)
                     - dm.mixture_log_density(x - 1e-5 * e, abar, means, s, w)) / 2e-5 for e in np.eye(2)], 1)  # fmt: skip
    assert np.allclose(eps, -math.sqrt(1 - abar) * num, rtol=1e-6, atol=1e-8)
    n = 400_000
    x0 = means[rng.choice(2, n, p=w)] + s * rng.standard_normal((n, 2))
    for i in range(len(x)):
        # E[eps | x_t] with x_0 drawn from the data and weights q(x_t | x_0)
        lw = dm.log_normal(np.repeat(x[i : i + 1], n, 0), math.sqrt(abar) * x0, 1 - abar)
        wt = np.exp(lw - lw.max())
        e_given = (x[i] - math.sqrt(abar) * x0) / math.sqrt(1 - abar)
        mc = (wt[:, None] * e_given).sum(0) / wt.sum()
        assert np.allclose(mc, eps[i], atol=0.02), (abar, i, mc, eps[i])
    x0 = means[rng.choice(2, 20_000, p=w)] + s * rng.standard_normal((20_000, 2))
    e = rng.standard_normal(x0.shape)
    xt = math.sqrt(abar) * x0 + math.sqrt(1 - abar) * e
    pred = dm.mixture_eps(xt, abar, means, s, w)
    base = np.mean(np.sum((e - pred) ** 2, 1))
    for shift in (np.array([0.05, 0.0]), np.array([0.0, -0.05])):
        assert np.mean(np.sum((e - pred - shift) ** 2, 1)) > base
print("tweedie-score: ok")
