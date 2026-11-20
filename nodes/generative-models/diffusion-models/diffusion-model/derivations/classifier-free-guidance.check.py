# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/diffusion-model/ (ai://NEW-diffusion-model).
"""Numeric check of the derivation classifier-free-guidance."""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import diffusion as dm  # noqa: E402

rng = np.random.default_rng(4)
means, s, prior = np.array([[-2.0, 0.0], [1.5, 1.0], [0.0, -2.0]]), 0.5, np.array([0.2, 0.5, 0.3])
for abar in (0.9, 0.4, 0.05):
    x = rng.standard_normal((5, 2)) * 1.5
    eu = dm.mixture_eps(x, abar, means, s, prior)
    for c in range(3):
        ec = dm.mixture_eps(x, abar, means[c : c + 1], s, [1.0])

        def log_class(y, c=c, abar=abar):
            return (math.log(prior[c]) + dm.mixture_log_density(y, abar, means[c : c + 1], s, [1.0])
                    - dm.mixture_log_density(y, abar, means, s, prior))  # fmt: skip

        grad = np.stack([(log_class(x + 1e-5 * e) - log_class(x - 1e-5 * e)) / 2e-5 for e in np.eye(2)], 1)
        for w in (0.0, 0.5, 2.0, 7.5):
            want = ec - w * math.sqrt(1 - abar) * grad
            assert np.allclose(dm.guided_eps(ec, eu, w), want, rtol=1e-5, atol=1e-7), (abar, c, w)
print("classifier-free-guidance: ok")
