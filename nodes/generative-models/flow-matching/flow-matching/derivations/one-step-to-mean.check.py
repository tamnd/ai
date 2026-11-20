# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/flow-matching/ (ai://NEW-flow-matching).
"""Numeric check of the derivation one-step-to-mean."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import flow_matching as fm  # noqa: E402

rng = np.random.default_rng(5)
means, s, w = np.array([[-2.0, 0.0], [2.0, 1.0], [0.0, 3.0]]), 0.5, np.array([0.2, 0.5, 0.3])
mean = w @ means
x = rng.standard_normal((50, 2))
for sm in (0.0, 0.2):

    def v(z, t, sm=sm):
        return fm.mixture_field(z, t, means, s, w, sm)

    assert np.allclose(v(x, np.zeros(len(x))), mean - (1 - sm) * x, atol=1e-12)
    assert np.allclose(fm.euler_sample(v, x, 1), mean + sm * x, atol=1e-12)
# with sigma_min = 0 every noise point lands on the mean: one Euler step collapses the distribution
assert np.ptp(fm.euler_sample(lambda z, t: fm.mixture_field(z, t, means, s, w), x, 1), 0).max() < 1e-12
print("one-step-to-mean: ok")
