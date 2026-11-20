# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/flow-matching/ (ai://NEW-flow-matching).
"""Numeric check of the derivation marginal-field."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import flow_matching as fm  # noqa: E402

rng = np.random.default_rng(1)
means, s, w = np.array([[-2.0, 0.0], [2.0, 1.0]]), 0.4, np.array([0.3, 0.7])
for sm in (0.0, 0.1):
    c = 1 - sm
    for t in (0.2, 0.5, 0.8):
        k = rng.choice(2, 5, p=w)
        x = t * (means[k] + s * rng.standard_normal((5, 2))) + (1 - c * t) * rng.standard_normal((5, 2))
        tt = np.full(len(x), t)
        u = fm.mixture_field(x, tt, means, s, w, sm)

        def p(z, tau, sm=sm):
            return np.exp(fm.mixture_log_density(z, np.full(len(z), tau), means, s, w, sm))

        # continuity: d/dt p_t + div(p_t u_t) = 0, by central differences
        h = 1e-4
        dpdt = (p(x, t + h) - p(x, t - h)) / (2 * h)
        div = sum(
            (
                p(x + h * e, t) * fm.mixture_field(x + h * e, tt, means, s, w, sm)[:, i]
                - p(x - h * e, t) * fm.mixture_field(x - h * e, tt, means, s, w, sm)[:, i]
            )
            / (2 * h)
            for i, e in enumerate(np.eye(2))
        )
        assert np.allclose(dpdt + div, 0, atol=1e-6), (t, dpdt + div)
        # u_t(x) = E[x_1 - c x_0 | x_t = x]: weight data draws by p_t(x | x_1)
        n = 400_000
        x1 = means[rng.choice(2, n, p=w)] + s * rng.standard_normal((n, 2))
        a = 1 - c * t
        for i in range(len(x)):
            lw = fm.log_normal(np.repeat(x[i : i + 1], n, 0), t * x1, a**2)
            wt = np.exp(lw - lw.max())
            x0 = (x[i] - t * x1) / a
            mc = (wt[:, None] * (x1 - c * x0)).sum(0) / wt.sum()
            assert np.allclose(mc, u[i], atol=0.03), (t, i, mc, u[i])
print("marginal-field: ok")
