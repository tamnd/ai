# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/flow-matching/ (ai://NEW-flow-matching).
"""Numeric check of the derivation conditional-ot-field."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import flow_matching as fm  # noqa: E402

rng = np.random.default_rng(0)
for sm in (0.0, 0.1):
    cfg = fm.config(d_data=2, sigma_min=sm)
    x1 = np.array([[1.5, -0.5]])
    x0 = rng.standard_normal((200_000, 2))
    for t in (0.0, 0.3, 0.7, 1.0):
        xt = fm.path(cfg, x0, x1, np.full(len(x0), t))
        sd = 1 - (1 - sm) * t
        # the path pushes N(0, I) to N(t x_1, sd^2 I)
        assert np.allclose(xt.mean(0), t * x1[0], atol=0.01)
        assert np.allclose(xt.std(0), sd, atol=0.01)
        h = 1e-6
        tp, tm = np.full(4, min(t + h, 1)), np.full(4, max(t - h, 0))
        dx = (fm.path(cfg, x0[:4], x1, tp) - fm.path(cfg, x0[:4], x1, tm)) / (tp - tm)[:, None]
        assert np.allclose(dx, fm.target(cfg, x0[:4], x1), atol=1e-6)
        if sd > 0:
            u = fm.cond_field(cfg, fm.path(cfg, x0[:4], x1, np.full(4, t)), x1, np.full(4, t))
            assert np.allclose(u, fm.target(cfg, x0[:4], x1), atol=1e-9)
print("conditional-ot-field: ok")
