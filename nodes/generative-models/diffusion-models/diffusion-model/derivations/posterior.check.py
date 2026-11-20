# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/diffusion-model/ (ai://NEW-diffusion-model).
"""Numeric check of the derivation posterior."""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import diffusion as dm  # noqa: E402

rng = np.random.default_rng(0)
grid = np.linspace(-4, 4, 81)[:, None]
for cfg in (dm.config(d_data=1), dm.config(d_data=1, n_steps=50, beta_1=1e-3, beta_T=0.2)):
    s = dm.schedule(cfg)
    for t in range(2, cfg["n_steps"] + 1):
        x0, xt = rng.standard_normal((1, 1)), rng.standard_normal((1, 1))
        m, v = dm.posterior(cfg, xt, x0, np.array([t]))
        b, abp = s["beta"][t - 1], s["abar_prev"][t - 1]
        joint = dm.log_normal(np.repeat(xt, len(grid), 0), math.sqrt(1 - b) * grid, b)
        joint = joint + dm.log_normal(grid, math.sqrt(abp) * x0[0, 0], 1 - abp)
        diff = joint - dm.log_normal(grid, m[0, 0], v[0, 0])
        assert np.ptp(diff) < 1e-8, t
        assert v[0, 0] < b
print("posterior: ok")
