# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/diffusion-model/ (ai://NEW-diffusion-model).
"""Numeric check of the derivation forward-marginal."""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import diffusion as dm  # noqa: E402

for cfg in (dm.config(d_data=1), dm.config(d_data=1, n_steps=1000, beta_1=1e-4, beta_T=0.02)):
    s = dm.schedule(cfg)
    x0 = 0.8
    m, v = x0, 0.0
    for t in range(1, cfg["n_steps"] + 1):
        tt = np.array([t])
        c = dm.forward_step(cfg, np.zeros((1, 1)), tt, np.zeros((1, 1)))[0, 0]
        a = dm.forward_step(cfg, np.ones((1, 1)), tt, np.zeros((1, 1)))[0, 0] - c
        g = dm.forward_step(cfg, np.zeros((1, 1)), tt, np.ones((1, 1)))[0, 0] - c
        m, v = a * m + c, a * a * v + g * g
        mean = dm.q_sample(cfg, np.array([[x0]]), tt, np.zeros((1, 1)))[0, 0]
        sd = dm.q_sample(cfg, np.array([[x0]]), tt, np.ones((1, 1)))[0, 0] - mean
        assert abs(m - mean) < 1e-12 and abs(v - sd**2) < 1e-12, t
        assert abs(v - (1 - s["abar"][t - 1])) < 1e-12
        assert s["abar"][t - 1] <= math.exp(-s["beta"][:t].sum()) + 1e-15
print("forward-marginal: ok")
