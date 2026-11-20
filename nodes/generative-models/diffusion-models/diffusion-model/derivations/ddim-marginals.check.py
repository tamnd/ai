# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/diffusion-model/ (ai://NEW-diffusion-model).
"""Numeric check of the derivation ddim-marginals."""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import diffusion as dm  # noqa: E402

cfg = dm.config(d_data=1)
ab0 = np.concatenate([[1.0], dm.schedule(cfg)["abar"]])
x0 = 0.6
one = np.ones((1, 1))
for t in range(1, cfg["n_steps"] + 1):
    for s in range(t):
        tt, ss = np.array([t]), np.array([s])
        for eta in (0.0, 0.5, 1.0):
            # x_s = A eps + c + g z with eps the true noise of x_t = sqrt(abar_t) x0 + sqrt(1 - abar_t) eps
            def step(e, z, tt=tt, ss=ss, eta=eta, t=t):
                xt = math.sqrt(ab0[t]) * x0 + math.sqrt(1 - ab0[t]) * e
                return dm.ddim_step(cfg, xt * one, tt, ss, e * one, z * one, eta)[0, 0]

            c = step(0.0, 0.0)
            a, g = step(1.0, 0.0) - c, step(0.0, 1.0) - c
            assert abs(c - math.sqrt(ab0[s]) * x0) < 1e-12
            assert abs(a * a + g * g - (1 - ab0[s])) < 1e-12, (t, s, eta)
rng = np.random.default_rng(3)
s = dm.schedule(cfg)
t = np.arange(2, cfg["n_steps"] + 1)
xt, eps, z = (rng.standard_normal((len(t), 2)) for _ in range(3))
assert np.allclose(dm.ddim_sigma(cfg, t, t - 1, 1.0)[:, 0] ** 2, s["beta_tilde"][t - 1], rtol=1e-12)
assert np.allclose(dm.ddim_step(cfg, xt, t, t - 1, eps, z, 1.0), dm.ddpm_step(cfg, xt, t, eps, z, "beta_tilde"))
print("ddim-marginals: ok")
