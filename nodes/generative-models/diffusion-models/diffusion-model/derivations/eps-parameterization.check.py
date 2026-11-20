# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/diffusion-model/ (ai://NEW-diffusion-model).
"""Numeric check of the derivation eps-parameterization."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import diffusion as dm  # noqa: E402

rng = np.random.default_rng(1)
for cfg in (dm.config(), dm.config(n_steps=1000, beta_1=1e-4, beta_T=0.02)):
    s = dm.schedule(cfg)
    T = cfg["n_steps"]
    t = np.arange(2, T + 1)
    x0, eps, guess = (rng.standard_normal((T - 1, 3)) for _ in range(3))
    xt = dm.q_sample(cfg, x0, t, eps)
    mq, vq = dm.posterior(cfg, xt, x0, t)
    assert np.allclose(dm.reverse_mean(cfg, xt, t, eps), mq, rtol=1e-9, atol=1e-10)
    b, a, ab = s["beta"][t - 1], s["alpha"][t - 1], s["abar"][t - 1]
    for var in ("beta", "beta_tilde"):
        sig = dm.sigma2(cfg, t, var)[:, 0]
        kl = dm.gauss_kl(mq, vq, dm.reverse_mean(cfg, xt, t, guess), sig[:, None])
        r = vq[:, 0] / sig
        want = b**2 / (2 * sig * a * (1 - ab)) * np.sum((eps - guess) ** 2, 1) + 1.5 * (r - 1 - np.log(r))
        assert np.allclose(kl, want, rtol=1e-8), var
    if T == 1000:
        w = b**2 / (2 * b * a * (1 - ab))
        print(f"weight with sigma^2 = beta: t = 2: {w[0]:.4f}, t = T: {w[-1]:.5f}, ratio {w[0] / w[-1]:.1f}")
print("eps-parameterization: ok")
