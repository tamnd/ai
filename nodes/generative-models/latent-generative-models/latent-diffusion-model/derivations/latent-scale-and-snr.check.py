# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/latent-diffusion-model/ (ai://NEW-latent-diffusion-model).
"""Numeric check of the derivation latent-scale-and-snr."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import latent_diffusion as ld  # noqa: E402

rng = np.random.default_rng(3)
abar = ld.alpha_bars(1000)
assert np.all(np.diff(abar) < 0)
z0 = rng.standard_normal((200000, 1))
for t in (50, 300, 700):
    for k in (0.5, 1.0, 7.0):
        sig = np.sqrt(abar[t]) * k * z0
        noise = ld.q_sample(k * z0, np.full(len(z0), abar[t]), np.zeros_like(z0)) - sig
        assert np.allclose(noise, 0.0)
        eps = rng.standard_normal(z0.shape)
        zt = ld.q_sample(k * z0, np.full(len(z0), abar[t]), eps)
        emp = sig.var() / (zt - sig).var()
        assert abs(emp / ld.snr(abar[t], k**2) - 1) < 0.02
        assert np.isclose(ld.snr(abar[t], k**2), k**2 * ld.snr(abar[t]))


def crossing(v):
    return int(np.argmax(ld.snr(abar, v) < 1.0))


assert (crossing(1.0), crossing(49.0), crossing(1 / 49)) == (259, 620, 40)
assert crossing(49.0) - crossing(1.0) > 1000 / 3
assert abs(1 / np.sqrt(abar[-1]) - 157.4) < 0.1
