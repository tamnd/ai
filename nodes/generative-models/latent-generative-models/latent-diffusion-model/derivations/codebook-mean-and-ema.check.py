# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/latent-diffusion-model/ (ai://NEW-latent-diffusion-model).
"""Numeric check of the derivation codebook-mean-and-ema."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import latent_diffusion as ld  # noqa: E402

rng = np.random.default_rng(2)
z = rng.standard_normal((300, 2)) + 4.0 * rng.integers(0, 3, (300, 1))
cb = np.vstack([z[:3], [[100.0, 100.0]]])
idx, _ = ld.quantize(z, cb)
new = ld.kmeans_step(z, cb)
for i in range(3):
    zi = z[idx == i]
    assert np.allclose(new[i], zi.mean(0), atol=1e-14)
    best = ((zi - new[i]) ** 2).sum()
    for _ in range(20):
        assert ((zi - new[i] - 0.01 * rng.standard_normal(2)) ** 2).sum() > best
assert np.array_equal(new[3], cb[3])
for gamma in (0.0, 0.5, 0.99):
    s = ld.ema_init(cb)
    for t in range(50):
        s = ld.ema_update(s, z, idx, gamma)
        assert np.allclose(s["e"], new, atol=1e-12)
        assert np.allclose(s["N"][:3], (1 - gamma ** (t + 1)) * np.bincount(idx, minlength=4)[:3], atol=1e-12)
