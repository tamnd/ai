# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/latent-diffusion-model/ (ai://NEW-latent-diffusion-model).
"""Numeric check of the derivation compression-ratio."""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import latent_diffusion as ld  # noqa: E402

rng = np.random.default_rng(0)
for f, c in ((2, 1), (4, 2), (8, 3)):
    x = rng.standard_normal((50, 16 * f))
    z = ld.encode(ld.fit_autoencoder(x, f, c), x, f)
    assert z.shape == (50, 16, c)
    assert x.size / z.size == f / c
img = np.zeros((256, 256, 3))
pos = img.reshape(32, 8, 32, 8, 3).transpose(0, 2, 1, 3, 4).reshape(32, 32, -1).shape[:2]
assert img.size / (pos[0] * pos[1] * 4) == 3 * 8**2 / 4 == 48
assert abs(128 * 128 * 3 * 8 / (32 * 32 * math.log2(512)) - 24 * 16 / 9) < 1e-12
assert abs(24 * 16 / 9 - 42.67) < 0.005
n2 = 64**2 + 32**2
assert 256 * 256 * 3 / n2 == 38.4
assert abs(256 * 256 * 24 / (n2 * 9) - 34.13) < 0.005
