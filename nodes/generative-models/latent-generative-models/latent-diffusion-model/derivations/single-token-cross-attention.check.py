# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/latent-diffusion-model/ (ai://NEW-latent-diffusion-model).
"""Numeric check of the derivation single-token-cross-attention."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import latent_diffusion as ld  # noqa: E402

rng = np.random.default_rng(4)
for M in (1, 2):
    cfg = ld.config(3, 2, 7, 4, M, 5, 3)
    p = ld.init(rng, cfg)
    p = {k: v + 0.3 * rng.standard_normal(v.shape) for k, v in p.items()}
    z, t, y = rng.standard_normal((6, 3, 2)), np.array([3, 90, 400, 999, 10, 700]), np.array([0, 1, 2, 3, 1, 0])
    out, cache = ld._forward(p, cfg, z, t, y)
    q = dict(p, Wq=rng.standard_normal(p["Wq"].shape), Wk=rng.standard_normal(p["Wk"].shape))
    out2 = ld.forward(q, cfg, z, t, y)
    if M == 1:
        assert np.allclose(cache[7], 1.0, atol=0)
        h = cache[2]
        bias = p["ctx"][y, 0] @ p["Wv"]
        assert np.allclose(out, z.reshape(6, -1) + np.tanh(h + bias) @ p["W2"] + p["b2"], atol=1e-14)
        assert np.allclose(out, out2, atol=1e-14)
    else:
        assert not np.allclose(out, out2, atol=1e-6)
