# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/latent-diffusion-model/ (ai://NEW-latent-diffusion-model).
"""Numeric check of the derivation vq-straight-through."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import latent_diffusion as ld  # noqa: E402

rng = np.random.default_rng(1)
beta, h = 0.25, 1e-6
cb = rng.standard_normal((7, 3))
cb[6] = 50.0  # a code that no vector is near
z = rng.standard_normal((4, 5, 3))
g = rng.standard_normal(z.shape)
idx, _ = ld.quantize(z, cb)
assert 6 not in idx
n = idx.size


def commit(ze):
    return beta * ((ze - cb[idx]) ** 2).sum(-1).mean()


def codebook_term(c):
    return ((z - c[idx]) ** 2).sum(-1).mean()


g_e, g_cb = ld.vq_grads(z, cb, idx, g, beta)
fd_e = np.zeros_like(z)
for i in np.ndindex(z.shape):
    d = np.zeros_like(z)
    d[i] = h
    fd_e[i] = (commit(z + d) - commit(z - d)) / (2 * h)
assert np.allclose(g_e, g + fd_e, atol=1e-8)
assert np.allclose(g_e, g + 2 * beta * (z - cb[idx]) / n, atol=1e-14)
fd_c = np.zeros_like(cb)
for i in np.ndindex(cb.shape):
    d = np.zeros_like(cb)
    d[i] = h
    fd_c[i] = (codebook_term(cb + d) - codebook_term(cb - d)) / (2 * h)
assert np.allclose(g_cb, fd_c, atol=1e-8)
assert np.all(g_cb[6] == 0.0)
_, g_cb2 = ld.vq_grads(z, cb, idx, 10.0 * rng.standard_normal(z.shape), beta)
assert np.array_equal(g_cb, g_cb2)
