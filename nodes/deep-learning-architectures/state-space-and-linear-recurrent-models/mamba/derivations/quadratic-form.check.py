# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/mamba/ (ai://NEW-mamba).
"""Numeric check of the derivation quadratic-form: the selective scan as a masked T x T matrix."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import mamba  # noqa: E402

rng = np.random.default_rng(2)
for disc in ("simple", "zoh"):
    for _ in range(10):
        Bsz, T, I, N = 2, int(rng.integers(1, 16)), 3, int(rng.integers(1, 6))
        dt = mamba.softplus(rng.normal(size=(Bsz, T, I)))
        A = -np.exp(rng.normal(size=(I, N)))
        Bm, C, x = rng.normal(size=(Bsz, T, N)), rng.normal(size=(Bsz, T, N)), rng.normal(size=(Bsz, T, I))
        dA, dB = mamba.discretize(dt, A, Bm, disc)
        y, _ = mamba.selective_scan(dA, dB, x, C)
        S = np.cumsum(dt, 1)  # (B, T, I)
        diff = S[:, :, None] - S[:, None, :]  # (B, t, s, I)
        mask = np.tril(np.ones((T, T)))[None, :, :, None]
        expo = (diff * mask)[..., None] * A  # masked before the exponential
        assert (expo <= 1e-12).all()
        M = np.einsum("btn,btsin,bsin->btsi", C, np.exp(expo) * mask[..., None], dB)
        assert np.allclose(np.triu(M[0, :, :, 0], 1), 0.0)
        assert np.allclose(np.einsum("btsi,bsi->bti", M, x), y, atol=1e-12), disc
print("quadratic-form: ok")
