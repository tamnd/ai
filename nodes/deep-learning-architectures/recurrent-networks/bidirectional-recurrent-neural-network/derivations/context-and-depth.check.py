# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/bidirectional-recurrent-neural-network/ (ai://NEW-bidirectional-recurrent-neural-network).
"""Numeric check of the derivation context-and-depth: which inputs each output reads, and the longest chain of cells."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import birnn  # noqa: E402

rng = np.random.default_rng(0)
for cell in ("lstm", "tanh"):
    for bi in (True, False):
        for L in (1, 2, 3):
            cfg = birnn.config(2, 3, L, 3, cell, bi)
            params = birnn.init(rng, cfg, scale=0.5)
            T = 7
            x = rng.normal(size=(1, T, 2))
            base = birnn.forward(params, cfg, x)[0][0]
            for s in range(T):
                y = x.copy()
                y[0, s] += 0.5
                moved = np.abs(birnn.forward(params, cfg, y)[0][0] - base).max(axis=-1) > 1e-12
                assert moved.tolist() == [bi or t >= s for t in range(T)], (cell, bi, L, s)


def depth(T, L, bi):
    """The longest chain of cell steps, by dynamic programming over the dependency graph."""
    f = np.zeros((L + 1, T + 2), dtype=int)  # finish time of the forward cell (l, t), 1-based, with borders
    b = np.zeros((L + 1, T + 2), dtype=int)
    for l in range(1, L + 1):
        below = np.maximum(f[l - 1], b[l - 1]) if bi else f[l - 1]
        for t in range(1, T + 1):
            f[l, t] = max(f[l, t - 1], below[t]) + 1
        if bi:
            for t in range(T, 0, -1):
                b[l, t] = max(b[l, t + 1], below[t]) + 1
    return int(max(f.max(), b.max()))


for T in range(1, 13):
    for L in range(1, 13):
        assert depth(T, L, False) == T + L - 1
        assert depth(T, L, True) == L * T
print("ok")
