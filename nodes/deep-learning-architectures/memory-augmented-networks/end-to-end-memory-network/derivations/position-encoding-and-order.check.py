# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/end-to-end-memory-network/ (ai://NEW-end-to-end-memory-network).
"""Numeric check of the derivation position-encoding-and-order."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import memn2n  # noqa: E402

for J in range(1, 9):
    for d in range(1, 9):
        lw = memn2n.position_encoding(J, d)
        a = (np.arange(1, J + 1) / J)[:, None]
        b = (np.arange(1, d + 1) / d)[None, :]
        # Statement 1: the bilinear form, the range, the slope, and the middle word.
        assert np.allclose(lw, (1 - a) * (1 - b) + a * b)
        assert lw.min() >= -1e-12 and lw.max() <= 1 + 1e-12
        if J > 1:
            assert np.allclose(np.diff(lw, axis=0), (2 * b - 1) / J)
        if J % 2 == 0:
            assert np.allclose(lw[J // 2 - 1], 0.5)
        # Statement 2: the difference of two rows.
        for j in range(J):
            for jj in range(J):
                assert np.allclose(lw[j] - lw[jj], (j - jj) / J * (2 * b[0] - 1))

rng = np.random.default_rng(13)
E = rng.normal(size=(9, 6))
E[0] = 0.0
lw = memn2n.position_encoding(5, 6)
s = np.array([3, 7, 2, 5, 0])
t = s.copy()
t[[1, 3]] = t[[3, 1]]
diff = memn2n.encode(E, t, lw) - memn2n.encode(E, s, lw)
# Word 5 moves to position 2 (1-based) and word 7 to position 4: j - j' = 2 - 4 with a = E[5], b = E[7].
want = (2 - 4) / 5 * (2 * np.arange(1, 7) / 6 - 1) * (E[5] - E[7])
assert np.allclose(diff, want)
assert abs(diff[2]) < 1e-12  # k = d/2 = 3
assert np.allclose(memn2n.encode(E, t), memn2n.encode(E, s))
print("position-encoding-and-order: ok")
