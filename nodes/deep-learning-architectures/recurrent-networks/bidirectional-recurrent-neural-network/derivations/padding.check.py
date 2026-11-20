# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/bidirectional-recurrent-neural-network/ (ai://NEW-bidirectional-recurrent-neural-network).
"""Numeric check of the derivation padding: per-sequence reversal, the masked reverse scan and the unmasked one."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import birnn  # noqa: E402


def reverse_scan(cell, W, U, b, x, mask=None):
    """The backward direction from t = T down to 1, the state set to zero at the padded steps if a mask is given."""
    B, T, _ = x.shape
    H = U.shape[0]
    h, c = np.zeros((B, H)), np.zeros((B, H))
    hs = np.zeros((B, T, H))
    for t in reversed(range(T)):
        z = x[:, t] @ W + h @ U + b
        if cell == "tanh":
            h = np.tanh(z)
        else:
            i, f, g, o = (z[:, k * H : (k + 1) * H] for k in range(4))
            c = birnn.sigmoid(f) * c + birnn.sigmoid(i) * np.tanh(g)
            h = birnn.sigmoid(o) * np.tanh(c)
        if mask is not None:
            h, c = h * mask[:, t], c * mask[:, t]
        hs[:, t] = h
    return hs


rng = np.random.default_rng(0)
for trial in range(20):
    cell = ("lstm", "tanh")[trial % 2]
    B, T, I, H = 4, int(rng.integers(2, 9)), 3, 4
    lengths = rng.integers(1, T + 1, B)
    lengths[0] = T
    lengths[1] = max(1, T - 1)
    G = 4 if cell == "lstm" else 1
    W, U, b = rng.uniform(-0.7, 0.7, (I, G * H)), rng.uniform(-0.7, 0.7, (H, G * H)), rng.uniform(-0.7, 0.7, G * H)
    x = rng.normal(size=(B, T, I))
    mask = birnn.mask_of(lengths, T)
    hs, _ = birnn.run_direction(cell, W, U, b, birnn.reverse_padded(x, lengths))
    reversal = birnn.reverse_padded(hs, lengths) * mask
    masked = reverse_scan(cell, W, U, b, x, mask)
    plain = reverse_scan(cell, W, U, b, x)
    np.testing.assert_allclose(reversal, masked, rtol=1e-13, atol=1e-14)
    for k, n in enumerate(lengths):
        alone = reverse_scan(cell, W, U, b, x[k : k + 1, :n])[0]
        np.testing.assert_allclose(reversal[k, :n], alone, rtol=1e-13, atol=1e-14)
        if n < T:
            assert (np.abs(plain[k, :n] - alone).max(axis=-1) > 1e-9).all()  # every real step reads the padding
print("ok")
