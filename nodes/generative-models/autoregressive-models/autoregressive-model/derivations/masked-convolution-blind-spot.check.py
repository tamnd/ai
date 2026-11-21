# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/autoregressive-model/ (ai://NEW-autoregressive-model).
"""Numeric check of the derivation masked-convolution-blind-spot."""

import pathlib
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import autoregressive as ar  # noqa: E402

for L in range(1, 16):
    field = ar.masked_stack_field(L)
    closed = {(-u, j) for u in range(1, L + 1) for j in range(-L, u + 1)} | {(0, j) for j in range(-L, 0)}
    box = {(-u, j) for u in range(1, L + 1) for j in range(-L, L + 1)} | {(0, j) for j in range(-L, 0)}
    assert field == closed and len(box) == 2 * L * (L + 1)
    assert len(box - field) == L * (L - 1) // 2
    assert Fraction(len(box - field), len(box)) == Fraction(L - 1, 4 * (L + 1))
    gated = ar.gated_stack_field(L)
    assert box <= gated and all(i < 0 or (i == 0 and j < 0) for i, j in gated)
assert len(ar.masked_stack_field(15)) == 480 - 105

# a random network of masked convolutions with ReLU: perturb each pixel and see which move the centre output
rng = np.random.default_rng(5)
L, n, c = 4, 13, 6
ws = [rng.standard_normal((3, 3, 1 if i == 0 else 3, 3)) for i in range(L)]
bs = [rng.standard_normal(3) + 2.0 for _ in range(L)]  # positive biases keep most units active


def net(img):
    h = img
    for i in range(L):
        h = np.maximum(ar.masked_conv2d(h, ws[i], ar.conv_mask(3, "A" if i == 0 else "B")) + bs[i], 0.0)
    return h[0, c, c]


img = rng.standard_normal((1, n, n, 1)) * 0.1
base = net(img)
seen = set()
for i in range(n):
    for j in range(n):
        y = img.copy()
        y[0, i, j, 0] += 0.05
        if np.abs(net(y) - base).max() > 1e-12:
            seen.add((i - c, j - c))
assert seen == ar.masked_stack_field(L)
print("masked-convolution-blind-spot: ok")
