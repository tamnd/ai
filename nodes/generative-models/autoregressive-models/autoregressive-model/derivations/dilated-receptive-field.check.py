# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/autoregressive-model/ (ai://NEW-autoregressive-model).
"""Numeric check of the derivation dilated-receptive-field."""

import itertools
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import autoregressive as ar  # noqa: E402


def offsets(k, dilations):
    return {sum(j * d for j, d in zip(js, dilations, strict=True)) for js in itertools.product(range(k), repeat=len(dilations))}


rng = np.random.default_rng(4)
cases = [(2, (1, 2, 4)), (2, (1, 2, 4, 8)), (3, (1, 3)), (2, (1, 1, 1, 1)), (3, (1, 2, 4)), (2, (1, 2, 1, 2))]
for k, dil in cases:
    R = ar.receptive_field(k, dil)
    assert offsets(k, dil) == set(range(R))
    K, T = 3, R + 6
    w = {n: rng.normal(0.0, 0.7, s) for n, s in ar.wavenet_shapes(K, 16, k, dil).items()}
    x = rng.integers(0, K, (1, T))
    base = ar.wavenet_logits(w, x, dil)
    for t in range(T):
        y = x.copy()
        y[0, t] = (y[0, t] + 1) % K
        diff = np.abs(ar.wavenet_logits(w, y, dil) - base).max(axis=(0, 2))
        assert set(np.nonzero(diff > 1e-12)[0]) == set(range(t + 1, min(t + R, T - 1) + 1)), (k, dil, t)
# the interval condition matters: a first dilation of 2 leaves gaps
assert offsets(2, (2,)) == {0, 2} and offsets(2, (1, 4)) == {0, 1, 4, 5}
# doubling blocks
block = tuple(2**i for i in range(10))
assert ar.receptive_field(2, block) == 1024 and offsets(2, block) == set(range(1024))
assert ar.receptive_field(2, block * 3) == 3070
assert abs(3070 / 16000 - 0.19) < 0.005
print("dilated-receptive-field: ok")
