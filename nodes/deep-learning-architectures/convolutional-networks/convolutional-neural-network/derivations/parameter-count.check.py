# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/convolutional-neural-network/ (ai://NEW-convolutional-neural-network).
"""Numeric check of the derivation parameter-count."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import cnn  # noqa: E402

rng = np.random.default_rng(0)
for k, c_in, c_out, n in [(1, 1, 1, 4), (3, 2, 5, 7), (5, 1, 6, 32), (5, 16, 120, 5)]:
    w, b = rng.normal(size=(c_out, c_in, k, k)), rng.normal(size=c_out)
    assert w.size + b.size == cnn.conv_params(k, c_in, c_out) == k * k * c_in * c_out + c_out
    # The count does not depend on n, and the layer makes n'^2 C_out k^2 C_in multiply-adds.
    y = cnn.conv2d(rng.normal(size=(1, c_in, n, n)), w, b)
    assert y.size * k * k * c_in == cnn.out_size(n, k) ** 2 * c_out * k * k * c_in

assert cnn.conv_params(5, 1, 6) == 156
assert 28 * 28 * 6 * (25 + 1) == 122_304
assert 32 * 32 * 28 * 28 * 6 + 28 * 28 * 6 == 4_821_600

# The connection scheme of Table I of LeCun et al. (1998), as its text describes it: six maps fed by three
# contiguous S2 maps, six by four contiguous, three by four non-contiguous, and one by all six.
table = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 0], [5, 0, 1],
         [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 0], [4, 5, 0, 1], [5, 0, 1, 2],
         [0, 1, 3, 4], [1, 2, 4, 5], [0, 2, 3, 5], [0, 1, 2, 3, 4, 5]]
assert sum(len(t) for t in table) == 60 and 60 * 25 + 16 == 1_516
assert cnn.conv_params(5, 6, 16) == 2_416
assert cnn.conv_params(5, 16, 120) == 48_120
assert 120 * 84 + 84 == 10_164

for m, single in [(1, 9), (2, 25), (3, 49)]:
    assert cnn.receptive_field([(3, 1)] * m)[0] == 2 * m + 1
    assert (2 * m + 1) ** 2 == single and m * cnn.conv_params(3, 1, 1, bias=False) == 9 * m
assert round(49 / 27 - 1, 2) == 0.81
print("parameter-count: ok")
