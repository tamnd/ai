# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/convolutional-neural-network/ (ai://GNXZCP).
"""Numeric check of the derivation receptive-field: measure receptive fields with gradients of the NumPy reference."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import cnn  # noqa: E402


def measured(stack, n=96, unit=1):
    """The input rows that unit (unit, 0) of the last layer depends on, and whether they form an interval."""
    shapes = [(1, 1, n, n)]
    for k, s in stack:
        shapes.append((1, 1, cnn.out_size(shapes[-1][2], k, s), cnn.out_size(shapes[-1][3], k, s)))
    dy = np.zeros(shapes[-1])
    dy[0, 0, unit, 0] = 1.0
    for (k, s), shp in zip(reversed(stack), reversed(shapes[:-1])):
        dy, _, _ = cnn.conv2d_backward(np.zeros(shp), np.ones((1, 1, k, k)), dy, stride=s)
    rows = np.nonzero(dy[0, 0].any(axis=1))[0]
    return rows.max() - rows.min() + 1, len(rows) == rows.max() - rows.min() + 1, rows.min()


def closed_form(stack):
    r, j = 1, 1
    for k, s in stack:
        r += (k - 1) * j
        j *= s
    return r, j


rng = np.random.default_rng(0)
for _ in range(200):
    depth = rng.integers(1, 5)
    stack = [(int(rng.integers(1, 6)), int(rng.integers(1, 4))) for _ in range(depth)]
    if np.prod([s for _, s in stack]) > 12:
        continue
    span, interval, start = measured(stack)
    r, j = cnn.receptive_field(stack)
    assert (r, j) == closed_form(stack), stack
    assert span == r, (stack, span, r)
    if all(s <= k for k, s in stack):
        assert interval, stack
    assert start == j  # unit 1 starts one jump after unit 0

assert [cnn.receptive_field([(3, 1)] * m)[0] for m in (1, 2, 3)] == [3, 5, 7]
lenet = [(5, 1), (2, 2), (5, 1), (2, 2), (5, 1)]
assert [cnn.receptive_field(lenet[:i]) for i in range(1, 6)] == [(5, 1), (6, 2), (14, 2), (16, 4), (32, 4)]
print("receptive-field: ok")
