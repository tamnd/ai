# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/randaugment/ (ai://88YEPR).
# Check of derivations/cutout-area.md: a clipped s x s Cutout patch with a uniform pixel centre on an L x L image masks
# (s - s^2 / (4L))^2 pixels on average, exactly on the pixel grid, for even s with 0 <= s <= 2L.
from fractions import Fraction

import numpy as np


def formula(s, length):
    return (Fraction(s) - Fraction(s * s, 4 * length)) ** 2


def mean_area(s, length):
    half = s // 2
    total = 0
    for r in range(length):
        for c in range(length):
            mask = np.zeros((length, length), dtype=bool)
            mask[
                max(r - half, 0) : min(r + half, length),
                max(c - half, 0) : min(c + half, length),
            ] = True
            total += int(mask.sum())
    return Fraction(total, length * length)


def main():
    for length in range(1, 25):
        for s in range(0, 2 * length + 1, 2):
            assert mean_area(s, length) == formula(s, length), (s, length)
        for s in range(1, 2 * length, 2):
            assert mean_area(s, length) == formula(s - 1, length), (s, length)

    # The continuous centre: the mean clipped side by midpoint quadrature.
    for s, length in [(16, 32), (8, 32), (40, 32)]:
        h, n = s / 2, 200000
        r = (np.arange(n) + 0.5) * length / n
        side = np.minimum(r + h, length) - np.maximum(r - h, 0)
        assert abs(side.mean() - (s - s * s / (4 * length))) < 1e-6

    assert formula(16, 32) == 196 and formula(16, 32) / 1024 == Fraction(49, 256)
    assert 1 - formula(16, 32) / 256 == Fraction(15, 64)
    assert formula(8, 32) == Fraction(225, 4)
    print("ok: E[A] = (s - s^2/(4L))^2 exactly for L <= 24; 16 x 16 on 32 x 32 masks 196 pixels, 0.1914 of the image")


if __name__ == "__main__":
    main()
