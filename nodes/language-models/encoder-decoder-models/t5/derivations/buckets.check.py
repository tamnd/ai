# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/t5/ (ai://NEW-t5).
"""Numeric check of the derivation buckets: where each position bucket begins, for the decoder and the encoder."""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import t5  # noqa: E402


def starts(K, d, bidirectional):
    b = t5.bucket(-np.arange(0, 4000), bidirectional, K, d)
    return {int(k): int(np.argmax(b == k)) for k in np.unique(b)}


def closed_form(B, d):
    e = B // 2
    out = {k: k for k in range(e)}
    for k in range(e, B):
        out[k] = math.ceil(e * (d / e) ** ((k - e) / e) - 1e-9)
    return out


for K, d in [(32, 128), (8, 16), (16, 64), (12, 40), (32, 1000), (64, 256)]:
    assert starts(K, d, False) == closed_form(K, d), (K, d)
    assert starts(K, d, True) == closed_form(K // 2, d), (K, d)

dec, enc = starts(32, 128, False), starts(32, 128, True)
assert [dec[k] for k in range(16, 32)] == [16, 19, 21, 24, 27, 31, 35, 40, 46, 52, 59, 67, 77, 87, 99, 113]
assert [enc[k] for k in range(8, 16)] == [8, 12, 16, 23, 32, 46, 64, 91]
assert round(16 * 8 ** (15 / 16), 1) == 112.4 and round(8 * 16 ** (7 / 8), 1) == 90.5
assert round(8 ** (1 / 16), 3) == 1.139 and abs(16 ** (1 / 8) - math.sqrt(2)) < 1e-15
both = t5.bucket(np.arange(-4000, 4001), True, 32, 128)
assert 16 not in both and len(np.unique(both)) == 31
assert t5.bucket(np.array([-128]), False, 32, 128)[0] == 31 and t5.bucket(np.array([-112]), False, 32, 128)[0] == 30
print("ok")
