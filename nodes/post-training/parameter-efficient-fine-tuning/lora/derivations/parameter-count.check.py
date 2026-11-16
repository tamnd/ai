# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/lora/ (ai://NEW-lora).
"""Numeric check of the derivation parameter-count: count the entries of built adapters, then the GPT-3 numbers."""

import numpy as np

# A small model: three adapted matrices of different shapes.
shapes = [(64, 32), (32, 32), (10, 32)]
for r in (1, 2, 4):
    built = sum(np.zeros((d, r)).size + np.zeros((r, k)).size for d, k in shapes)
    assert built == sum(r * (d + k) for d, k in shapes), (r, built)
    for d, k in shapes:
        assert np.isclose(r * (d + k) / (d * k), r / d + r / k)

d_model, layers, matrices = 12288, 96, 2
per_r = matrices * layers * (d_model + d_model)
assert per_r == 4_718_592
counts = {r: per_r * r for r in (1, 4, 8)}
assert counts == {1: 4_718_592, 4: 18_874_368, 8: 37_748_736}
assert 1 * layers * (2 * d_model) * 8 == counts[4]  # r = 8 on one type of matrix, the same budget
assert round(counts[1] / 1e6, 1) == 4.7 and round(counts[8] / 1e6, 1) == 37.7
fp16 = 2 * counts[4]
assert fp16 == 37_748_736 and round(fp16 / 2**20, 1) == 36.0
share = counts[4] / 175e9
assert 9_000 < 1 / share < 9_500
print(f"parameter-count: r(d+k) holds on built arrays; GPT-3 Wq, Wv: {counts}; r=4 in FP16 is {fp16 / 2**20:.1f} MiB, 1 in {1 / share:,.0f}")
