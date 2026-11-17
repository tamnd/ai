# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/group-equivariant-convolutional-network/ (ai://NEW-group-equivariant-convolutional-network).
"""Numeric check of the derivation parameter-and-flop-count."""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import gcnn  # noqa: E402

rng = np.random.default_rng(4)
for c_in, c1, c2, k, m in ((1, 4, 4, 3, 3), (3, 6, 6, 3, 5), (1, 2, 3, 5, 2), (2, 5, 7, 1, 4)):
    cfg = gcnn.config(c_in, c1, c2, k, m)
    params = gcnn.init(rng, cfg)
    # Statement 1: the parameter count is the sum of the arrays.
    assert gcnn.param_count(cfg) == sum(v.size for v in params.values())
    # The FLOPs: multiply-adds of the two planar correlations with the bank, plus the head.
    B, n = 2, 5
    bank1, bank2 = gcnn.lift_bank(params["W1"]), gcnn.gconv_bank(params["W2"])
    macs = B * n * n * (bank1.size + bank2.size) + B * params["Wc"].size
    assert gcnn.flops(cfg, B, n) == 2 * macs
    # Statement 2: a planar CNN with the same plane counts has 4 times the convolution weights and the same FLOPs.
    planar = (4 * c1) * c_in * k * k + (4 * c2) * (4 * c1) * k * k
    assert planar == 4 * (params["W1"].size + params["W2"].size)
    assert planar == bank1.size + bank2.size

# Statement 3: the sqrt rule, for p4 (|S| = 4) and p4m (|S| = 8).
for S, C_planar in ((4, 20), (4, 64), (8, 64)):
    C = C_planar / math.sqrt(S)
    weights = S * C * C * 9
    assert abs(weights - C_planar * C_planar * 9) < 1e-9
    assert abs(S * C - math.sqrt(S) * C_planar) < 1e-9
    flops_g, flops_planar = 2 * 9 * (S * C) ** 2, 2 * 9 * C_planar ** 2
    assert abs(flops_g / flops_planar - S) < 1e-9
print("parameter-and-flop-count: ok")
