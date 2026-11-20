# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/autoregressive-model/ (ai://NEW-autoregressive-model).
"""Numeric check of the derivation parameter-and-flop-count."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import autoregressive as ar  # noqa: E402

for D, H in ((4, 6), (8, 48), (784, 500)):
    cfg = ar.config(D, (H,), True)
    assert ar.param_count(cfg) == 2 * D * H + H + D + D * D
    for B in (1, 100):
        assert ar.flops_forward(cfg, B) == 2 * B * (2 * D * H + D * D)
    cfg = ar.config(D, (H,), False)
    assert ar.param_count(cfg) == 2 * D * H + H + D and ar.flops_forward(cfg, 1) == 4 * D * H
cfg = ar.config(5, (7, 3), False)
assert ar.flops_forward(cfg, 2) == 2 * 2 * (5 * 7 + 7 * 3 + 3 * 5)
assert ar.param_count(ar.config()) == 74 and ar.flops_forward(ar.config(), 1) == 128
assert ar.param_count(ar.config(784, (500,))) == 1_399_940
assert abs(ar.flops_forward(ar.config(784, (500,)), 1) / 1e6 - 2.80) < 0.01

# the masked fraction of the example: about half of W and V, and D(D - 1)/2 of the D^2 direct weights kept
cfg = ar.config()
mk = ar.masks(cfg, ar.degrees(cfg))
assert mk["A"].sum() == 4 * 3 / 2
assert 0.3 < mk["W1"].mean() < 0.8 and 0.2 < mk["V"].mean() < 0.7

# sampling takes one pass per variable
calls = []
orig = ar.logits


def counted(*a):
    calls.append(1)
    return orig(*a)


ar.logits = counted
p = ar.init(np.random.default_rng(0), cfg)
ar.sample(p, cfg, mk, np.arange(1, 5), np.random.default_rng(1), 3)
ar.logits = orig
assert len(calls) == 4
print("parameter-and-flop-count: ok")
