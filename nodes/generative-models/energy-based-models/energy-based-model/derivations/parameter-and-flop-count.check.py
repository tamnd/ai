# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/energy-based-model/ (ai://NEW-energy-based-model).
"""Numeric check of the derivation parameter-and-flop-count."""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import energy_based as eb  # noqa: E402

for D, H in ((4, 3), (8, 8), (256, 500), (784, 500), (1, 1)):
    cfg = eb.config(D, H)
    assert eb.param_count(cfg) == D * H + D + H
    assert eb.param_count(cfg) == sum(math.prod(s) for s in eb.param_shapes(cfg).values())
    for B in (1, 5, 100):
        assert eb.flops_forward(cfg, B) == 4 * B * D * H
assert eb.param_count(eb.config()) == 19 and eb.flops_forward(eb.config(), 5) == 240
assert eb.param_count(eb.config(784, 500)) == 393_284 and eb.param_count(eb.config(256, 500)) == 128_756
assert eb.flops_forward(eb.config(784, 500), 1) == 1_568_000

# count the multiply-adds of one Gibbs step by recording the shapes of the products with W
calls = []


class Counted(np.ndarray):
    def __matmul__(self, other):
        a, b = np.asarray(self), np.asarray(other)
        calls.append(a.shape[0] * a.shape[1] * b.shape[-1])
        return a @ b

    def __rmatmul__(self, other):
        a, b = np.asarray(other), np.asarray(self)
        calls.append(a.shape[0] * a.shape[1] * b.shape[-1])
        return a @ b


rng = np.random.default_rng(8)
B, D, H = 6, 5, 4
p = {k: rng.standard_normal(s) for k, s in eb.param_shapes(eb.config(D, H)).items()}
p["W"] = p["W"].view(Counted)
v = rng.integers(0, 2, (B, D)).astype(float)
eb.gibbs_step(p, v, rng.uniform(size=(B, H)), rng.uniform(size=(B, D)))
assert 2 * sum(calls) == eb.flops_forward(eb.config(D, H), B), calls
