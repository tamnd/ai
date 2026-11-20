# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/flow-matching/ (ai://NEW-flow-matching).
"""Numeric check of the derivation parameter-and-flop-count."""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import flow_matching as fm  # noqa: E402

for D, H, E in ((3, 5, 4), (2, 64, 8), (784, 512, 16)):
    cfg = fm.config(d_data=D, d_hidden=H, d_time=E)
    assert fm.param_count(cfg) == (D + E + 1) * H + (H + 1) * D
    for B in (1, 256):
        assert fm.flops_forward(cfg, B) == 2 * B * (D + E) * H + 2 * B * H * D == 2 * B * H * (2 * D + E)
calls = []


def v(z, t):
    calls.append(len(z))
    return -z


x = np.ones((3, 2))
for n in (1, 7):
    calls.clear()
    fm.euler_sample(v, x, n)
    assert len(calls) == n
    calls.clear()
    fm.midpoint_sample(v, x, n)
    assert len(calls) == 2 * n
# both samplers are consistent with dx/dt = -x, whose solution at t = 1 is x / e
assert abs(fm.midpoint_sample(lambda z, t: -z, x, 100)[0, 0] - math.exp(-1)) < 1e-5
print("parameter-and-flop-count: ok")
