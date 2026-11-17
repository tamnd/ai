# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/gated-recurrent-unit/ (ai://NEW-gated-recurrent-unit).
"""Numeric check of the derivation parameter-and-flop-count: the counts, the redundant biases and the speech sizes."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import gru  # noqa: E402


def counted_flops(params, B, T):
    # Replay the two products of cell, 2 m k n each, for T steps.
    n_in, H3 = params["W"].shape
    H = H3 // 3
    return T * (2 * B * n_in * 3 * H + 2 * B * H * 3 * H)


rng = np.random.default_rng(108)
assert gru.param_count(gru.config(3, 4)) == 108 and gru.flops(gru.config(3, 4), 2, 5) == 1680
for _ in range(100):
    n_in, H, B, T = (int(v) for v in rng.integers(1, 8, size=4))
    cfg = gru.config(n_in, H)
    params = gru.init(rng, cfg)
    assert sum(v.size for v in params.values()) == gru.param_count(cfg) == 3 * H * n_in + 3 * H * H + 6 * H
    assert counted_flops(params, B, T) == gru.flops(cfg, B, T) == 6 * B * T * H * (n_in + H)
    one_bias = 3 * H * n_in + 3 * H * H + 3 * H
    assert 4 * one_bias == 3 * gru.lstm_param_count(cfg) and 4 * gru.flops(cfg, B, T) == 3 * 8 * B * T * H * (n_in + H)

# The redundant biases: shift b_i by +d and b_h by -d on the blocks where only the sum matters.
cfg = gru.config(3, 5)
params = gru.init(rng, cfg)
x, h0 = rng.normal(size=(2, 6, 3)), rng.normal(size=(2, 5)) * 0.5
d = rng.normal(size=15)
for reset, free in (("after", slice(0, 10)), ("before", slice(0, 15))):
    mask = np.zeros(15)
    mask[free] = 1.0
    shifted = params | {"b_i": params["b_i"] + d * mask, "b_h": params["b_h"] - d * mask}
    want, _ = gru.forward(params, x, h0, reset)
    got, _ = gru.forward(shifted, x, h0, reset)
    assert np.allclose(got, want, atol=1e-12), reset
# With the reset after the matrix, the candidate biases are not interchangeable.
mask = np.zeros(15)
mask[10:] = 1.0
shifted = params | {"b_i": params["b_i"] + d * mask, "b_h": params["b_h"] - d * mask}
assert not np.allclose(gru.forward(shifted, x, h0, "after")[0], gru.forward(params, x, h0, "after")[0])

# The speech models of Chung et al., one bias per block, the LSTM with peepholes.
assert 3 * 227 * 20 + 3 * 227**2 + 3 * 227 == 168_888
assert gru.lstm_param_count(gru.config(20, 195), peepholes=True) == 169_065
assert gru.tanh_param_count(gru.config(20, 400)) == 168_400
assert [round(v, -2) for v in (168_888, 169_065, 168_400)] == [168_900, 169_100, 168_400]
print("parameter-and-flop-count: ok")
