# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/recurrent-neural-network/ (ai://NEW-recurrent-neural-network).
"""Numeric check of the derivation parameter-and-flop-count: the counts, the redundant bias and the gated ratios."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import rnn  # noqa: E402

rng = np.random.default_rng(0)
for _ in range(50):
    n_in, H, B, T = (int(v) for v in rng.integers(1, 9, size=4))
    cfg = rnn.config(n_in, H)
    params = rnn.init(rng, cfg)
    N = sum(a.size for a in params.values())
    assert N == rnn.param_count(cfg) == H * n_in + H * H + 2 * H
    assert rnn.one_bias_param_count(cfg) == N - H
    assert rnn.flops(cfg, B, T) == T * (2 * B * n_in * H + 2 * B * H * H)
    # Only b_i + b_h matters.
    x, h0 = rng.normal(size=(B, T, n_in)), rng.normal(size=(B, H))
    c = rng.normal(size=H)
    moved = params | {"b_i": params["b_i"] + c, "b_h": params["b_h"] - c}
    for nl in rnn.NONLINEARITIES:
        assert np.allclose(rnn.forward(params, x, h0, nl)[0], rnn.forward(moved, x, h0, nl)[0], atol=1e-13)

assert rnn.param_count(rnn.config(3, 4)) == 36 and rnn.flops(rnn.config(3, 4), 2, 5) == 560
assert rnn.param_count(rnn.config(1, 100)) == 10_300

try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    for n_in, H in ((3, 4), (1, 100), (7, 2)):
        N = rnn.param_count(rnn.config(n_in, H))
        count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        assert count(torch.nn.RNN(n_in, H)) == N
        assert count(torch.nn.GRU(n_in, H)) == 3 * N
        assert count(torch.nn.LSTM(n_in, H)) == 4 * N
print("parameter-and-flop-count: ok")
