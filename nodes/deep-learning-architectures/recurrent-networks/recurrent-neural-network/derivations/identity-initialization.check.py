# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/recurrent-neural-network/ (ai://NEW-recurrent-neural-network).
"""Numeric check of the derivation identity-initialization: the IRNN keeps its state and its gradient."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import rnn  # noqa: E402

rng = np.random.default_rng(6)
cfg = rnn.config(3, 7)
params = rnn.irnn_init(rng, cfg)
assert np.array_equal(params["U"], np.eye(7)) and not params["b_i"].any() and not params["b_h"].any()
x = np.zeros((4, 1000, 3))
h0 = np.abs(rng.normal(size=(4, 7))) + 0.01
_, cache = rnn.forward(params, x, h0, "relu")
assert np.array_equal(cache["h"], np.broadcast_to(h0[:, None], cache["h"].shape))
for b in range(4):
    for k in range(7):
        dhT = np.zeros((4, 7))
        dhT[b, k] = 1.0
        _, _, dh0 = rnn.backward(params, cache, np.zeros_like(cache["h"]), dhT)
        assert np.array_equal(dh0, dhT)
# A zero coordinate keeps its value but passes no gradient.
h0z = h0.copy()
h0z[:, 0] = 0.0
_, cache = rnn.forward(params, x[:, :5], h0z, "relu")
assert np.array_equal(cache["h_T"], h0z)
_, _, dh0 = rnn.backward(params, cache, np.zeros_like(cache["h"]), np.ones((4, 7)))
assert not dh0[:, 0].any() and np.array_equal(dh0[:, 1:], np.ones((4, 6)))

# The scaled identity forgets as c^t.
for c in (0.5, 0.9, 0.99):
    p = rnn.irnn_init(rng, cfg, scale=c)
    _, cache = rnn.forward(p, x[:, :50], h0, "relu")
    assert np.allclose(cache["h_T"], c**50 * h0, rtol=1e-12)
    _, _, dh0 = rnn.backward(p, cache, np.zeros_like(cache["h"]), np.ones((4, 7)))
    assert np.allclose(dh0, c**50, rtol=1e-12)

# With tanh the identity does not keep the state: h_t ~ sqrt(3 / (2 t)).
h = 1.0
for t in range(1, 100_001):
    h_new = np.tanh(h)
    assert h_new < h
    h = h_new
assert abs(h * np.sqrt(2 * 100_000 / 3) - 1.0) < 0.01
print("identity-initialization: ok")
