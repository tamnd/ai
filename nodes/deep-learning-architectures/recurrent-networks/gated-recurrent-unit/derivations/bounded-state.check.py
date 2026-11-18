# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/gated-recurrent-unit/ (ai://NEW-gated-recurrent-unit).
"""Numeric check of the derivation bounded-state: the box [-1, 1]^H and the saturated special cases."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import gru  # noqa: E402

rng = np.random.default_rng(2018)
for reset in gru.RESETS:
    for scale in (1.0, 5.0, 30.0):
        params = {k: v * scale for k, v in gru.init(rng, gru.config(3, 6)).items()}
        x = rng.normal(size=(4, 500, 3)) * 1e3
        h, _ = gru.forward(params, x, rng.uniform(-1, 1, size=(4, 6)), reset)
        assert np.all(np.abs(h) <= 1.0), (reset, scale)
        # From outside the box: |h_t| <= max(|h_{t-1}|, |n_t|) at every step.
        h0 = rng.normal(size=(4, 6)) * 10
        h, c = gru.forward(params, rng.normal(size=(4, 50, 3)), h0, reset)
        prev = np.concatenate([h0[:, None], h[:, :-1]], axis=1)
        assert np.all(np.abs(h) <= np.maximum(np.abs(prev), np.abs(c["n"])) + 1e-15)

# The saturated gates. Biases of +-800 give sigmoids of exactly 0 or 1 in float64.
n_in, H = 3, 4
params = gru.init(rng, gru.config(n_in, H))
x, h0 = rng.normal(size=(2, 7, n_in)), rng.uniform(-1, 1, size=(2, H))
W, U, b_i, b_h = params["W"], params["U"], params["b_i"], params["b_h"]


def with_gates(r_bias, z_bias):
    zero = np.zeros_like(W[:, :H])
    Wg = np.concatenate([zero, zero, W[:, 2 * H :]], axis=1)
    Ug = np.concatenate([np.zeros((H, 2 * H)), U[:, 2 * H :]], axis=1)
    big = np.concatenate([np.full(H, r_bias), np.full(H, z_bias), b_i[2 * H :]])
    return {"W": Wg, "U": Ug, "b_i": big, "b_h": np.concatenate([np.zeros(2 * H), b_h[2 * H :]])}


tanh_rnn = with_gates(800.0, -800.0)
for reset in gru.RESETS:
    h, _ = gru.forward(tanh_rnn, x, h0, reset)
    want, prev = np.empty_like(h), h0
    for t in range(7):
        prev = np.tanh(x[:, t] @ W[:, 2 * H :] + b_i[2 * H :] + prev @ U[:, 2 * H :] + b_h[2 * H :])
        want[:, t] = prev
    assert np.allclose(h, want, atol=1e-13), reset
    h, _ = gru.forward(with_gates(0.0, 800.0), x, h0, reset)
    assert np.array_equal(h, np.broadcast_to(h0[:, None], h.shape)), reset
forget = with_gates(-800.0, 0.0)
_, c = gru.forward(forget, x, h0, "after")
assert np.allclose(c["n"], np.tanh(x @ W[:, 2 * H :] + b_i[2 * H :]), atol=1e-13)
_, c = gru.forward(forget, x, h0, "before")
assert np.allclose(c["n"], np.tanh(x @ W[:, 2 * H :] + b_i[2 * H :] + b_h[2 * H :]), atol=1e-13)
print("bounded-state: ok")
