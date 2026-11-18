# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/gated-recurrent-unit/ (ai://NEW-gated-recurrent-unit).
"""Numeric check of the derivation update-gate-path: the step Jacobian, the copy path and the leaky integrator."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import gru  # noqa: E402


def numeric_jacobian(params, x, h, reset, eps=1e-6):
    H = h.shape[-1]
    J = np.zeros((H, H))
    for j in range(H):
        up, down = h.copy(), h.copy()
        up[0, j] += eps
        down[0, j] -= eps
        J[:, j] = (gru.cell(params, x, up, reset)[0] - gru.cell(params, x, down, reset)[0])[0] / (2 * eps)
    return J


def gate_jacobians(params, x, h, reset):
    # dz/dh, dn/dh by finite differences of the gates alone, to assemble statement 1.
    H = h.shape[-1]
    dz, dn = np.zeros((H, H)), np.zeros((H, H))
    eps = 1e-6
    for j in range(H):
        up, down = h.copy(), h.copy()
        up[0, j] += eps
        down[0, j] -= eps
        su, sd = gru.cell(params, x, up, reset)[1], gru.cell(params, x, down, reset)[1]
        dz[:, j] = (su["z"] - sd["z"])[0] / (2 * eps)
        dn[:, j] = (su["n"] - sd["n"])[0] / (2 * eps)
    return dz, dn


rng = np.random.default_rng(4)
for reset in gru.RESETS:
    for _ in range(20):
        n_in, H = (int(v) for v in rng.integers(1, 6, size=2))
        params = gru.init(rng, gru.config(n_in, H))
        x, h = rng.normal(size=(1, n_in)), rng.uniform(-1, 1, size=(1, H))
        _, s = gru.cell(params, x, h, reset)
        dz, dn = gate_jacobians(params, x, h, reset)
        z, n = s["z"][0], s["n"][0]
        J = np.diag(z) + np.diag(1 - z) @ dn + np.diag(h[0] - n) @ dz
        assert np.allclose(J, numeric_jacobian(params, x, h, reset), atol=1e-8), reset
        # With U = 0 the Jacobian is diag(z).
        flat = params | {"U": np.zeros_like(params["U"])}
        _, s = gru.cell(flat, x, h, reset)
        assert np.allclose(numeric_jacobian(flat, x, h, reset), np.diag(s["z"][0]), atol=1e-9)

# The leaky integrator: U = 0, b_z = 2, candidate tanh(x) with x = 1.
p = {"W": np.array([[0.0, 0.0, 1.0]]), "U": np.zeros((1, 3)), "b_i": np.array([0.0, 2.0, 0.0]), "b_h": np.zeros(3)}
hs, _ = gru.forward(p, np.ones((1, 10, 1)), np.zeros((1, 1)))
z, n = 1 / (1 + np.exp(-2.0)), np.tanh(1.0)
assert np.allclose(hs[0, :, 0], [(1 - z**t) * n for t in range(1, 11)], atol=1e-14)
assert abs(hs[0, 0, 0] - 0.0907842488) < 1e-10 and abs(z**10 - 0.2810338623) < 1e-10
lags = np.arange(10_000)
w = (1 - z) * z**lags
assert abs(w.sum() - 1) < 1e-12 and abs((lags * w).sum() - np.exp(2.0)) < 1e-9
print("update-gate-path: ok")
