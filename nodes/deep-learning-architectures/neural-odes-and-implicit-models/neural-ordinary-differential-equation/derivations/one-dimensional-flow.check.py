# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-ordinary-differential-equation/ (ai://WNCNQE).
"""Numeric check of the derivation one-dimensional-flow."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import neural_ode as node  # noqa: E402

rng = np.random.default_rng(3)

# Statement 1: random one-dimensional flows keep the order of their inputs.
for _ in range(20):
    cfg = node.config(1, 6, n_steps=40, T=2.0)
    params = node.init(rng, cfg, 2.0)
    params["b1"], params["b2"] = rng.normal(0, 1, 6), rng.normal(0, 1, 1)
    x = np.sort(rng.uniform(-3, 3, size=(30, 1)), axis=0)
    assert np.all(np.diff(node.forward(params, x, cfg)[:, 0]) > 0)

# Statement 2: training the plain model on g keeps its loss above 1/2.
x, t = np.array([[-1.0], [1.0]]), np.array([[1.0], [-1.0]])
for seed in range(3):
    cfg = node.config(1, 8, n_steps=10)
    _, losses = node.train(node.init(np.random.default_rng(seed), cfg), x, t, cfg, lr=0.2, steps=200)
    assert min(losses) > 0.5
assert node.mse_loss(np.zeros((2, 1)), t)[0] == 0.5  # the bound, met by u = v = 0

# Statement 3: one residual step with F(h) = -2h fits g.
assert np.array_equal(x + (-2.0 * x), t)

# Statement 4: the rotation by pi in the augmented plane maps (x, 0) to (-x, 0).
R = np.pi * np.array([[0.0, -1.0], [1.0, 0.0]])
h0 = node.augment(np.array([[-1.0], [1.0], [0.3]]), node.config(1, 1, d_state=2))
y = node.rk4(lambda h, s: h @ R.T, h0, 0.0, 1.0, 200)
np.testing.assert_allclose(y, [[1.0, 0.0], [-1.0, 0.0], [-0.3, 0.0]], atol=1e-8)
print("one-dimensional-flow: ok")
