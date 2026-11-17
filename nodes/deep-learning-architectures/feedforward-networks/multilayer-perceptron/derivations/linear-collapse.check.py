# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/multilayer-perceptron/ (ai://YA59M7).
"""Numeric check of the derivation linear-collapse."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import multilayer_perceptron as mlp  # noqa: E402

rng = np.random.default_rng(0)
for d_in, d_hidden, d_out, n_hidden in [(3, 5, 2, 1), (4, 6, 3, 3), (2, 7, 5, 5)]:
    cfg = mlp.config(d_in, d_hidden, d_out, n_hidden, "identity")
    p = mlp.init(rng, cfg, "glorot")
    p["b"] = [rng.normal(size=b.shape) for b in p["b"]]
    W, b = np.eye(d_in), np.zeros(d_in)
    for Wl, bl in zip(p["W"], p["b"]):
        W, b = Wl @ W, Wl @ b + bl  # statement 1, one layer at a time
    x = rng.normal(size=(9, d_in))
    np.testing.assert_allclose(mlp.forward(p, x, cfg)[0], x @ W.T + b, rtol=1e-12, atol=1e-12)
    assert np.linalg.matrix_rank(W) <= min(mlp.widths(cfg))

# Statement 2 with a bottleneck: widths 6 -> k -> 6 give a rank-k map of 6 inputs to 6 outputs.
for k in (1, 2):
    W0, W1 = rng.normal(size=(k, 6)), rng.normal(size=(6, k))
    assert np.linalg.matrix_rank(W1 @ W0) == k

# A ReLU breaks the identity: the same weights are no longer affine (f(x) + f(-x) != 2 f(0) somewhere).
cfg = mlp.config(3, 5, 2, 2, "relu")
p = mlp.init(rng, cfg, "he")
x = rng.normal(size=(20, 3))
f = lambda v: mlp.forward(p, v, cfg)[0]  # noqa: E731
assert np.abs(f(x) + f(-x) - 2 * f(np.zeros((20, 3)))).max() > 1e-3
print("linear-collapse: ok")
