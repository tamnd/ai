# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/multilayer-perceptron/ (ai://NEW-multilayer-perceptron).
"""Numeric check of the derivation backpropagation."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import multilayer_perceptron as mlp  # noqa: E402


class Counted(np.ndarray):
    """An array that adds the multiply-adds of each matrix product it takes part in to a counter."""

    macs = 0

    def __matmul__(self, other):
        a, b = np.asarray(self), np.asarray(other)
        Counted.macs += a.shape[0] * a.shape[1] * b.shape[1]
        return (a @ b).view(Counted)

    def __rmatmul__(self, other):
        a, b = np.asarray(other), np.asarray(self)
        Counted.macs += a.shape[0] * a.shape[1] * b.shape[1]
        return (a @ b).view(Counted)


rng = np.random.default_rng(0)
for act in ("relu", "tanh", "sigmoid", "gelu"):
    cfg = mlp.config(3, 5, 2, 3, act)
    p = mlp.init(rng, cfg, "glorot")
    p["b"] = [rng.normal(0.0, 0.3, b.shape) for b in p["b"]]
    x, t = rng.normal(size=(6, 3)), rng.normal(size=(6, 2))
    y, cache = mlp.forward(p, x, cfg)
    dx, g = mlp.backward(p, cache, mlp.mse_loss(y, t)[1])

    def E(q, xx=x):
        return mlp.mse_loss(mlp.forward(q, xx, cfg)[0], t)[0]

    eps = 1e-6
    for key in ("W", "b"):
        for l, a in enumerate(p[key]):
            for idx in np.ndindex(a.shape):
                hi = {k: [v.copy() for v in vs] for k, vs in p.items()}
                lo = {k: [v.copy() for v in vs] for k, vs in p.items()}
                hi[key][l][idx] += eps
                lo[key][l][idx] -= eps
                num = (E(hi) - E(lo)) / (2 * eps)
                assert abs(g[key][l][idx] - num) <= 1e-6 * max(1.0, abs(num)), (act, key, l, idx)
    # Statement 2 written out for the first layer: dE/dW_0 = delta_0^T x.
    delta = mlp.mse_loss(y, t)[1]
    for l in reversed(range(1, 4)):
        delta = (delta @ p["W"][l]) * mlp.ACTIVATIONS[act][1](cache["z"][l - 1])
    np.testing.assert_allclose(g["W"][0], delta.T @ x, rtol=1e-12)
    np.testing.assert_allclose(dx, delta @ p["W"][0], rtol=1e-12)

# Statement 3: the backward pass makes twice the multiply-adds of the forward pass.
for d_in, d, k, L, B in [(3, 4, 2, 2, 5), (7, 9, 3, 4, 8)]:
    cfg = mlp.config(d_in, d, k, L)
    p = mlp.init(rng, cfg)
    p["W"] = [w.view(Counted) for w in p["W"]]
    Counted.macs = 0
    y, cache = mlp.forward(p, rng.normal(size=(B, d_in)).view(Counted), cfg)
    fwd = Counted.macs
    Counted.macs = 0
    mlp.backward(p, cache, np.ones((B, k)).view(Counted))
    assert Counted.macs == 2 * fwd and 2 * fwd == mlp.flops(cfg, B)
print("backpropagation: ok")
