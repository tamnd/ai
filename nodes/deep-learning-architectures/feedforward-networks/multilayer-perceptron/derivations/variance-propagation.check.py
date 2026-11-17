# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/multilayer-perceptron/ (ai://YA59M7).
"""Numeric check of the derivation variance-propagation."""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import multilayer_perceptron as mlp  # noqa: E402

rng = np.random.default_rng(0)

# Statement 2, step 2: E[relu(z)^2] = Var[z] / 2 for a symmetric z, while Var[relu(z)] is smaller.
z = rng.normal(0.0, 1.7, size=2_000_000)
r = np.maximum(z, 0.0)
assert abs(np.mean(r * r) / z.var() - 0.5) < 0.005
assert abs(r.var() / z.var() - (0.5 - 1 / (2 * math.pi))) < 0.005

# Statements 2 and 3 on a ReLU network of width 512 and 12 hidden layers.
cfg = mlp.config(512, 512, 512, 12, "relu")
x = rng.normal(size=(1000, 512))
he = mlp.init(rng, cfg, "he")
v = np.array([a.var() for a in mlp.forward(he, x, cfg)[1]["z"]])
assert abs(v[0] - 2.0) < 0.1  # n Var[w] E[x^2] = 512 (2/512) 1
# One draw of finite width: each layer keeps the variance within 15%, and 11 layers within a factor 1.5.
assert np.all(np.abs(v[1:] / v[:-1] - 1.0) < 0.15)
assert 1 / 1.5 < v[-1] / v[0] < 1.5
half = {"W": [w / math.sqrt(2.0) for w in he["W"]], "b": he["b"]}
vh = np.array([a.var() for a in mlp.forward(half, x, cfg)[1]["z"]])
ratios = vh[1:] / vh[:-1]
assert np.all(np.abs(ratios - 0.5) < 0.1)
assert abs(math.log2(vh[-1] / vh[0]) - (-11)) < 0.5

# Statement 4: the bound sqrt(6 / (n_in + n_out)) gives the variance 2 / (n_in + n_out).
for n_in, n_out in [(300, 100), (64, 64), (784, 10)]:
    a = math.sqrt(6.0 / (n_in + n_out))
    assert math.isclose(a * a / 3, 2.0 / (n_in + n_out))
    assert math.isclose(2.0 / (n_in + n_out), 2.0 / (1.0 / (1.0 / n_in) + 1.0 / (1.0 / n_out)))
    w = mlp.init(rng, mlp.config(n_in, n_out, 1, 1), "glorot")["W"][0]
    assert abs(w.var() * (n_in + n_out) / 2.0 - 1.0) < 0.05 and np.abs(w).max() <= a
print("variance-propagation: ok")
