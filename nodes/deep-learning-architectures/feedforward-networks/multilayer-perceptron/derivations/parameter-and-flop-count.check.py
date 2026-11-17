# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/multilayer-perceptron/ (ai://YA59M7).
"""Numeric check of the derivation parameter-and-flop-count."""

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


rng = np.random.default_rng(0)
for n, d, k, L, B in [(2, 2, 1, 1, 4), (3, 4, 2, 2, 5), (6, 8, 4, 4, 6), (784, 512, 10, 3, 2)]:
    cfg = mlp.config(n, d, k, L)
    p = mlp.init(rng, cfg)
    P = sum(a.size for v in p.values() for a in v)
    widths = [n] + [d] * L + [k]
    assert P == sum(a * b + b for a, b in zip(widths[:-1], widths[1:])) == mlp.param_count(cfg)
    assert P == (n + 1) * d + (L - 1) * (d + 1) * d + (d + 1) * k
    Counted.macs = 0
    mlp.forward(p, rng.normal(size=(B, n)).view(Counted), cfg)
    assert 2 * Counted.macs == 2 * B * sum(a * b for a, b in zip(widths[:-1], widths[1:])) == mlp.flops(cfg, B)

# The example config of shapes.yaml: 3 -> 4 -> 4 -> 2 with a batch of 5.
assert mlp.param_count(mlp.config(3, 4, 2, 2)) == 16 + 20 + 10
assert mlp.flops(mlp.config(3, 4, 2, 2), 5) == 2 * 5 * (12 + 16 + 8)
# An MNIST-sized network 784-512-512-512-10 has 932,362 parameters.
assert mlp.param_count(mlp.config(784, 512, 10, 3)) == 932_362
print("parameter-and-flop-count: ok")
