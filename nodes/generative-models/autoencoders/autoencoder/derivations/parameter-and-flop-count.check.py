# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/autoencoder/ (ai://NEW-autoencoder).
"""Numeric check of the derivation parameter-and-flop-count."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import autoencoder as ae  # noqa: E402


def closed(w, tied):
    weights = sum(a * b for a, b in zip(w[:-1], w[1:]))
    return (1 if tied else 2) * weights + sum(w[1:]) + sum(w[:-1])


rng = np.random.default_rng(0)
for _ in range(50):
    w = list(rng.integers(1, 9, size=rng.integers(2, 6)))
    for tied in (False, True):
        cfg = ae.from_widths(w, tied=tied)
        p = ae.init(rng, cfg)
        assert sum(a.size for v in p.values() for a in v) == closed(w, tied) == ae.param_count(cfg)

# statement 2, the uniform widths of the shape contract
for D, H, L, C in [(6, 5, 1, 2), (784, 100, 3, 6), (3, 7, 4, 1)]:
    w = [D] + [H] * L + [C]
    assert closed(w, False) == 2 * (D * H + (L - 1) * H * H + H * C) + L * H + C + L * H + D
    assert closed(w, True) == D * H + (L - 1) * H * H + H * C + 2 * L * H + C + D

# statement 3, by counting the multiply-adds of every product
count = [0]
matmul = np.matmul


class Counted(np.ndarray):
    def __matmul__(self, other):
        a, b = np.asarray(self), np.asarray(other)
        count[0] += 2 * a.shape[0] * a.shape[1] * b.shape[-1] if a.ndim == 2 else 0
        return matmul(a, b).view(Counted)

    def __rmatmul__(self, other):
        return Counted.__matmul__(np.asarray(other).view(Counted), self)


for w, B in [([6, 5, 2], 4), ([7, 4, 3, 2], 3)]:
    for tied in (False, True):
        cfg = ae.from_widths(w, tied=tied)
        p = ae.init(rng, cfg)
        p = {k: [a.view(Counted) for a in v] for k, v in p.items()}
        x = rng.uniform(size=(B, w[0])).view(Counted)
        count[0] = 0
        out, cache = ae.forward(p, cfg, x)
        fwd = count[0]
        assert (fwd, 2 * fwd) == ae.flops(cfg, B), (fwd, ae.flops(cfg, B))
        count[0] = 0
        ae.backward(p, cfg, cache, np.asarray(out["a"]).view(Counted))
        assert count[0] == 2 * fwd, (count[0], fwd)

# statement 4, the autoencoders of Hinton and Salakhutdinov (2006)
nets = {"mnist": [784, 1000, 500, 250, 30], "curves": [784, 400, 200, 100, 50, 25, 6],
        "faces": [625, 2000, 1000, 500, 30], "documents": [2000, 500, 250, 125, 10]}  # fmt: skip
want = {"mnist": (2837314, 1420814), "curves": (842340, 422340), "faces": (7537655, 3772655),
        "documents": (2318760, 1161260)}  # fmt: skip
for name, w in nets.items():
    assert (closed(w, False), closed(w, True)) == want[name], name
print("parameter-and-flop-count: ok")
