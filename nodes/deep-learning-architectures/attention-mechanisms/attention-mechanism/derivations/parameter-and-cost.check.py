# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/attention-mechanism/ (ai://NEW-attention-mechanism).
"""Numeric check of the derivation parameter-and-cost: the weights of RNNsearch and the FLOPs of attention."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import attention  # noqa: E402

COUNT = [0]


class Counted(np.ndarray):
    def __matmul__(self, other):
        a, b = np.asarray(self), np.asarray(other)
        COUNT[0] += 2 * a.size * (b.shape[-1] if b.ndim > 1 else 1)
        return np.matmul(a, b).view(Counted)

    def __rmatmul__(self, other):
        a, b = np.asarray(other), np.asarray(self)
        COUNT[0] += 2 * a.size * (b.shape[-1] if b.ndim > 1 else 1)
        return np.matmul(a, b).view(Counted)


def n_params(Kx, Ky, m, n, a, l):
    return (Kx + Ky) * m + 2 * (3 * n * m + 3 * n * n) + (3 * n * m + 3 * n * n + 6 * n * n + n * n) + (
        n * a + 2 * n * a + a) + (2 * l * n + 2 * l * m + 4 * l * n + l * Ky)  # fmt: skip


def att_flops(Tx, Ty, n, a, precompute=True):
    return 4 * Tx * n * a * (1 if precompute else Ty) + 2 * Ty * n * a + 2 * Tx * Ty * (a + 2 * n)


def rest_flops(Tx, Ty, Ky, m, n, l):
    enc = 2 * Tx * 2 * (3 * n * m + 3 * n * n)
    dec = Ty * 2 * (3 * n * m + 3 * n * n + 6 * n * n)
    out = Ty * 2 * (2 * l * n + 2 * l * m + 4 * l * n + l * Ky)
    return enc + 2 * n * n + dec + out


rng = np.random.default_rng(2)
for _ in range(30):
    Kx, Ky, m, n, a, l, Tx, Ty = (int(v) for v in rng.integers([2, 2, 1, 1, 1, 1, 1, 1], [9, 9, 5, 6, 6, 4, 7, 6]))
    cfg = attention.config(Kx, Ky, m, n, a, l)
    assert attention.param_count(cfg) == n_params(Kx, Ky, m, n, a, l)
    p = {k: v.view(Counted) for k, v in attention.init(rng, cfg, 0.5).items()}
    src, tgt = rng.integers(0, Kx, (1, Tx)), rng.integers(0, Ky, (1, Ty))
    COUNT[0] = 0
    attention.forward(p, cfg, src, tgt)
    context = 2 * Tx * Ty * 2 * n  # the einsum of eq-context, not seen by Counted
    assert COUNT[0] + context == att_flops(Tx, Ty, n, a) + rest_flops(Tx, Ty, Ky, m, n, l), (COUNT[0], cfg)
    COUNT[0] = 0
    attention.forward(p, attention.config(Kx, Ky, m, n, a, l, attention=False), src, tgt)
    assert COUNT[0] == rest_flops(Tx, Ty, Ky, m, n, l) + 4 * Tx * n * a  # U_a h_j is still computed, then unused

sizes = (30000, 30000, 620, 1000, 1000, 500)
assert n_params(*sizes) == 80_401_000
assert 3 * 1000 * 1000 + 1000 == 3_001_000 and round(100 * 3_001_000 / 80_401_000, 2) == 3.73
assert att_flops(30, 30, 1000, 1000) == 185_400_000
assert att_flops(30, 30, 1000, 1000, precompute=False) == 3_665_400_000
total = att_flops(30, 30, 1000, 1000) + rest_flops(30, 30, 30000, 620, 1000, 500)
assert total == 2_539_400_000 and round(100 * 185_400_000 / total, 1) == 7.3
assert round(3_665_400_000 / 185_400_000, 1) == 19.8
print("parameter-and-cost: ok")
