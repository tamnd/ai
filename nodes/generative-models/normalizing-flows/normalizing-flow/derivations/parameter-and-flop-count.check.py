# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/normalizing-flow/ (ai://NEW-normalizing-flow).
"""Numeric check of the derivation parameter-and-flop-count."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import flow  # noqa: E402

COUNT = [0]


def cost(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return 2 * a.size * (b.shape[-1] if b.ndim == 2 else 1)


class Counted(np.ndarray):
    def __matmul__(self, other):
        COUNT[0] += cost(self, other)
        return np.matmul(np.asarray(self), np.asarray(other)).view(Counted)

    def __rmatmul__(self, other):
        COUNT[0] += cost(other, self)
        return np.matmul(np.asarray(other), np.asarray(self)).view(Counted)


def n_params(D, H, L):
    return L * (3 * D * H + H + 2 * D)


def n_flops(B, D, H, L):
    return 6 * L * B * D * H


rng = np.random.default_rng(6)
for _ in range(40):
    D, H, L, B = (int(v) for v in rng.integers([2, 1, 1, 1], [7, 9, 6, 7]))
    cfg = flow.config(D, H, L)
    p = flow.init(rng, cfg)
    assert sum(v.size for v in p.values()) == n_params(D, H, L) == flow.param_count(cfg)
    q = {k: v.view(Counted) for k, v in p.items()}
    COUNT[0] = 0
    flow.forward(q, cfg, rng.standard_normal((B, D)))
    assert COUNT[0] == n_flops(B, D, H, L) == flow.flops_forward(cfg, B), (COUNT[0], D, H, L, B)
    COUNT[0] = 0
    flow.inverse(q, cfg, rng.standard_normal((B, D)))
    assert COUNT[0] == n_flops(B, D, H, L)

assert n_params(3, 5, 3) == 168 and n_flops(4, 3, 5, 3) == 1_080
assert n_params(2, 32, 4) == 912 and n_flops(256, 2, 32, 4) == 393_216
print("parameter-and-flop-count: ok")
