# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/deep-sets/ (ai://NEW-deep-sets).
"""Numeric check of the derivation parameter-and-flop-count: the formulas, the reference and the point-cloud models."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import deepsets  # noqa: E402

COUNT = [0]


class Counted(np.ndarray):
    def __matmul__(self, other):
        a, b = np.asarray(self), np.asarray(other)
        COUNT[0] += 2 * a.size * b.shape[-1]
        return np.matmul(a, b).view(Counted)

    def __rmatmul__(self, other):
        a, b = np.asarray(other), np.asarray(self)
        COUNT[0] += 2 * a.size * b.shape[-1]
        return np.matmul(a, b).view(Counted)


def n_params(D, H, L, K):
    return 2 * D * H + H + (L - 1) * (2 * H * H + H) + H * H + H + H * K + K


def n_flops(M, D, H, L, K):
    return 2 * (M + 1) * D * H + (L - 1) * 2 * (M + 1) * H * H + 2 * H * H + 2 * H * K


rng = np.random.default_rng(0)
for _ in range(60):
    D, H, L, K, M, B = (int(v) for v in rng.integers([1, 1, 1, 2, 1, 1], [5, 9, 4, 6, 9, 4]))
    head = ("set", "element")[rng.integers(2)]
    interact = bool(rng.integers(2)) if head == "element" else True
    pool = deepsets.POOLS[rng.integers(3)]
    cfg = deepsets.config(D, H, L, K, pool=pool, readout=pool, interact=interact, head=head)
    params = deepsets.init(rng, cfg)
    assert sum(a.size for a in params.values()) == deepsets.param_count(cfg)
    if head == "set":
        assert deepsets.param_count(cfg) == n_params(D, H, L, K)
        assert deepsets.flops(cfg, B, M) == B * n_flops(M, D, H, L, K)
    COUNT[0] = 0
    deepsets.forward({k: v.view(Counted) for k, v in params.items()}, cfg, rng.normal(size=(B, M, D)).view(Counted))
    assert COUNT[0] == deepsets.flops(cfg, B, M), (COUNT[0], deepsets.flops(cfg, B, M))

assert n_params(3, 256, 3, 40) == 1792 + 262656 + 76072 == 340520
assert n_params(3, 512, 3, 40) == 3584 + 1049600 + 283176 == 1336360
C = n_flops(1000, 3, 256, 3, 40)
assert C == 1537536 + 262406144 + 151552 == 264095232
pool_terms = 2 * 3 * 256 + 2 * 2 * 256 * 256
assert pool_terms == 263680 and round(100 * pool_terms / C, 4) == 0.0998
attn = 4 * 1000**2 * 256
assert attn == 1024000000 and round(attn / C, 2) == 3.88
cfg = deepsets.config(3, 256, 3, 40)
assert deepsets.param_count(cfg) == 340520 and deepsets.flops(cfg, 1, 1000) == C

print("parameter-and-flop-count: ok")
