# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/generative-adversarial-network/ (ai://NEW-generative-adversarial-network).
"""Numeric check of the derivation parameter-and-flop-count."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import gan  # noqa: E402

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


def n_params(J, H, D, K):
    return (J + 1) * H + (H + 1) * D, (D + 1) * K + K + 1


def n_flops(B, J, H, D, K):
    return 2 * B * (J * H + H * D) + 4 * B * (D * K + K)


rng = np.random.default_rng(0)
for _ in range(40):
    J, H, D, K, B = (int(v) for v in rng.integers(1, [6, 9, 7, 9, 7]))
    cfg = gan.config(J, H, D, K)
    p = gan.init(rng, cfg)
    pg, pd = n_params(J, H, D, K)
    assert sum(p[k].size for k in gan.GEN) == pg and sum(p[k].size for k in gan.DIS) == pd
    assert gan.param_count(cfg) == (pg, pd)
    COUNT[0] = 0
    q = {k: v.view(Counted) for k, v in p.items()}
    fake = gan.generate(q, rng.standard_normal((B, J)))[0]
    gan.critic(q, fake)
    gan.critic(q, rng.standard_normal((B, D)))
    assert COUNT[0] == n_flops(B, J, H, D, K) == gan.flops_forward(cfg, B), (COUNT[0], J, H, D, K, B)

assert n_params(1, 16, 1, 16) == (49, 49) and n_flops(128, 1, 16, 1, 16) == 24_576
assert n_params(2, 16, 2, 16) == (82, 65) and n_flops(128, 2, 16, 2, 16) == 40_960
print("parameter-and-flop-count: ok")
