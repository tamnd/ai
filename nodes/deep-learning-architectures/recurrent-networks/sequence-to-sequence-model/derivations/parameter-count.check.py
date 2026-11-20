# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/sequence-to-sequence-model/ (ai://NEW-sequence-to-sequence-model).
"""Numeric check of the derivation parameter-count: the counts of random configs and those of Sutskever et al."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import seq2seq  # noqa: E402

COUNT = [0]


def counted_matmul(a, b):
    COUNT[0] += 2 * a.size * b.shape[-1]  # (..., k) @ (k, n): 2 m k n
    return np.matmul(a, b)


class Counted(np.ndarray):
    def __matmul__(self, other):
        return counted_matmul(np.asarray(self), np.asarray(other)).view(Counted)

    def __rmatmul__(self, other):
        return counted_matmul(np.asarray(other), np.asarray(self)).view(Counted)


rng = np.random.default_rng(0)
for _ in range(30):
    Vx, Vy, E, H, L, B, S, T = (int(v) for v in rng.integers([2, 2, 1, 1, 1, 1, 1, 1], [9, 9, 5, 6, 4, 4, 6, 6]))
    cfg = seq2seq.config(Vx, Vy, E, H, L)
    params = seq2seq.init(rng, cfg)
    stack = 4 * H * (E + H) + 4 * H + (L - 1) * (8 * H * H + 4 * H)
    N = (Vx + Vy) * E + 2 * stack + H * Vy + Vy
    assert sum(a.size for a in params.values()) == seq2seq.param_count(cfg) == N
    src, tgt = rng.integers(0, Vx, (B, S)), rng.integers(0, Vy, (B, T))
    COUNT[0] = 0
    seq2seq.forward({k: v.view(Counted) for k, v in params.items()}, cfg, src, tgt)
    C = B * (S + T) * (8 * H * (E + H) + 16 * (L - 1) * H * H) + 2 * B * T * H * Vy
    assert COUNT[0] == C == seq2seq.flops(cfg, B, S, T), (COUNT[0], C)

big = seq2seq.config(160_000, 80_000, 1000, 1000, 4)
assert seq2seq.weight_count(big) == 384_000_000
assert seq2seq.recurrent_weight_count(big) == 64_000_000
assert (160_000 + 80_000) * 1000 == 240_000_000 and 1000 * 80_000 == 80_000_000
assert seq2seq.state_size(big) == 8000
assert seq2seq.param_count(big) == 384_112_000
per_step = 8 * 1000 * 2000 + 3 * 16 * 1000**2
assert per_step == 64_000_000 and 2 * 1000 * 80_000 / per_step == 2.5
example = seq2seq.config()
assert seq2seq.param_count(example) == 613 and seq2seq.flops(example, 2, 5, 4) == 9024
print("ok")
