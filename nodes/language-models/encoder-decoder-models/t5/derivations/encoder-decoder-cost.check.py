# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/t5/ (ai://NEW-t5).
"""Numeric check of the derivation encoder-decoder-cost: the parameters and FLOPs of an L + L encoder-decoder against
those of a language model of 2L and of L layers."""

import pathlib
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import t5  # noqa: E402


def lm_layer_params(M, I, F):  # noqa: E741
    return 4 * M * I + 2 * M * F


def lm_flops(L, M, I, F, n):  # noqa: E741
    return L * (n * (8 * M * I + 4 * M * F) + 4 * n * n * I)


rng = np.random.default_rng(0)
for _ in range(60):
    V, M, H, d, F, L, S, T = (int(v) for v in rng.integers([2, 1, 1, 1, 1, 1, 1, 1], [20, 9, 4, 4, 9, 4, 12, 12]))
    I = H * d  # noqa: E741
    cfg = t5.config(V, M, H, d, F, L, pos="none")
    layers = t5.param_count(cfg) - V * M - 2 * M - 5 * M * L
    assert layers == L * (12 * M * I + 4 * M * F)
    assert 2 * L * lm_layer_params(M, I, F) == L * (8 * M * I + 4 * M * F)
    ed = t5.flops(cfg, 1, S, T) - 2 * T * M * V
    matrices = L * (S + T) * (12 * M * I + 4 * M * F)
    assert ed == matrices + 4 * L * I * (S * S + S * T + T * T)
    assert lm_flops(L, M, I, F, S + T) - L * (S + T) * (8 * M * I + 4 * M * F) == 4 * L * I * (S + T) ** 2
    assert lm_flops(L, M, I, F, S + T) - (ed - matrices) - L * (S + T) * (8 * M * I + 4 * M * F) == 4 * L * I * S * T

M, F = 768, 3072
assert Fraction(12 * M * M + 4 * M * F, 8 * M * M + 4 * M * F) == Fraction(7, 6)
assert Fraction(4 * M * M, 12 * M * M + 4 * M * F) == Fraction(1, 7)
n, V = 512, 32000
base = t5.config(V, 768, 12, 64, 3072, 12, 32, 128)
ed = t5.flops(base, 1, n, n) - 2 * n * 768 * V
lm = lm_flops(12, 768, 768, 3072, 2 * n)
assert ed == 231_928_233_984 and lm == 212_600_881_152 and Fraction(ed, lm) == Fraction(12, 11)
assert ed == 12 * (2 * n * 28 * M * M + 12 * n * n * M) and lm == 12 * (2 * n * 24 * M * M + 16 * n * n * M)
assert 56 * M + 12 * n == 49_152 and 48 * M + 16 * n == 45_056
assert Fraction(4 * n * n * 3, 4 * (2 * n) ** 2) == Fraction(3, 4)
half = t5.config(V, 768, 12, 64, 3072, 6, 32, 128)
assert 2 * (t5.flops(half, 1, n, n) - 2 * n * 768 * V) == ed
print("ok")
