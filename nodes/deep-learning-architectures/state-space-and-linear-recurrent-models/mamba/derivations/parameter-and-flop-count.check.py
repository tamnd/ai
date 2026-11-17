# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/mamba/ (ai://NEW-mamba).
"""Numeric check of the derivation parameter-and-flop-count: the formulas, the reference and the released models."""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import mamba  # noqa: E402

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


def n_params(V, D, L, I, N, K, R):
    return V * D + D + L * (3 * I * D + I * (K + 1) + I * (R + 2 * N) + (R + 1) * I + I * N + I + D)


def n_flops(B, T, V, D, L, I, N, K, R):
    return 2 * B * T * (L * (2 * D * I + I * K + I * (R + 2 * N) + R * I + 2 * I * N + I * D) + D * V)


def sizes(cfg):
    return tuple(cfg[k] for k in ("vocab", "d_model", "n_layers", "d_inner", "d_state", "d_conv", "dt_rank"))


rng = np.random.default_rng(0)
for _ in range(40):
    V, D, L, E, N, K, R, B, T = (int(v) for v in rng.integers([3, 1, 1, 1, 1, 1, 1, 1, 1], [20, 12, 4, 4, 7, 5, 4, 4, 7]))
    cfg = mamba.config(vocab=V, d_model=D, n_layers=L, expand=E, d_state=N, d_conv=K, dt_rank=R)
    shapes = mamba.param_shapes(cfg)
    assert sum(math.prod(s) for s in shapes.values()) == n_params(*sizes(cfg)) == mamba.param_count(cfg)
    p = {k: v.view(Counted) for k, v in mamba.init(rng, cfg).items()}
    COUNT[0] = 0
    mamba.forward(p, cfg, rng.integers(0, V, (B, T)))
    I = cfg["d_inner"]
    conv_and_scan = B * T * L * (2 * I * K + 4 * I * N)
    assert COUNT[0] + conv_and_scan == n_flops(B, T, *sizes(cfg)) == mamba.flops(cfg, B, T), (COUNT[0], cfg)

released = {768: (24, 129_135_360), 1024: (48, 371_516_416), 1536: (48, 793_204_224), 2048: (48, 1_372_178_432),
            2560: (64, 2_768_345_600)}  # fmt: skip
for D, (L, want) in released.items():
    R = math.ceil(D / 16)
    assert n_params(50280, D, L, 2 * D, 16, 4, R) == want, D
assert 50277 % 8 != 0 and math.ceil(50277 / 8) * 8 == 50280

block = 3 * 5120 * 2560 + 5120 * 5 + 5120 * (160 + 32) + 161 * 5120 + 5120 * 16 + 5120 + 2560
assert block == 41_244_160 and 64 * block + 50280 * 2560 + 2560 == 2_768_345_600
assert 2 * 3 * 2 * 64**2 == 12 * 64**2  # two blocks with E = 2 have the 12 D^2 of a Transformer layer

abl = {(16, 64): 371.5, (1, 64): 367.1, (16, 1): 359.1, (16, 0): 358.9}
for (N, R), want in abl.items():
    assert round(n_params(50280, 1024, 48, 2048, N, 4, R) / 1e6, 1) == want, (N, R)

C = n_flops(1, 2048, 50280, 2560, 64, 5120, 16, 4, 160)
scan = 4 * 2048 * 64 * 5120 * 16
assert C == 11_355_910_307_840 and C // 2048 == 5_544_878_080 and scan == 42_949_672_960
assert abs(scan / C - 0.00378) < 1e-5
assert abs(C / 2048 / (2 * 2_768_345_600) - 1 - 0.00148) < 1e-5
print("parameter-and-flop-count: ok")
