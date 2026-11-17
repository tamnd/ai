# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/vision-transformer/ (ai://NEW-vision-transformer).
"""Numeric check of the derivation compute: the FLOPs of the products of the reference and Table 6 of the paper."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import vit  # noqa: E402

COUNT = [0]


def counted_matmul(a, b):
    COUNT[0] += 2 * a.size * b.shape[-1]  # (..., m, k) @ (..., k, n) with a broadcast batch: 2 m k n per batch
    return np.matmul(a, b)


class Counted(np.ndarray):
    def __matmul__(self, other):
        return counted_matmul(np.asarray(self), np.asarray(other)).view(Counted)

    def __rmatmul__(self, other):
        return counted_matmul(np.asarray(other), np.asarray(self)).view(Counted)


rng = np.random.default_rng(0)
for _ in range(30):
    G, P, C, H, Dh, F, L, K, B = (int(v) for v in rng.integers([1, 1, 1, 1, 1, 1, 1, 2, 1], [5, 4, 4, 4, 4, 9, 4, 6, 4]))
    D = H * Dh
    cfg = vit.config(G * P, P, C, D, H, F, L, K)
    n = G * G + 1
    params = {k: v.view(Counted) for k, v in vit.init(rng, cfg).items()}
    COUNT[0] = 0
    vit.forward(params, cfg, rng.normal(size=(B, G * P, G * P, C)).view(Counted))
    want = B * (2 * G * G * P * P * C * D + L * (8 * n * D * D + 4 * n * n * D + 4 * n * D * F) + 2 * D * K)
    assert COUNT[0] == want == vit.flops(cfg, B), (COUNT[0], want)

models = {
    "B/32": ((7, 32, 768, 12, 3072, 12), 7, 55),
    "B/16": ((14, 16, 768, 12, 3072, 12), 7, 224),
    "L/32": ((7, 32, 1024, 16, 4096, 24), 7, 196),
    "L/16": ((14, 16, 1024, 16, 4096, 24), 7, 783),
    "L/16 14": ((14, 16, 1024, 16, 4096, 24), 14, 1567),
    "H/14": ((16, 14, 1280, 16, 5120, 32), 14, 4262),
}
for name, ((G, P, D, H, F, L), epochs, table) in models.items():
    cfg = vit.config(G * P, P, 3, D, H, F, L, 1000)
    ef = 3 * vit.flops(cfg, 1) * epochs * 303e6 / 1e18
    rel = ef / table - 1
    assert abs(rel) < (0.021 if name == "B/32" else 0.003), (name, ef, table)
    extra = 2 * D * (D + 18_000 - 1000)  # the pre-training head: D hidden units and the 18k classes of JFT
    assert extra / vit.flops(cfg, 1) < 0.0032

b16 = vit.config(224, 16, 3, 768, 12, 3072, 12, 1000)
b16_384 = vit.config(384, 16, 3, 768, 12, 3072, 12, 1000)
assert vit.flops(b16, 1) == 35_127_656_448 and vit.flops(b16_384, 1) == 110_968_700_928
for cfg, share in [(b16, 0.0410), (b16_384, 0.1113)]:
    n, D, F = vit.n_tokens(cfg), cfg["d_model"], cfg["d_ff"]
    layer = 8 * n * D * D + 4 * n * n * D + 4 * n * D * F
    assert 4 * n * n * D / layer == n / (n + 2 * D + F) and round(n / (n + 2 * D + F), 4) == share
print("ok")
