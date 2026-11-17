# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/vision-transformer/ (ai://NEW-vision-transformer).
"""Numeric check of the derivation parameter-count: random configs, the models of the paper, torch.nn.TransformerEncoderLayer."""

import importlib.util
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import vit  # noqa: E402

rng = np.random.default_rng(0)
for _ in range(60):
    G, P, C, H, Dh, F, L, K = (int(v) for v in rng.integers([1, 1, 1, 1, 1, 1, 1, 2], [5, 4, 4, 4, 4, 9, 4, 6]))
    D = H * Dh
    pool, pos = ("cls", "gap")[rng.integers(2)], ("learned", "none")[rng.integers(2)]
    Hh = int(rng.integers(0, 4))
    cfg = vit.config(G * P, P, C, D, H, F, L, K, Hh, pool, pos)
    N = G * G
    n = N + (pool == "cls")
    embed = P * P * C * D + D + D * (pool == "cls") + n * D * (pos == "learned")
    head = 2 * D + (D * Hh + Hh + Hh * K + K if Hh else D * K + K)
    want = embed + L * (4 * D * D + 2 * D * F + 9 * D + F) + head
    params = vit.init(rng, cfg)
    assert sum(a.size for a in params.values()) == vit.param_count(cfg) == want

models = {
    "B/16": (14, 16, 768, 12, 3072, 12),
    "B/32": (7, 32, 768, 12, 3072, 12),
    "L/16": (14, 16, 1024, 16, 4096, 24),
    "L/32": (7, 32, 1024, 16, 4096, 24),
    "H/14": (16, 14, 1280, 16, 5120, 32),
    "B/16@384": (24, 16, 768, 12, 3072, 12),
}
count = {k: vit.param_count(vit.config(G * P, P, 3, D, H, F, L, 1000)) for k, (G, P, D, H, F, L) in models.items()}
assert count["B/16"] == 742_656 + 12 * 7_087_872 + 770_536 == 86_567_656
assert (count["L/16"], count["L/32"], count["H/14"], count["B/32"]) == (304_326_632, 306_535_400, 632_045_800, 88_224_232)
assert math.floor(count["B/16"] / 1e6) == 86 and round(count["L/32"] / 1e6) == 307 and round(count["H/14"] / 1e6) == 632
assert round(12 * 7_087_872 / count["B/16"], 3) == 0.983
assert count["B/16@384"] - count["B/16"] == (577 - 197) * 768 == 291_840

if importlib.util.find_spec("torch") is not None:
    import torch

    for D, H, F in [(8, 2, 16), (768, 12, 3072), (1024, 16, 4096)]:
        layer = torch.nn.TransformerEncoderLayer(D, H, F, activation="gelu", norm_first=True, batch_first=True)
        assert sum(p.numel() for p in layer.parameters()) == 4 * D * D + 2 * D * F + 9 * D + F
print("ok")
