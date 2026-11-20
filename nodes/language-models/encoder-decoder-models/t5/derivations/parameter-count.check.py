# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/t5/ (ai://NEW-t5).
"""Numeric check of the derivation parameter-count: random configs, the five sizes of the paper, and the share of the
attention to the encoder."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import t5  # noqa: E402

rng = np.random.default_rng(0)
for _ in range(60):
    V, M, H, d, F, L = (int(v) for v in rng.integers([2, 1, 1, 1, 1, 1], [20, 9, 4, 4, 9, 4]))
    K = int(rng.integers(4, 10))
    pos = ("relative", "none")[rng.integers(2)]
    cfg = t5.config(V, M, H, d, F, L, K, 2 * K, pos)
    I = H * d  # noqa: E741
    want = V * M + 2 * K * H * (pos == "relative") + L * (12 * M * I + 4 * M * F + 5 * M) + 2 * M
    assert sum(a.size for a in t5.init(rng, cfg).values()) == t5.param_count(cfg) == want

V = 32000
sizes = {  # d_model, n_heads, d_kv, d_ff, n_layers of Section 3.7, and the count each gives
    "small": ((512, 8, 64, 2048, 6), 60_441_088),
    "base": ((768, 12, 64, 3072, 12), 222_805_248),
    "large": ((1024, 16, 64, 4096, 24), 737_537_024),
    "3b": ((1024, 32, 128, 16384, 24), 2_851_467_264),
    "11b": ((1024, 128, 128, 65536, 24), 11_307_190_272),
}
counts, cross = {}, {}
for name, ((M, H, d, F, L), want) in sizes.items():
    cfg = t5.config(V, M, H, d, F, L, 32, 128)
    counts[name] = t5.param_count(cfg)
    assert counts[name] == want, name
    cross[name] = 4 * L * M * H * d / counts[name]
assert 12 * 768**2 + 4 * 768 * 3072 + 5 * 768 == 16_518_912 and 12 * 16_518_912 == 198_226_944
assert 24_576_000 + 768 + 198_226_944 + 1536 == counts["base"]
assert round(counts["small"] / 1e6) == 60 and round(counts["base"] / 1e7) == 22
assert round(counts["3b"] / 1e8) == 29 and counts["3b"] // 10**8 == 28 and round(counts["11b"] / 1e9) == 11
assert round(counts["large"] / 1e6) == 738 and 770e6 - counts["large"] > 32e6
assert counts["large"] + V * 1024 == 770_305_024 and counts["base"] + V * 768 == 247_381_248
assert 12 * 4 * 768**2 == 28_311_552 and round(cross["base"], 4) == 0.1271
assert 24 * 4 * 1024 * 16384 == 1_610_612_736 and round(cross["11b"], 4) == 0.1424
assert round(cross["large"], 3) == 0.136 and all(cross[k] > 0.12 for k in sizes if k != "small")
one_table = t5.param_count(t5.config(V, 1024, 128, 128, 65536, 24, 32, 128)) - 32 * 128
assert round(one_table / 1e9) == 11 and 32 * 128 == 4096
print("ok")
