# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/decoder-only-transformer/ (ai://NEW-decoder-only-transformer).
"""Numeric check of the derivation parameter-count: random configs, the models of GPT-2 and GPT-3, and torch."""

import importlib.util
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import decoder_lm as lm  # noqa: E402

rng = np.random.default_rng(0)
for _ in range(60):
    V, T, H, Dh, F, L = (int(v) for v in rng.integers([2, 1, 1, 1, 1, 1], [20, 10, 4, 4, 9, 4]))
    D = H * Dh
    tied, pos = bool(rng.integers(2)), ("learned", "none")[rng.integers(2)]
    cfg = lm.config(V, T, D, H, F, L, tied, pos)
    want = V * D + T * D * (pos == "learned") + L * (4 * D * D + 2 * D * F + 9 * D + F) + 2 * D + D * V * (not tied)
    params = lm.init(rng, cfg)
    assert sum(a.size for a in params.values()) == lm.param_count(cfg) == want
    assert lm.non_embedding_params(cfg) == L * (4 * D * D + 2 * D * F + 9 * D + F)

V = 50257
gpt3 = {  # (n_layers, d_model, n_heads) of Table 2.1, d_model 5120 for 13B; Table D.1 in millions, and its precision
    "small": ((12, 768, 12), 125, 1),
    "medium": ((24, 1024, 16), 356, 1),
    "large": ((24, 1536, 16), 760, 1),
    "xl": ((24, 2048, 16), 1320, 10),
    "2.7b": ((32, 2560, 32), 2650, 10),
    "6.7b": ((32, 4096, 32), 6660, 10),
    "13b": ((40, 5120, 40), 12850, 10),
    "175b": ((96, 12288, 96), 174600, 100),
}
want3 = [125_226_240, 355_871_744, 760_300_032, 1_315_723_264, 2_651_553_280, 6_658_404_352, 12_853_386_240]
want3.append(174_604_259_328)
for ((L, D, H), table, unit), want in zip(gpt3.values(), want3, strict=True):
    n = lm.param_count(lm.config(V, 2048, D, H, 4 * D, L))
    assert n == want and round(n / 1e6 / unit) * unit == table, (n, table)
small = lm.config(V, 2048, 768, 12, 3072, 12)
assert lm.non_embedding_params(small) == 12 * 7_087_872 == 85_054_464 and (V + 2048) * 768 == 40_170_240
assert 85_054_464 + 40_170_240 + 1536 == lm.param_count(small)

wide = lm.param_count(lm.config(V, 2048, 5140, 20, 4 * 5140, 40))  # 20 heads divide 5140; the heads change no count
assert wide == 12_952_938_780 and round(wide / 1e7) * 10 == 12950 and wide - 12_850e6 > 100e6 and 40 * 128 == 5120

gpt2 = {117: (12, 768), 345: (24, 1024), 762: (36, 1280), 1542: (48, 1600)}
counts = [lm.param_count(lm.config(V, 1024, D, 4, 4 * D, L)) for L, D in gpt2.values()]
assert counts == [124_439_808, 354_823_168, 774_030_080, 1_557_611_200]
assert all(c > m * 1e6 for c, m in zip(counts, gpt2, strict=True))
assert (V + 1024) * 768 == 39_383_808 and round(39_383_808 / counts[0], 4) == 0.3165

for D, L in [(768, 12), (12288, 96), (1600, 48)]:
    N = lm.non_embedding_params(lm.config(V, 1024, D, 4, 4 * D, L))
    assert N == 12 * L * D * D + 13 * L * D and abs(N / (12 * L * D * D) - 1 - 13 / (12 * D)) < 1e-15
assert round(13 / (12 * 768), 4) == 0.0014 and round(13 / (12 * 12288), 6) == 0.000088
big = lm.config(V, 2048, 12288, 96, 49152, 96)
emb = (V + 2048) * 12288
assert emb == 642_723_840 and lm.param_count(big) - lm.non_embedding_params(big) == emb + 2 * 12288
assert round(emb / lm.param_count(big), 5) == 0.00368

if importlib.util.find_spec("torch") is not None:
    import torch

    for D, H, F in [(8, 2, 16), (768, 12, 3072), (1600, 25, 6400)]:
        layer = torch.nn.TransformerEncoderLayer(D, H, F, activation="gelu", norm_first=True, batch_first=True)
        assert sum(p.numel() for p in layer.parameters()) == 4 * D * D + 2 * D * F + 9 * D + F
print("ok")
