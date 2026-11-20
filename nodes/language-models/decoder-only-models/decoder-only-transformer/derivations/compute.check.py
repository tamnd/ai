# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/decoder-only-transformer/ (ai://NEW-decoder-only-transformer).
"""Numeric check of the derivation compute: the FLOPs of the products of the reference, dense and with the cache, and
the GPT-3 figures of statements 2 to 4 against Table D.1."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import decoder_lm as lm  # noqa: E402

COUNT = [0]


def counted_matmul(a, b):
    COUNT[0] += 2 * a.size * b.shape[-1]  # (..., m, k) @ (..., k, p) with a broadcast batch: 2 m k p per batch
    return np.matmul(a, b)


class Counted(np.ndarray):
    def __matmul__(self, other):
        return counted_matmul(np.asarray(self), np.asarray(other)).view(Counted)

    def __rmatmul__(self, other):
        return counted_matmul(np.asarray(other), np.asarray(self)).view(Counted)


_softmax = lm.softmax
lm.softmax = lambda s: _softmax(s).view(Counted)  # the cached values come back from np.concatenate uncounted


def causal(cfg, B, n):
    V, D, F, L = (cfg[k] for k in ("vocab", "d_model", "d_ff", "n_layers"))
    return B * (L * (8 * n * D * D + 2 * n * (n + 1) * D + 4 * n * D * F) + 2 * n * D * V)


rng = np.random.default_rng(0)
for _ in range(30):
    V, H, Dh, F, L, B, n = (int(v) for v in rng.integers([2, 1, 1, 1, 1, 1, 1], [12, 4, 4, 9, 4, 4, 9]))
    D = H * Dh
    cfg = lm.config(V, 8, D, H, F, L, bool(rng.integers(2)), ("learned", "none")[rng.integers(2)])
    params = {k: v.view(Counted) for k, v in lm.init(rng, cfg).items()}
    ids = rng.integers(0, V, (B, n))
    COUNT[0] = 0
    lm.forward(params, cfg, ids)
    want = B * (L * (8 * n * D * D + 4 * n * n * D + 4 * n * D * F) + 2 * n * D * V)
    assert COUNT[0] == want == lm.flops(cfg, B, n), (COUNT[0], want)
    COUNT[0] = 0  # one token at a time with the cache: query t meets t keys, the causal count
    _, cache = lm.forward(params, cfg, ids[:, :1])
    for t in range(1, n):
        _, cache = lm.forward(params, cfg, ids[:, t : t + 1], past=cache["kv"])
    assert COUNT[0] == causal(cfg, B, n), (COUNT[0], causal(cfg, B, n))

V, n = 50257, 2048
gpt3 = {  # (n_layers, d_model, n_heads); Table D.1: params (M), total train compute (FLOPs), (PF-days)
    "small": ((12, 768, 12), 125, 2.25e20, 2.60),
    "medium": ((24, 1024, 16), 356, 6.41e20, 7.42),
    "large": ((24, 1536, 16), 760, 1.37e21, 15.8),
    "xl": ((24, 2048, 16), 1320, 2.38e21, 27.5),
    "2.7b": ((32, 2560, 32), 2650, 4.77e21, 55.2),
    "6.7b": ((32, 4096, 32), 6660, 1.20e22, 139),
    "13b": ((40, 5120, 40), 12850, 2.31e22, 268),
    "175b": ((96, 12288, 96), 174600, 3.14e23, 3640),
}
ratios, shares, gaps, printed = {}, {}, {}, {}
for name, ((L, D, H), params_m, flops_d1, pf_days) in gpt3.items():
    cfg = lm.config(V, n, D, H, 4 * D, L)
    N_total, N = lm.param_count(cfg), lm.non_embedding_params(cfg)
    per_token = causal(cfg, 1, n) / n
    assert per_token == L * (8 * D * D + 16 * D * D) + 2 * D * V + 2 * L * (n + 1) * D
    assert per_token == 2 * N - 2 * L * (9 * D + 4 * D) + 2 * D * V + 2 * L * (n + 1) * D
    ratios[name] = per_token / (2 * N_total)
    layer = 24 * D * D + 2 * (n + 1) * D
    shares[name] = 2 * (n + 1) * D / layer
    assert abs(shares[name] - (n + 1) / (n + 1 + 12 * D)) < 1e-15
    gaps[name] = 6 * N_total * 300e9 / flops_d1 - 1
    printed[name] = 6 * params_m * 1e6 * 300e9 / flops_d1 - 1
    assert abs(6 * params_m * 1e6 * 300e9 / 8.64e19 / pf_days - 1) < 0.0025, name
assert round(ratios["small"], 3) == 1.137 and round(ratios["175b"], 3) == 1.014
assert all(ratios["175b"] <= r <= ratios["small"] for r in ratios.values())
assert causal(lm.config(V, n, 768, 12, 3072, 12), 1, n) / n == 284_831_232
assert causal(lm.config(V, n, 12288, 96, 49152, 96), 1, n) / n == 353_961_664_512
rounded = {k: round(100 * v, 1) for k, v in shares.items()}
assert (rounded["small"], rounded["medium"], rounded["large"], rounded["xl"], rounded["175b"]) == (
    18.2,
    14.3,
    10.0,
    7.7,
    1.4,
)
assert shares["large"] > 0.1 > shares["xl"] and 12 * 1536 <= 9 * (n + 1) < 12 * 1537
assert max(abs(g) for g in gaps.values()) == abs(gaps["xl"]) and abs(gaps["xl"]) < 0.005
assert max(abs(g) for g in printed.values()) < 0.0017
big = 6 * 174_604_259_328 * 300e9
assert round(big / 1e21) / 100 == 3.14 and round(big / 8.64e19, 1) == 3637.6
print("ok")
