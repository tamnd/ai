# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/decoder-only-transformer/ (ai://NEW-decoder-only-transformer).
"""Numeric check of the derivation kv-cache: the size of the cache of the reference, the products of greedy decoding
with and without it, and the GPT-3 figures."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import decoder_lm as lm  # noqa: E402

COUNT = [0]


def counted_matmul(a, b):
    COUNT[0] += 2 * a.size * b.shape[-1]
    return np.matmul(a, b)


class Counted(np.ndarray):
    def __matmul__(self, other):
        return counted_matmul(np.asarray(self), np.asarray(other)).view(Counted)

    def __rmatmul__(self, other):
        return counted_matmul(np.asarray(other), np.asarray(self)).view(Counted)


_softmax = lm.softmax
lm.softmax = lambda s: _softmax(s).view(Counted)  # the cached values come back from np.concatenate uncounted


def per_token(cfg):
    V, D, F, L = (cfg[k] for k in ("vocab", "d_model", "d_ff", "n_layers"))
    return L * (8 * D * D + 4 * D * F) + 2 * D * V


rng = np.random.default_rng(0)
for _ in range(20):
    V, H, Dh, F, L, B, n = (int(v) for v in rng.integers([2, 1, 1, 1, 1, 1, 1], [12, 4, 4, 9, 4, 4, 9]))
    D = H * Dh
    cfg = lm.config(V, 8, D, H, F, L, bool(rng.integers(2)), ("learned", "none")[rng.integers(2)])
    params = {k: v.view(Counted) for k, v in lm.init(rng, cfg, scale=0.5).items()}
    ids = rng.integers(0, V, (B, n))
    _, cache = lm.forward(params, cfg, ids)
    assert sum(k.size + v.size for k, v in cache["kv"]) == B * lm.kv_cache_size(cfg, n) == 2 * B * L * D * n
    P = per_token(cfg)
    COUNT[0] = 0
    fast = lm.generate(params, cfg, ids[:, :1], n)  # feeds the prompt and n - 1 new tokens: n tokens in all
    assert COUNT[0] == B * (n * P + 2 * L * D * n * (n + 1)), (COUNT[0], n)
    COUNT[0] = 0
    slow = lm.generate_without_cache(params, cfg, ids[:, :1], n)  # forward passes over 1, 2, ..., n tokens
    want = P * n * (n + 1) // 2 + 4 * L * D * n * (n + 1) * (2 * n + 1) // 6
    assert COUNT[0] == B * want and 3 * want == 3 * P * n * (n + 1) // 2 + 2 * L * D * n * (n + 1) * (2 * n + 1)
    np.testing.assert_array_equal(fast, slow)

gpt3 = lm.config(50257, 2048, 12288, 96, 49152, 96)
L, D, H, n = 96, 12288, 96, 2048
assert lm.kv_cache_size(gpt3, 1) == 2 * L * D == 2_359_296 and 2 * 2_359_296 == 4.5 * 2**20
assert 2 * lm.kv_cache_size(gpt3, n) == 9 * 2**30 == 9_663_676_416
assert 2 * L * (D // H) == 24_576 == 2_359_296 // 96 and 2 * 24_576 == 48 * 2**10
P = per_token(gpt3)
assert P == 349_127_467_008
cached = n * P + 2 * L * D * n * (n + 1)
again = P * n * (n + 1) // 2 + 4 * L * D * n * (n + 1) * (2 * n + 1) // 6
assert round(cached / 1e10) == 72491 and round(again / 1e13) == 74605 and round(again / cached) == 1029
assert (P * n * (n + 1) / 2) / (n * P) == (n + 1) / 2 == 1024.5
print("ok")
