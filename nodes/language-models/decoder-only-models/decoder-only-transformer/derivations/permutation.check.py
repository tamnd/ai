# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/decoder-only-transformer/ (ai://NEW-decoder-only-transformer).
"""Numeric check of the derivation permutation: what a network without position embeddings can tell apart."""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import decoder_lm as lm  # noqa: E402

rng = np.random.default_rng(0)
changed = 0
for _ in range(40):
    V, H, Dh, F, n = (int(v) for v in rng.integers([3, 2, 2, 1, 3], [12, 4, 4, 9, 9]))
    one, two = (lm.config(V, 8, H * Dh, H, F, L, bool(rng.integers(2)), "none") for L in (1, 2))
    ids = rng.integers(0, V, (2, n))
    perm = np.concatenate([rng.permutation(n - 1), [n - 1]])
    p1 = lm.init(rng, one, scale=0.5)
    np.testing.assert_allclose(
        lm.forward(p1, one, ids[:, perm])[0][:, -1], lm.forward(p1, one, ids)[0][:, -1], rtol=1e-12, atol=1e-13
    )
    p2 = lm.init(rng, two, scale=0.5)
    gap = np.abs(lm.forward(p2, two, ids[:, perm])[0][:, -1] - lm.forward(p2, two, ids)[0][:, -1]).max()
    changed += gap > 1e-3 or (ids[:, perm] == ids).all()
assert changed == 40  # two layers of width 4 to 9: every reordering that moves a token changes the last logits

learned = lm.config(9, 8, 8, 2, 12, 1)
p = lm.init(rng, learned, scale=0.5)
ids = rng.integers(0, 9, (3, 8))
perm = np.concatenate([[1, 0], np.arange(2, 8)])
moved = (ids[:, perm] != ids).any(1)
gap = np.abs(lm.forward(p, learned, ids[:, perm])[0][:, -1] - lm.forward(p, learned, ids)[0][:, -1]).max(1)
assert (gap[moved] > 1e-6).all()  # with positions, one layer already sees the order

cfg = lm.config(vocab=4, n_ctx=8, d_model=8, n_heads=2, d_ff=12, n_layers=1, pos="none")
for _ in range(10):
    params = lm.init(rng, cfg, scale=0.5)
    same = lm.forward(params, cfg, np.full((1, 8), 2))[0][0]
    np.testing.assert_allclose(same, np.broadcast_to(same[0], same.shape), rtol=1e-12, atol=1e-13)
    ids = np.array([[0] + [2] * 7])
    logits, cache = lm.forward(params, cfg, ids)
    c = cache["layers"][0]["attn"]
    for h in range(2):
        s = c["q"][0, h, 1] @ c["k"][0, h, :2].T / math.sqrt(4)  # the scores s_a and s_b of a query of b
        for i in range(2, 9):
            w = math.exp(s[0]) / (math.exp(s[0]) + (i - 1) * math.exp(s[1]))
            assert abs(c["weights"][0, h, i - 1, 0] - w) < 1e-12
            np.testing.assert_allclose(c["weights"][0, h, i - 1, 1:i], (1 - w) / (i - 1), rtol=1e-12)
    rows = logits[0, 1:]
    assert min(np.abs(rows[i] - rows[j]).max() for i in range(7) for j in range(i)) > 1e-6
print("ok")
