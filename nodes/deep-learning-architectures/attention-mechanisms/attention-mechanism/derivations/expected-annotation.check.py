# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/attention-mechanism/ (ai://NEW-attention-mechanism).
"""Numeric check of the derivation expected-annotation: the context as an average of the annotations."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import attention  # noqa: E402

rng = np.random.default_rng(1)
for _ in range(10):
    cfg = attention.config(embed=int(rng.integers(1, 5)), hidden=int(rng.integers(1, 6)), align=int(rng.integers(1, 6)))
    p = attention.init(rng, cfg, 1.0)
    B, S, T = 3, int(rng.integers(2, 7)), int(rng.integers(1, 5))
    src, tgt = rng.integers(1, 7, (B, S)), rng.integers(0, 6, (B, T))
    lengths = rng.integers(1, S + 1, B)
    mask = np.arange(S) < lengths[:, None]
    _, alpha, cache = attention.forward(p, cfg, src, tgt, mask)
    H = cache["H"]
    assert np.all(alpha >= 0) and np.allclose(alpha.sum(-1), 1) and np.all(alpha[~np.broadcast_to(mask[:, None], alpha.shape)] == 0)
    for b in range(B):
        real = H[b, : lengths[b]]
        for i, step in enumerate(cache["steps"]):
            c = step[1][b]
            assert np.allclose(c, alpha[b, i, : lengths[b]] @ real)
            assert np.all(c >= real.min(0) - 1e-12) and np.all(c <= real.max(0) + 1e-12)
    p["v_a"][:] = 0.0
    _, alpha, cache = attention.forward(p, cfg, src, tgt, mask)
    for b in range(B):
        assert np.allclose(alpha[b, :, : lengths[b]], 1.0 / lengths[b])
        assert np.allclose(cache["steps"][-1][1][b], H[b, : lengths[b]].mean(0))
    fixed = attention.config(**{**cfg, "attention": False})
    before = attention.forward(p, fixed, src, tgt, mask)[0]
    q = {**p, "W_a": rng.normal(size=p["W_a"].shape), "U_a": rng.normal(size=p["U_a"].shape), "v_a": rng.normal(size=p["v_a"].shape)}
    assert np.array_equal(attention.forward(q, fixed, src, tgt, mask)[0], before)
print("expected-annotation: ok")
