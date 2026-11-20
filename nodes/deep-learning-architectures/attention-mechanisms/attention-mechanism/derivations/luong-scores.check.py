# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/attention-mechanism/ (ai://NEW-attention-mechanism).
"""Numeric check of the derivation luong-scores: the relations between the scores and variants of Luong et al."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import attention as A  # noqa: E402

rng = np.random.default_rng(3)
for _ in range(50):
    B, S, n, k = (int(v) for v in rng.integers([1, 2, 1, 1], [4, 10, 6, 6]))
    h, Hs = rng.normal(size=(B, n)), rng.normal(size=(B, S, n))
    Wa = rng.normal(size=(n, n))
    # 1. general with the identity is dot, and general is dot of the mapped query
    assert np.allclose(A.luong_score("general", h, Hs, np.eye(n)), A.luong_score("dot", h, Hs))
    assert np.allclose(A.luong_score("general", h, Hs, Wa), A.luong_score("dot", h @ Wa, Hs))
    # 2. concat is the additive score with the matrices stacked
    W1, W2, va = rng.normal(size=(n, k)), rng.normal(size=(n, k)), rng.normal(size=k)
    additive = np.tanh((h @ W1)[:, None] + Hs @ W2) @ va
    assert np.allclose(A.luong_score("concat", h, Hs, np.vstack([W1, W2]), va), additive)
    # 3. local-m with a window that covers the sentence is global attention, and a smaller one is not
    t = int(rng.integers(0, S))
    pos = np.arange(S)
    full = np.ones((B, S), bool)
    g, cg = A.global_attention("dot", h, Hs, full)
    D = max(t, S - 1 - t)
    a, c = A.global_attention("dot", h, Hs, np.broadcast_to(np.abs(pos - t) <= D, (B, S)))
    assert np.allclose(a, g) and np.allclose(c, cg)
    if D > 0:
        a, _ = A.global_attention("dot", h, Hs, np.broadcast_to(np.abs(pos - t) <= D - 1, (B, S)))
        assert np.all(a[:, np.abs(pos - t) > D - 1] == 0) and not np.allclose(a, g)
    # 4. the local-p weights sum to at most 1
    lengths = rng.integers(1, S + 1, B)
    Wp, vp = rng.normal(size=(n, n)), rng.normal(size=n) * 2
    p_t, a, c = A.local_p_attention("general", h, Hs, lengths, int(rng.integers(1, 4)), Wp, vp, Wa)
    assert np.all(a >= 0) and np.all(a.sum(-1) <= 1 + 1e-12)
    assert np.allclose(c, np.einsum("bs,bsk->bk", a, Hs))
    off = np.abs(p_t - np.round(p_t)) > 1e-9  # p_t not on a position: some weight has g < 1
    assert np.all(a.sum(-1)[off] < 1)
    # 5. location ignores the source
    Wl = rng.normal(size=(n, S))
    assert np.array_equal(A.luong_score("location", h, Hs, Wl), A.luong_score("location", h, rng.normal(size=Hs.shape), Wl))
print("luong-scores: ok")
