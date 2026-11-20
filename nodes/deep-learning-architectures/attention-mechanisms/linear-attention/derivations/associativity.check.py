# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/linear-attention/ (ai://NEW-linear-attention).
"""Numeric check of the derivation associativity: the linear form, softmax as a kernel, the causal recurrence, the
chunked form and the size of the state."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import linear_attention as L  # noqa: E402


def chunked(fQ, fK, V, T):
    N = fQ.shape[-2]
    S, z = np.zeros((fK.shape[-1], V.shape[-1])), np.zeros(fK.shape[-1])
    out = []
    for c in range(0, N, T):
        q, k, v = fQ[c : c + T], fK[c : c + T], V[c : c + T]
        A = np.tril(q @ k.T)
        out.append((q @ S + A @ v) / (q @ z + A.sum(-1))[:, None])
        S, z = S + k.T @ v, z + k.sum(0)
    return np.concatenate(out)


rng = np.random.default_rng(0)
for _ in range(40):
    N, D, M = (int(v) for v in rng.integers([2, 1, 1], [13, 6, 6]))
    Q, K, V = rng.normal(size=(N, D)), rng.normal(size=(N, D)), rng.normal(size=(N, M))
    for name in ("elu", "poly2", "dpfp", "relu"):
        phi = L.FEATURES[name][0]
        fQ, fK = phi(Q), phi(K)
        if name in ("relu", "dpfp"):  # these maps can vanish; keep every denominator positive
            fQ, fK = fQ + 0.1, fK + 0.1
        A = fQ @ fK.T
        # 1. associativity
        assert np.allclose(L.linear_attention(fQ, fK, V), (A @ V) / A.sum(-1, keepdims=True))
        # 3. the causal recurrence is the masked form
        out, num, den = L.causal_linear_attention(fQ, fK, V)
        At = np.tril(A)
        assert np.allclose(out, (At @ V) / At.sum(-1, keepdims=True))
        # 4. every chunk length
        for T in range(1, N + 1):
            assert np.allclose(chunked(fQ, fK, V, T), out)
    # 2. softmax is the exponential kernel
    E = np.exp(Q @ K.T / np.sqrt(D))
    for causal in (False, True):
        Em = np.tril(E) if causal else E
        assert np.allclose(L.softmax_attention(Q, K, V, causal), (Em @ V) / Em.sum(-1, keepdims=True))
# 5. the size of the state against the size of a cache
cfg = L.config(8, 2)
P = L.init(rng, cfg)
state = L.initial_state(cfg, 1)
for i in range(1, 50):
    _, state = L.step(P, cfg, state, rng.normal(size=(1, 8)))
    assert sum(a.size for a in state) == 2 * (4 * 4 + 4)
    assert 2 * i * (4 + 4) > 2 * (4 * 4 + 4) or i <= 2
print("associativity: ok")
