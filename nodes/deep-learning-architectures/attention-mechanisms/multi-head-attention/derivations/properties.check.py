# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/multi-head-attention/ (ai://NEW-multi-head-attention).
# Check of derivations/properties.md: the head sum, the convex combinations, the permutation properties, the
# causal and padding masks, the weight count, the FLOP count and the size of the key-value cache.
import numpy as np


def mha(P, X, M, H, dk, dv, mask=None):
    heads, weights = [], []
    for r in range(H):
        Q, K = X @ P["Wq"][:, r * dk : (r + 1) * dk], M @ P["Wk"][:, r * dk : (r + 1) * dk]
        V = M @ P["Wv"][:, r * dv : (r + 1) * dv]
        Z = Q @ K.T / np.sqrt(dk)
        if mask is not None:
            Z = np.where(mask, Z, -np.inf)
        E = np.exp(Z - Z.max(axis=1, keepdims=True))
        A = E / E.sum(axis=1, keepdims=True)
        heads.append(A @ V)
        weights.append(A)
    return np.concatenate(heads, axis=1) @ P["Wo"], heads, weights


def count_flops(T, S, d, H, dk, dv):
    """Count the multiply-adds of one sequence with loops over the products, times 2."""
    madds = 0
    for _ in range(T):
        madds += d * H * dk  # one query row
    for _ in range(S):
        madds += d * H * dk + d * H * dv  # one key row and one value row
    for _ in range(H):
        for _ in range(T):
            for _ in range(S):
                madds += dk + dv  # one score and one term of the weighted sum
    for _ in range(T):
        madds += H * dv * d  # one output row
    return 2 * madds


def main():
    rng = np.random.default_rng(20170612)
    for H, dk, dv in [(1, 4, 4), (2, 2, 2), (4, 3, 5), (8, 1, 1)]:
        d, T, S = 8, 5, 4
        P = {"Wq": rng.normal(size=(d, H * dk)), "Wk": rng.normal(size=(d, H * dk)),
             "Wv": rng.normal(size=(d, H * dv)), "Wo": rng.normal(size=(H * dv, d))}
        X, M = rng.normal(size=(T, d)), rng.normal(size=(S, d))
        Y, heads, weights = mha(P, X, M, H, dk, dv)
        assert np.allclose(Y, sum(heads[r] @ P["Wo"][r * dv : (r + 1) * dv] for r in range(H)), atol=1e-12)
        for A in weights:
            assert np.all(A >= 0) and np.allclose(A.sum(axis=1), 1, atol=1e-12)
        perm = rng.permutation(T)
        Ys = mha(P, X, X, H, dk, dv)[0]
        assert np.allclose(mha(P, X[perm], X[perm], H, dk, dv)[0], Ys[perm], atol=1e-12)
        assert np.allclose(mha(P, X, M[rng.permutation(S)], H, dk, dv)[0], Y, atol=1e-12)
        causal = np.tril(np.ones((T, T), bool))
        Yc = mha(P, X, X, H, dk, dv, causal)[0]
        for t in range(T):
            X2 = X.copy()
            X2[t + 1 :] = rng.normal(size=X2[t + 1 :].shape)
            assert np.allclose(mha(P, X2, X2, H, dk, dv, causal)[0][: t + 1], Yc[: t + 1], atol=1e-12)
        L = 2
        pad = np.tile(np.arange(S) < L, (T, 1))
        assert np.allclose(mha(P, X, M, H, dk, dv, pad)[0], mha(P, X, M[:L], H, dk, dv)[0], atol=1e-12)
        n_weights = sum(w.size for w in P.values())
        assert n_weights == 2 * d * H * (dk + dv)
        if H * dk == d and H * dv == d:
            assert n_weights == 4 * d * d
        for B in (1, 3):
            want = 2 * B * T * d * H * dk + 2 * B * S * d * H * (dk + dv) + 2 * B * T * d * H * dv + 2 * B * H * T * S * (dk + dv)
            assert B * count_flops(T, S, d, H, dk, dv) == want
            if H * dk == d and H * dv == d:
                assert B * count_flops(T, T, d, H, dk, dv) == 8 * B * T * d * d + 4 * B * T * T * d
    h, dk = 8, 64
    mha_cache = 2 * h * dk
    for G in (1, 2, 4, 8):
        gqa_cache = 2 * G * dk
        assert mha_cache // gqa_cache == h // G
    assert 2 * 1 * dk * h == mha_cache  # G = 1 is multi-query attention, a factor h smaller
    print("ok: the head sum, the convex rows, the permutation properties, the masks and the counts hold")


if __name__ == "__main__":
    main()
