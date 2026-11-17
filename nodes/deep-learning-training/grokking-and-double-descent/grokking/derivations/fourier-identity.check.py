# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/grokking/ (ai://52PVZH).
# Check of derivations/fourier-identity.md: sum_k alpha_k cos(w_k (a + b - c)), w_k = 2 pi k / p, computed through the
# angle-sum identities, is largest at c = a + b mod p and only there for a prime p and positive weights; with all
# frequencies 1..(p-1)/2 it is (p-1)/2 at the answer and -1/2 elsewhere; at p = 113 with k in {14, 35, 41, 42, 52} the
# right logit is 5 and the largest wrong one 2.5898 at c = a + b + 16; a composite p can tie.
import numpy as np


def logits(p, freqs, alpha):
    a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    a, b = a.ravel(), b.ravel()
    w = 2 * np.pi * np.asarray(freqs, dtype=float) / p
    ca, sa, cb, sb = np.cos(np.outer(a, w)), np.sin(np.outer(a, w)), np.cos(np.outer(b, w)), np.sin(np.outer(b, w))
    cab, sab = ca * cb - sa * sb, sa * cb + ca * sb
    c = np.arange(p)
    out = (cab * alpha) @ np.cos(np.outer(w, c)) + (sab * alpha) @ np.sin(np.outer(w, c))
    direct = (alpha[None, None, :] * np.cos(w[None, None, :] * (a[:, None, None] + b[:, None, None] - c[None, :, None]))).sum(-1)
    assert np.allclose(out, direct, atol=1e-9)
    return out, (a + b) % p


primes = [q for q in range(3, 60) if all(q % r for r in range(2, int(q**0.5) + 1))]
rng = np.random.default_rng(0)
for p in primes:
    for _ in range(3):
        n = rng.integers(1, p)
        freqs = rng.choice(np.arange(1, p), size=n, replace=False)
        alpha = rng.uniform(0.1, 3.0, size=n)
        L, y = logits(p, freqs, alpha)
        right = L[np.arange(p * p), y]
        assert np.allclose(right, alpha.sum())
        L[np.arange(p * p), y] = -np.inf
        assert (L.max(axis=1) < right - 1e-9).all()
    allk = np.arange(1, (p - 1) // 2 + 1)
    L, y = logits(p, allk, np.ones(len(allk)))
    mask = np.ones_like(L, dtype=bool)
    mask[np.arange(p * p), y] = False
    assert np.allclose(L[~mask], (p - 1) / 2) and np.allclose(L[mask], -0.5)

K = np.array([14, 35, 41, 42, 52])
d = np.arange(1, 113)
wrong = np.cos(2 * np.pi * np.outer(d, K) / 113).sum(axis=1)
assert abs(wrong.max() - 2.5898235660602) < 1e-9 and d[wrong.argmax()] == 97 and (113 - 97) == 16
L, y = logits(113, K, np.ones(5))
assert (L.argmax(axis=1) == y).all() and L.shape == (12769, 113)
assert np.allclose(L[np.arange(12769), y], 5.0)

L, y = logits(12, [3], np.ones(1))
assert np.allclose(L[np.arange(144), (y - 4) % 12], L[np.arange(144), y])
print("fourier-identity: ok")
