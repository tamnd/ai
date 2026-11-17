# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/sparse-mixture-of-experts/ (ai://1H67K0).
# Check of derivations/balance-loss-minimum.md: the Switch balance loss alpha N sum f_i P_i is alpha under uniform
# routing, lies in [alpha, alpha N] when P = f, is at least alpha sum f_i^2 in general and can fall to 0.875 alpha with
# two experts, and its gradient in a token's logits is (alpha N / T) p_j (f_j - sum_i f_i p_i) with f held fixed.
import itertools

import numpy as np


def softmax(h):
    e = np.exp(h - h.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def fractions(p):
    t, n = p.shape
    return np.bincount(p.argmax(axis=-1), minlength=n) / t


def loss(p, alpha, f=None):
    n = p.shape[1]
    f = fractions(p) if f is None else f
    return alpha * n * float(f @ p.mean(axis=0))


def main():
    rng = np.random.default_rng(20261116)
    alpha = 0.01

    # 1. Uniform routing gives alpha for every N.
    for n in (2, 3, 4, 8, 16, 64):
        f = np.full(n, 1.0 / n)
        assert abs(alpha * n * float(f @ f) - alpha) < 1e-15

    # 2 and 3. One-hot batches: P = f, alpha <= L <= alpha N.
    for _ in range(2000):
        n = int(rng.integers(2, 12))
        t = int(rng.integers(1, 50))
        p = np.eye(n)[rng.integers(0, n, size=t)]
        f = fractions(p)
        assert np.allclose(p.mean(axis=0), f)
        val = loss(p, alpha)
        assert abs(val - alpha * n * (f**2).sum()) < 1e-15
        assert alpha - 1e-15 <= val <= alpha * n + 1e-15
    for n in (2, 3, 4):
        best = min(
            (loss(np.eye(n)[list(c)], alpha), c) for c in itertools.product(range(n), repeat=n)
        )
        assert abs(best[0] - alpha) < 1e-15
        assert sorted(best[1]) == list(range(n))
        for c in itertools.product(range(n), repeat=n):
            if sorted(c) != list(range(n)):
                assert loss(np.eye(n)[list(c)], alpha) > alpha + 1e-12
    collapsed = np.zeros((10, 5))
    collapsed[:, 2] = 1.0
    assert abs(loss(collapsed, alpha) - 5 * alpha) < 1e-15

    # 4. General batches: alpha sum f^2 <= L <= alpha N, and the two-expert example.
    for _ in range(2000):
        n = int(rng.integers(2, 12))
        t = int(rng.integers(1, 50))
        p = softmax(rng.normal(scale=rng.uniform(0.1, 6.0), size=(t, n)))
        f = fractions(p)
        assert (p.mean(axis=0) >= f / n - 1e-15).all()
        val = loss(p, alpha)
        assert alpha * (f**2).sum() - 1e-15 <= val <= alpha * n + 1e-15
    for eps in (1e-3, 1e-6, 1e-9):
        p = np.array([[0.5 + eps, 0.5 - eps]] * 3 + [[0.0, 1.0]])
        assert abs(loss(p, 1.0) - 0.875) < 2 * eps

    # 5. The logit gradient with f fixed, against central differences.
    for _ in range(200):
        n = int(rng.integers(2, 8))
        t = int(rng.integers(1, 12))
        h = rng.normal(scale=2.0, size=(t, n))
        p = softmax(h)
        f = fractions(p)
        grad = alpha * n / t * p * (f[None, :] - (p * f[None, :]).sum(axis=-1, keepdims=True))
        step = 1e-6
        for x in range(t):
            for j in range(n):
                hp, hm = h.copy(), h.copy()
                hp[x, j] += step
                hm[x, j] -= step
                fd = (loss(softmax(hp), alpha, f) - loss(softmax(hm), alpha, f)) / (2 * step)
                assert abs(fd - grad[x, j]) < 1e-9, (fd, grad[x, j])

    # The worked example of the page.
    p = np.eye(4)[[0, 1, 0, 2, 0, 1, 3, 0]]
    assert np.allclose(fractions(p), [0.5, 0.25, 0.125, 0.125])
    assert abs(loss(p, 1.0) - 1.375) < 1e-15
    print("ok: uniform gives alpha; P = f gives alpha <= L <= alpha N; general L >= alpha sum f^2; 0.875 alpha example; gradient")


if __name__ == "__main__":
    main()
