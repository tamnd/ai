# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/multi-head-attention/ (ai://NEW-multi-head-attention).
# Check of derivations/scale.md: the variance of q.k is d_k and the scaled score has variance 1; the softmax
# Jacobian is diag(a) - a a^T and its entries are at most (n - 1) exp(-Delta) when one score leads by Delta.
import numpy as np


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def main():
    rng = np.random.default_rng(20170612)
    N = 200_000
    for d_k in (1, 4, 64, 512):
        for draw in ("normal", "rademacher"):
            if draw == "normal":
                q, k = rng.normal(size=(N, d_k)), rng.normal(size=(N, d_k))
            else:
                q, k = rng.choice([-1.0, 1.0], size=(N, d_k)), rng.choice([-1.0, 1.0], size=(N, d_k))
            s = np.einsum("nc,nc->n", q, k)
            assert abs(s.mean()) / np.sqrt(d_k) < 0.03, (d_k, draw)
            assert abs(s.var() / d_k - 1) < 0.03, (d_k, draw, s.var())
            z = s / np.sqrt(d_k)
            assert abs(z.var() - 1) < 0.03
    for _ in range(2000):
        n = int(rng.integers(2, 51))
        delta = float(rng.uniform(0.1, 20.0))
        z = rng.normal(size=n) * 3
        m = int(rng.integers(n))
        others = np.delete(np.arange(n), m)
        z[m] = z[others].max() + delta + rng.exponential() * (rng.random() < 0.5)
        a = softmax(z)
        J = np.diag(a) - np.outer(a, a)
        assert np.all(np.delete(a, m) <= np.exp(-delta) * (1 + 1e-12))
        assert 1 - a[m] <= (n - 1) * np.exp(-delta) * (1 + 1e-12) + 1e-15
        assert np.abs(J).max() <= (n - 1) * np.exp(-delta) * (1 + 1e-12) + 1e-15
    z = rng.normal(size=6)
    h = 1e-6
    num = np.stack([(softmax(z + h * e) - softmax(z - h * e)) / (2 * h) for e in np.eye(6)], axis=1)
    a = softmax(z)
    assert np.allclose(num, np.diag(a) - np.outer(a, a), atol=1e-9)
    print("ok: Var(q.k) = d_k, the scaled score has variance 1, and the softmax Jacobian bound holds")


if __name__ == "__main__":
    main()
