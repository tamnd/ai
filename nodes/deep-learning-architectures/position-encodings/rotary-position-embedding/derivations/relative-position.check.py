# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/rotary-position-embedding/ (ai://0V8E5T).
# Check of derivations/relative-position.md: R_m is orthogonal, R_0 = I, R_m R_n = R_{m+n}, the score depends only
# on n - m, the complex, the matrix and the elementwise forms agree, the operation count, and the half pairing.
import numpy as np


def theta(d, b=10000.0):
    return b ** (-2.0 * np.arange(d // 2) / d)


def rmat(m, th):
    d = 2 * len(th)
    r = np.zeros((d, d))
    for j, t in enumerate(th):
        c, s = np.cos(m * t), np.sin(m * t)
        r[2 * j : 2 * j + 2, 2 * j : 2 * j + 2] = [[c, -s], [s, c]]
    return r


def elementwise(x, m, th):
    c, s = np.repeat(np.cos(m * th), 2), np.repeat(np.sin(m * th), 2)
    xt = np.empty_like(x)
    xt[0::2], xt[1::2] = -x[1::2], x[0::2]
    return x * c + xt * s


def main():
    rng = np.random.default_rng(20261116)
    for d in [2, 8, 64]:
        th = theta(d)
        assert np.array_equal(rmat(0.0, th), np.eye(d))
        assert np.count_nonzero(rmat(1.0, th)) == 2 * d
        for _ in range(50):
            m, n, t = rng.uniform(-3000, 3000, size=3)
            q, k = rng.normal(size=d), rng.normal(size=d)
            rm, rn = rmat(m, th), rmat(n, th)
            assert np.allclose(rm.T @ rm, np.eye(d), atol=1e-12)
            assert np.isclose(np.linalg.norm(rm @ q), np.linalg.norm(q), rtol=1e-12)
            assert np.allclose(rm @ rn, rmat(m + n, th), atol=1e-9)
            assert np.allclose(rm.T, rmat(-m, th), atol=1e-12)
            s1 = (rm @ q) @ (rn @ k)
            assert np.isclose(s1, q @ rmat(n - m, th) @ k, rtol=1e-10, atol=1e-10)
            assert np.isclose(s1, (rmat(m + t, th) @ q) @ (rmat(n + t, th) @ k), rtol=1e-9, atol=1e-9)
            qh, kh = q[0::2] + 1j * q[1::2], k[0::2] + 1j * k[1::2]
            assert np.isclose(s1, np.real(np.sum(qh * np.conj(kh) * np.exp(1j * (m - n) * th))), rtol=1e-9, atol=1e-9)
            z = qh * np.exp(1j * m * th)
            assert np.allclose(np.ravel(np.column_stack([z.real, z.imag])), rm @ q, atol=1e-10)
            assert np.allclose(elementwise(q, m, th), rm @ q, atol=1e-10)
            # pair j with j + d/2: P^T R_m P, and the scores are those of the paired form on P q and P k.
            p = np.eye(d)[np.ravel(np.column_stack([np.arange(d // 2), np.arange(d // 2) + d // 2]))]
            half_q, half_k = p.T @ rm @ p @ q, p.T @ rn @ p @ k
            assert np.isclose(half_q @ half_k, (rm @ p @ q) @ (rn @ p @ k), rtol=1e-10, atol=1e-10)
    print("relative-position: ok")


if __name__ == "__main__":
    main()
