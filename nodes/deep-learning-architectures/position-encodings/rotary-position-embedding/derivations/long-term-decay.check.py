# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/rotary-position-embedding/ (ai://0V8E5T).
# Check of derivations/long-term-decay.md: the Abel identity, the bound on the score and the averages of |S_j|.
import numpy as np


def theta(d, b=10000.0):
    return b ** (-2.0 * np.arange(d // 2) / d)


def main():
    rng = np.random.default_rng(20261116)
    d = 128
    th = theta(d)
    for r in [0, 1, 2, 5, 10, 50, 100, 1000, 2000, 37.5, -250]:
        e = np.exp(1j * r * th)
        s = np.concatenate([[0], np.cumsum(e)])  # S_0 .. S_{d/2}
        for _ in range(20):
            q, k = rng.normal(size=d), rng.normal(size=d)
            h = np.concatenate([(q[0::2] + 1j * q[1::2]) * np.conj(k[0::2] + 1j * k[1::2]), [0]])
            total = np.sum(h[:-1] * e)
            abel = -np.sum(s[1:] * (h[1:] - h[:-1]))
            assert abs(total - abel) < 1e-10 * (1 + abs(total))
            # the real score from the rotated vectors, through the complex form
            qh = (q[0::2] + 1j * q[1::2]) * np.exp(1j * r * th)
            kh = k[0::2] + 1j * k[1::2]
            score = np.sum(qh.real * kh.real + qh.imag * kh.imag)
            assert abs(score - total.real) < 1e-10 * (1 + abs(score))
            bound = np.max(np.abs(h[1:] - h[:-1])) * np.sum(np.abs(s[1:]))
            assert abs(score) <= abs(total) + 1e-12 <= bound + 1e-9
    table = {0: 32.5, 1: 31.538, 10: 17.954, 100: 10.227, 1000: 4.471, 2000: 5.047}
    for r, want in table.items():
        got = np.mean(np.abs(np.cumsum(np.exp(1j * r * th))))
        assert abs(got - want) < 5e-4, (r, got)
    assert table[2000] > table[1000]
    print("long-term-decay: ok")


if __name__ == "__main__":
    main()
