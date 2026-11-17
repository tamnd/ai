# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/rotary-position-embedding/ (ai://NEW-rotary-position-embedding).
# Check of derivations/context-extension.md: position interpolation, the NTK-aware base, the NTK-by-parts ends,
# the YaRN length-scaling trick and its recommended values.
import numpy as np


def theta(d, b=10000.0):
    return b ** (-2.0 * np.arange(d // 2) / d)


def rotate(x, m, th):
    c, s = np.repeat(np.cos(m * th), 2), np.repeat(np.sin(m * th), 2)
    xt = np.empty_like(x)
    xt[0::2], xt[1::2] = -x[1::2], x[0::2]
    return x * c + xt * s


def main():
    rng = np.random.default_rng(20261116)
    d, b = 64, 10000.0
    th = theta(d, b)
    for s in [2.0, 4.0, 8.0, 16.0, 32.0]:
        q, k = rng.normal(size=d), rng.normal(size=d)
        m, n = rng.integers(0, 8192, size=2)
        # part 1: R_{m/s} equals RoPE with theta / s, and the score depends on (n - m) / s.
        assert np.allclose(rotate(q, m / s, th), rotate(q, m, th / s), atol=1e-10)
        assert np.isclose(rotate(q, m / s, th) @ rotate(k, n / s, th), q @ rotate(k, (n - m) / s, th), atol=1e-9)
        L = 2048
        assert (int(L * s) - 1) / s < L
        # part 2: the NTK-aware base.
        th2 = theta(d, b * s ** (d / (d - 2)))
        assert np.allclose(th2 / th, s ** (-2.0 * np.arange(d // 2) / (d - 2)), rtol=1e-12)
        assert np.isclose(th2[0], th[0]) and np.isclose(th2[-1], th[-1] / s, rtol=1e-12)
        # part 3: the ends of the ramp.
        for g, want in [(1.0, th), (0.0, th / s)]:
            assert np.allclose((1 - g) * th / s + g * th, want, rtol=1e-15)
        # part 4: scaling q and k by sqrt(1/t) divides the score by t.
        t = (0.1 * np.log(s) + 1) ** -2
        c = np.sqrt(1 / t)
        plain = rotate(q, m, th) @ rotate(k, n, th)
        assert np.isclose((c * rotate(q, m, th)) @ (c * rotate(k, n, th)), plain / t, rtol=1e-12)
    # part 5: the recommended values.
    for s, want in [(8, 1.2079), (16, 1.2773), (32, 1.3466)]:
        assert abs(0.1 * np.log(s) + 1 - want) < 5e-5
    # the wavelength split for d = 128, b = 10000, L = 2048, alpha = 1, beta = 32.
    th = theta(128, b)
    lam = 2 * np.pi / th
    r = 2048 / lam
    assert np.isclose(lam[0], 2 * np.pi) and abs(lam[-1] - 54410) < 1
    assert np.sum(r < 1) == 23 and np.all(r[41:] < 1) and r[40] >= 1
    assert np.sum(r > 32) == 17 and np.all(r[:17] > 32) and r[17] <= 32
    print("context-extension: ok")


if __name__ == "__main__":
    main()
