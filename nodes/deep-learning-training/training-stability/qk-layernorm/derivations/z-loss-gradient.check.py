# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/qk-layernorm/ (ai://NEW-qk-layernorm).
# Check of derivations/z-loss-gradient.md: the gradient 2 alpha log Z softmax(z) against finite differences,
# the shift that leaves the cross-entropy alone and zeroes log Z, and the first-order change of log Z.
import numpy as np


def lse(z):
    m = z.max()
    return float(m + np.log(np.exp(z - m).sum()))


def zl(z, a):
    return a * lse(z) ** 2


def ce(z, y):
    return lse(z) - z[y]


def main():
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(50):
        n = int(rng.integers(2, 30))
        a = float(10 ** rng.uniform(-5, 0))
        z = rng.normal(size=n) * 3 + rng.normal() * 5
        p = np.exp(z - lse(z))
        g = 2 * a * lse(z) * p
        h = 1e-6
        # The loss is linear in alpha, so the finite differences use alpha = 1 and are then scaled by a.
        fd = a * np.array([(zl(z + h * e, 1.0) - zl(z - h * e, 1.0)) / (2 * h) for e in np.eye(n)])
        worst = max(worst, np.abs(fd - g).max() / np.abs(g).max())
        assert np.allclose(fd, g, rtol=1e-5, atol=1e-6 * a * (1 + lse(z) ** 2))
        assert np.isclose(g.sum(), 2 * a * lse(z))
        y = int(rng.integers(n))
        s = -lse(z)
        assert np.isclose(ce(z + s, y), ce(z, y))
        assert abs(lse(z + s)) < 1e-12
        assert np.allclose(np.exp(z + s), p)
        eta = 1e-3 / a
        first = -eta * 2 * a * lse(z) * (p**2).sum()
        actual = lse(z - eta * g) - lse(z)
        assert np.sign(actual) == -np.sign(lse(z)) or abs(lse(z)) < 1e-9
        assert np.isclose(actual, first, rtol=2e-2, atol=1e-10)
        assert 1 / n - 1e-12 <= (p**2).sum() <= 1 + 1e-12
    print(f"largest relative gap between finite differences and 2 alpha log Z softmax(z): {worst:.2e}")
    print("z-loss-gradient: ok")


if __name__ == "__main__":
    main()
