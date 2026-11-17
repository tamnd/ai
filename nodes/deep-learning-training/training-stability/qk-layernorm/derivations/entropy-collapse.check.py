# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/qk-layernorm/ (ai://B6PWZA).
# Check of derivations/entropy-collapse.md: dH/dc = -c Var_p(z), the softmax of c z tends to the one-hot
# vector of the argmax, H tends to 0 within the bound h(eps) + eps log(n - 1), and ties give log k.
import numpy as np


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def entropy(p):
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())


def main():
    rng = np.random.default_rng(7)
    for _ in range(20):
        n = int(rng.integers(2, 40))
        z = rng.normal(size=n) * 2
        m = int(np.argmax(z))
        gap = z[m] - np.max(np.delete(z, m))
        cs = np.concatenate([[0.0], np.geomspace(1e-2, 1e3, 60)])
        hs = np.array([entropy(softmax(c * z)) for c in cs])
        assert np.isclose(hs[0], np.log(n))
        assert np.all(np.diff(hs) <= 1e-12)
        for c in [0.3, 1.0, 4.0]:
            p = softmax(c * z)
            var = (p * z**2).sum() - (p * z).sum() ** 2
            h = 1e-6
            fd = (entropy(softmax((c + h) * z)) - entropy(softmax((c - h) * z))) / (2 * h)
            assert np.isclose(fd, -c * var, rtol=1e-5, atol=1e-9)
        for c in [1.0, 10.0, 100.0]:
            p = softmax(c * z)
            eps = 1.0 - p[m]
            assert eps <= (n - 1) * np.exp(-c * gap) * (1 + 1e-9)
            hb = 0.0 if eps <= 0 else -(eps * np.log(eps) + (1 - eps) * np.log1p(-eps))
            assert entropy(p) <= hb + eps * np.log(n - 1) + 1e-12
    z = np.array([2.0, 0.0, -2.0])
    for c in [0, 1, 2, 4, 8, 16]:
        print(f"c={c:>2}: H = {entropy(softmax(c * z)):.6g} nats, max weight = {softmax(c * z).max():.6f}")
    tie = np.array([1.0, 1.0, 1.0, -1.0, 0.0])
    assert np.isclose(entropy(softmax(500.0 * tie)), np.log(3), rtol=1e-12)
    print("entropy-collapse: ok")


if __name__ == "__main__":
    main()
