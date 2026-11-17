# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/adam/ (ai://88GHGT).
# Check of derivations/step-bound.md: with eps = 0, |m_hat_t| / sqrt(v_hat_t) is at most B_t for every gradient
# sequence, the sequence g_{t-k} = (b1/b2)^k reaches it, B_1 = 1, the limit of B_t exceeds the bounds of Section 2.1
# of Kingma and Ba, B_t grows without bound when b1^2 >= b2, and the single spike gives (1 - b1)/sqrt(1 - b2).
import numpy as np


def ratio(g, b1, b2):
    t = len(g)
    k = np.arange(t - 1, -1, -1)  # the age of each gradient
    m = (1 - b1) * np.sum(b1**k * g)
    v = (1 - b2) * np.sum(b2**k * g**2)
    return (m / (1 - b1**t)) / np.sqrt(v / (1 - b2**t))


def bound(b1, b2, t):
    gamma = b1**2 / b2
    s = t if gamma == 1 else (1 - gamma**t) / (1 - gamma)
    return (1 - b1) / np.sqrt(1 - b2) * np.sqrt(1 - b2**t) / (1 - b1**t) * np.sqrt(s)


def limit(b1, b2):
    return (1 - b1) / np.sqrt((1 - b2) * (1 - b1**2 / b2))


def main():
    rng = np.random.default_rng(20261114)
    settings = [(0.9, 0.999), (0.9, 0.99), (0.9, 0.98), (0.9, 0.95), (0.5, 0.96), (0.0, 0.5), (0.99, 0.999)]
    for b1, b2 in settings:
        for t in [1, 2, 7, 50, 400]:
            assert np.isclose(bound(b1, b2, 1), 1.0, rtol=1e-12)
            # Random sequences, with magnitudes over several orders, never exceed the bound.
            for _ in range(200):
                g = rng.standard_normal(t) * np.exp(rng.normal(0, 3, t))
                assert abs(ratio(g, b1, b2)) <= bound(b1, b2, t) * (1 + 1e-12)
            # The extremal sequence reaches it.
            g = (b1 / b2) ** np.arange(t - 1, -1, -1) if b1 > 0 else np.eye(1, t, t - 1)[0]
            assert np.isclose(ratio(g, b1, b2), bound(b1, b2, t), rtol=1e-9), (b1, b2, t)
    # The limits for gamma = b1^2/b2 < 1, against the bounds stated in Section 2.1.
    assert abs(limit(0.9, 0.999) - 7.27) < 0.005 and abs(0.1 / np.sqrt(0.001) - 3.16) < 0.005
    assert abs(limit(0.9, 0.99) - 2.35) < 0.005  # (1 - b1) = sqrt(1 - b2) here, and the paper states |ratio| < 1
    assert abs(limit(0.9, 0.95) - 1.165) < 0.001 and abs(limit(0.9, 0.98) - 1.698) < 0.001  # the paper: at most 1
    assert bound(0.9, 0.999, 50) < bound(0.9, 0.999, 500) < limit(0.9, 0.999)
    assert abs(limit(0.9, 0.999) - bound(0.9, 0.999, 100_000)) < 1e-9
    # gamma >= 1: the bound grows without limit.
    assert bound(0.99, 0.9, 20) < bound(0.99, 0.9, 200) and bound(0.99, 0.9, 200) > 600
    assert bound(0.99, 0.98, 2000) > 2 * bound(0.99, 0.98, 200)
    # The single spike of Section 2.1: zero gradients and then one gradient.
    for t in [1, 10, 10_000]:
        g = np.zeros(t)
        g[-1] = 1.0
        want = 0.1 / np.sqrt(0.001) * np.sqrt(1 - 0.999**t) / (1 - 0.9**t)
        assert np.isclose(ratio(g, 0.9, 0.999), want, rtol=1e-12)
    # |E[g]| <= sqrt(E[g^2]) for any distribution: a discrete one with random weights.
    for _ in range(100):
        x, p = rng.normal(size=6) * 3, rng.dirichlet(np.ones(6))
        assert abs(p @ x) <= np.sqrt(p @ x**2) * (1 + 1e-12)
    print("ok: the ratio never exceeds B_t, the extremal sequence reaches it, and the limits are 7.27 and 2.35")


if __name__ == "__main__":
    main()
