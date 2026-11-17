# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/adam/ (ai://88GHGT).
# Check of derivations/bias-correction.md: the moving averages unrolled, their expectations under stationary
# gradients, the step of a constant gradient, the first step without the correction, and the efficient form with
# epsilon hat.
import numpy as np


def moments(g, b1, b2):
    m = v = 0.0
    for gi in g:
        m = b1 * m + (1 - b1) * gi
        v = b2 * v + (1 - b2) * gi * gi
    return m, v


def main():
    rng = np.random.default_rng(20261114)
    # 1. The unrolled sums (eq-unrolled).
    for b1, b2, t in [(0.9, 0.999, 1), (0.9, 0.999, 17), (0.5, 0.96, 6), (0.0, 0.3, 4)]:
        g = rng.normal(size=t)
        m, v = moments(g, b1, b2)
        w1 = (1 - b1) * b1 ** np.arange(t - 1, -1, -1)
        w2 = (1 - b2) * b2 ** np.arange(t - 1, -1, -1)
        assert np.isclose(m, w1 @ g, rtol=1e-12) and np.isclose(v, w2 @ g**2, rtol=1e-12)
        assert np.isclose(w1.sum(), 1 - b1**t, rtol=1e-12) and np.isclose(w2.sum(), 1 - b2**t, rtol=1e-12)
    # 2. The expectations: E[m_t] = (1 - b1^t) mu and E[v_t] = (1 - b2^t) s, by Monte Carlo over 200,000 runs.
    b1, b2, t, n = 0.9, 0.999, 10, 200_000
    mu, sd = 0.5, 2.0
    g = rng.normal(mu, sd, size=(n, t))
    w1 = (1 - b1) * b1 ** np.arange(t - 1, -1, -1)
    w2 = (1 - b2) * b2 ** np.arange(t - 1, -1, -1)
    m, v = g @ w1, g**2 @ w2
    s = mu**2 + sd**2
    assert abs(m.mean() / (1 - b1**t) - mu) < 0.02, m.mean()
    assert abs(v.mean() / (1 - b2**t) - s) < 0.05, v.mean()
    assert abs(v.mean() - s * (1 - b2**t)) < 0.001  # without the correction, v_t is about 0.01 s at t = 10
    # 3. A constant gradient: m_hat = g and v_hat = g^2, so the step is lr sign(g) when eps = 0.
    for b1, b2, gc in [(0.9, 0.999, -3.0), (0.5, 0.96, 2.0), (0.99, 0.9999, 1e-6)]:
        for t in [1, 2, 10, 1000]:
            m, v = moments([gc] * t, b1, b2)
            assert np.isclose((m / (1 - b1**t)) / np.sqrt(v / (1 - b2**t)), np.sign(gc), rtol=1e-9)
    # 4. Without the correction the first step is (1 - b1)/sqrt(1 - b2) times lr: about 3.16 for the defaults.
    m, v = moments([0.7], 0.9, 0.999)
    assert np.isclose(m / np.sqrt(v), 0.1 / np.sqrt(0.001), rtol=1e-12) and abs(m / np.sqrt(v) - 3.1623) < 1e-4
    # 5. The efficient form: lr_t m / (sqrt(v) + eps_hat) = lr m_hat / (sqrt(v_hat) + eps) with
    # eps_hat = eps sqrt(1 - b2^t).
    b1, b2, lr, eps = 0.9, 0.999, 1e-3, 1e-8
    for t in [1, 5, 100]:
        g = rng.normal(size=t) * 1e-7
        m, v = moments(g, b1, b2)
        a = lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)
        lr_t = lr * np.sqrt(1 - b2**t) / (1 - b1**t)
        b = lr_t * m / (np.sqrt(v) + eps * np.sqrt(1 - b2**t))
        assert np.isclose(a, b, rtol=1e-12), t
    assert np.isclose(1 / np.sqrt(1 - 0.999), 31.62, rtol=1e-3)
    print("ok: unrolled moments, their bias, the corrected expectations, the constant-gradient step and epsilon hat")


if __name__ == "__main__":
    main()
