# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/label-smoothing/ (ai://98TJG9).
# Check of derivations/optimal-logits.md: with eps > 0 the smoothed loss of one example is smallest at p = q', every
# incorrect logit then sits log(1 + K (1 - eps) / eps) below the correct one, and the minimum is H(q'); with eps = 0
# gradient descent widens the gap at every step; K = 1000, eps = 0.1 gives log 9001; and eps / (K - 1) on the wrong
# classes is uniform smoothing with eps K / (K - 1).
import numpy as np


def log_softmax(z):
    m = z.max()
    return z - m - np.log(np.exp(z - m).sum())


def target(y, k, eps):
    q = np.full(k, eps / k)
    q[y] += 1.0 - eps
    return q


def gap(k, eps):
    return np.log1p(k * (1.0 - eps) / eps)


def main():
    rng = np.random.default_rng(20261115)

    # 1 and 2: gradient descent on free logits finds p = q', the gap, and the minimum H(q').
    for k, eps in [(2, 0.1), (3, 0.3), (10, 0.1), (10, 0.5), (100, 0.2), (5, 1.0)]:
        y = 1
        q = target(y, k, eps)
        z = rng.normal(size=k)
        for _ in range(20000):
            z -= 1.0 * (np.exp(log_softmax(z)) - q)
        p = np.exp(log_softmax(z))
        assert np.allclose(p, q, atol=1e-9), (k, eps)
        gaps = z[y] - np.delete(z, y)
        assert np.allclose(gaps, gap(k, eps), atol=1e-7), (k, eps, gaps)
        h_q = -(q * np.log(q)).sum()
        best = -(q * log_softmax(z)).sum()
        assert abs(best - h_q) < 1e-12 and h_q > 0
        for _ in range(500):
            assert -(q * log_softmax(rng.normal(scale=5.0, size=k))).sum() >= h_q - 1e-12

    # 3: with hard targets the gap grows at every step and has no finite limit.
    k, y = 4, 0
    q = target(y, k, 0.0)
    z = np.zeros(k)
    prev = 0.0
    for _ in range(20000):
        z -= 1.0 * (np.exp(log_softmax(z)) - q)
        g = z[y] - np.delete(z, y).max()
        assert g > prev
        prev = g
    assert prev > 9.0, prev
    assert -log_softmax(z)[y] > 0

    # 4: the ImageNet setting of Szegedy et al.
    q = target(0, 1000, 0.1)
    assert abs(q[0] - 0.9001) < 1e-12 and abs(q[1] - 0.0001) < 1e-15
    assert abs(gap(1000, 0.1) - np.log(9001)) < 1e-12
    assert abs(np.log(9001) - 9.105) < 1e-3

    # 5: eps / (K - 1) on each wrong class is uniform smoothing with eps K / (K - 1).
    for k in range(2, 21):
        for eps in np.linspace(0.0, (k - 1) / k, 7):
            other = np.full(k, eps / (k - 1))
            other[3 % k] = 1.0 - eps
            assert np.allclose(other, target(3 % k, k, eps * k / (k - 1)), atol=1e-15)
    print("ok: the smoothed loss is minimized at p = q', with the logit gap log(1 + K(1 - eps)/eps)")


if __name__ == "__main__":
    main()
