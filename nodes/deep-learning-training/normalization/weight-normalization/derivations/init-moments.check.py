# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/weight-normalization/ (ai://5PC0VR).
# Check of derivations/init-moments.md: after g <- 1 / sigma[t] and b <- -mu[t] / sigma[t] with t = v . x / ||v||,
# the pre-activations w . x + b on the initialization batch have mean 0 and biased variance 1; with the unbiased
# standard deviation the biased variance is (m - 1) / m; a second batch is not normalized.
import numpy as np


def init(v, x, ddof=0):
    t = x @ v.T / np.linalg.norm(v, axis=1)
    sigma = t.std(axis=0, ddof=ddof)
    return 1.0 / sigma, -t.mean(axis=0) / sigma


def pre(v, g, b, x):
    return x @ (g[:, None] * v / np.linalg.norm(v, axis=1)[:, None]).T + b


def main():
    rng = np.random.default_rng(3)
    worst = 0.0
    for m in (2, 3, 10, 100):
        for _ in range(25):
            v = rng.normal(0.0, 0.05, size=(7, 5))
            x = rng.normal(size=(m, 5)) * rng.uniform(0.1, 20) + rng.normal(size=5) * 5
            g, b = init(v, x)
            y = pre(v, g, b, x)
            assert np.allclose(y.mean(axis=0), 0.0, atol=1e-10) and np.allclose(y.var(axis=0), 1.0, rtol=1e-10)
            gu, bu = init(v, x, ddof=1)
            assert np.allclose(pre(v, gu, bu, x).var(axis=0), (m - 1) / m, rtol=1e-10)
            other = pre(v, g, b, rng.normal(size=(m, 5)) * 3)
            worst = max(worst, float(np.max(np.abs(other.var(axis=0) - 1.0))))
    assert worst > 0.1
    # the worked example: v = (3, 4), t = (0, 3, 4, 7)
    x = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0], [5.0, 5.0]])
    g, b = init(np.array([[3.0, 4.0]]), x)
    assert np.allclose([g[0], b[0]], [0.4, -1.4])
    assert np.allclose(pre(np.array([[3.0, 4.0]]), g, b, x).ravel(), [-1.4, -0.2, 0.2, 1.4])
    print("ok: mean 0 and variance 1 on the initialization batch; (m - 1) / m with the unbiased std; not on a new batch")


if __name__ == "__main__":
    main()
