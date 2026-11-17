# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/expected-calibration-error/ (ai://01R1CF).
# Check of derivations/temperature-scaling.md: softmax(z / T) keeps the order of the logits, so the argmax and the
# accuracy do not change; the confidence does not increase in T and tends to 1/K and to 1; the mean NLL is convex in
# beta = 1/T with slope mean(E_q[z] - z_y) and second derivative mean(Var_q[z]); and at the minimizer the slope is 0.
import numpy as np


def softmax(z, beta):
    s = beta * z
    e = np.exp(s - s.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def nll(z, y, beta):
    s = beta * z
    m = s.max(axis=1)
    return float(np.mean(m + np.log(np.exp(s - m[:, None]).sum(axis=1)) - s[np.arange(len(y)), y]))


def main():
    rng = np.random.default_rng(20261116)
    for _ in range(500):
        n, k = int(rng.integers(1, 30)), int(rng.integers(2, 8))
        z = rng.normal(size=(n, k)) * rng.uniform(0.1, 5)
        y = rng.integers(0, k, size=n)
        betas = np.sort(rng.uniform(0.01, 10, size=6))
        # 1. Order and argmax are kept for every T > 0, hence the accuracy.
        for b in betas:
            q = softmax(z, b)
            assert np.array_equal(np.argsort(-q, axis=1, kind="stable"), np.argsort(-z, axis=1, kind="stable"))
            assert np.array_equal(q.argmax(axis=1), z.argmax(axis=1))
            assert np.mean(q.argmax(axis=1) == y) == np.mean(z.argmax(axis=1) == y)
        # 2. The confidence max_k q_k does not decrease in beta (does not increase in T).
        conf = np.array([softmax(z, b).max(axis=1) for b in betas])
        assert np.all(np.diff(conf, axis=0) >= -1e-15)
        # 3. Slope and curvature of the mean NLL in beta, against central differences.
        b, h = float(rng.uniform(0.2, 3)), 1e-5
        q = softmax(z, b)
        mean_z = np.sum(q * z, axis=1)
        slope = float(np.mean(mean_z - z[np.arange(n), y]))
        curv = float(np.mean(np.sum(q * z * z, axis=1) - mean_z**2))
        assert abs((nll(z, y, b + h) - nll(z, y, b - h)) / (2 * h) - slope) < 1e-6
        assert abs((nll(z, y, b + h) - 2 * nll(z, y, b) + nll(z, y, b - h)) / h**2 - curv) < 1e-3
        assert curv >= 0
        # convexity along a grid: second differences are not negative
        grid = np.linspace(0.01, 6, 200)
        vals = np.array([nll(z, y, g) for g in grid])
        assert np.all(vals[2:] - 2 * vals[1:-1] + vals[:-2] >= -1e-10)
    # 4. Limits: T -> infinity gives 1/K, T -> 0 gives a point mass on a unique maximum.
    z = np.array([[2.0, 1.0, 0.0, -1.0]])
    assert np.allclose(softmax(z, 1e-9), 0.25, atol=1e-8)
    assert abs(softmax(z, 1e3)[0, 0] - 1.0) < 1e-12
    # 5. At the minimizer of the NLL over beta the mean true-class logit equals the mean softmax-weighted logit.
    z = rng.normal(size=(2000, 10)) * 3
    y = np.array([rng.choice(10, p=p) for p in softmax(z, 0.5)])
    lo, hi = 1e-3, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        s = float(np.mean(np.sum(softmax(z, mid) * z, axis=1) - z[np.arange(2000), y]))
        lo, hi = (lo, mid) if s > 0 else (mid, hi)
    beta = 0.5 * (lo + hi)
    q = softmax(z, beta)
    assert abs(float(np.mean(np.sum(q * z, axis=1))) - float(np.mean(z[np.arange(2000), y]))) < 1e-9
    assert all(nll(z, y, beta) <= nll(z, y, beta * f) + 1e-12 for f in (0.9, 0.99, 1.01, 1.1))
    assert abs(beta - 0.5) < 0.05  # labels drawn at T = 2 give back T near 2
    print("ok: temperature scaling keeps the argmax and accuracy, NLL convex in 1/T, stationarity at the fit")


if __name__ == "__main__":
    main()
