# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/few-shot-in-context-learning/ (ai://JNEP9D).
# Check of derivations/bayes-optimal-in-context-regression.md: under a Gaussian prior ridge with lambda =
# sigma^2 / tau^2 is the posterior mean and the Bayes-optimal in-context predictor, minimum-norm least squares is
# the noiseless limit and a projection, and the noiseless error curve is (d - k)/d.
import numpy as np


def ridge(x, y, lam):
    d = x.shape[1]
    return np.linalg.solve(x.T @ x + lam * np.eye(d), x.T @ y)


def main():
    rng = np.random.default_rng(20261116)
    # 1. Ridge equals the posterior mean from Gaussian conditioning.
    for k, d, tau, sigma in [(3, 5, 1.0, 1.0), (8, 4, 0.5, 2.0), (20, 20, 2.0, 0.3)]:
        x = rng.normal(size=(k, d))
        y = rng.normal(size=k)
        post = tau**2 * x.T @ np.linalg.solve(tau**2 * x @ x.T + sigma**2 * np.eye(k), y)
        assert np.allclose(ridge(x, y, sigma**2 / tau**2), post, atol=1e-10)
    # Monte Carlo: lambda* = sigma^2 / tau^2 beats other penalties and least squares on fresh tasks.
    d, k, tau, sigma, trials = 10, 10, 1.0, 1.0, 20000
    lam = sigma**2 / tau**2
    err = {c: 0.0 for c in ["star", "small", "big", "ols"]}
    for _ in range(trials):
        w = rng.normal(0, tau, size=d)
        x = rng.normal(size=(k, d))
        y = x @ w + rng.normal(0, sigma, size=k)
        xq = rng.normal(size=d)
        yq = w @ xq + rng.normal(0, sigma)
        for c, l in [("star", lam), ("small", lam / 4), ("big", 4 * lam)]:
            err[c] += (ridge(x, y, l) @ xq - yq) ** 2
        err["ols"] += (np.linalg.pinv(x) @ y @ xq - yq) ** 2
    assert err["star"] < min(err["small"], err["big"], err["ols"]), err
    assert err["ols"] > 2 * err["star"], err  # least squares blows up near k = d with noise
    # 2. Noiseless: lambda -> 0 gives the pseudo-inverse, which is the projection of w onto the span of the inputs.
    for k, d in [(3, 7), (7, 7), (12, 5)]:
        x = rng.normal(size=(k, d))
        w = rng.normal(size=d)
        y = x @ w
        mn = np.linalg.pinv(x) @ y
        assert np.allclose(ridge(x, y, 1e-10), mn, atol=1e-6)
        q, _ = np.linalg.qr(x.T)
        proj = q @ q.T
        assert np.allclose(mn, proj @ w, atol=1e-10)
        # Posterior mean by conditioning on y = X w with a tiny noise level agrees.
        post = x.T @ np.linalg.solve(x @ x.T + 1e-12 * np.eye(k), y) if k <= d else mn
        assert np.allclose(post, mn, atol=1e-5)
    # 3. Error curve (d - k)/d at d = 20.
    d, trials = 20, 4000
    for k in [0, 5, 10, 15, 19, 20, 30]:
        tot = 0.0
        for _ in range(trials):
            w = rng.normal(size=d)
            xq = rng.normal(size=d)
            if k == 0:
                pred = 0.0
            else:
                x = rng.normal(size=(k, d))
                pred = np.linalg.pinv(x) @ (x @ w) @ xq
            tot += (pred - w @ xq) ** 2 / d
        want = max(d - k, 0) / d
        assert abs(tot / trials - want) < 0.06 + 0.05 * want, (k, tot / trials, want)
    # 4. Garg et al.'s noisy task: tau^2 = 1 and sigma^2 = 1 give lambda = 1.
    assert 1.0**2 / 1.0**2 == 1.0
    print("bayes-optimal-in-context-regression: ok")


if __name__ == "__main__":
    main()
