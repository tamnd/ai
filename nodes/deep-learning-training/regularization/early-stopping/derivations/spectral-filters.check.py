# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/early-stopping/ (ai://NEW-early-stopping).
# Check of derivations/spectral-filters.md: gradient descent from zero on least squares is the spectral filter
# 1 - (1 - eta s)^t, ridge is s / (s + lam), the two have the same slope at s = 0 when lam = 1 / (eta t), and for
# 0 <= eta s <= 1 the ridge filter at that lam is below the gradient descent filter, which is below twice it.
import numpy as np


def gd_filter(s, eta, t):
    return 1.0 - (1.0 - eta * s) ** t


def ridge_filter(s, lam):
    return s / (s + lam)


def spectral_fit(s, V, c, filt):
    keep = s > 1e-12 * s.max()
    coef = np.zeros_like(s)
    coef[keep] = filt(s[keep]) * c[keep] / s[keep]
    return V @ coef


def main():
    rng = np.random.default_rng(20261116)
    for n, p in ((60, 8), (10, 16)):
        X = rng.normal(size=(n, p)) * np.linspace(0.1, 2.0, p)
        y = X @ rng.normal(size=p) + rng.normal(size=n)
        s, V = np.linalg.eigh(X.T @ X / n)
        s = np.clip(s, 0.0, None)
        b = X.T @ y / n
        c = V.T @ b
        assert np.all(np.abs(c[s <= 1e-12 * s.max()]) < 1e-10), "b has no part in the null space of X"
        eta = 1.0 / s.max()
        w = np.zeros(p)
        for t in range(1, 201):
            w = w + eta * X.T @ (y - X @ w) / n
            if t in (1, 5, 25, 200):
                gd = spectral_fit(s, V, c, lambda v, t=t: gd_filter(v, eta, t))
                assert np.allclose(w, gd, rtol=1e-10, atol=1e-12), "the gradient descent iterate is the filter"
                lam = 1.0 / (eta * t)
                r = np.linalg.solve(X.T @ X / n + lam * np.eye(p), b)
                rs = spectral_fit(s, V, c, lambda v, lam=lam: ridge_filter(v, lam))
                assert np.allclose(r, rs, rtol=1e-9, atol=1e-12), "ridge is the filter"
                nr, ng = np.linalg.norm(r), np.linalg.norm(w)
                assert nr <= ng + 1e-12 and ng <= 2 * nr + 1e-12, "the norms inherit the sandwich"
                near = np.linalg.norm(w - r)
                print(f"ok: n={n} p={p} t={t}: |w_gd - w_ridge| / |w_ridge| = {near / nr:.4f}, |w_gd| / |w_ridge| = {ng / nr:.4f}")
    # slope at zero: both filters are eta t s to first order, and differ at second order
    eta, t = 0.05, 20
    lam = 1.0 / (eta * t)
    for h in (1e-4, 1e-5):
        assert abs(gd_filter(h, eta, t) / h - eta * t) < 2 * eta**2 * t**2 * h
        assert abs(ridge_filter(h, lam) / h - eta * t) < 2 * eta**2 * t**2 * h
    h = 1e-3
    second_gd = (gd_filter(h, eta, t) - eta * t * h) / h**2
    second_ridge = (ridge_filter(h, lam) - eta * t * h) / h**2
    assert abs(second_gd + eta**2 * t * (t - 1) / 2) < 1e-2 and abs(second_ridge + (eta * t) ** 2) < 1e-2
    print(f"ok: slope eta t = {eta * t} for both; second-order coefficients {second_gd:.3f} and {second_ridge:.3f}")
    # the sandwich ridge(1 / (eta t)) <= gd <= min(1, eta t s) <= 2 ridge(1 / (eta t)) on a grid of 0 <= eta s <= 1
    x = np.linspace(0.0, 1.0, 2001)
    for t in (1, 2, 3, 10, 100, 1000):
        g = 1.0 - (1.0 - x) ** t
        r = t * x / (1.0 + t * x)
        m = np.minimum(1.0, t * x)
        assert np.all(r <= g + 1e-15) and np.all(g <= m + 1e-15) and np.all(m <= 2 * r + 1e-15)
        ratio = np.max(g[1:] / r[1:])
        print(f"ok: t={t}: max gd / ridge on (0, 1] = {ratio:.4f}")


if __name__ == "__main__":
    main()
