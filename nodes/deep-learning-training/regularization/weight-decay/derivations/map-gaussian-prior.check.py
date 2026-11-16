# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/weight-decay/ (ai://NEW-weight-decay).
# Check of derivations/map-gaussian-prior.md: the MAP estimate under a Gaussian prior N(0, s_p^2 I) minimizes the
# negative log likelihood plus |theta|^2 / (2 s_p^2); for a linear model with Gaussian noise of variance s^2 it is
# ridge regression with lam = s^2 / s_p^2; that lam minimizes the expected squared error of the weights (the optimal
# weight decay of Krogh and Hertz); and as lam -> 0 the ridge solution tends to the minimum-norm solution.
import numpy as np


def ridge(X, y, lam):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def main():
    rng = np.random.default_rng(20261117)
    n, d = 30, 8
    X = rng.normal(size=(n, d)) / np.sqrt(d)
    s, s_p = 0.5, 1.3
    theta_true = s_p * rng.normal(size=d)
    y = X @ theta_true + s * rng.normal(size=n)
    # 1. Gradient descent on the negative log posterior reaches the closed-form ridge solution with lam = s^2 / s_p^2.
    theta = np.zeros(d)
    for _ in range(20000):
        g = -X.T @ (y - X @ theta) / s**2 + theta / s_p**2
        theta = theta - 0.05 * g
    assert np.allclose(theta, ridge(X, y, s**2 / s_p**2), rtol=1e-10, atol=1e-12)
    # The same MAP from the mean loss over n examples needs the coefficient lam / n.
    theta_m = np.zeros(d)
    for _ in range(40000):
        g = -X.T @ (y - X @ theta_m) / n + (s**2 / s_p**2) / n * theta_m
        theta_m = theta_m - 0.5 * g
    assert np.allclose(theta_m, theta, rtol=1e-9, atol=1e-11)
    # 2. The expected error E|theta_hat - theta|^2 over the prior and the noise, computed exactly:
    # lam^2 s_p^2 tr(A^2) + s^2 tr(A X^T X A) with A = (X^T X + lam I)^-1, is smallest at lam = s^2 / s_p^2.
    XtX = X.T @ X

    def expected_error(lam):
        A = np.linalg.inv(XtX + lam * np.eye(d))
        return lam**2 * s_p**2 * np.trace(A @ A) + s**2 * np.trace(A @ XtX @ A)

    lams = np.exp(np.linspace(np.log(1e-4), np.log(1e2), 4001))
    errs = np.array([expected_error(lam) for lam in lams])
    best = lams[np.argmin(errs)]
    assert abs(np.log(best / (s**2 / s_p**2))) < 0.01, (best, s**2 / s_p**2)
    assert expected_error(s**2 / s_p**2) < expected_error(0.0)
    # A Monte Carlo estimate agrees with the exact formula at the optimum.
    errs_mc = []
    for _ in range(20000):
        th = s_p * rng.normal(size=d)
        yy = X @ th + s * rng.normal(size=n)
        errs_mc.append(np.sum((ridge(X, yy, s**2 / s_p**2) - th) ** 2))
    assert abs(np.mean(errs_mc) / expected_error(s**2 / s_p**2) - 1) < 0.03
    # 3. With fewer examples than weights, the limit lam -> 0 is the minimum-norm solution pinv(X) y, which has no
    # component in the null space of X.
    Xs = rng.normal(size=(4, d))
    ys = rng.normal(size=4)
    # In the SVD X = U diag(sv) V^T the ridge solution is V diag(sv / (sv^2 + lam)) U^T y, which tends to
    # V diag(1 / sv) U^T y = pinv(X) y; the d x d normal equations are singular at lam = 0, so use this form.
    U, sv, Vt = np.linalg.svd(Xs, full_matrices=False)

    def ridge_svd(lam):
        return Vt.T @ ((sv / (sv**2 + lam)) * (U.T @ ys))

    assert np.allclose(ridge_svd(0.3), ridge(Xs, ys, 0.3), rtol=1e-10)
    pinv_sol = np.linalg.pinv(Xs) @ ys
    gaps = [np.linalg.norm(ridge_svd(lam) - pinv_sol) for lam in (1e-2, 1e-4, 1e-6, 1e-8)]
    assert all(g2 < g1 for g1, g2 in zip(gaps, gaps[1:])) and gaps[-1] < 1e-6
    # Gradient descent with decay lam shrinks a null-space component by exactly (1 - lr lam) per step, since the
    # loss gradient -Xs^T (ys - Xs th) lies in the row space and has no component along a null vector.
    null = np.linalg.svd(Xs)[2][-1]
    assert np.allclose(Xs @ null, 0.0, atol=1e-12)
    lr, lam = 0.1, 0.2
    th = 3.0 * null
    comp = []
    for _ in range(10):
        th = th - lr * (-Xs.T @ (ys - Xs @ th) + lam * th)
        comp.append(th @ null)
    assert np.allclose(comp, [3.0 * (1 - lr * lam) ** (k + 1) for k in range(10)], rtol=1e-10)
    print("ok: a Gaussian prior gives L2 at the MAP, ridge with s^2/s_p^2 is the best decay, and lam -> 0 is min-norm")


if __name__ == "__main__":
    main()
