# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/rms-normalization/ (ai://NEW-rms-normalization).
# Check of derivations/gradient.md: the Jacobian of RMSNorm, (diag g / r)(I - a a^T / (n r^2)), against central
# finite differences; the input gradient is orthogonal to a when eps = 0 and leaks eps / (m + eps) otherwise; the gain
# gradient; the 1/delta scaling of the gradient (Eq. 10 of Zhang and Sennrich); and the same facts for pRMSNorm.
import numpy as np


def rn(a, g, eps, k=None):
    k = a.size if k is None else k
    return g * a / np.sqrt((a[:k] ** 2).mean() + eps)


def jac_fd(f, a, h=1e-6):
    cols = []
    for i in range(a.size):
        e = np.zeros_like(a)
        e[i] = h
        cols.append((f(a + e) - f(a - e)) / (2 * h))
    return np.stack(cols, axis=1)  # J[i, j] = d f_i / d a_j


def grad_input(a, g, dy, eps, k=None):
    """The input gradient: (1/r)(u - s a_K (a^T u) / (k r^2)), with u = g * dy and s_K the mask of the first k."""
    n = a.size
    k = n if k is None else k
    mask = (np.arange(n) < k).astype(float)
    r = np.sqrt((a[:k] ** 2).mean() + eps)
    u = g * dy
    return (u - mask * a * (a @ u) / (k * r ** 2)) / r


def main():
    rng = np.random.default_rng(20261116)
    for n in [2, 3, 8, 33]:
        for eps in [0.0, 1e-4, 0.5]:
            a = rng.normal(size=n) + 0.4
            g, dy = rng.normal(size=n), rng.normal(size=n)
            r = np.sqrt((a ** 2).mean() + eps)
            J = np.diag(g) @ (np.eye(n) - np.outer(a, a) / (n * r ** 2)) / r
            assert np.allclose(J, jac_fd(lambda v: rn(v, g, eps), a), rtol=1e-6, atol=1e-8)
            dx = grad_input(a, g, dy, eps)
            assert np.allclose(J.T @ dy, dx, rtol=1e-12, atol=1e-12)
            # orthogonality to a, exact with eps = 0, with the leak eps / (m + eps) otherwise
            m = (a ** 2).mean()
            assert np.isclose(a @ dx, (a @ (g * dy)) / r * eps / (m + eps), rtol=1e-9, atol=1e-12)
            if eps == 0.0:
                assert abs(a @ dx) < 1e-12 * (1 + np.abs(a).sum() * np.abs(dy).sum())
                assert np.allclose(J @ a, 0.0, atol=1e-12)  # moving along a does not change the output
            # the gain gradient is dy times the normalized input
            gfd = jac_fd(lambda q: np.array([dy @ rn(a, q, eps)]), g)[0]
            assert np.allclose(gfd, dy * a / r, rtol=1e-6, atol=1e-8)
            # pRMSNorm: the first k entries enter r, the rest only scale
            for k in sorted({1, max(1, n // 4), n}):
                rk = np.sqrt((a[:k] ** 2).mean() + eps)
                Jk = jac_fd(lambda v: rn(v, g, eps, k), a)
                dxk = grad_input(a, g, dy, eps, k)
                assert np.allclose(Jk.T @ dy, dxk, rtol=1e-6, atol=1e-7)
                mk = (a[:k] ** 2).mean()
                assert np.isclose(a @ dxk, (a @ (g * dy)) / rk * eps / (mk + eps), rtol=1e-8, atol=1e-11)
    # Eq. 10: with eps = 0, scaling the input of the normalization by delta divides the gradient by delta
    a, g, dy = rng.normal(size=6), rng.normal(size=6), rng.normal(size=6)
    for delta in [0.1, 3.0, 50.0]:
        assert np.allclose(grad_input(delta * a, g, dy, 0.0), grad_input(a, g, dy, 0.0) / delta, rtol=1e-12)
        assert np.allclose(grad_input(delta * a, g, dy, 0.0, 3), grad_input(a, g, dy, 0.0, 3) / delta, rtol=1e-12)
    # the worked example: a = (4, 0, -2, 4), g = 1, dy = e_1
    a = np.array([4.0, 0.0, -2.0, 4.0])
    dx = grad_input(a, np.ones(4), np.array([1.0, 0, 0, 0]), 0.0)
    assert np.allclose(dx, np.array([5.0, 0.0, 2.0, -4.0]) / 27)
    assert np.isclose(a @ dx, 0.0)
    assert np.allclose(grad_input(2 * a, np.ones(4), np.array([1.0, 0, 0, 0]), 0.0), dx / 2)
    print("ok: the Jacobian matches finite differences, the gradient is orthogonal to a at eps = 0, Eq. 10 holds, pRMSNorm too")


if __name__ == "__main__":
    main()
