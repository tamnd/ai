# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/weight-decay/ (ai://NEW-weight-decay).
# Check of derivations/preconditioned.md: with a fixed preconditioner P, an L2 term is multiplied by P and decoupled
# decay is not; they agree for all theta only if P is a multiple of the identity; decoupled decay is preconditioned
# descent on the loss plus (delta / (2 lr)) theta^T P^-1 theta; and the two forms have different fixed points.
import numpy as np


def main():
    rng = np.random.default_rng(20261116)
    d = 5
    A = rng.normal(size=(d, d))
    H = A @ A.T / d + 0.2 * np.eye(d)
    c = rng.normal(size=d)
    s = np.array([100.0, 10.0, 1.0, 0.1, 0.01])  # typical gradient sizes, as sqrt(v_hat) in Adam
    P = np.diag(1.0 / s)
    lr, lam = 0.01, 0.5
    delta = lr * lam

    def grad(theta):
        return H @ theta - c

    # 1. One step from the same theta: the L2 decay of coordinate i is lr * lam * theta_i / s_i, the decoupled one
    # is delta * theta_i, so they agree only on coordinates with s_i = 1.
    theta = rng.normal(size=d)
    l2_step = theta - lr * P @ (grad(theta) + lam * theta)
    wd_step = (1 - delta) * theta - lr * P @ grad(theta)
    diff = l2_step - wd_step
    assert np.allclose(diff, delta * theta * (1 - 1 / s), rtol=1e-12, atol=1e-15)
    assert np.isclose(diff[2], 0.0, atol=1e-15) and not np.isclose(diff[0], 0.0)
    # 2. For P = k I the L2 coefficient delta / (lr k) reproduces the decay exactly.
    k = 3.0
    l2_k = theta - lr * k * (grad(theta) + delta / (lr * k) * theta)
    wd_k = (1 - delta) * theta - lr * k * grad(theta)
    assert np.allclose(l2_k, wd_k, rtol=1e-13)
    # 3. Decoupled decay is a preconditioned step on f + (delta / (2 lr)) theta^T P^-1 theta.
    Pinv = np.diag(s)
    reg_step = theta - lr * P @ (grad(theta) + (delta / lr) * Pinv @ theta)
    assert np.allclose(reg_step, wd_step, rtol=1e-12, atol=1e-15)
    # 4. Fixed points: L2 solves (H + lam I) theta = c whatever P is; decoupled decay solves
    # (H + (delta / lr) P^-1) theta = c, so it shrinks the coordinates with large s_i more.
    # (A milder spread of s keeps the iteration fast enough to converge in a few thousand steps.)
    s2 = np.array([4.0, 2.0, 1.0, 0.5, 0.25])
    P2, P2inv = np.diag(1.0 / s2), np.diag(s2)
    lr2 = 0.05
    delta2 = lr2 * lam
    a = b = np.zeros(d)
    for _ in range(20000):
        a = a - lr2 * P2 @ (grad(a) + lam * a)
        b = (1 - delta2) * b - lr2 * P2 @ grad(b)
    assert np.allclose(a, np.linalg.solve(H + lam * np.eye(d), c), rtol=1e-9, atol=1e-11)
    assert np.allclose(b, np.linalg.solve(H + (delta2 / lr2) * P2inv, c), rtol=1e-9, atol=1e-11)
    assert not np.allclose(a, b, rtol=1e-2)
    print("ok: with a preconditioner, decoupled decay is L2 on theta^T P^-1 theta, not on |theta|^2")


if __name__ == "__main__":
    main()
