# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/weight-decay/ (ai://DD5QVM).
# Check of derivations/sgd-equivalence.md: for plain SGD an L2 coefficient lam is a decay of delta = lr * lam per
# step; with no gradient the weights decay geometrically; with heavy-ball momentum the two forms leave step 1 equal
# and differ from step 2 by lr * mu * lam * theta_0; their fixed points agree when delta = lr * lam / (1 - mu).
import numpy as np


def sgd_l2(theta, grad_fn, lr, lam, mu, steps):
    """torch.optim.SGD with weight_decay=lam, momentum=mu, dampening 0: the L2 term enters the buffer."""
    b = np.zeros_like(theta)
    out = []
    for t in range(steps):
        g = grad_fn(theta, t) + lam * theta
        b = mu * b + g if t else g
        theta = theta - lr * b
        out.append(theta)
    return out


def sgd_decoupled(theta, grad_fn, lr, delta, mu, steps):
    """Decoupled decay beside heavy-ball momentum: theta <- (1 - delta) theta - lr b, with b fed by the loss only."""
    b = np.zeros_like(theta)
    out = []
    for t in range(steps):
        g = grad_fn(theta, t)
        b = mu * b + g if t else g
        theta = (1 - delta) * theta - lr * b
        out.append(theta)
    return out


def main():
    rng = np.random.default_rng(20261115)
    d = 6
    theta0 = rng.normal(size=d)
    noise = [rng.normal(size=d) for _ in range(40)]
    A = rng.normal(size=(d, d))
    H = A @ A.T / d + 0.1 * np.eye(d)
    c = rng.normal(size=d)

    def stoch(theta, t):  # a quadratic with a fresh noise term at each step
        return H @ theta - c + 0.3 * noise[t]

    def det(theta, t):
        return H @ theta - c

    # 1. Plain SGD: L2 with lam equals decoupled decay with delta = lr * lam, step by step.
    lr, lam = 0.05, 0.3
    a = sgd_l2(theta0, stoch, lr, lam, 0.0, 40)
    b = sgd_decoupled(theta0, stoch, lr, lr * lam, 0.0, 40)
    for x, y in zip(a, b):
        assert np.allclose(x, y, rtol=1e-12, atol=1e-14)
    # 2. With no gradient the weights shrink by the factor 1 - lr * lam per step, an exponential decay.
    zero = sgd_l2(theta0, lambda th, t: np.zeros(d), lr, lam, 0.0, 25)
    assert np.allclose(zero[-1], (1 - lr * lam) ** 25 * theta0, rtol=1e-12)
    # 3. Momentum: the two forms agree at step 1 with delta = lr * lam, and at step 2 the L2 form has moved further
    # by exactly lr * mu * lam * theta_0, the decay remembered by the buffer.
    mu = 0.9
    a = sgd_l2(theta0, stoch, lr, lam, mu, 2)
    b = sgd_decoupled(theta0, stoch, lr, lr * lam, mu, 2)
    assert np.allclose(a[0], b[0], rtol=1e-12)
    assert np.allclose(b[1] - a[1], lr * mu * lam * theta0, rtol=1e-10, atol=1e-14)
    # No constant delta matches the L2 form over many steps: fit delta on step 1, check the gap grows.
    a = sgd_l2(theta0, stoch, lr, lam, mu, 40)
    b = sgd_decoupled(theta0, stoch, lr, lr * lam, mu, 40)
    assert np.linalg.norm(a[-1] - b[-1]) > 1e-3
    # 4. Fixed points on the deterministic quadratic: the L2 form solves (H + lam I) theta = c; the decoupled form
    # with delta solves (H + delta (1 - mu) / lr I) theta = c. They agree when delta = lr * lam / (1 - mu).
    a = sgd_l2(theta0, det, lr, lam, mu, 3000)[-1]
    b = sgd_decoupled(theta0, det, lr, lr * lam / (1 - mu), mu, 3000)[-1]
    star = np.linalg.solve(H + lam * np.eye(d), c)
    assert np.allclose(a, star, rtol=1e-9, atol=1e-11)
    assert np.allclose(b, star, rtol=1e-9, atol=1e-11)
    b_wrong = sgd_decoupled(theta0, det, lr, lr * lam, mu, 3000)[-1]
    assert np.allclose(b_wrong, np.linalg.solve(H + lam * (1 - mu) * np.eye(d), c), rtol=1e-9, atol=1e-11)
    assert not np.allclose(b_wrong, star, rtol=1e-3)
    print("ok: L2 is decay for plain SGD; with momentum the trajectories differ and the fixed points need 1/(1-mu)")


if __name__ == "__main__":
    main()
