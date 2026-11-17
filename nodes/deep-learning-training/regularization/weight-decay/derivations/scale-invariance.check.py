# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/weight-decay/ (ai://DD5QVM).
# Check of derivations/scale-invariance.md: for a loss with L(c w) = L(w) for all c > 0, the gradient is orthogonal to
# w and scales as 1/c; without decay, gradient descent only grows |w|; a step moves the direction w/|w| like a step of
# size lr/|w|^2 on the unit sphere; decay by (1 - lr lam) is the same as an exponentially growing learning rate
# lr (1 - lr lam)^(-2t-1) on an undecayed copy (Li and Arora, momentum 0); and under the model |g| = G/|w| the squared
# norm settles at sqrt(lr G^2 / (lam (2 - lr lam))), where the angular step is about sqrt(2 lr lam).
import numpy as np


def make_loss(rng, d):
    """A scale-invariant loss: a quadratic in the direction w / |w|."""
    A = rng.normal(size=(d, d))
    H = A @ A.T / d
    b = rng.normal(size=d)

    def loss(w):
        u = w / np.linalg.norm(w)
        return 0.5 * u @ H @ u - b @ u

    def grad(w):
        r = np.linalg.norm(w)
        u = w / r
        gu = H @ u - b
        return (gu - (u @ gu) * u) / r  # chain rule through u = w / |w|

    return loss, grad


def main():
    rng = np.random.default_rng(20261118)
    d = 7
    loss, grad = make_loss(rng, d)
    w = rng.normal(size=d)
    # 1. Invariance, orthogonality and the 1/c scaling, with a finite-difference check of the gradient.
    for c in (0.1, 3.0, 50.0):
        assert np.isclose(loss(c * w), loss(w), rtol=1e-12)
        assert np.allclose(grad(c * w), grad(w) / c, rtol=1e-10, atol=1e-14)
    assert abs(w @ grad(w)) < 1e-12 * np.linalg.norm(w) * np.linalg.norm(grad(w)) + 1e-15
    e = np.eye(d)
    fd = np.array([(loss(w + 1e-6 * e[i]) - loss(w - 1e-6 * e[i])) / 2e-6 for i in range(d)])
    assert np.allclose(fd, grad(w), rtol=1e-6, atol=1e-9)
    # 2. Without decay the squared norm grows by exactly lr^2 |g|^2 per step.
    lr = 0.5
    v = w.copy()
    for _ in range(50):
        g = grad(v)
        new = v - lr * g
        assert np.isclose(new @ new, v @ v + lr**2 * (g @ g), rtol=1e-12)
        v = new
    # 3. The direction after one step equals a step of size lr / |w|^2 with the gradient at the unit vector,
    # renormalized: (w - lr g(w)) / |.| = (u - (lr / |w|^2) g(u)) / |.|.
    r = np.linalg.norm(w)
    u = w / r
    a = w - lr * grad(w)
    b = u - (lr / r**2) * grad(u)
    assert np.allclose(a / np.linalg.norm(a), b / np.linalg.norm(b), rtol=1e-12)
    # 4. Li and Arora (momentum 0): decayed GD w_{t+1} = (1 - lr lam) w_t - lr g(w_t) and undecayed GD with learning
    # rate lr_t = lr (1 - lr lam)^(-2t-1) give iterates related by wt_t = (1 - lr lam)^(-t) w_t.
    lam = 0.02
    q = 1 - lr * lam
    x = w.copy()
    y = w.copy()
    for t in range(60):
        assert np.allclose(y, x / q**t, rtol=1e-9)
        x = q * x - lr * grad(x)
        y = y - lr * q ** (-2 * t - 1) * grad(y)
    # Same directions, so the same loss at every step.
    assert np.isclose(loss(x), loss(y), rtol=1e-10)
    # 5. Equilibrium of the norm under |g| = G / |w| (g orthogonal to w): r_{t+1} = q^2 r_t + lr^2 G^2 / r_t for
    # r = |w|^2. The fixed point is r* = sqrt(lr G^2 / (lam (2 - lr lam))), and the map has slope 2 q^2 - 1 there.
    G = 1.7
    for lr_, lam_ in ((0.1, 5e-4), (0.5, 1e-2), (1.0, 1e-4)):
        q2 = (1 - lr_ * lam_) ** 2
        r = 1.0
        for _ in range(200000):
            r = q2 * r + lr_**2 * G**2 / r
        r_star = np.sqrt(lr_ * G**2 / (lam_ * (2 - lr_ * lam_)))
        assert np.isclose(r, r_star, rtol=1e-8), (r, r_star)
        assert abs(2 * q2 - 1) < 1
        # The angular step: tan(phi) = lr |g| / (q |w|) = sqrt(lr lam (2 - lr lam)) / q, about sqrt(2 lr lam).
        tan_phi = lr_ * (G / np.sqrt(r_star)) / ((1 - lr_ * lam_) * np.sqrt(r_star))
        assert np.isclose(tan_phi, np.sqrt(lr_ * lam_ * (2 - lr_ * lam_)) / (1 - lr_ * lam_), rtol=1e-12)
        assert np.isclose(tan_phi, np.sqrt(2 * lr_ * lam_), rtol=0.02)
        # The effective learning rate lr / |w|^2 at equilibrium is sqrt(lr lam (2 - lr lam)) / G.
        assert np.isclose(lr_ / r_star, np.sqrt(lr_ * lam_ * (2 - lr_ * lam_)) / G, rtol=1e-12)
    print("ok: scale invariance makes decay a learning-rate control; equilibrium angular step is about sqrt(2 lr lam)")


if __name__ == "__main__":
    main()
