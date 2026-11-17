# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/adam/ (ai://NEW-adam).
# Check of derivations/decoupled-weight-decay.md: weight decay and L2 agree for SGD with lambda' = lambda / lr, they
# disagree for Adam when the preconditioner is not a multiple of the identity, an L2 term is normalized away at the
# first step, and the PyTorch AdamW step is Algorithm 2 of Loshchilov and Hutter with lambda = lr * weight_decay.
import numpy as np


def adam(theta, grads, lr, b1=0.9, b2=0.999, eps=1e-8, l2=0.0, decay=0.0):
    """Adam with an L2 coefficient l2 added to the gradient, and a decoupled decay theta <- theta - decay theta."""
    m = v = np.zeros_like(theta)
    for t, g in enumerate(grads, start=1):
        g = g + l2 * theta
        theta = theta - decay * theta
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        theta = theta - lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)
    return theta


def main():
    rng = np.random.default_rng(20261114)
    # 1. SGD: theta - lr (g + l2 theta) = (1 - lambda) theta - lr g with lambda = lr l2 (Proposition 1).
    theta0, lr, lam = rng.normal(size=5), 0.05, 0.01
    a = b = theta0
    for _ in range(10):
        g = rng.normal(size=5)
        a = a - lr * (g + (lam / lr) * a)
        b = (1 - lam) * b - lr * g
    assert np.allclose(a, b, rtol=1e-12, atol=1e-14)
    # 2. Adam: for a fixed decoupled decay, the L2 coefficient that matches one coordinate misses the other, because
    # the decay of coordinate i passes through 1 / (sqrt(v_hat_i) + eps) (Proposition 2).
    theta0 = np.array([1.0, 1.0])
    grads = [np.array([10.0, 0.1]) * rng.choice([-1, 1], size=2) for _ in range(20)]
    lr, lam = 1e-2, 1e-3
    target = adam(theta0, grads, lr, decay=lam)
    fits = []
    for i in range(2):
        lo, hi = 0.0, 100.0
        for _ in range(200):  # bisection on the L2 coefficient for coordinate i
            mid = (lo + hi) / 2
            if adam(theta0, grads, lr, l2=mid)[i] > target[i]:
                lo = mid
            else:
                hi = mid
        fits.append(lo)
    assert fits[0] > 10 * fits[1], fits
    # 3. At the first step with eps = 0, the L2 term changes nothing unless it flips the sign of the gradient.
    for theta, g, l2 in [(1.0, 2.0, 0.1), (3.0, -0.5, 0.1), (-2.0, 4.0, 1.0)]:
        x = adam(np.array([theta]), [np.array([g])], 0.1, eps=0.0, l2=l2)
        y = adam(np.array([theta]), [np.array([g])], 0.1, eps=0.0)
        assert np.allclose(x, y, rtol=1e-14)
    # 4. Algorithm 2, line 12, with schedule multiplier eta_t = 1 and step size alpha:
    # theta <- theta - (alpha m_hat / (sqrt(v_hat) + eps) + lambda theta). torch.optim.AdamW with lr = alpha and
    # weight_decay = wd multiplies theta by 1 - lr wd first: the same step with lambda = lr wd.
    theta0, grads, lr, wd = rng.normal(size=4), [rng.normal(size=4) for _ in range(8)], 1e-3, 0.1
    torch_style = theta0
    m = v = np.zeros(4)
    lh = theta0
    for t, g in enumerate(grads, start=1):
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        step = (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
        torch_style = torch_style * (1 - lr * wd) - lr * step
        lh = lh - (lr * step + (lr * wd) * lh)
    assert np.allclose(torch_style, lh, rtol=1e-12, atol=1e-15)
    assert np.allclose(adam(theta0, grads, lr, decay=lr * wd), lh, rtol=1e-12, atol=1e-15)
    print("ok: weight decay is L2 for SGD, not for Adam, and the PyTorch AdamW step is Algorithm 2 with lr * wd")


if __name__ == "__main__":
    main()
