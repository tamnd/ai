# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/sharpness-aware-minimization/ (ai://NEW-sharpness-aware-minimization).
# Check of derivations/dual-norm-perturbation.md: the perturbation rho sign(g) |g|^(q-1) / (||g||_q^q)^(1/p) has
# p-norm rho and attains rho ||g||_q, the largest value of e.g on the p-ball, which no sampled point of the ball beats;
# it is rho g / ||g||_2 at p = 2 and rho sign(g) at p = infinity; it does not change when g is scaled by c > 0; and
# the first-order objective L(w) + e.g is L(w) + rho ||g||_q, which matches max L(w + e) on the ball to O(rho^2).
import numpy as np


def eps_hat(g, rho, p):
    if np.isinf(p):
        return rho * np.sign(g)
    q = p / (p - 1.0)
    g = g / np.max(np.abs(g))  # the formula is unchanged by g -> c g, c > 0; this keeps |g|^q finite
    return rho * np.sign(g) * np.abs(g) ** (q - 1) / np.sum(np.abs(g) ** q) ** (1.0 / p)


def dual(p):
    return 1.0 if np.isinf(p) else p / (p - 1.0)


def pnorm(x, p):
    return np.max(np.abs(x), axis=-1) if np.isinf(p) else np.sum(np.abs(x) ** p, axis=-1) ** (1.0 / p)


rng = np.random.default_rng(0)
for p in [1.25, 1.5, 2.0, 3.0, 4.0, np.inf]:
    q = dual(p)
    for _ in range(50):
        d = int(rng.integers(2, 6))
        g = rng.normal(size=d)
        rho = float(rng.uniform(0.01, 2.0))
        e = eps_hat(g, rho, p)
        assert abs(pnorm(e, p) - rho) < 1e-9 * (1 + rho)
        best = e @ g
        assert abs(best - rho * pnorm(g, q)) < 1e-9 * (1 + abs(best))
        # brute force: points on the p-sphere of radius rho (the maximum of a linear function is on the boundary)
        z = rng.normal(size=(20000, d))
        z = rho * z / pnorm(z, p)[:, None]
        assert pnorm(z, p).max() <= rho * (1 + 1e-9)
        sampled = (z @ g).max()
        assert sampled <= best + 1e-9, (p, sampled, best)
        assert sampled >= best * 0.9 - 1e-9 if d <= 3 else True
        c = float(rng.uniform(0.1, 10.0))
        assert np.allclose(eps_hat(c * g, rho, p), e, atol=1e-10)

# p = 2 and p = infinity in closed form
g = rng.normal(size=7)
assert np.allclose(eps_hat(g, 0.3, 2.0), 0.3 * g / np.linalg.norm(g))
assert np.allclose(eps_hat(g, 0.3, np.inf), 0.3 * np.sign(g))
# p -> 1: the perturbation concentrates on the entry with the largest |g|
e = eps_hat(np.array([0.5, -2.0, 1.0]), 1.0, 1.01)
assert e[1] < -0.999 and np.abs(e[[0, 2]]).max() < 1e-20

# the first-order SAM objective versus the exact inner maximum on a smooth loss
A = rng.normal(size=(4, 4))
A = A @ A.T
b = rng.normal(size=4)


def loss(w):
    return 0.5 * w @ A @ w + b @ w + np.sum(np.cos(w))


def grad(w):
    return A @ w + b - np.sin(w)


w = rng.normal(size=4)
gaps = []
for rho in [0.1, 0.05, 0.025]:
    z = rng.normal(size=(200000, 4))
    z = rho * z / np.linalg.norm(z, axis=1, keepdims=True)
    exact = max(loss(w + zi) for zi in z[:4000])
    exact = max(exact, loss(w + eps_hat(grad(w), rho, 2.0)))
    first = loss(w) + rho * np.linalg.norm(grad(w))
    gaps.append(abs(exact - first))
assert gaps[1] < gaps[0] / 2.5 and gaps[2] < gaps[1] / 2.5, gaps  # O(rho^2): the gap shrinks about 4x per halving
print("ok")
