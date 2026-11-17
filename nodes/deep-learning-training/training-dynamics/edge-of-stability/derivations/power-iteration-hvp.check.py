# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/edge-of-stability/ (ai://17Q1PD).
# Check of derivations/power-iteration-hvp.md: the central-difference HVP is exact on quadratics and O(eps^2) otherwise;
# power iteration on HVPs converges to the eigenvalue of largest modulus at the rate (mu2 / mu1)^(2k); on a PSD Hessian
# and on a small tanh network near a minimum this is the top eigenvalue from eigvalsh.
import numpy as np


def hvp(grad, theta, v, eps):
    return (grad(theta + eps * v) - grad(theta - eps * v)) / (2 * eps)


def power(mv, d, iters, seed=0):
    q = np.random.default_rng(seed).normal(size=d)
    q /= np.linalg.norm(q)
    lam = 0.0
    for _ in range(iters):
        hq = mv(q)
        lam = q @ hq
        q = hq / np.linalg.norm(hq)
    return lam


rng = np.random.default_rng(0)
for d in (3, 8, 30):
    A = rng.normal(size=(d, d))
    H = A @ A.T / d
    b = rng.normal(size=d)
    grad = lambda t: H @ t + b  # noqa: E731
    th, v = rng.normal(size=d), rng.normal(size=d)
    for eps in (1.0, 1e-3):
        assert np.allclose(hvp(grad, th, v, eps), H @ v, atol=1e-9)
    lam = power(lambda q: hvp(grad, th, q, 1e-3), d, 3000)
    assert abs(lam / np.linalg.eigvalsh(H)[-1] - 1) < 1e-8

# quartic: grad of sum t^4 / 4 is t^3; error of the central difference is eps^2 v^3 (third-order term)
th, v = np.array([0.3, -0.7, 1.1]), np.array([1.0, -2.0, 0.5])
for eps in (0.1, 0.01, 0.001):
    err = hvp(lambda t: t**3, th, v, eps) - 3 * th**2 * v
    assert np.allclose(err, eps**2 * v**3, rtol=1e-5, atol=1e-15)

# rate: Rayleigh error ~ (mu2 / mu1)^(2k)
H = np.diag([1.0, 0.8, 0.1])
errs = [1 - power(lambda q: H @ q, 3, k) for k in (10, 20, 30)]
ratios = [errs[1] / errs[0], errs[2] / errs[1]]
assert all(abs(r / 0.8**20 - 1) < 0.2 for r in ratios)

# largest modulus, not largest value
assert abs(power(lambda q: np.diag([2.0, -5.0, 1.0]) @ q, 3, 500) + 5) < 1e-9
# a shift restores the top eigenvalue
assert abs(power(lambda q: (np.diag([2.0, -5.0, 1.0]) + 6 * np.eye(3)) @ q, 3, 2000) - 6 - 2) < 1e-6


def loss_grad(t, X, Y, h=5):
    W1, b1, W2 = t[: 2 * h].reshape(2, h), t[2 * h : 3 * h], t[3 * h :].reshape(h, 1)
    a = np.tanh(X @ W1 + b1)
    r = a @ W2 - Y
    n = len(X)
    g2 = a.T @ r / n
    ga = (r @ W2.T) * (1 - a**2) / n
    return 0.5 * float(np.sum(r * r)) / n, np.concatenate([(X.T @ ga).ravel(), ga.sum(0), g2.ravel()])


X = rng.normal(size=(15, 2))
Y = np.sin(X[:, :1] - X[:, 1:])
t = 0.5 * rng.normal(size=20)
for _ in range(500):
    t -= 0.3 * loss_grad(t, X, Y)[1]
g = lambda s: loss_grad(s, X, Y)[1]  # noqa: E731
Hm = np.stack([hvp(g, t, e, 1e-5) for e in np.eye(20)])
assert np.allclose(Hm, Hm.T, atol=1e-7)
ev = np.linalg.eigvalsh((Hm + Hm.T) / 2)
assert ev[-1] > abs(ev[0])
assert abs(power(lambda q: hvp(g, t, q, 1e-5), 20, 4000) / ev[-1] - 1) < 1e-6
print("power-iteration-hvp: ok")
