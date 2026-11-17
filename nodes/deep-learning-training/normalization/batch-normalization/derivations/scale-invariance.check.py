# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/batch-normalization/ (ai://TD3H76).
# Check of derivations/scale-invariance.md: batch normalization with eps = 0 is invariant to the scale of the weights
# before it, the gradient scales as 1/a, it is orthogonal to each row of the weights, and a plain step grows the norm.
import numpy as np


def bn(z):
    mu = z.mean(axis=0)
    sd = np.sqrt(((z - mu) ** 2).mean(axis=0))
    return (z - mu) / sd


def f(w, u, c):
    return float(np.sum(c * np.tanh(bn(u @ w.T))))


def grad(w, u, c, h=1e-6):
    g = np.zeros_like(w)
    for idx in np.ndindex(*w.shape):
        e = np.zeros_like(w)
        e[idx] = h
        g[idx] = (f(w + e, u, c) - f(w - e, u, c)) / (2 * h)
    return g


def main():
    rng = np.random.default_rng(20261114)
    m, d_in, d_out = 8, 5, 3
    u = rng.normal(size=(m, d_in))
    w = rng.normal(size=(d_out, d_in))
    c = rng.normal(size=(m, d_out))
    g = grad(w, u, c)
    for a in [0.1, 0.5, 2.0, 7.0]:
        assert np.allclose(bn(u @ (a * w).T), bn(u @ w.T), rtol=1e-12, atol=1e-12), a
        ga = grad(a * w, u, c, h=1e-6 * a)
        assert np.linalg.norm(ga - g / a) <= 1e-6 * np.linalg.norm(g / a), a
    for k in range(d_out):
        assert abs(np.dot(w[k], g[k])) <= 1e-6 * np.linalg.norm(w[k]) * np.linalg.norm(g[k]), k
    eta = 0.3
    w2 = w - eta * g
    lhs = np.sum(w2 ** 2, axis=1)
    rhs = np.sum(w ** 2, axis=1) + eta ** 2 * np.sum(g ** 2, axis=1)
    assert np.allclose(lhs, rhs, rtol=1e-6) and np.all(lhs >= np.sum(w ** 2, axis=1)), (lhs, rhs)
    print("ok: scale invariance, 1/a gradient scaling, orthogonality and norm growth hold")


if __name__ == "__main__":
    main()
