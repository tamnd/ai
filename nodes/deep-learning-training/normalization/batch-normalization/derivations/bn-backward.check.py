# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/batch-normalization/ (ai://NEW-batch-normalization).
# Check of derivations/bn-backward.md: the gradients of batch normalization against finite differences, against the
# separate chain-rule equations of Ioffe and Szegedy (2015), and the sum and orthogonality properties.
import numpy as np


def forward(x, gamma, beta, eps):
    mu = x.mean()
    var = ((x - mu) ** 2).mean()
    s = np.sqrt(var + eps)
    xhat = (x - mu) / s
    return gamma * xhat + beta, xhat, s, mu, var


def loss(x, gamma, beta, eps, c):
    y = forward(x, gamma, beta, eps)[0]
    return float(np.sum(c * np.tanh(y)))


def compact(g, xhat, s, gamma):
    m = g.size
    return gamma / (m * s) * (m * g - g.sum() - xhat * np.sum(g * xhat))


def ioffe(x, g, gamma, mu, var, eps):
    # The equations of Ioffe and Szegedy (2015, p. 4), term by term.
    m = x.size
    xhat = (x - mu) / np.sqrt(var + eps)
    dxhat = g * gamma
    dvar = np.sum(dxhat * (x - mu)) * -0.5 * (var + eps) ** -1.5
    dmu = np.sum(dxhat * -1.0 / np.sqrt(var + eps)) + dvar * np.sum(-2.0 * (x - mu)) / m
    dx = dxhat / np.sqrt(var + eps) + dvar * 2.0 * (x - mu) / m + dmu / m
    return dx, np.sum(g * xhat), np.sum(g)


def main():
    rng = np.random.default_rng(20261114)
    for m in [2, 3, 5, 16, 64]:
        for eps in [0.0, 1e-5, 0.3]:
            x = rng.normal(1.5, 2.0, size=m)
            gamma, beta = float(rng.uniform(0.5, 2.0)), float(rng.normal())
            c = rng.normal(size=m)
            y, xhat, s, mu, var = forward(x, gamma, beta, eps)
            g = c * (1.0 - np.tanh(y) ** 2)
            dx = compact(g, xhat, s, gamma)
            dgamma, dbeta = np.sum(g * xhat), np.sum(g)
            h = 1e-6
            fd = np.array([(loss(x + h * e, gamma, beta, eps, c) - loss(x - h * e, gamma, beta, eps, c)) / (2 * h) for e in np.eye(m)])
            assert np.linalg.norm(dx - fd) <= 1e-7 * max(1.0, np.linalg.norm(fd)), (m, eps)
            fdg = (loss(x, gamma + h, beta, eps, c) - loss(x, gamma - h, beta, eps, c)) / (2 * h)
            fdb = (loss(x, gamma, beta + h, eps, c) - loss(x, gamma, beta - h, eps, c)) / (2 * h)
            assert abs(dgamma - fdg) <= 1e-7 * max(1.0, abs(fdg)) and abs(dbeta - fdb) <= 1e-7 * max(1.0, abs(fdb)), (m, eps)
            dx2, dg2, db2 = ioffe(x, g, gamma, mu, var, eps)
            assert np.allclose(dx, dx2, rtol=1e-12, atol=1e-12) and np.isclose(dgamma, dg2) and np.isclose(dbeta, db2), (m, eps)
            assert abs(dx.sum()) <= 1e-12 * max(1.0, np.abs(dx).sum()), (m, eps)
            expected = gamma / s * eps / (var + eps) * np.sum(g * xhat)
            assert abs(np.dot(xhat, dx) - expected) <= 1e-12 * max(1.0, np.abs(dx).sum() * np.abs(xhat).max()), (m, eps)
            if eps == 0.0:
                p = np.eye(m) - np.ones((m, m)) / m - np.outer(xhat, xhat) / m
                assert np.allclose(p @ p, p, atol=1e-12) and np.allclose(p, p.T), m
                assert np.allclose(dx, gamma / s * p @ g, atol=1e-12), m
    print("ok: batch normalization gradients match finite differences and the equations of Ioffe and Szegedy")


if __name__ == "__main__":
    main()
