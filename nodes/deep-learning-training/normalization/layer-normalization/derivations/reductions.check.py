# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/layer-normalization/ (ai://NEW-layer-normalization).
# Check of derivations/reductions.md: group normalization with one group is layer normalization over (C, H, W), and
# layer normalization of one example does not depend on the other examples of the batch.
import numpy as np


def ln(X, g, b, eps):
    axes = tuple(range(1, X.ndim))
    mu = X.mean(axes, keepdims=True)
    var = ((X - mu) ** 2).mean(axes, keepdims=True)
    return g * (X - mu) / np.sqrt(var + eps) + b


def gn(X, G, gamma, beta, eps):
    N, C, H, W = X.shape
    Y = X.reshape(N, G, C // G, H, W)
    mu = Y.mean((2, 3, 4), keepdims=True)
    var = ((Y - mu) ** 2).mean((2, 3, 4), keepdims=True)
    Y = ((Y - mu) / np.sqrt(var + eps)).reshape(N, C, H, W)
    return gamma[None, :, None, None] * Y + beta[None, :, None, None]


def main():
    rng = np.random.default_rng(20261115)
    N, C, H, W = 3, 4, 2, 5
    for eps in [0.0, 1e-5, 0.3]:
        X = rng.normal(size=(N, C, H, W)) * 3 + 1
        gamma, beta = rng.normal(size=C), rng.normal(size=C)
        g = np.broadcast_to(gamma[:, None, None], (C, H, W))
        b = np.broadcast_to(beta[:, None, None], (C, H, W))
        assert np.allclose(gn(X, 1, gamma, beta, eps), ln(X, g, b, eps), atol=1e-12)
        assert not np.allclose(gn(X, 2, gamma, beta, eps), ln(X, g, b, eps))
    # The Jacobian over a batch of 4 examples with 3 features is block diagonal.
    X = rng.normal(size=(4, 3))
    g, b = rng.normal(size=3), rng.normal(size=3)
    f = lambda Z: ln(Z, g, b, 1e-5).ravel()  # noqa: E731
    J = np.zeros((12, 12))
    for k in range(12):
        E = np.zeros(12)
        E[k] = 1e-6
        J[:, k] = (f(X + E.reshape(4, 3)) - f(X - E.reshape(4, 3))) / 2e-6
    for i in range(4):
        for j in range(4):
            block = J[3 * i : 3 * i + 3, 3 * j : 3 * j + 3]
            if i == j:
                assert np.abs(block).max() > 1e-2
            else:
                assert np.abs(block).max() < 1e-9
    # One example alone, or among other examples, gives the same output.
    Y = ln(X, g, b, 1e-5)
    for i in range(4):
        assert np.allclose(ln(X[i : i + 1], g, b, 1e-5)[0], Y[i], atol=1e-14)
        other = np.concatenate([rng.normal(size=(7, 3)) * 50, X[i : i + 1]])
        assert np.allclose(ln(other, g, b, 1e-5)[-1], Y[i], atol=1e-14)
    print("ok: group normalization with G = 1 is layer normalization, and the batch Jacobian is block diagonal")


if __name__ == "__main__":
    main()
