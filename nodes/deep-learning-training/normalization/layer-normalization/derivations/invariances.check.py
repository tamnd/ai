# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/layer-normalization/ (ai://RA5R92).
# Check of derivations/invariances.md: the invariances of layer normalization to weight matrix re-scaling and
# re-centering and to single case re-scaling, the two counterexamples, and the gradient with respect to the weights.
import numpy as np


def ln(A, g, b, eps=0.0):
    mu = A.mean(-1, keepdims=True)
    s = np.sqrt(((A - mu) ** 2).mean(-1, keepdims=True) + eps)
    return g * (A - mu) / s + b


def loss(W, X, g, b, T):
    # The examples are the rows of X; a = W x is a row of X @ W.T.
    return 0.5 * float(((ln(X @ W.T, g, b) - T) ** 2).sum())


def grad(W, X, g, b, T):
    A = X @ W.T
    H = A.shape[-1]
    mu = A.mean(-1, keepdims=True)
    s = np.sqrt(((A - mu) ** 2).mean(-1, keepdims=True))
    Ah = (A - mu) / s
    U = g * (ln(A, g, b) - T)
    D = (H * U - U.sum(-1, keepdims=True) - Ah * (Ah * U).sum(-1, keepdims=True)) / (H * s)
    return D.T @ X


def main():
    rng = np.random.default_rng(20261115)
    H, Din, N = 6, 4, 9
    for _ in range(20):
        W, X = rng.normal(size=(H, Din)), rng.normal(size=(N, Din))
        g, b, T = rng.normal(size=H), rng.normal(size=H), rng.normal(size=(N, H))
        delta, gamma = rng.uniform(0.1, 10), rng.normal(size=Din)
        W2 = delta * W + np.outer(np.ones(H), gamma)
        # Item 1 and item 2, with a different scale for each example.
        assert np.allclose(ln(X @ W2.T, g, b), ln(X @ W.T, g, b), atol=1e-12)
        scales = rng.uniform(0.1, 10, size=(N, 1))
        assert np.allclose(ln((scales * X) @ W.T, g, b), ln(X @ W.T, g, b), atol=1e-12)
        # Item 5: the gradient, by finite differences, then its scaling, its zero column sums and its orthogonality.
        G = grad(W, X, g, b, T)
        num = np.zeros_like(W)
        for i in range(H):
            for j in range(Din):
                E = np.zeros_like(W)
                E[i, j] = 1e-6
                num[i, j] = (loss(W + E, X, g, b, T) - loss(W - E, X, g, b, T)) / 2e-6
        assert np.linalg.norm(G - num) <= 1e-6 * np.linalg.norm(G)
        assert np.allclose(grad(W2, X, g, b, T), G / delta, atol=1e-12)
        assert np.allclose(G.sum(0), 0, atol=1e-12)
        assert abs((W * G).sum()) <= 1e-12 * np.linalg.norm(W) * np.linalg.norm(G) * H
        # With eps > 0 the scaling changes the output by sqrt(v + eps) / sqrt(delta^2 v + eps).
        a = X[0] @ W.T
        v = a.var()
        want = delta * (a - a.mean()) / np.sqrt(delta**2 * v + 0.1)
        assert np.allclose(ln(delta * a, 1.0, 0.0, 0.1), want)
        assert not np.allclose(ln(delta * a, 1.0, 0.0, 0.1), ln(a, 1.0, 0.0, 0.1)) or abs(delta - 1) < 1e-3
    # Plain gradient descent keeps the column sums and only grows the Frobenius norm.
    W, X = rng.normal(size=(H, Din)), rng.normal(size=(N, Din))
    g, b, T = rng.normal(size=H), rng.normal(size=H), rng.normal(size=(N, H))
    col, eta = W.sum(0), 0.3
    for _ in range(50):
        G = grad(W, X, g, b, T)
        before = (W**2).sum()
        W = W - eta * G
        assert np.isclose((W**2).sum(), before + eta**2 * (G**2).sum(), rtol=1e-10)
    assert np.allclose(W.sum(0), col, atol=1e-10)
    # Items 3 and 4: the counterexamples.
    one = lambda a: ln(np.asarray(a, float), 1.0, 0.0)  # noqa: E731
    assert np.allclose(one([1, 2, 3]), [-1.2247449, 0.0, 1.2247449])
    Wr = np.eye(3)
    Wr[0] *= 2
    x = np.array([1.0, 2.0, 3.0])
    assert np.allclose(one(Wr @ x), [-0.7071068, -0.7071068, 1.4142136])
    assert np.allclose(one(np.eye(3) @ (x + np.array([1.0, 0.0, 0.0]))), one(Wr @ x))
    assert not np.allclose(one(Wr @ x), one(x))
    print("ok: the matrix and case invariances, the two counterexamples and the weight gradient identities hold")


if __name__ == "__main__":
    main()
