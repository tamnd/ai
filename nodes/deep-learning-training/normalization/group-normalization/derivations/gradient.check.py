# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/group-normalization/ (ai://QFEJMP).
# Check of derivations/gradient.md: the input gradient of group normalization, (u - mean(u) - xhat mean(u xhat)) / s
# per group, against central finite differences; its sum over a group is 0; its inner product with xhat leaks
# eps / (var + eps); and the gradients of the per-channel gamma and beta.
import numpy as np


def layer(x, g, gamma, beta, eps):
    n, c = x.shape[:2]
    xg = x.reshape(n, g, -1)
    h = ((xg - xg.mean(-1, keepdims=True)) / np.sqrt(xg.var(-1, keepdims=True) + eps)).reshape(x.shape)
    shape = (1, c) + (1,) * (x.ndim - 2)
    return h * gamma.reshape(shape) + beta.reshape(shape), h


def grad_input(x, dy, g, gamma, eps):
    n, c = x.shape[:2]
    shape = (1, c) + (1,) * (x.ndim - 2)
    u = (dy * gamma.reshape(shape)).reshape(n, g, -1)
    xg = x.reshape(n, g, -1)
    s = np.sqrt(xg.var(-1, keepdims=True) + eps)
    h = (xg - xg.mean(-1, keepdims=True)) / s
    return ((u - u.mean(-1, keepdims=True) - h * (u * h).mean(-1, keepdims=True)) / s).reshape(x.shape)


def fd(f, a, h=1e-6):
    out = np.zeros_like(a)
    for i in np.ndindex(*a.shape):
        e = np.zeros_like(a)
        e[i] = h
        out[i] = (f(a + e) - f(a - e)) / (2 * h)
    return out


def main():
    rng = np.random.default_rng(20261116)
    cases = 0
    for shape in [(2, 4, 3), (2, 4, 2, 2), (1, 6, 5)]:
        c = shape[1]
        for g in [d for d in range(1, c + 1) if c % d == 0]:
            for eps in [0.0, 1e-4, 0.5]:
                x, dy = rng.normal(size=shape) + 0.3, rng.normal(size=shape)
                gamma, beta = rng.normal(size=c), rng.normal(size=c)
                dx = grad_input(x, dy, g, gamma, eps)
                num = fd(lambda a: float(np.sum(dy * layer(a, g, gamma, beta, eps)[0])), x)
                assert np.allclose(dx, num, rtol=1e-6, atol=1e-8)
                h = layer(x, g, gamma, beta, eps)[1]
                n = shape[0]
                dxg, hg = dx.reshape(n, g, -1), h.reshape(n, g, -1)
                assert np.allclose(dxg.sum(-1), 0.0, atol=1e-10)
                u = (dy * gamma.reshape((1, c) + (1,) * (len(shape) - 2))).reshape(n, g, -1)
                xg = x.reshape(n, g, -1)
                var = xg.var(-1)
                s = np.sqrt(var + eps)
                leak = (u * hg).mean(-1) / s * eps / (var + eps)
                assert np.allclose((hg * dxg).mean(-1), leak, rtol=1e-8, atol=1e-11)
                other = (0,) + tuple(range(2, len(shape)))
                dgamma = fd(lambda q: float(np.sum(dy * layer(x, g, q, beta, eps)[0])), gamma)
                dbeta = fd(lambda q: float(np.sum(dy * layer(x, g, gamma, q, eps)[0])), beta)
                assert np.allclose(dgamma, np.sum(dy * h, axis=other), rtol=1e-6, atol=1e-8)
                assert np.allclose(dbeta, np.sum(dy, axis=other), rtol=1e-6, atol=1e-8)
                cases += 1
    # the worked example: group 2 is (0, 6, 6, 8), dy = 1 on its first entry, gamma = 1, eps = 0
    x = np.array([[[0.0, 0.0], [4.0, 4.0], [0.0, 6.0], [6.0, 8.0]]])
    dy = np.zeros_like(x)
    dy[0, 2, 0] = 1.0
    dx = grad_input(x, dy, 2, np.ones(4), 0.0)
    assert np.allclose(dx[0, 2:].ravel() * 54, [1.0, -2.0, -2.0, 3.0]) and np.allclose(dx[0, :2], 0.0)
    print(f"ok: the group gradient matches finite differences in {cases} cases, sums to 0, and leaks eps / (var + eps)")


if __name__ == "__main__":
    main()
