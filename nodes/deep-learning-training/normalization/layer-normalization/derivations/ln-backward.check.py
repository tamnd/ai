# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/layer-normalization/ (ai://NEW-layer-normalization).
# Check of derivations/ln-backward.md: the gradient of layer normalization, its sum and its orthogonality, the
# Jacobian as a scaled projection, and its spectral norm 1/sqrt(v + eps).
import numpy as np


def ln(a, g, b, eps):
    mu = a.mean()
    s = np.sqrt(((a - mu) ** 2).mean() + eps)
    return g * (a - mu) / s + b


def num_grad(f, x, h=1e-6):
    out = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = h
        out[i] = (f(x + e) - f(x - e)) / (2 * h)
    return out


def main():
    rng = np.random.default_rng(20261115)
    for H in [2, 3, 5, 16]:
        for eps in [0.0, 1e-3, 0.7]:
            a = rng.normal(size=H) * 2 + 1.5
            g, b, d = rng.normal(size=H), rng.normal(size=H), rng.normal(size=H)
            mu, v = a.mean(), ((a - a.mean()) ** 2).mean()
            s = np.sqrt(v + eps)
            ah = (a - mu) / s
            u = g * d
            J = (np.eye(H) - np.ones((H, H)) / H - np.outer(ah, ah) / H) / s
            assert np.allclose(J, J.T)
            da = (H * u - u.sum() - ah * (ah @ u)) / (H * s)
            loss = lambda x: float(d @ ln(x, g, b, eps))  # noqa: E731
            assert np.linalg.norm(da - num_grad(loss, a)) <= 1e-6 * (1 + np.linalg.norm(da)), (H, eps)
            assert np.allclose(J @ u, da, atol=1e-12)
            dg = num_grad(lambda q: float(d @ ln(a, q, b, eps)), g)
            assert np.allclose(dg, d * ah, rtol=1e-6, atol=1e-8)
            assert np.allclose(num_grad(lambda q: float(d @ ln(a, g, q, eps)), b), d, rtol=1e-6)
            assert abs(da.sum()) <= 1e-12 * (1 + np.abs(da).max()) * H
            if eps == 0.0:
                assert abs(ah @ da) <= 1e-12 * (1 + np.abs(da).max()) * H
                assert np.isclose(np.linalg.norm(ah), np.sqrt(H), rtol=1e-12)
                P = J * s
                assert np.allclose(P @ P, P, atol=1e-12)
            norm = np.linalg.norm(J, 2)
            if H >= 3:
                assert np.isclose(norm, 1 / s, rtol=1e-10), (H, eps, norm, 1 / s)
            if H >= 3 and eps == 0.0:
                assert np.isclose(norm, np.sqrt(H) / np.linalg.norm(a - mu), rtol=1e-10)
            if eps > 0:
                ev = np.sort(np.linalg.eigvalsh(J))
                want = np.sort([0.0, eps / (v + eps) / s] + [1 / s] * (H - 2))
                assert np.allclose(ev, want, atol=1e-10), (ev, want)
    # The worked example of the page.
    a = np.array([3.0, -4.0, 7.0, -1.0, 5.0])
    g, b = np.array([2.0, 1.0, 2.0, 1.0, 4.0]), np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    assert np.allclose(ln(a, g, b, 0.0), [0.5, -0.5, 2.5, 0.25, 3.0])
    d = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    assert np.allclose(num_grad(lambda x: float(d @ ln(x, g, b, 0.0)), a), np.array([-0.625, 1.75, 0.875, -2.125, 0.125]) / 20)
    print("ok: the gradient, its zero sum and orthogonality, the projection and the norm 1/sqrt(v + eps) hold")


if __name__ == "__main__":
    main()
