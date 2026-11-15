# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/layer-normalization/ (ai://NEW-layer-normalization).
# Check of derivations/rmsnorm.md: RMSNorm of the centered vector is layer normalization, its invariances and the
# counterexample to re-centering, its gradient and Jacobian norm, the R of Eq. 9 of Zhang and Sennrich (2019), and the
# sphere of radius sqrt(n).
import numpy as np


def rn(a, g, eps):
    return g * a / np.sqrt((a**2).mean() + eps)


def ln(a, g, eps):
    return g * (a - a.mean()) / np.sqrt(a.var() + eps)


def num_grad(f, x, h=1e-6):
    out = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = h
        out[i] = (f(x + e) - f(x - e)) / (2 * h)
    return out


def main():
    rng = np.random.default_rng(20261115)
    for n in [2, 3, 8]:
        for eps in [0.0, 1e-3, 0.5]:
            a = rng.normal(size=n) + 0.7
            g, d = rng.normal(size=n), rng.normal(size=n)
            assert np.allclose(ln(a, g, eps), rn(a - a.mean(), g, eps), atol=1e-12)
            c = a - a.mean()
            assert np.allclose(rn(c, g, eps), ln(c, g, eps), atol=1e-12)
            r = np.sqrt((a**2).mean() + eps)
            ah, u = a / r, g * d
            J = (np.eye(n) - np.outer(ah, ah) / n) / r
            da = (u - ah * (ah @ u) / n) / r
            assert np.allclose(J @ u, da, atol=1e-12)
            num = num_grad(lambda x: float(d @ rn(x, g, eps)), a)
            assert np.linalg.norm(da - num) <= 1e-6 * (1 + np.linalg.norm(da))
            assert np.allclose(num_grad(lambda q: float(d @ rn(a, q, eps)), g), d * ah, rtol=1e-6, atol=1e-8)
            assert np.isclose(np.linalg.norm(J, 2), 1 / r, rtol=1e-10)
            m = (a**2).mean()
            ev = np.sort(np.linalg.eigvalsh(J))
            assert np.allclose(ev, np.sort([eps / (m + eps) / r] + [1 / r] * (n - 1)), atol=1e-10)
            if eps == 0.0:
                assert np.isclose(np.linalg.norm(ah), np.sqrt(n), rtol=1e-12)
                assert np.allclose(J @ ah, 0, atol=1e-12)
                for delta in [0.01, 3.0, 250.0]:
                    assert np.allclose(rn(delta * a, g, 0.0), rn(a, g, 0.0), atol=1e-12)
            else:
                assert not np.allclose(rn(3.0 * a, g, eps), rn(a, g, eps))
            if n == 8:
                assert abs(da.sum()) > 1e-3
    # Eq. 9 and Eq. 10: R for a = W x, and R / delta for delta W.
    W, x = rng.normal(size=(5, 4)), rng.normal(size=4)
    for delta in [0.5, 4.0]:
        R = []
        for M in [W, delta * W]:
            a = M @ x
            rms = np.sqrt((a**2).mean())
            R.append((np.eye(5) - np.outer(a, a) / (5 * rms**2)) / rms)
            ah = a / rms
            assert np.allclose(R[-1], (np.eye(5) - np.outer(ah, ah) / 5) / rms)
            assert np.allclose(rn(M @ (2.5 * x), 1.0, 0.0), rn(W @ x, 1.0, 0.0))
        assert np.allclose(R[1], R[0] / delta)
    # Re-centering: (1, -1) and (2, 0).
    one = np.ones(2)
    assert np.allclose(rn(np.array([1.0, -1.0]), one, 0.0), [1.0, -1.0])
    assert np.allclose(rn(np.array([2.0, 0.0]), one, 0.0), [np.sqrt(2.0), 0.0])
    assert np.allclose(ln(np.array([2.0, 0.0]), one, 0.0), ln(np.array([1.0, -1.0]), one, 0.0))
    # The worked example: RMSNorm of the centered vector is the layer norm output, and RMS(a) = sqrt(20).
    a = np.array([3.0, -4.0, 7.0, -1.0, 5.0])
    assert np.allclose(rn(a - 2.0, np.ones(5), 0.0), [0.25, -1.5, 1.25, -0.75, 0.75])
    assert np.isclose(np.sqrt((a**2).mean()), np.sqrt(20.0))
    print("ok: LN(a) = RN(a - mean), the invariances, the counterexample, the gradient, Eq. 9 and the sphere hold")


if __name__ == "__main__":
    main()
