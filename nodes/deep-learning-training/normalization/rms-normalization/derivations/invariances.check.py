# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/rms-normalization/ (ai://NEW-rms-normalization).
# Check of derivations/invariances.md: RMSNorm and pRMSNorm do not change when the input, the weight matrix or
# both are scaled by delta > 0 (eps = 0); a positive eps breaks this; scaling one row of W does change the output;
# adding a constant does change the output; and layer normalization equals RMSNorm of the centred input.
import numpy as np


def rn(a, g, eps, k=None):
    k = a.size if k is None else k
    return g * a / np.sqrt((a[:k] ** 2).mean() + eps)


def ln(a, g, eps):
    return g * (a - a.mean()) / np.sqrt(((a - a.mean()) ** 2).mean() + eps)


def main():
    rng = np.random.default_rng(20261116)
    for n in [2, 3, 7, 16]:
        W, x, g = rng.normal(size=(n, 5)), rng.normal(size=5), rng.normal(size=n)
        a = W @ x
        for delta in [1e-3, 0.37, 2.0, 1e3]:
            for k in sorted({1, max(1, n // 2), n}):
                base = rn(a, g, 0.0, k)
                assert np.allclose(rn((delta * W) @ x, g, 0.0, k), base, rtol=1e-12, atol=1e-12)
                assert np.allclose(rn(W @ (delta * x), g, 0.0, k), base, rtol=1e-12, atol=1e-12)
                assert np.allclose(rn(delta * a, g, 0.0, k), base, rtol=1e-12, atol=1e-12)
            if delta != 1.0:
                # with eps > 0 the scale leaks through: r(delta a)^2 = delta^2 m + eps
                assert not np.allclose(rn(delta * a, g, 0.5), rn(a, g, 0.5))
        for eps in [0.0, 1e-6, 0.3]:
            # LN(a) = RN(a - mean), for every eps and every gain
            assert np.allclose(ln(a, g, eps), rn(a - a.mean(), g, eps), rtol=1e-12, atol=1e-12)
            # and LN is unchanged by a shift, which RN is not
            assert np.allclose(ln(a + 4.0, g, eps), ln(a, g, eps), atol=1e-10)
        if n >= 2:
            W2 = W.copy()
            W2[0] *= 3.0  # scale one weight vector only
            assert not np.allclose(rn(W2 @ x, np.ones(n), 0.0)[1:], rn(a, np.ones(n), 0.0)[1:])
    one = np.ones(2)
    assert np.allclose(rn(np.array([1.0, -1.0]), one, 0.0), [1.0, -1.0])
    assert np.allclose(rn(np.array([2.0, 0.0]), one, 0.0), [np.sqrt(2.0), 0.0])
    assert np.allclose(ln(np.array([2.0, 0.0]), one, 0.0), [1.0, -1.0])
    # the worked example: a = (4, 0, -2, 4) has RMS 3; 2a gives the same output
    a = np.array([4.0, 0.0, -2.0, 4.0])
    assert np.isclose(np.sqrt((a ** 2).mean()), 3.0)
    assert np.allclose(rn(a, np.ones(4), 0.0), [4 / 3, 0.0, -2 / 3, 4 / 3])
    assert np.allclose(rn(2 * a, np.ones(4), 0.0), rn(a, np.ones(4), 0.0))
    assert np.allclose(rn(a, np.ones(4), 0.0, k=2), a / np.sqrt(8.0))
    print("ok: scaling invariance of RMSNorm and pRMSNorm, its loss with eps > 0, the row and shift counterexamples, LN = RN(a - mean)")


if __name__ == "__main__":
    main()
