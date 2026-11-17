# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/early-stopping/ (ai://VR3ZRP).
# Check of derivations/gradient-flow.md: gradient flow from zero on least squares is the filter 1 - exp(-s tau); k
# steps of gradient descent with step eta, eta s_max <= 1, are within (eta / 2) |X^T y / n| of the flow at tau = k eta
# for every k; and the flow filter lies between the ridge filter at lam = 1 / tau and 1.2985 times it.
import numpy as np


def main():
    rng = np.random.default_rng(20261116)
    n, p = 50, 6
    X = rng.normal(size=(n, p)) * np.linspace(0.2, 2.0, p)
    y = X @ rng.normal(size=p) + rng.normal(size=n)
    s, V = np.linalg.eigh(X.T @ X / n)
    b = X.T @ y / n
    c = V.T @ b

    def flow(tau):
        return V @ (-np.expm1(-s * tau) * c / s)

    # the flow solves dw/dtau = X^T (y - X w) / n with w(0) = 0
    assert np.allclose(flow(0.0), 0.0)
    for tau in (0.3, 2.0, 10.0):
        h = 1e-6
        deriv = (flow(tau + h) - flow(tau - h)) / (2 * h)
        assert np.allclose(deriv, X.T @ (y - X @ flow(tau)) / n, rtol=1e-6, atol=1e-8)
    print("ok: w(tau) = sum (1 - exp(-s tau)) (c / s) v solves the flow")

    # gradient descent tracks the flow within (eta / 2) |b| uniformly in k, and the error is first order in eta
    worst = {}
    for eta in (1.0 / s.max(), 0.5 / s.max(), 0.25 / s.max(), 0.125 / s.max(), 0.0625 / s.max()):
        w = np.zeros(p)
        err = 0.0
        for k in range(1, int(round(40.0 / eta)) + 1):
            w = w + eta * X.T @ (y - X @ w) / n
            err = max(err, np.linalg.norm(w - flow(k * eta)))
        assert err <= eta / 2 * np.linalg.norm(b), "the uniform bound"
        worst[eta] = err
        print(f"ok: eta s_max = {eta * s.max():.3f}: max_k |w_k - w(k eta)| = {err:.3e} <= {eta / 2 * np.linalg.norm(b):.3e}")
    e = list(worst.values())
    for a, b2 in zip(e[1:], e[2:]):
        assert 1.6 < a / b2 < 2.4, "halving eta halves the error"
    print(f"ok: error ratios on halving eta {[round(float(a / b2), 3) for a, b2 in zip(e, e[1:])]}")

    # the scalar bound |(1 - x)^k - exp(-k x)| <= k (x^2 / 2) exp(-(k - 1) x) <= x e^x / (2 e) on 0 <= x <= 1
    x = np.linspace(0.0, 1.0, 1001)
    for k in (1, 2, 5, 20, 100, 1000):
        gap = np.abs((1 - x) ** k - np.exp(-k * x))
        assert np.all(gap <= k * x**2 / 2 * np.exp(-(k - 1) * x) + 1e-15)
        assert np.all(gap <= x * np.exp(x - 1) / 2 + 1e-15)
    print("ok: the scalar filter gap is below x e^(x - 1) / 2 for k up to 1000")

    # ridge(1 / tau) <= flow(tau) <= 1.2985 ridge(1 / tau): with u = s tau the ratio is (1 - e^-u)(1 + u) / u
    u = np.linspace(1e-6, 50.0, 2_000_001)
    ratio = -np.expm1(-u) * (1 + u) / u
    assert np.all(ratio >= 1 - 1e-12) and ratio.max() <= 1.2985
    print(f"ok: max flow / ridge = {ratio.max():.5f} at s tau = {u[np.argmax(ratio)]:.4f}")


if __name__ == "__main__":
    main()
