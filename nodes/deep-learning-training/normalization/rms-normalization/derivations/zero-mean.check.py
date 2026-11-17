# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/rms-normalization/ (ai://3FZB17).
# Check of derivations/zero-mean.md: RMSNorm equals layer normalization on an input of mean zero, for every eps;
# for a mean mu != 0, with rho = mu / sigma and eps = 0, RN(a) = g * (LN_1(a) + rho) / sqrt(1 + rho^2), the
# root-mean-square gap between the two (gain 1) is sqrt(2 - 2 / sqrt(1 + rho^2)), and the mean of RN(a) is
# rho / sqrt(1 + rho^2).
import numpy as np


def rn(a, g, eps):
    return g * a / np.sqrt((a ** 2).mean() + eps)


def ln(a, g, eps):
    c = a - a.mean()
    return g * c / np.sqrt((c ** 2).mean() + eps)


def main():
    rng = np.random.default_rng(20261116)
    for n in [2, 5, 64]:
        for eps in [0.0, 1e-5, 0.7]:
            a = rng.normal(size=n) * 3.0
            g = rng.normal(size=n)
            c = a - a.mean()
            assert abs(c.mean()) < 1e-12
            assert np.allclose(rn(c, g, eps), ln(c, g, eps), rtol=1e-12, atol=1e-12)
        a = rng.normal(size=n)
        a = a - a.mean() + 0.0
        sigma = np.sqrt((a ** 2).mean())
        for mu in [-5.0, -0.3, 0.01, 0.5, 1.0, 10.0]:
            b = a + mu
            rho = mu / sigma
            one = np.ones(n)
            assert np.allclose(rn(b, one, 0.0), (ln(b, one, 0.0) + rho) / np.sqrt(1 + rho ** 2), atol=1e-12)
            gap = np.sqrt(((rn(b, one, 0.0) - ln(b, one, 0.0)) ** 2).mean())
            assert np.isclose(gap, np.sqrt(2 - 2 / np.sqrt(1 + rho ** 2)), rtol=1e-9)
            assert gap <= abs(rho) + 1e-12  # the gap is below |rho| and close to it for small rho
            assert np.isclose(rn(b, one, 0.0).mean(), rho / np.sqrt(1 + rho ** 2), rtol=1e-9)
            if abs(rho) < 0.05:
                assert abs(gap - abs(rho)) < abs(rho) ** 2
    # the gap grows with rho and tends to sqrt(2) as |rho| grows
    rhos = np.array([0.0, 0.1, 0.5, 1.0, 3.0, 100.0])
    gaps = np.sqrt(2 - 2 / np.sqrt(1 + rhos ** 2))
    assert np.all(np.diff(gaps) > 0) and gaps[0] == 0.0 and abs(gaps[-1] - np.sqrt(2)) < 1e-2
    # the worked example: a = (4, 0, -2, 4), mu = 1.5, sigma^2 = 6.75, rho^2 = 1/3
    a = np.array([4.0, 0.0, -2.0, 4.0])
    rho = a.mean() / a.std()
    assert np.isclose(rho ** 2, 1 / 3)
    gap = np.sqrt(((rn(a, 1.0, 0.0) - ln(a, 1.0, 0.0)) ** 2).mean())
    assert np.isclose(gap, np.sqrt(2 - np.sqrt(3)))
    assert np.isclose(rn(a, 1.0, 0.0).mean(), 0.5)
    assert np.allclose(ln(a, 1.0, 0.0), np.array([2.5, -1.5, -3.5, 2.5]) / np.sqrt(6.75))
    print("ok: RN = LN at mean zero for every eps, RN = (LN + rho) / sqrt(1 + rho^2), and the gap formula hold")


if __name__ == "__main__":
    main()
