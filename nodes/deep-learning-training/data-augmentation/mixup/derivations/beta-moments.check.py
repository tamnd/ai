# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/mixup/ (ai://8X8XEV).
# Check of derivations/beta-moments.md: for lam ~ Beta(alpha, alpha), E[lam] = 1/2 and Var[lam] = 1 / (4 (2 alpha + 1));
# small alpha puts the mass near 0 and 1 (ERM), large alpha near 1/2, and alpha = 1 is uniform. Monte Carlo and
# numerical integration.
import math

import numpy as np


def var(alpha):
    return 1.0 / (4.0 * (2.0 * alpha + 1.0))


def main():
    rng = np.random.default_rng(20261116)
    n = 1_000_000

    # 1. Mean and variance by Monte Carlo, within five standard errors.
    for alpha in [0.01, 0.1, 0.2, 0.4, 1.0, 2.0, 8.0, 32.0, 1000.0]:
        lam = rng.beta(alpha, alpha, size=n)
        v = var(alpha)
        assert abs(lam.mean() - 0.5) < 5 * math.sqrt(v / n), (alpha, lam.mean())
        # The variance of the sample variance is (mu4 - v^2) / n; mu4 <= v for a variable in [0, 1] around 1/2.
        assert abs(lam.var() - v) < 5 * math.sqrt(v / n), (alpha, lam.var(), v)

    # E[lam^2] = (alpha + 1) / (2 (2 alpha + 1)) by integrating the density on a midpoint grid.
    m = 200_000
    t = (np.arange(m) + 0.5) / m
    for alpha in [1.0, 1.5, 2.0, 3.0, 8.0]:
        dens = np.exp((alpha - 1) * (np.log(t) + np.log1p(-t)) - (2 * math.lgamma(alpha) - math.lgamma(2 * alpha)))
        assert abs(dens.mean() - 1.0) < 1e-6
        assert abs((t**2 * dens).mean() - (alpha + 1) / (2 * (2 * alpha + 1))) < 1e-6
        assert abs(((t - 0.5) ** 2 * dens).mean() - var(alpha)) < 1e-6

    # 2. Small alpha: almost all mass within 0.01 of an end, split evenly. Var tends to 1/4.
    lam = rng.beta(0.01, 0.01, size=n)
    lo, hi = (lam <= 0.01).mean(), (lam >= 0.99).mean()
    assert lo + hi > 0.90 and abs(lo - hi) < 0.01, (lo, hi)
    assert abs(var(1e-9) - 0.25) < 1e-8

    # 3. Large alpha: all mass near 1/2.
    lam = rng.beta(1000.0, 1000.0, size=n)
    assert np.abs(lam - 0.5).max() < 0.07

    # 4. alpha = 1 is uniform: Kolmogorov-Smirnov distance to the identity CDF.
    lam = np.sort(rng.beta(1.0, 1.0, size=n))
    ecdf = np.arange(1, n + 1) / n
    assert np.abs(ecdf - lam).max() < 1.63 / math.sqrt(n) * 1.5
    assert abs(var(1.0) - 1 / 12) < 1e-15 and abs(var(0.2) - 1 / 5.6) < 1e-15
    assert abs(var(0.2) - 0.178571) < 1e-6 and abs(math.sqrt(var(0.2)) - 0.4226) < 1e-4
    print("ok: Beta(alpha, alpha) has mean 1/2 and variance 1/(4(2 alpha + 1)); alpha -> 0 gives {0, 1}, -> inf gives 1/2")


if __name__ == "__main__":
    main()
