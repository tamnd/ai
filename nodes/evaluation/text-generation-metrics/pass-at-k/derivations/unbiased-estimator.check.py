# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/pass-at-k/ (ai://AH634X).
# Check of derivations/unbiased-estimator.md: the estimator 1 - C(n - c, k) / C(n, k) has mean 1 - (1 - p)^k when
# c ~ Binomial(n, p); it equals the product of eq-product-form; the plug-in 1 - (1 - c/n)^k is biased low for k >= 2;
# and the special cases of part 4.
import math
from fractions import Fraction

import numpy as np


def ratio(n, c, k):
    return Fraction(math.comb(n - c, k), math.comb(n, k))  # math.comb is 0 when n - c < k


def product(n, c, k):
    if n - c < k:
        return 0.0
    return float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1, dtype=np.float64)))


def main():
    rng = np.random.default_rng(20261115)
    # 1. Part 1, exactly: sum over c of P[c] (1 - C(n - c, k) / C(n, k)) equals 1 - (1 - p)^k, in fractions.
    for _ in range(1000):
        n = int(rng.integers(1, 41))
        k = int(rng.integers(1, n + 1))
        p = Fraction(int(rng.integers(0, 101)), 100)
        mean = sum(math.comb(n, c) * p**c * (1 - p) ** (n - c) * (1 - ratio(n, c, k)) for c in range(n + 1))
        assert mean == 1 - (1 - p) ** k, (n, k, p)
    # 2. Part 2: the product equals the ratio of binomials for every c and k, n <= 60, within 1e-12.
    for n in range(1, 61):
        for k in range(1, n + 1):
            for c in range(n + 1):
                assert abs(product(n, c, k) - float(ratio(n, c, k))) <= 1e-12, (n, c, k)
    # 3. Part 3: the plug-in is strictly below 1 - (1 - p)^k for k >= 2 and 0 < p < 1, and equal at k = 1.
    for n, k, p in [(5, 2, Fraction(1, 3)), (200, 10, Fraction(3, 100)), (40, 40, Fraction(1, 2)), (3, 2, Fraction(99, 100))]:
        plug = sum(math.comb(n, c) * p**c * (1 - p) ** (n - c) * (1 - (1 - Fraction(c, n)) ** k) for c in range(n + 1))
        assert plug < 1 - (1 - p) ** k, (n, k, p)
        plug1 = sum(math.comb(n, c) * p**c * (1 - p) ** (n - c) * Fraction(c, n) for c in range(n + 1))
        assert plug1 == p
    # 4. Part 4: range, c = 0, c = 1, k = n, and monotone in c and in k.
    for n in range(1, 31):
        for k in range(1, n + 1):
            est = [1 - ratio(n, c, k) for c in range(n + 1)]
            assert est[0] == 0 and all(0 < e <= 1 for e in est[1:])
            assert est[1] == Fraction(k, n)
            assert all(a <= b for a, b in zip(est, est[1:]))
            if k < n:
                assert all(1 - ratio(n, c, k) <= 1 - ratio(n, c, k + 1) for c in range(n + 1))
        assert all(1 - ratio(n, c, n) == 1 for c in range(1, n + 1))
    # 5. Monte Carlo: 400,000 draws of c ~ Binomial(200, 0.03); the unbiased mean is within 4 standard errors of
    # 1 - 0.97^10, and the plug-in mean is lower by more than 10 standard errors.
    n, k, p = 200, 10, 0.03
    cs = rng.binomial(n, p, size=400_000)
    unb = np.array([1.0 - product(n, c, k) for c in range(n + 1)])[cs]
    plug = np.array([1.0 - (1.0 - c / n) ** k for c in range(n + 1)])[cs]
    true = 1.0 - (1.0 - p) ** k
    se = unb.std() / math.sqrt(len(cs))
    assert abs(unb.mean() - true) < 4 * se, (unb.mean(), true, se)
    assert plug.mean() < true - 10 * se, (plug.mean(), true, se)
    print(f"ok: unbiased mean {unb.mean():.5f}, plug-in mean {plug.mean():.5f}, pass@10 {true:.5f}, se {se:.1e}")


if __name__ == "__main__":
    main()
