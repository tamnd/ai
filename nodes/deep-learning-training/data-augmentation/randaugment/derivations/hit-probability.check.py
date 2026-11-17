# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/randaugment/ (ai://NEW-randaugment).
# Check of derivations/hit-probability.md: with N uniform draws with replacement from K operations, a given one is drawn
# at least once with probability 1 - (1 - 1/K) ** N; exact enumeration and Monte Carlo.
import itertools
import math
from fractions import Fraction

import numpy as np


def p_hit(k, n):
    return 1 - (1 - Fraction(1, k)) ** n


def main():
    for k in range(1, 7):
        for n in range(0, 5):
            lists = list(itertools.product(range(k), repeat=n))
            hits = Fraction(sum(0 in lst for lst in lists), len(lists))
            assert hits == p_hit(k, n), (k, n)
            count = Fraction(sum(lst.count(0) for lst in lists), len(lists))
            assert count == Fraction(n, k)
            distinct = Fraction(sum(len(set(lst)) for lst in lists), len(lists))
            assert distinct == k * p_hit(k, n)
            if n <= k:
                alldiff = Fraction(sum(len(set(lst)) == n for lst in lists), len(lists))
                assert alldiff == Fraction(math.perm(k, n), k**n)

    expected = {1: Fraction(1, 14), 2: Fraction(27, 196), 3: Fraction(547, 2744)}
    for n, p in expected.items():
        lists = list(itertools.product(range(14), repeat=n))
        assert Fraction(sum(0 in lst for lst in lists), len(lists)) == p == p_hit(14, n)
    assert 14 * p_hit(14, 2) == Fraction(27, 14)
    assert 1 - Fraction(math.perm(14, 2), 14**2) == Fraction(1, 14)

    rng = np.random.default_rng(20261116)
    trials = 200000
    for n, p in expected.items():
        draws = rng.choice(14, size=(trials, n), replace=True)
        share = float((draws == 0).any(axis=1).mean())
        se = math.sqrt(float(p) * (1 - float(p)) / trials)
        assert abs(share - float(p)) < 4 * se, (n, share, float(p))
    print("ok: p_hit(14, N) = 1/14, 27/196, 547/2744 by enumeration and Monte Carlo; distinct ops at N = 2 is 27/14")


if __name__ == "__main__":
    main()
