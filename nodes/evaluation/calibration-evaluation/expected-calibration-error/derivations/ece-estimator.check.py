# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/expected-calibration-error/ (ai://01R1CF).
# Check of derivations/ece-estimator.md: binning lowers the population calibration error; refining the bins never
# lowers the sample ECE; given the confidences, the sample ECE is biased up, and is positive in expectation for a
# calibrated model; the expected squared gap of a bin is the squared gap plus a variance; de Moivre's formula for the
# mean absolute deviation of a binomial; and the exact expected ECE of a calibrated model quoted on the page.
import itertools
import math
from fractions import Fraction as F

import numpy as np


def bin_of(c, m):
    return max(0, math.ceil(c * m) - 1)  # (b/m, (b+1)/m], and 0 in bin 0; c is a Fraction here


def ece_hat(c, a, m):
    gaps = {}
    for ci, ai in zip(c, a):
        b = bin_of(ci, m)
        gaps[b] = gaps.get(b, 0) + (ai - ci)
    return sum(abs(g) for g in gaps.values()) / len(c)


def pb(ps):
    """The distribution of a sum of independent Bernoulli(p_i), as floats."""
    d = np.ones(1)
    for p in ps:
        nd = np.zeros(len(d) + 1)
        nd[:-1] += d * (1 - p)
        nd[1:] += d * p
        d = nd
    return d


def expected_ece_calibrated(c, m):
    """E[sample ECE] when A_i ~ Bernoulli(c_i) independently: sum_m E|S_m - sum c_i| / n."""
    idx = np.clip(np.ceil(c * m).astype(int) - 1, 0, m - 1)
    total = 0.0
    for b in range(m):
        cs = c[idx == b]
        if len(cs):
            d = pb(cs)
            total += float(np.sum(d * np.abs(np.arange(len(d)) - cs.sum())))
    return total / len(c)


def de_moivre(n, p):
    v = math.floor(n * p) + 1
    return 2 * v * math.comb(n, v) * p**v * (1 - p) ** (n - v + 1)


def main():
    rng = np.random.default_rng(20261116)
    # 1. Population: sum_m P(bin m) |E[r(C) - C | bin m]| <= E|r(C) - C|, for C on a finite set.
    for _ in range(500):
        k, m = int(rng.integers(1, 10)), int(rng.integers(1, 12))
        c = [F(int(v), 100) for v in rng.integers(1, 101, size=k)]
        r = [F(int(v), 100) for v in rng.integers(0, 101, size=k)]
        w = [F(int(v)) for v in rng.integers(1, 10, size=k)]
        w = [x / sum(w) for x in w]
        true = sum(wi * abs(ri - ci) for wi, ri, ci in zip(w, r, c))
        binned = F(0)
        for b in range(m):
            binned += abs(sum(wi * (ri - ci) for wi, ri, ci in zip(w, r, c) if bin_of(ci, m) == b))
        assert binned <= true
    # the example of the page: confidence 5/8 right 7/8 of the time and 7/8 right 5/8 of the time, half each
    c, r = [F(5, 8), F(7, 8)], [F(7, 8), F(5, 8)]
    assert sum(F(1, 2) * abs(ri - ci) for ri, ci in zip(r, c)) == F(1, 4)
    assert abs(sum(F(1, 2) * (ri - ci) for ri, ci in zip(r, c))) == 0
    # 2. Sample: refining the bins never lowers the sample ECE.
    for _ in range(500):
        n, m, f = int(rng.integers(1, 30)), int(rng.integers(1, 12)), int(rng.integers(2, 5))
        c = [F(int(v), 1000) for v in rng.integers(1, 1001, size=n)]
        a = [int(v) for v in rng.integers(0, 2, size=n)]
        assert ece_hat(c, a, m) <= ece_hat(c, a, m * f)
    # 3. Given the confidences, E[sample ECE] >= sum_m |sum_{i in B_m} (r_i - c_i)| / n, exactly, by enumeration.
    for _ in range(200):
        n, m = int(rng.integers(1, 8)), int(rng.integers(1, 6))
        c = [F(int(v), 20) for v in rng.integers(1, 21, size=n)]
        r = [F(int(v), 10) for v in rng.integers(0, 11, size=n)]
        mean = F(0)
        for a in itertools.product((0, 1), repeat=n):
            pr = math.prod(ri if ai else 1 - ri for ai, ri in zip(a, r))
            mean += pr * ece_hat(c, a, m)
        target = ece_hat(c, r, m)  # the same sum with r_i in place of the outcomes
        assert mean >= target
        if r == c and any(0 < ri < 1 for ri in r):
            assert mean > 0  # a calibrated model still gets a positive expected ECE
    # 4. Squared gap of one bin: E[(acc - conf)^2] = (mean r - mean c)^2 + sum r(1 - r) / k^2, exactly.
    for _ in range(200):
        k = int(rng.integers(1, 7))
        c = [F(int(v), 20) for v in rng.integers(1, 21, size=k)]
        r = [F(int(v), 10) for v in rng.integers(0, 11, size=k)]
        mean = F(0)
        for a in itertools.product((0, 1), repeat=k):
            pr = math.prod(ri if ai else 1 - ri for ai, ri in zip(a, r))
            mean += pr * (F(sum(a), k) - sum(c) / k) ** 2
        assert mean == (sum(r) / k - sum(c) / k) ** 2 + sum(ri * (1 - ri) for ri in r) / k**2
    # 5. de Moivre: E|X - np| = 2 v C(n, v) p^v (1 - p)^(n - v + 1), v = floor(np) + 1, for X ~ Binomial(n, p).
    for _ in range(300):
        n, p = int(rng.integers(1, 41)), F(int(rng.integers(0, 101)), 100)
        direct = sum(math.comb(n, j) * p**j * (1 - p) ** (n - j) * abs(j - n * p) for j in range(n + 1))
        assert direct == de_moivre(n, p)
    for n, want in ((10, 0.0966), (100, 0.0318), (1000, 0.0101)):
        assert round(float(de_moivre(n, F(4, 5)) / n), 4) == want
    # 6. The table of the page: a calibrated model with n confidences spread evenly over [0.5, 1].
    table = {(100, 15): 0.090, (100, 100): 0.229, (1000, 15): 0.028, (1000, 100): 0.070,
             (10000, 15): 0.0089, (10000, 100): 0.022}  # fmt: skip
    for (n, m), want in table.items():
        c = 0.5 + 0.5 * (np.arange(n) + 0.5) / n
        got = expected_ece_calibrated(c, m)
        assert round(got, 3 if want >= 0.01 else 4) == want, (n, m, got)
        idx = np.clip(np.ceil(c * m).astype(int) - 1, 0, m - 1)
        approx = sum(math.sqrt(2 / math.pi * float(np.sum(c[idx == b] * (1 - c[idx == b])))) for b in set(idx)) / n
        assert abs(approx / got - 1) < 0.04
    # the same, by simulation with the outcomes drawn at random
    c = 0.5 + 0.5 * (np.arange(1000) + 0.5) / 1000
    sims = [float(ece_hat_float(c, rng.random(1000) < c, 15)) for _ in range(4000)]
    assert abs(np.mean(sims) - 0.028) < 0.001
    print("ok: binning bias down, sampling bias up, de Moivre, and the expected ECE of a calibrated model")


def ece_hat_float(c, a, m):
    idx = np.clip(np.ceil(c * m).astype(int) - 1, 0, m - 1)
    return np.sum(np.abs(np.bincount(idx, weights=a - c, minlength=m))) / len(c)


if __name__ == "__main__":
    main()
