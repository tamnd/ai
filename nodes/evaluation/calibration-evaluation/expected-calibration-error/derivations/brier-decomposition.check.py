# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/expected-calibration-error/ (ai://NEW-expected-calibration-error).
# Check of derivations/brier-decomposition.md, in exact rational arithmetic: the population identity
# E[(C - A)^2] = E[(C - r(C))^2] + E[r(C)(1 - r(C))]; the exact binned identity of the sample top-label Brier score;
# and ECE <= RMS calibration error, both with the same bins.
from fractions import Fraction as F
import math

import numpy as np


def binned_terms(c, a, m):
    """Per bin: weight, gap, acc(1 - acc), within-bin variance of c, within-bin covariance of c and a."""
    n = len(c)
    out = []
    for b in range(m):
        idx = [i for i in range(n) if (F(b, m) < c[i] <= F(b + 1, m)) or (b == 0 and c[i] == 0)]
        if not idx:
            continue
        k = len(idx)
        cb = sum(c[i] for i in idx) / k
        ab = F(sum(a[i] for i in idx), k)
        var = sum((c[i] - cb) ** 2 for i in idx) / k
        cov = sum((c[i] - cb) * (a[i] - ab) for i in idx) / k
        out.append((F(k, n), cb - ab, ab * (1 - ab), var, cov))
    return out


def main():
    rng = np.random.default_rng(20261116)
    # 1. Population identity, with C on a finite set of values and r(c) = P(A = 1 | C = c).
    for _ in range(300):
        vals = [F(int(v), 100) for v in rng.integers(0, 101, size=int(rng.integers(1, 8)))]
        w = [F(int(v), 1) for v in rng.integers(1, 20, size=len(vals))]
        w = [x / sum(w) for x in w]
        r = [F(int(v), 50) for v in rng.integers(0, 51, size=len(vals))]
        brier = sum(wi * (ri * (c - 1) ** 2 + (1 - ri) * c**2) for wi, c, ri in zip(w, vals, r))
        assert brier == sum(wi * (c - ri) ** 2 for wi, c, ri in zip(w, vals, r)) + sum(
            wi * ri * (1 - ri) for wi, ri in zip(w, r)
        )
    # 2. The exact binned identity of the sample Brier score, and ECE <= RMS.
    for _ in range(300):
        n, m = int(rng.integers(1, 40)), int(rng.integers(1, 16))
        c = [F(int(v), 1000) for v in rng.integers(0, 1001, size=n)]
        a = [int(v) for v in rng.integers(0, 2, size=n)]
        brier = sum((ci - ai) ** 2 for ci, ai in zip(c, a)) / n
        terms = binned_terms(c, a, m)
        assert sum(t[0] for t in terms) == 1
        assert brier == sum(wt * (g * g + s + v - 2 * cv) for wt, g, s, v, cv in terms)
        ece = sum(wt * abs(g) for wt, g, *_ in terms)
        ms = sum(wt * g * g for wt, g, *_ in terms)
        assert ece * ece <= ms
        assert math.sqrt(ms) >= float(ece) - 1e-15
        # with one confidence value per bin the within-bin terms vanish: Brier = RMS^2 + sum w acc(1 - acc)
        cq = [F(math.ceil(ci * m), m) if ci > 0 else F(1, m) for ci in c]
        tq = binned_terms(cq, a, m)
        assert all(v == 0 and cv == 0 for *_, v, cv in tq)
        bq = sum((ci - ai) ** 2 for ci, ai in zip(cq, a)) / n
        assert bq == sum(wt * g * g for wt, g, *_ in tq) + sum(wt * s for wt, _, s, *_ in tq)
    print("ok: Brier = calibration + refinement in the population and, with the within-bin terms, in every sample")


if __name__ == "__main__":
    main()
