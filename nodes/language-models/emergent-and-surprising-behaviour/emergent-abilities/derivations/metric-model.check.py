# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/emergent-abilities/ (ai://NEW-emergent-abilities).
# Check of derivations/metric-model.md: with per-token loss (c/N)^alpha, exact-match accuracy exp(-L l) rises smoothly
# in ln N, passes 1/e at c L^(1/alpha), has maximum slope alpha/e and a 10%-90% width ln(ln 10/ln(10/9))/alpha for every
# L, ln(-ln A) is linear in ln N, the expected token errors lie in [L l e^-l, L l], and a test set of n questions
# scores zero with probability (1 - A)^n >= 1 - nA.
import math

import numpy as np


def loss(u, c, alpha):
    return (c / np.exp(u)) ** alpha


def acc(u, c, alpha, length):
    return np.exp(-length * loss(u, c, alpha))


def level(a, c, alpha, length, lo, hi):
    # bisection for the u at which acc = a; acc is increasing in u
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if acc(mid, c, alpha, length) < a:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


WIDTH = math.log(math.log(10.0) / math.log(10.0 / 9.0))
rng = np.random.default_rng(7)
for _ in range(300):
    c = float(np.exp(rng.uniform(math.log(1e6), math.log(1e10))))
    alpha = float(rng.uniform(0.05, 1.5))
    length = int(rng.integers(1, 40))
    ustar = math.log(c) + math.log(length) / alpha
    u = np.linspace(ustar - 30.0 / alpha, ustar + 30.0 / alpha, 20001)
    a = acc(u, c, alpha, length)
    # item 1: strictly increasing wherever it is not already at 0 or 1 in floating point
    inner = (a > 1e-300) & (a < 1.0 - 1e-15)
    assert np.all(np.diff(a[inner]) > 0)
    # item 2: 1/e at N* = c L^(1/alpha)
    assert abs(acc(ustar, c, alpha, length) - math.exp(-1.0)) < 1e-12
    # item 3: maximum slope alpha/e at N*, and the width of the 10%-90% rise
    h = 1e-5
    slope = (acc(u + h, c, alpha, length) - acc(u - h, c, alpha, length)) / (2 * h)
    k = int(np.argmax(slope))
    assert abs(slope[k] - alpha / math.e) < 1e-6 * max(1.0, alpha)
    assert abs(u[k] - ustar) <= (u[1] - u[0]) * 1.01
    lo, hi = u[0], u[-1]
    w = level(0.9, c, alpha, length, lo, hi) - level(0.1, c, alpha, length, lo, hi)
    assert abs(w - WIDTH / alpha) < 1e-8 * max(1.0, 1 / alpha)
    # item 4: ln(-ln A) is linear in u with slope -alpha, and the bounds on the expected errors
    v = np.log(length * loss(u, c, alpha))
    fit = np.polyfit(u, v, 1)
    assert abs(fit[0] + alpha) < 1e-9
    assert abs(fit[1] - (math.log(length) + alpha * math.log(c))) < 1e-6
    ell = loss(u, c, alpha)
    errors = -length * np.expm1(-ell)
    assert np.all(errors <= length * ell * (1 + 1e-12))
    assert np.all(errors >= length * ell * np.exp(-ell) * (1 - 1e-12))
    big = ell < 1e-6
    assert np.all(np.abs(errors[big] / (length * ell[big]) - 1.0) < 1e-6)

# item 5: the chance that a test set of n questions scores zero
for n in (10, 100, 1000):
    for p in (1e-4, 1e-3, 1.0 / (10 * n), 0.02):
        exact = (1.0 - p) ** n
        assert exact >= 1.0 - n * p - 1e-15
        sims = rng.binomial(n, p, size=200000)
        assert abs(float(np.mean(sims == 0)) - exact) < 0.005
    assert (1.0 - 1.0 / (10 * n)) ** n >= 0.9

# the picture: the rise moves right with L and keeps its width
alpha, c = 0.5, 1e8
for length in (1, 4, 16):
    lo, hi = math.log(c) - 80, math.log(c) + 80
    u10 = level(0.1, c, alpha, length, lo, hi)
    u90 = level(0.9, c, alpha, length, lo, hi)
    print(
        f"L = {length:2d}: A = 0.1 at N = {math.exp(u10):.3g}, A = 0.9 at N = {math.exp(u90):.3g}, width {u90 - u10:.3f}"
    )
    assert abs((u90 - u10) - WIDTH / alpha) < 1e-9
print("metric-model: ok")
