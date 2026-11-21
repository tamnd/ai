# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/prompt-sensitivity/ (ai://NEW-prompt-sensitivity).
"""Checks every number of calibration-orderings-and-format-spread.md."""

import itertools
import math

import numpy as np


def close(x, y, tol=5e-5):
    assert abs(x - y) < tol, (x, y)


def phi_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def emax(m):
    x = np.linspace(-12, 12, 480_001)
    pdf = np.exp(-x * x / 2) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))
    return float(np.trapezoid(x * m * pdf * cdf ** (m - 1), x))


rng = np.random.default_rng(0)

# 1. calibration is a threshold at the content-free probability
for a in [0.2, 0.5, 0.62, 0.9]:
    p = rng.uniform(size=100_000)
    cal = (p / a) > ((1 - p) / (1 - a))
    assert np.array_equal(cal, p > a)
    close(
        math.log(p[0] / (1 - p[0])) - math.log(a / (1 - a)),
        math.log((p[0] / a) / ((1 - p[0]) / (1 - a))),
    )
close(math.log(0.62 / 0.38), 0.4895)
s = rng.normal(size=50_000)
y = s + 0.3 * rng.normal(size=s.size) > 0
b = 1.3
pb = 1 / (1 + np.exp(-(s + b)))
a = 1 / (1 + np.exp(-b))
assert np.array_equal(pb > a, s > 0)
assert ((pb > 0.5) == y).mean() < ((pb > a) == y).mean()

# 2. orderings
assert math.factorial(4) == 24 and math.factorial(8) == 40320 and math.comb(4, 2) == 6
assert len(set(itertools.permutations("PPNN"))) == 6

# 3. expected max and range
for m, e in [(2, 0.5642), (10, 1.5388), (24, 1.9477), (320, 2.8983)]:
    close(emax(m), e, 1e-4)
close(2 * emax(10) / (2 * emax(320)), 0.531, 5e-4)
z = rng.normal(size=(20_000, 10))
close((z.max(1) - z.min(1)).mean(), 3.0775, 0.03)

# 4. noise floor
close(100 * math.sqrt(0.25 / 1000), 1.58, 0.005)
close(2 * emax(10) * 100 * math.sqrt(0.25 / 1000), 4.87, 0.005)
close(100 * math.sqrt(0.1 / 1000), 1.0)
n, d = 1000, 0.1
diffs = []
for _ in range(4000):
    u = rng.uniform(size=n)
    x = np.where(u < d / 2, 1, np.where(u < d, -1, 0))
    diffs.append(x.mean())
close(100 * np.std(diffs), 1.0, 0.04)

# 5. flips
flip = lambda dl, sg, r: phi_cdf(-dl / (sg * math.sqrt(2 * (1 - r))))
for dl, sg, r, v in [
    (2, 5, 0, 0.3886),
    (5, 5, 0, 0.2398),
    (10, 5, 0, 0.0786),
    (2, 5, 0.5, 0.3446),
]:
    close(flip(dl, sg, r), v)
e1 = rng.normal(size=200_000) * 5
e2 = rng.normal(size=200_000) * 5
close(((2 + e1 - e2) < 0).mean(), 0.3886, 0.005)

# 6. selection optimism
se = 100 * math.sqrt(0.7 * 0.3 / 100)
close(se, 4.583, 5e-4)
close(se * emax(10), 7.05, 0.005)
close(se * emax(24), 8.93, 0.005)
sims = rng.binomial(100, 0.7, size=(20_000, 24)).max(1)
close(sims.mean() - 70, 8.93, 0.4)
print("ok")
