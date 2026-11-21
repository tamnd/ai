"""Check every number in text-to-image-arithmetic.md."""

import itertools
import math
from collections import Counter
from fractions import Fraction

import numpy as np


def close(x, y, tol=5e-4):
    assert abs(x - y) <= tol * max(1.0, abs(y)), (x, y)


# 1. Tokenizer arithmetic.
assert 256 * 256 * 3 == 196608 and 32 * 32 == 1024 and 196608 // 1024 == 192 and 196608 % 1024 == 0
assert math.log2(8192) == 13 and 1024 * 13 == 13312 and 196608 * 8 == 1572864
assert 256 + 1024 == 1280
assert 256 // 32 == 8
assert Fraction(1, 8) + Fraction(7, 8) == 1 and Fraction(7, 8) / Fraction(1, 8) == 7

# 2. The three notations of guidance agree.
rng = np.random.default_rng(0)
gc, gu = rng.standard_normal(5), rng.standard_normal(5)
for lam in (0.0, 1.0, 3.0, 7.5):
    assert np.allclose(gu + lam * (gc - gu), lam * gc + (1 - lam) * gu)
z, a, s = rng.standard_normal(5), 0.6, 0.8
ec, eu = rng.standard_normal(5), rng.standard_normal(5)
w = 4.0
x_from_eps = (z - s * (w * ec + (1 - w) * eu)) / a
x_mixed = w * (z - s * ec) / a + (1 - w) * (z - s * eu) / a
assert np.allclose(x_from_eps, x_mixed)
# On logits: softmax(l_u + lam (l_c - l_u)) is p_c^lam p_u^(1-lam), renormalized, and p_c (p_c/p_u)^(lam-1).
lc, lu = rng.standard_normal(7), rng.standard_normal(7)
pc, pu = np.exp(lc) / np.exp(lc).sum(), np.exp(lu) / np.exp(lu).sum()
lam = 3.0
g = np.exp(lu + lam * (lc - lu))
g /= g.sum()
h = pc**lam * pu ** (1 - lam)
h /= h.sum()
k = pc * (pc / pu) ** (lam - 1)
k /= k.sum()
assert np.allclose(g, h) and np.allclose(g, k)


# 3. Range of the guided prediction and thresholding.
for w in (1.0, 2.0, 8.0):
    worst = max(abs(w * xc + (1 - w) * xu) for xc in (-1, 1) for xu in (-1, 1))
    assert worst == 2 * w - 1
close(8 * 0.9 + (1 - 8) * 0.0, 7.2)
assert round(100 - 99.5, 1) == 0.5
# The toy of the numpy file.
D = 4096
r = np.random.default_rng(0)
mu_c = r.uniform(-0.6, 0.6, D)
S = 0.1
mu_u, s_u = np.zeros(D), np.sqrt(S**2 + 0.12)


def post(zz, al, sg, mu, sd):
    return mu + al * sd**2 / (al**2 * sd**2 + sg**2) * (zz - al * mu)


rr = np.random.default_rng(1)
x = mu_c + S * rr.standard_normal(D)
al, sg = math.cos(math.pi / 4), math.sin(math.pi / 4)
zz = al * x + sg * rr.standard_normal(D)
sat = {}
for w in (4, 8):
    xg = w * post(zz, al, sg, mu_c, S) + (1 - w) * post(zz, al, sg, mu_u, s_u)
    st = np.clip(xg, -1, 1)
    sv = max(np.percentile(np.abs(xg), 99.5), 1.0)
    dy = np.clip(xg, -sv, sv) / sv
    sat[w] = (np.mean(np.abs(st) >= 1 - 1e-6), np.mean(np.abs(dy) >= 1 - 1e-6))
assert round(100 * sat[4][0]) == 54 and round(100 * sat[8][0]) == 76
assert round(100 * sat[4][1], 1) == 0.5 and round(100 * sat[8][1], 1) == 0.5

# 4. The pooled ceiling: the mean of 1/m over all 6^4 images.
assert 6**4 == 1296
tot = Fraction(0)
for img in itertools.product(range(6), repeat=4):
    c = Counter(img)
    m = math.factorial(4) // math.prod(math.factorial(v) for v in c.values())
    tot += Fraction(1, m)
assert tot == 126 and tot / 1296 == Fraction(7, 72)
close(float(Fraction(7, 72)), 0.0972, 1e-3)
assert round(float(Fraction(7, 72)), 3) == 0.097
assert 0.091 < 7 / 72 < 0.998

# 5. Best of N.
p, q = 0.1, 0.9
n_star = math.log(2) / -math.log(q)
close(n_star, 6.579, 1e-3)
assert round(n_star, 1) == 6.6
gain = [q**n * (1 - q**n) for n in range(1, 60)]
assert int(np.argmax(gain)) + 1 in (6, 7)
assert round(100 * (1 - q**32), 1) == 96.6
assert 100 * ((1 - q**512) - (1 - q**32)) < 3.5
assert 512 // 16 == 32
close(1 - (1 - 0.459) ** 4, 0.914, 1e-3)
assert round(1 - 0.541**4, 3) == 0.914

# 6. unCLIP.
assert round(1024 / 319, 2) == 3.21

# 7. Budgets.
assert 2048 * 2_500_000 == 5_120_000_000
assert 460 + 400 == 860
assert round(5.12e9 / 8.6e8, 1) == 6.0
assert round(2.0 + 0.6 + 0.4, 1) == 3.0
assert 1024 * 430_000 == 440_320_000
assert round(440.32e6 / 250e6, 2) == 1.76

# 8. PowerSGD.
for rk, d, pct in ((512, 1920, 83.3), (640, 2688, 85.1), (896, 3968, 85.9)):
    assert round(100 * (1 - 5 * rk / (8 * d)), 1) == pct
print("ok")
