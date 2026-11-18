# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/deep-sets/ (ai://NEW-deep-sets).
"""Numeric check of the derivation decompositions: the examples of the appendix of Zaheer et al. (2017)."""

import decimal

import numpy as np

rng = np.random.default_rng(0)


def phi_cubic(x):
    return np.stack([x, x**2, x**3], -1)


def rho_cubic(s):
    u, v, w = s[..., 0], s[..., 1], s[..., 2]
    return u * v - w + 1.5 * (u * u - v)


def phi_soft(x, alpha):
    e = np.exp(alpha * (x - x.max()))  # a common factor, which v / u cancels
    return np.stack([e, x * e], -1)


def rho_second(s, alpha):
    u, v = s[..., 0], s[..., 1]
    e = np.exp(alpha * v / u)
    return (v - (v / u) * e) / (u - e)


# 1. x1 x2 (x1 + x2 + 3)
x = rng.normal(size=(1000, 2))
assert np.allclose(rho_cubic(phi_cubic(x).sum(1)), x[:, 0] * x[:, 1] * (x[:, 0] + x[:, 1] + 3))
assert np.allclose(rho_cubic(phi_cubic(x[:, ::-1]).sum(1)), rho_cubic(phi_cubic(x).sum(1)))

# 2. the mean
for n in range(1, 9):
    x = rng.normal(size=n)
    s = np.stack([np.ones(n), x], -1).sum(0)
    assert np.isclose(s[1] / s[0], x.mean())

# 3. the softmax-weighted mean tends to the maximum
for _ in range(100):
    x = rng.uniform(0, 1, size=int(rng.integers(2, 9)))
    s = phi_soft(x, 1e4).sum(0)
    assert abs(s[1] / s[0] - x.max()) < 1e-3

# 4. the expression offered for the second largest is v / u
for _ in range(1000):
    u, v, alpha = rng.uniform(0.1, 5), rng.normal(), rng.normal()
    if abs(u - np.exp(alpha * v / u)) > 1e-3:
        assert np.isclose(rho_second(np.array([u, v]), alpha), v / u)
# On {0.3, 0.9, 0.5, 0.1} the denominator u - e^{alpha v / u} cancels in floating point for large alpha, so evaluate
# the expression exactly as written in decimal arithmetic with 600 digits.
decimal.getcontext().prec = 600
X = [decimal.Decimal(t) for t in ("0.3", "0.9", "0.5", "0.1")]
vals = []
for alpha in (1, 10, 100, 1000):
    a = decimal.Decimal(alpha)
    u = sum((a * t).exp() for t in X)
    v = sum(t * (a * t).exp() for t in X)
    e = (a * v / u).exp()
    vals.append((v - (v / u) * e) / (u - e))
    assert abs(vals[-1] - v / u) < decimal.Decimal(10) ** -150
assert abs(vals[-1] - decimal.Decimal("0.9")) < decimal.Decimal(10) ** -100 and vals[0] < vals[1] < vals[2] < vals[3]

print("decompositions: ok")
