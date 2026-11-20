# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/deep-sets/ (ai://NEW-deep-sets).
"""Numeric check of the derivation power-sums: the Newton-Girard formulae recover a set from its power sums."""

import numpy as np


def elementary(p):
    """e_0, ..., e_M from the power sums p_1, ..., p_M by the Newton-Girard formulae."""
    e = [1.0]
    for k in range(1, len(p) + 1):
        e.append(sum((-1) ** (i - 1) * e[k - i] * p[i - 1] for i in range(1, k + 1)) / k)
    return e


def recover(p):
    e = elementary(p)
    coeffs = [(-1) ** k * e[k] for k in range(len(e))]
    return np.sort(np.roots(coeffs).real)


rng = np.random.default_rng(0)
for M in range(1, 7):
    for _ in range(50):
        x = rng.uniform(-1, 1, size=M)
        p = [float((x**k).sum()) for k in range(1, M + 1)]
        assert np.allclose(recover(p), np.sort(x), atol=1e-6), (x, recover(p))
        assert np.allclose(np.sum([[1.0] + [t**k for k in range(1, M + 1)] for t in x], 0), [M] + p)

# the worked example
assert [sum(t**k for t in (1, 2, 4)) for k in (1, 2, 3)] == [7, 21, 73]
assert elementary([7, 21, 73]) == [1.0, 7.0, 14.0, 8.0]
assert np.allclose(recover([7, 21, 73]), [1, 2, 4])

# two sets of three numbers with the same first two power sums
a, b = (1, 5, 6), (2, 3, 7)
assert sum(a) == sum(b) == 12 and sum(t * t for t in a) == sum(t * t for t in b) == 62
assert sum(t**3 for t in a) == 342 and sum(t**3 for t in b) == 378

print("power-sums: ok")
