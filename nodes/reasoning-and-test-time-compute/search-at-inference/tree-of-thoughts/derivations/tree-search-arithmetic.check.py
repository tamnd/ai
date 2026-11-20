# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/tree-of-thoughts/ (ai://NEW-tree-of-thoughts).
"""Checks every step of tree-search-arithmetic.md numerically."""

import math
from fractions import Fraction

import numpy as np

rng = np.random.default_rng(0)


def moves(nums):
    out = []
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            a, b = nums[i], nums[j]
            rest = [nums[t] for t in range(len(nums)) if t not in (i, j)]
            for r in (a + b, a * b, a - b, b - a, a / b if b else None, b / a if a else None):
                if r is not None:
                    out.append(rest + [r])
    return out


def leaves(nums):
    return 1 if len(nums) == 1 else sum(leaves(s) for s in moves(nums))


# 1. Tree size.
assert math.comb(4, 2) * 6 == 36 and math.comb(3, 2) * 6 == 18 and math.comb(2, 2) * 6 == 6
assert leaves([Fraction(x) for x in (1, 2, 4, 7)]) == 3888
assert leaves([Fraction(x) for x in (4, 9, 10, 13)]) == 3880

# 2. Chains against a tree.
q, T, k = 0.4, 3, 5
chain = q**T
five = 1 - (1 - q**T) ** 5
tree = (1 - (1 - q) ** k) ** T
assert math.isclose(chain, 0.064)
assert round(five, 4) == 0.2816 and round(1 - (1 - q) ** k, 5) == 0.92224 and round(tree, 4) == 0.7844
assert 5 * T == k * T == 15  # both generate fifteen thoughts
# Monte Carlo of the tree with a perfect evaluator and of five chains.
good = rng.random((200_000, T, k)) < q
assert abs(good.any(2).all(1).mean() - tree) < 4e-3
assert abs(good[:, :, :5].all(1).any(1).mean() - five) < 4e-3  # column c is chain c

# 3. Evaluator errors compound.
assert round(0.9**3, 3) == 0.729 and round(0.9**10, 4) == 0.3487

# 4. Breadth-first call counts.
T, b, k, v = 3, 5, 5, 3
E = 1 + b * (T - 1)
assert (E, k * E, v * k * E) == (11, 55, 165)
assert v * k * E // E == 15  # evaluation calls per expansion call


# 5. UCT.
def uct(value, n, n_parent, w=1.0):
    return math.inf if n == 0 else value + w * math.sqrt(math.log(n_parent) / n)


assert round(uct(0.6, 5, 10), 4) == 1.2786 and round(uct(0.2, 1, 10), 4) == 1.7174
assert uct(0.2, 1, 10) > uct(0.6, 5, 10) and uct(0.0, 0, 10) == math.inf


# 6. Self-evaluation score.
def score(p, c, lam=0.5):
    return c**lam * p ** (1 - lam)


assert round(score(0.9, 0.2), 4) == 0.4243 and round(score(0.5, 0.9), 4) == 0.6708
assert score(0.5, 0.9) > score(0.9, 0.2) and score(0.9, 0.2, lam=0.0) == 0.9

# 7. Token accounting.
assert 6700 / 100 == 67 and round(5500 / 67) == 82
assert round(0.74 / 0.74, 2) == 1.00 and round(0.47 / 0.49, 2) == 0.96
print("ok")
