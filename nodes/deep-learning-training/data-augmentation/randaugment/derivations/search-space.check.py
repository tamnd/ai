# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/randaugment/ (ai://NEW-randaugment).
# Check of derivations/search-space.md: AutoAugment has (16 * 10 * 11) ** 10 policies, about 2.9e32, and a RandAugment
# grid search tries |N values| * |M values| settings.
import itertools
import math


def aa_space(n_ops=16, n_mag=10, n_prob=11, slots=2, n_sub=5):
    return (n_ops * n_mag * n_prob) ** (slots * n_sub)


def main():
    total = aa_space()
    assert total == 1760**10 == 285184999433627773173760000000000
    assert f"{total:.1e}" == "2.9e+32" and f"{total:.2e}" == "2.85e+32"
    assert abs(math.log10(total) - 32.455) < 1e-3
    assert aa_space(n_sub=1) == 3097600

    # The rule of product, by enumeration on a small space: 3 ops, 2 magnitudes, 2 probabilities, 2 slots, 2 sub-policies.
    slot = list(itertools.product(range(3), range(2), range(2)))
    sub = list(itertools.product(slot, repeat=2))
    policies = sum(1 for _ in itertools.product(sub, repeat=2))
    assert len(slot) == 12 and policies == aa_space(3, 2, 2, 2, 2) == 12**4

    grid = set(itertools.product([1, 2, 3], [5, 7, 9, 11, 13, 15]))
    assert len(grid) == 18
    assert len(set(itertools.product([1, 2, 3], range(11)))) == 33
    assert len(set(itertools.product([1, 2], [2, 6, 10, 14]))) == 8

    lists = set(itertools.product(range(14), repeat=2))
    assert len(lists) == 14**2 == 196
    print(f"ok: |AA| = 1760^10 = {total:.3e}, one sub-policy 3097600, RA grids 18 and 33, 196 operation lists")


if __name__ == "__main__":
    main()
