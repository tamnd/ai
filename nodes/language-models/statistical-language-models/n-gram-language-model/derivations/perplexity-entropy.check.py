# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/n-gram-language-model/ (ai://NEW-n-gram-language-model).
"""Check of perplexity-entropy: perplexity is the reciprocal geometric mean of the predicted probabilities and equals
2 to the cross-entropy in bits per token; a model that spreads its mass uniformly over k words has perplexity k; a
perplexity cut by the fraction r lowers the cross-entropy by log2(1 / (1 - r)) bits; and the numbers that Goodman
(2001) and Chelba et al. (2013) quote follow."""

import math
import random


def main():
    rng = random.Random(7)
    for _ in range(100):
        ps = [rng.uniform(1e-6, 1.0) for _ in range(rng.randint(1, 500))]
        T = len(ps)
        geo = math.exp(sum(math.log(p) for p in ps) / T)
        bits = -sum(math.log2(p) for p in ps) / T
        assert math.isclose(1 / geo, 2**bits, rel_tol=1e-12)
        assert math.isclose(1 / geo, math.exp(-sum(math.log(p) for p in ps) / T), rel_tol=1e-12)
    for k in (2, 7, 100, 50_000):
        assert math.isclose(2 ** (-math.log2(1 / k)), k)
    # Goodman p. 52: 100 -> 90 is 6.64 -> 6.49 bits, about 2% of the entropy.
    a, b = math.log2(100), math.log2(90)
    assert f"{a:.2f}" == "6.64" and f"{b:.2f}" == "6.49" and math.isclose(a - b, math.log2(1 / 0.9))
    goodman_drop = (a - b) / a
    assert 0.02 < goodman_drop < 0.025
    # Goodman p. 2: a 50% perplexity reduction is one bit.
    assert math.log2(1 / 0.5) == 1.0
    # Chelba et al.: 67.6 -> 43.8 is a 35% perplexity reduction and about 10% of the cross-entropy.
    c0, c1 = 67.6, 43.8
    ppl_drop = 1 - c1 / c0
    ce_drop = (math.log2(c0) - math.log2(c1)) / math.log2(c0)
    assert round(100 * ppl_drop) == 35 and round(100 * ce_drop) == 10
    # Jozefowicz et al.: 51.3 -> 30.0 and 41.0 -> 23.7.
    j1 = (math.log2(51.3) - math.log2(30.0)) / math.log2(51.3)
    print(
        f"perplexity-entropy: 100 -> 90 is {a:.3f} -> {b:.3f} bits ({100 * goodman_drop:.1f}% of the entropy); "
        f"67.6 -> 43.8 is {100 * ppl_drop:.1f}% in perplexity, {math.log2(c0):.3f} -> {math.log2(c1):.3f} bits "
        f"({100 * ce_drop:.1f}%); 51.3 -> 30.0 is {math.log2(51.3) - math.log2(30.0):.3f} bits ({100 * j1:.1f}%)"
    )


if __name__ == "__main__":
    main()
