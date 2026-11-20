# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-probabilistic-language-model/ (ai://NEW-neural-probabilistic-language-model).
"""Check of mixture-bound: the perplexity of a linear mixture of two models is at most the weighted geometric mean
of their perplexities, with equality only when the models agree on every token."""

import math
import random


def ppl(ps):
    return math.exp(-sum(math.log(p) for p in ps) / len(ps))


def main():
    rng = random.Random(9)
    best_gain = 0.0
    for _ in range(500):
        T = rng.randint(1, 200)
        p = [rng.uniform(1e-5, 1) for _ in range(T)]
        q = [rng.uniform(1e-5, 1) for _ in range(T)]
        w = rng.uniform(0.01, 0.99)
        mixed = [w * a + (1 - w) * b for a, b in zip(p, q, strict=True)]
        bound = ppl(p) ** w * ppl(q) ** (1 - w)
        assert ppl(mixed) <= bound * (1 + 1e-12) and bound <= max(ppl(p), ppl(q)) * (1 + 1e-12)
        best_gain = max(best_gain, 1 - ppl(mixed) / min(ppl(p), ppl(q)))
        assert math.isclose(ppl([w * a + (1 - w) * a for a in p]), ppl(p) ** w * ppl(p) ** (1 - w))
    assert best_gain > 0  # the mixture beats both models on some lists
    brown = math.sqrt(276 * 336)
    assert round(brown, 1) == 304.5 and 252 < brown
    print(
        f"mixture-bound: bound held on 500 lists; largest gain over the better model {100 * best_gain:.1f}%; "
        f"Brown bound {brown:.1f}"
    )


if __name__ == "__main__":
    main()
