# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/n-gram-language-model/ (ai://NEW-n-gram-language-model).
"""Check of good-turing-leftover: n1 / N is exactly the share of tokens whose type is unseen once that token is held
out; Brown et al.'s 14.7 and 2.2 percent follow from their Table 1; and for N draws from a fixed distribution,
E[n1 / N] exceeds the expected unseen mass by sum p^2 (1 - p)^(N - 1), which is at most 1 / N."""

import random
from collections import Counter

BROWN_WORDS = 365_893_263
BROWN_ONCE = {"2-grams": 8_045_024, "3-grams": 53_737_350}


def main():
    share = {k: v / BROWN_WORDS for k, v in BROWN_ONCE.items()}
    assert round(100 * share["3-grams"], 1) == 14.7 and round(100 * share["2-grams"], 1) == 2.2

    rng = random.Random(3)
    for _ in range(50):
        toks = [min(int(rng.paretovariate(rng.uniform(0.6, 2.0))), 10_000) for _ in range(rng.randint(10, 3000))]
        c = Counter(toks)
        n1 = sum(1 for v in c.values() if v == 1)
        held_out_unseen = sum(1 for i, t in enumerate(toks) if c[t] - 1 == 0)
        assert held_out_unseen == n1

    # Exact expectations for a Zipf distribution over 50,000 types.
    K = 50_000
    w = [1 / (i + 1) ** 1.1 for i in range(K)]
    Z = sum(w)
    p = [x / Z for x in w]
    rows = []
    for N in (1_000, 10_000, 100_000, 1_000_000):
        e_n1_over_n = sum(q * (1 - q) ** (N - 1) for q in p)
        e_missing = sum(q * (1 - q) ** N for q in p)
        gap = sum(q * q * (1 - q) ** (N - 1) for q in p)
        assert abs(e_n1_over_n - e_missing - gap) < 1e-12 and 0 <= gap <= 1 / N
        rows.append(f"N={N}: E[n1/N]={e_n1_over_n:.4f}, E[missing]={e_missing:.4f}")
    # One simulated sample: the estimate tracks the realised missing mass.
    N = 100_000
    draws = Counter(rng.choices(range(K), weights=p, k=N))
    est = sum(1 for v in draws.values() if v == 1) / N
    missing = sum(q for i, q in enumerate(p) if i not in draws)
    assert abs(est - missing) < 0.01
    print(
        f"good-turing-leftover: Brown 3-grams {100 * share['3-grams']:.2f}%, 2-grams {100 * share['2-grams']:.2f}%; "
        + "; ".join(rows)
        + f"; one sample of {N}: n1/N={est:.4f}, missing mass={missing:.4f}"
    )


if __name__ == "__main__":
    main()
