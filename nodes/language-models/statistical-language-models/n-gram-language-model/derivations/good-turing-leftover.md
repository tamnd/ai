---
id: good-turing-leftover
derives: [c-gt-leave-one-out, c-gt-brown-numbers]
kind: proof
assumes:
  - A training sample has N n-gram tokens, and n1 n-gram types occur exactly once.
  - For the expectation, the N tokens are independent draws from one fixed distribution p over types.
uses: []
status: full
check: good-turing-leftover.check.py
lean: null
sources:
  - paper: brown-1992-class-based-n-gram-models
    locator: p. 4
---

## Statement

1. Hold out one token at a time. The share of held-out tokens whose type no longer appears in the remaining sample
   is exactly n1/N.
2. For N independent draws, E[n1/N] = Σ_w p(w)(1 − p(w))^(N−1). The expected probability of the types not drawn is
   Σ_w p(w)(1 − p(w))^N. They differ by Σ_w p(w)²(1 − p(w))^(N−1), which lies between 0 and 1/N.
3. Brown et al.'s figures follow from their Table 1. Of 365,893,263 words, 53,737,350 3-grams and 8,045,024
   2-grams occurred once, so n1/N is 14.7 percent and 2.2 percent.

## Assumptions

Turing's estimate of the probability that the next token is of an unseen type is n1/N. Brown et al. use it to say
how often maximum likelihood will give a new sample probability zero.

## Steps

1. Removing a token of a type with count c leaves c − 1 copies. The type is unseen afterwards exactly when c = 1.
   There are n1 such tokens, so the share is n1/N.
2. A type w occurs exactly once with probability N p(w)(1 − p(w))^(N−1), so E[n1] = N Σ_w p(w)(1 − p(w))^(N−1).
   A type is missing with probability (1 − p(w))^N, and it then contributes p(w) to the missing mass. The difference
   of the two expectations is Σ_w p(w)(1 − p(w))^(N−1)(1 − (1 − p(w))) = Σ_w p(w)²(1 − p(w))^(N−1). Each term is at
   most p(w) · max_x x(1 − x)^(N−1) ≤ p(w)/N, so the sum is at most 1/N.
3. 53,737,350/365,893,263 = 0.1469 and 8,045,024/365,893,263 = 0.0220.

## Result and check

The check confirms the leave-one-out identity on 50 random samples. It computes both expectations exactly for a
Zipf distribution over 50,000 types at N = 10³ to 10⁶. For example, at N = 10⁵ both are 0.0902. In one sample of
100,000 draws, n1/N is 0.0912 and the missing mass is 0.0904. Brown et al.'s percentages come out as 14.69 and 2.20.
