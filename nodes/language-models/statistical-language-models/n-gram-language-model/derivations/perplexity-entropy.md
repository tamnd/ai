---
id: perplexity-entropy
derives: [c-perplexity-bits]
kind: proof
assumes:
  - A model gives probabilities p_1, ..., p_T > 0 to the T predicted tokens of a test text.
uses: []
status: full
check: perplexity-entropy.check.py
lean: null
sources:
  - paper: brown-1992-class-based-n-gram-models
    locator: p. 3
  - paper: goodman-2001-bit-of-progress-language-modeling
    locator: p. 52
  - paper: chelba-2013-one-billion-word-benchmark
    locator: p. 1
---

## Statement

Perplexity, the reciprocal of the geometric mean of p_1, ..., p_T, equals 2^H. Here H = −(1/T) Σ_i log2 p_i is the
cross-entropy in bits per token. A model that spreads its mass evenly over k words has perplexity k. Cutting
perplexity by a fraction r lowers H by log2(1/(1 − r)) bits, whatever the starting value. So a 10% cut from 100 to
90 is 6.64 to 6.49 bits, about 2.3% of the entropy. The cut from 67.6 to 43.8 is 35% of the perplexity and about
10% of the cross-entropy.

## Assumptions

Brown et al. define perplexity as the reciprocal geometric average of the probabilities of the predictions. Any
base of logarithm gives the same perplexity.

## Steps

1. (Π p_i)^(−1/T) = 2^(−(1/T) Σ log2 p_i) = 2^H.
2. If every p_i = 1/k, H = log2 k and the perplexity is k.
3. If the perplexity goes from P to (1 − r) P, H goes from log2 P to log2 P + log2(1 − r), a drop of log2(1/(1 − r))
   bits. A 50% cut is one bit, and a 10% cut is 0.152 bits.
4. log2 100 = 6.644 and log2 90 = 6.492, a drop of 2.3%, which Goodman rounds to 2%. log2 67.6 = 6.079 and
   log2 43.8 = 5.453, a drop of 10.3%.

## Result and check

The check compares the three forms of perplexity on 100 random probability lists. It reproduces the numbers above,
and for the drop from 51.3 to 30.0 it gives 0.774 bits, 13.6% of the cross-entropy.
