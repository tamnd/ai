---
id: span-corruption-lengths
derives: [c-t5-lengths]
kind: count
assumes:
  - A sequence of n tokens has k noise tokens in s maximal runs; each run becomes one sentinel in the input, and the target is each run after its sentinel followed by one final sentinel.
  - For i.i.d. noise each token is noise independently with probability r.
uses: []
status: full
check: span-corruption-lengths.check.py
lean: null
sources:
  - paper: raffel-2020-t5
    locator: p. 13
  - paper: raffel-2020-t5
    locator: p. 24
---

## Statement

Span corruption turns n tokens with k noise tokens in s runs into an input of n − k + s tokens and a target of
k + s + 1 tokens. For Raffel et al.'s example of 500 tokens, 15% noise and 25 spans, the input has 450 tokens and the
target 101. With i.i.d. noise at rate r the expected number of runs is r + (n − 1) r (1 − r), which for n = 500 and
r = 0.15 is 63.77, a mean run length of 1.18 tokens; the expected input then has 488.77 tokens and the target 139.77.
Corrupting spans of mean length 3 instead shortens the target by 28% and the input by 8%.

## Assumptions

The counts ignore the end-of-sequence token, which adds one to each side in either scheme.

## Steps

1. The input keeps the n − k tokens that are not noise and adds one sentinel per run: n − k + s.
2. The target holds the k noise tokens, one sentinel before each run and one at the end: k + s + 1.
3. Example: k = 0.15 × 500 = 75 and s = 25, so the input is 500 − 75 + 25 = 450 and the target 75 + 25 + 1 = 101;
   the mean span length is 75 / 25 = 3.
4. With i.i.d. noise, a run starts at position 1 if that token is noise (probability r) and at position t > 1 if
   token t is noise and token t − 1 is not (probability r(1 − r)). Summing over positions gives
   E[s] = r + (n − 1) r (1 − r) = 0.15 + 499 × 0.1275 = 63.7725, and E[k] = rn = 75.
5. By linearity the expected lengths are 500 − 75 + 63.7725 = 488.7725 and 75 + 63.7725 + 1 = 139.7725, and
   1 − 101/139.7725 = 0.277 and 1 − 450/488.7725 = 0.079.

## Result and check

The check builds the input and target with the NumPy reference's span_corrupt for random noise masks and compares
their lengths with the formulas, reproduces the example, and compares the expected run count of i.i.d. noise with
a simulation of 20,000 sequences.
