---
id: infinigram-normalisation
derives: [c-infinigram-valid, c-backoff-not-normalised]
kind: proof
assumes:
  - The training data is a list of documents. A pattern occurs at a position if its tokens appear there in order inside one document.
  - For a context h, s is the longest suffix of h that occurs at least once followed by another token of the same document, and cnt(s ·) is the number of such occurrences.
uses: []
status: full
check: infinigram-normalisation.check.py
lean: null
sources:
  - paper: liu-2024-infini-gram
    locator: p. 3
  - paper: brants-2007-large-language-models-machine-translation
    locator: p. 2
---

## Statement

1. The estimate P(w | h) = cnt(s w)/cnt(s ·) sums to one over w, because s is chosen before looking at w.
2. If the denominator is cnt(s), the number of all occurrences of s, the estimate sums to less than one whenever s
   also ends a document.
3. Naive backoff takes, for each w separately, the longest suffix s_w with cnt(s_w w) > 0. Its probabilities can sum
   to more than one. Stupid Backoff scores, which do not discount, can sum to more than one as well.

## Assumptions

Liu et al. (2024) back off only when the denominator is zero and state that Σ_w cnt(s w) = cnt(s). That holds when
every occurrence of s is followed by a token. Their effective n is one plus the length of s.

## Steps

1. Each occurrence of s counted in cnt(s ·) is followed by exactly one token, so Σ_w cnt(s w) = cnt(s ·). The suffix s
   depends on h only, so the sum over w uses one denominator, and Σ_w P(w | h) = 1.
2. An occurrence of s at the end of a document is counted in cnt(s) but is followed by no token. Then
   Σ_w cnt(s w) = cnt(s) − (number of such occurrences) < cnt(s). Dividing by cnt(s) leaves the difference as
   missing mass. Counting only the occurrences that are followed, or appending an end marker, avoids it.
3. Under naive backoff, the words seen after the longest suffix already take probability one. Any other word that is
   seen after a shorter suffix adds its own positive share on top. Liu et al. call this the reason Katz backoff needs
   discounting. Stupid Backoff gives f(h w)/f(h) to the words seen after h, which sum to one when an end marker
   follows every sentence, and then adds α f(w)/N for the rest. On "a b", "a c", "b b" with α = 0.4, the scores
   after "a" sum to 11/9.

## Result and check

The check draws 300 random corpora and contexts and computes in exact fractions. The estimate sums to exactly one
every time. Dividing by cnt(s) loses mass in 98 cases, and naive backoff sums above one in 105 and never below. The
Stupid Backoff example sums to 11/9.
