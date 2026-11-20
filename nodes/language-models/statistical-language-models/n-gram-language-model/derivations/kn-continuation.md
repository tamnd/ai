---
id: kn-continuation
derives: [c-kn-continuation]
kind: proof
assumes:
  - The bigram counts c(v w) come from padded training sentences, with c(v ·) = Σ_w c(v w) and c(· w) = Σ_v c(v w).
  - The bigram model is interpolated absolute discounting, p(w | v) = max(c(v w) − D, 0)/c(v ·) + D N1+(v ·)/c(v ·) q(w), with one discount 0 < D ≤ 1.
  - The lower-order distribution q is chosen so that the model keeps the observed marginal of the predicted word, Σ_v c(v ·) p(w | v) = c(· w) for every w.
uses: []
status: full
check: kn-continuation.check.py
lean: null
sources:
  - paper: goodman-2001-bit-of-progress-language-modeling
    locator: p. 8
  - paper: goodman-2001-bit-of-progress-language-modeling
    locator: p. 61
---

## Statement

Under the assumptions, the only lower-order distribution that keeps the marginal is

$$q(w) = \frac{N_{1+}(\cdot\, w)}{N_{1+}(\cdot\,\cdot)},$$

where N1+(· w) is the number of distinct words v with c(v w) > 0 and N1+(· ·) is the number of distinct bigram
types. That is the share of bigram types that end in w, not the share of tokens. With D > 1 the continuation counts
no longer satisfy the constraint exactly.

## Assumptions

N1+(v ·) is the number of distinct words that follow v. Goodman (2001) gives the reason for keeping marginals: a
model that does not keep a known marginal can be improved by keeping it. Absolute discounting, interpolation and
kept marginals together give interpolated Kneser-Ney.

## Steps

1. Weight the model by the context counts and sum over v:
   Σ_v c(v ·) p(w | v) = Σ_v max(c(v w) − D, 0) + D q(w) Σ_v N1+(v ·).
   The last sum counts every bigram type once, so it is N1+(· ·).
2. Every bigram that occurs has c(v w) ≥ 1 ≥ D, so max(c(v w) − D, 0) = c(v w) − D for each of the N1+(· w) words v
   that precede w, and 0 for the others. The first sum is c(· w) − D N1+(· w).
3. Setting the total equal to c(· w) leaves D q(w) N1+(· ·) = D N1+(· w). Since D > 0, q(w) = N1+(· w)/N1+(· ·),
   and no other q satisfies the constraint.
4. If D > 1, a bigram seen once gives max(1 − D, 0) = 0 rather than 1 − D. The solution becomes
   q(w) = (c(· w) − Σ_v max(c(v w) − D, 0)) / (D N1+(· ·)), which differs from the continuation share whenever some
   bigram ending in w was seen once.
5. The same argument at each order, with contexts one word longer, gives the adjusted counts of Heafield et al.
   (2013): continuation counts for every order below the top.

## Result and check

The check builds 200 random corpora in exact fractions and draws D from (0, 1]. The continuation share keeps every
marginal exactly, and solving the constraint returns that share. With D in (1, 2], the solution differs. A corpus
in the spirit of Goodman's San Francisco example uses a frequent word that only ever follows "san". It gives
p(francisco | on) = 0.0114 with continuation counts and 0.0532 with relative frequencies, and only the first keeps
the marginal.
