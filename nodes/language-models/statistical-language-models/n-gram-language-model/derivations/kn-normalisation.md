---
id: kn-normalisation
derives: [c-kn-normalised]
kind: proof
assumes:
  - Every sentence is padded with N − 1 copies of <s> in front and </s> at the end, and <s> is never predicted.
  - The adjusted counts are those of Heafield et al. (2013), so every n-gram in the model of order n > 1 has its suffix of order n − 1 in the model too.
  - Each order n has discounts with D_n(0) = 0 and 0 ≤ D_n(k) ≤ k for k = 1, 2, 3, and D_n(k) = D_n(3) for k > 3.
  - The vocabulary is the set of words that end a unigram entry, together with one unknown word.
uses: []
status: full
check: kn-normalisation.check.py
lean: null
sources:
  - paper: heafield-2013-scalable-modified-kneser-ney
    locator: p. 2
  - paper: heafield-2013-scalable-modified-kneser-ney
    locator: p. 3
---

## Statement

For every context h, seen in training or not, the interpolated Kneser-Ney probabilities p(w | h) are non-negative
and sum to exactly one over the vocabulary. They are all positive when every discount D_n(k) with k ≥ 1 is positive.

## Assumptions

The model is the one of Heafield et al. (2013). For a context h of n − 1 words seen in training, with adjusted
counts a(h x) of its extensions,

$$u(w \mid h) = \frac{a(h w) - D_n(a(h w))}{\sum_x a(h x)}, \qquad b(h) = \frac{\sum_x D_n(a(h x))}{\sum_x a(h x)},$$

and p(w | h) = u(w | h) + b(h) p(w | h′), where h′ drops the first word of h. The recursion ends in the uniform
distribution 1/|V|. An unseen context has u = 0 and b = 1. Heafield et al. write b with the three sets
{x : a(h x) = i} for i = 1, 2, 3, the last standing for every count of 3 or more. That is the same sum.

## Steps

1. Take a seen context h. Each u(w | h) is non-negative because D_n(k) ≤ k, and it is zero unless h w is an entry.
   Summing over w,
   Σ_w u(w | h) = (Σ_x a(h x) − Σ_x D_n(a(h x))) / Σ_x a(h x) = 1 − b(h).
   This needs every word that ends an entry to be in the vocabulary. An entry of order n has its suffix of order
   n − 1 as an entry, so the word that ends it also ends a unigram entry, and the vocabulary contains it.
2. Induction on the length of h. The empty context gives the uniform distribution, which sums to one. Suppose
   p(· | h′) sums to one. If h was seen, Σ_w p(w | h) = (1 − b(h)) + b(h) · 1 = 1. If h was not seen,
   p(· | h) = p(· | h′), which sums to one.
3. Non-negativity follows by the same induction, since u ≥ 0 and b ≥ 0. If every D_n(k) with k ≥ 1 is positive, every
   seen context has b(h) > 0. Then p(w | h) ≥ b(h) p(w | h′) > 0, down to the uniform distribution.
4. The bound D_n(k) ≤ k is needed. With D = 1.5 on an n-gram seen once, u is negative, and some probability can fall
   below zero.

## Result and check

The check builds 60 random models in exact fractions, with orders 1 to 4 and discounts drawn on a grid in [0, k],
including zero. For 2,259 contexts, the seen ones and random unseen ones, every probability is non-negative and
the probabilities sum to exactly one. For each seen context, the u terms sum to exactly 1 − b(h). With every
discount positive, every probability is positive. A discount of 1.5 on a count of 1 gives a negative probability.
