---
id: ikn-as-hpy
derives: [c-ikn-hpy]
kind: proof
assumes:
  - The hierarchical Pitman-Yor language model of Teh (2006) has predictive probability p(w | u) = (c_uw − d t_uw)/(θ + c_u··) + (θ + d t_u··)/(θ + c_u··) p(w | π(u)), which ends with p(w) = 1/V below the empty context.
  - Every strength θ is zero, and each context length has one discount d in (0, 1].
  - The seating is t_uw = min(1, c_uw). The count of w in the restaurant of u is the number of customers from the data when u has N − 1 words, and otherwise the sum of t_u′w over the children u′ with π(u′) = u.
uses: [kn-normalisation]
status: full
check: ikn-as-hpy.check.py
lean: null
sources:
  - paper: teh-2006-hierarchical-pitman-yor-language-model
    locator: p. 5
  - paper: teh-2006-hierarchical-pitman-yor-language-model
    locator: p. 6
---

## Statement

Under these choices, Teh's predictive probabilities equal interpolated Kneser-Ney with discount d for each order.
The top order uses raw counts and every lower order uses continuation counts, the number of distinct words that
precede an n-gram. Heafield et al. keep raw counts for an n-gram that starts with <s>, and that rule falls outside
the correspondence.

## Assumptions

π(u) drops the first word of u. The counts c_u·· and t_u·· sum over w. The modified variant gives each count 1, 2
and 3+ its own discount, which can exceed one. Teh notes that the Pitman-Yor discounts lie between 0 and 1, so the
modified variant is not an approximation of this model.

## Steps

1. With t_uw = min(1, c_uw), the term c_uw − d t_uw is c_uw − d if c_uw > 0 and 0 otherwise, so it equals
   max(c_uw − d, 0) because d ≤ 1. The number t_u·· is the number of distinct w with c_uw > 0, which is N1+(u ·).
2. With θ = 0 the predictive probability is
   p(w | u) = max(c_uw − d, 0)/c_u·· + d N1+(u ·)/c_u·· · p(w | π(u)),
   the interpolated Kneser-Ney recursion with a single discount.
3. At the top, c_uw is the raw count of u w. One level down, c_uw = Σ_v min(1, c_vuw) is the number of words v with
   v u w seen, the continuation count. Deeper levels repeat the step. A child's count is positive exactly when the
   child n-gram is an entry, so each level counts the distinct left extensions. These are the adjusted counts of
   Heafield et al. except at <s>.
4. A context made of <s> has one child in the tree, the longer run of <s>. Its counts are therefore 0 or 1, whereas
   Heafield et al. keep the raw count of <s> w. The two models differ whenever a sentence-initial n-gram is seen
   more than once.

## Result and check

The check builds both models in exact fractions on 40 random corpora of orders 1 to 4. Teh's model fills the
restaurants bottom-up by Eq. 15 and Eq. 16, and interpolated Kneser-Ney is computed from the n-gram counts. All
7,410 probabilities agree exactly, and the <s> rule of Heafield et al. changes 990 of them.
