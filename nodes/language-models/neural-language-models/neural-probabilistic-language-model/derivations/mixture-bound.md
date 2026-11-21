---
id: mixture-bound
derives: [c-mixture-bound]
kind: bound
assumes:
  - Two models give probabilities p_i > 0 and q_i > 0 to the same T predicted tokens, and the mixture gives w p_i + (1 − w) q_i with 0 < w < 1.
uses: []
status: full
check: mixture-bound.check.py
lean: null
sources:
  - paper: bengio-2003-neural-probabilistic-language-model
    locator: p. 7
  - paper: bengio-2003-neural-probabilistic-language-model
    locator: p. 13
---

## Statement

The perplexity of the mixture is at most PP_p^w PP_q^(1 − w), the weighted geometric mean of the two perplexities. With
w = 1/2 the bound is the square root of their product, so the mixture is never worse than the worse model and can be
better than both. Bengio et al. report that mixing with the interpolated trigram always lowered the perplexity of
their networks.

## Assumptions

The mixture weight w is fixed, not tuned on the test text.

## Steps

1. log is concave, so log(w p_i + (1 − w) q_i) ≥ w log p_i + (1 − w) log q_i for every token.
2. Averaging over the T tokens and negating: the cross-entropy of the mixture is at most w H_p + (1 − w) H_q.
3. Exponentiating gives PP_mix ≤ PP_p^w PP_q^(1 − w) ≤ max(PP_p, PP_q).
4. Equality in step 1 needs p_i = q_i for every token. The gain grows with how differently the two models spread
   their errors.

## Result and check

On Brown, Bengio et al.'s MLP9 has test perplexity 276 and the deleted-interpolation trigram 336. The bound for their
half-half mixture is (276 × 336)^(1/2) = 304.5; the reported MLP10 has 252. The check tests the bound on random
probability lists.
