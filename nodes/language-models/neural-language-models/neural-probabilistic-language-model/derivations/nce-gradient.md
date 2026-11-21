---
id: nce-gradient
derives: [c-nce-limit]
kind: limit
assumes:
  - For one context the model scores each word w with an unnormalised probability P(w) = exp(s_w) whose normalising constant is fixed to 1.
  - Data words come from P_d, noise words from P_n with P_n(w) > 0 for every word, and there are k noise words per data word.
uses: []
status: full
check: nce-gradient.check.py
lean: null
sources:
  - paper: mnih-2012-fast-simple-algorithm-neural-language-models
    locator: p. 3
  - paper: mnih-2012-fast-simple-algorithm-neural-language-models
    locator: p. 4
---

## Statement

The noise-contrastive objective for one context is J = E_{P_d}[log σ(Δ(w))] + k E_{P_n}[log σ(−Δ(w))] with
Δ(w) = log P(w) − log(k P_n(w)). Its gradient with respect to the score s_w is

  k P_n(w) / (P(w) + k P_n(w)) · (P_d(w) − P(w)).

The weight in front tends to 1 as k grows, which leaves P_d(w) − P(w). That limit is the gradient of
Σ_w P_d(w) log P(w) − Σ_w P(w), which equals the log-likelihood gradient whenever P sums to one. The gradient is zero
at P = P_d for every k, so the data distribution is a stationary point at any number of noise samples. The weight
is at least 1 − P(w) / (k P_n(w)), so the gap to the limit shrinks like 1/k.

## Assumptions

The scores s_w are free parameters, one per word. Through the chain rule the same holds for any parametrisation.

## Steps

1. dΔ(w)/ds_v = 1{v = w}, d log σ(Δ)/dΔ = σ(−Δ) and d log σ(−Δ)/dΔ = −σ(Δ).
2. So dJ/ds_w = P_d(w) σ(−Δ(w)) − k P_n(w) σ(Δ(w)).
3. σ(Δ) = P / (P + k P_n) and σ(−Δ) = k P_n / (P + k P_n). Substituting gives k P_n (P_d − P) / (P + k P_n).
4. As k → ∞, k P_n / (P + k P_n) = 1 / (1 + P / (k P_n)) → 1, and it is at least 1 − P / (k P_n).
5. The derivative of Σ P_d log P − Σ P with respect to s_w is P_d(w) − P(w). If Σ P = 1, the second sum has zero
   derivative along the normalised model, so the limit is the maximum-likelihood gradient.

## Result and check

The check compares the closed form with finite differences of the exact expectation J on random distributions, and
shows the error to the limit falling with k from 1 to 10,000 with k times the error bounded.
