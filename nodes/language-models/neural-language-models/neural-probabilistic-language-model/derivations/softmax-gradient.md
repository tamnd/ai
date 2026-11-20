---
id: softmax-gradient
derives: [c-softmax-gradient]
kind: gradient
assumes:
  - p = softmax(y) over the vocabulary and the loss of one example is −log p_w for the observed word w.
  - The hidden layer is a = tanh(o) with o = d + H x.
uses: []
status: full
check: softmax-gradient.check.py
lean: null
sources:
  - paper: bengio-2003-neural-probabilistic-language-model
    locator: p. 10
---

## Statement

The gradient of −log p_w with respect to the logit y_j is p_j − 1{j = w}. These entries sum to zero over j. So the
gradient at the output layer touches every word of the vocabulary, not only w. Through the tanh layer, the gradient at
o_k is (1 − a_k²) times the gradient at a_k. Bengio et al.'s ascent rule uses the negatives of these quantities.

## Assumptions

The logits y are any real vector; nothing else about the network matters for the first part.

## Steps

1. −log p_w = −y_w + log Σ_i exp(y_i).
2. The derivative of log Σ_i exp(y_i) with respect to y_j is exp(y_j) / Σ_i exp(y_i) = p_j, and that of −y_w is
   −1{j = w}. So the derivative is p_j − 1{j = w}.
3. Σ_j (p_j − 1{j = w}) = 1 − 1 = 0. Each p_j > 0, so every entry is non-zero: the word w gets p_w − 1 < 0 and
   every other word gets p_j > 0.
4. d tanh(o) / do = 1 − tanh(o)² = 1 − a², which gives the factor at o.

## Result and check

The check compares both formulas with central finite differences on random logits and pre-activations, in exact
rational arithmetic for the sum to zero.
