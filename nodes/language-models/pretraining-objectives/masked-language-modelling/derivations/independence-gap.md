---
id: independence-gap
derives: [c-independence-gap]
kind: inequality
assumes:
  - Two positions a and b are masked in the same example; each prediction sees the visible context c and [MASK] at the other position, so the model's predictions are q(a | c) and q(b | c).
  - The masked LM loss of the pair is −log q(a | c) − log q(b | c), averaged over the data distribution.
uses: []
status: full
check: independence-gap.check.py
lean: null
sources:
  - paper: yang-2019-xlnet
    locator: p. 3
  - paper: yang-2019-xlnet
    locator: p. 6
---

## Statement

The lowest expected masked LM loss for the pair is H(A | C) + H(B | C), reached when q is the true conditional of each
position. This exceeds the joint entropy H(A, B | C), the best a model of the pair could do, by the conditional mutual
information I(A; B | C) ≥ 0, with equality exactly when the two masked tokens are independent given the context. In
XLNet's example, where "[MASK] [MASK] is a city" is "New York" or "Los Angeles" with equal probability, the gap is
log 2 nats: the masked LM loss is at best 2 log 2 while the joint needs only log 2, and the product of the best
predictions puts probability 1/4 on each of "New Angeles" and "Los York".

## Assumptions

The model class is rich enough to represent any conditional q(· | c).

## Steps

1. For fixed c the expected loss is E[−log q(a | c)] + E[−log q(b | c)], two separate cross-entropies. By Gibbs'
   inequality each is at least the entropy of the true conditional of its own position, with equality when q equals
   it, so the minimum is H(A | C) + H(B | C).
2. The chain rule gives H(A, B | C) = H(A | C) + H(B | A, C), so the difference is H(B | C) − H(B | A, C) =
   I(A; B | C), which is non-negative and zero exactly under conditional independence.
3. In the example the two tokens are a fair coin each, H(A) = H(B) = log 2, and the pair is one fair coin,
   H(A, B) = log 2. The best q puts 1/2 on each first word and 1/2 on each second word, so its product gives 1/4 to
   each of the four pairs, half of it to the two that never occur.

## Result and check

The check minimises the pair loss over q by gradient descent on random joint distributions and compares the minimum
with H(A) + H(B), checks the gap against the mutual information computed directly, and reproduces the example with the
NumPy reference's independence_gap.
