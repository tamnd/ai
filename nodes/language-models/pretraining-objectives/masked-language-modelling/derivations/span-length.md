---
id: span-length
derives: [c-span-mean]
kind: closed-form
assumes:
  - Span lengths follow a geometric distribution P(l) = p(1 − p)^(l − 1) on l = 1, 2, ..., truncated at l_max and renormalised.
uses: []
status: full
check: span-length.check.py
lean: null
sources:
  - paper: joshi-2019-spanbert
    locator: p. 3
---

## Statement

With p = 0.2 and l_max = 10 the truncated geometric distribution has mean 3.797, which matches the mean span length
of 3.8 that Joshi et al. report. Capping the lengths instead (setting every l > 10 to 10) would give 4.463, and the
untruncated distribution has mean 1/p = 5. The truncation removes a mass of 0.8^10 = 0.107.

## Assumptions

The lengths are counted in words, as in SpanBERT; the formula does not depend on the unit.

## Steps

1. With q = 1 − p, the mass kept by the truncation is Σ_{l=1}^{L} p q^(l − 1) = 1 − q^L.
2. The partial mean is Σ_{l=1}^{L} l p q^(l − 1) = (1 − (L + 1) q^L + L q^(L + 1)) / p, from differentiating the
   geometric sum Σ q^l with respect to q.
3. The truncated mean is the ratio: (1 − (L + 1) q^L + L q^(L + 1)) / (p (1 − q^L)). At p = 0.2 and L = 10:
   q^10 = 0.1073741824, the numerator is 1 − 11 × 0.1073741824 + 10 × 0.08589934592 = 0.67779, and the mean is
   0.67779 / (0.2 × 0.89263) = 3.797.
4. Capping instead gives E[min(l, L)] = Σ_{k=0}^{L−1} P(l > k) = Σ_{k=0}^{L−1} q^k = (1 − q^L)/p = 4.463.

## Result and check

The check computes both means with exact fractions, compares the closed form with the direct sum, and compares the
truncated mean with the mean of the NumPy reference's truncated_geometric distribution.
