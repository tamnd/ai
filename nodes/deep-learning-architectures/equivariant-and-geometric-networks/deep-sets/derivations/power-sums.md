---
id: power-sums
derives: [c-power-sums, c-power-sums-collision]
kind: proof
assumes: ["sets of exactly M real numbers, counted with multiplicity"]
uses: []
status: full
check: power-sums.check.py
lean: null
sources:
  - {work: zaheer-2017-deep-sets, locator: "pp. 14 to 15", note: "Theorem 7 and the map phi(x) = (1, x, ..., x^M)"}
  - {work: zaheer-2017-deep-sets, locator: "p. 12", note: "Lemma 4, the power sums as a sum-decomposition"}
---

### Statement

For a multiset $X$ of $M$ real numbers, $\sum_{x \in X} (1, x, x^2, \dots, x^M) = (M, p_1, \dots, p_M)$ with $p_k = \sum_x x^k$, and the power sums $p_1, \dots, p_M$ determine $X$. The first $M - 1$ power sums do not: $\{1, 5, 6\}$ and $\{2, 3, 7\}$ share $p_1 = 12$ and $p_2 = 62$.

### Assumptions

1. $|X| = M$, read from the first coordinate. Used in step 2.

### Steps

1. Let $e_k$ be the elementary symmetric polynomials of $X$, so that $\prod_{x \in X}(t - x) = \sum_{k=0}^{M} (-1)^k e_k t^{M-k}$.
2. The Newton-Girard formulae give $k e_k = \sum_{i=1}^{k} (-1)^{i-1} e_{k-i} p_i$ for $1 \leq k \leq M$, with $e_0 = 1$. Over the reals one can divide by $k$, so $p_1, \dots, p_M$ determine $e_1, \dots, e_M$ one after another.
3. The $e_k$ are the coefficients of the polynomial whose roots, with multiplicity, are the elements of $X$; the roots of a polynomial are determined by it. So $X$ is determined.
4. For $M = 3$: $1 + 5 + 6 = 2 + 3 + 7 = 12$ and $1 + 25 + 36 = 4 + 9 + 49 = 62$, while $1 + 125 + 216 = 342 \neq 8 + 27 + 343 = 378$. Two different sets of 3 numbers share their first two power sums, so $M - 1$ power sums do not suffice.

### Result and check

$$ \Big(\textstyle\sum_{x \in X} x^k\Big)_{k = 1}^{M} \text{ determines the multiset } X \text{ of } M \text{ numbers.} $$

This $\phi$ is continuous and its latent space has dimension $M$, which by Wagstaff et al. cannot be made smaller for all continuous functions. The check `power-sums.check.py` recovers random sets of up to 6 numbers from their power sums by step 2 and the roots of the polynomial, recovers $\{1, 2, 4\}$ of the worked example from $(7, 21, 73)$, and checks the collision of step 4.
