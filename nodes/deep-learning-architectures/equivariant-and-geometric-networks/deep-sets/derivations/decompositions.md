---
id: decompositions
derives: [c-examples, c-second-largest]
kind: calculation
assumes: ["sets of real numbers", "for the limits, finite sets whose largest element is unique"]
uses: []
status: full
check: decompositions.check.py
lean: null
sources:
  - {work: zaheer-2017-deep-sets, locator: "p. 16", note: "the examples at the end of Appendix A.2"}
---

### Statement

1. With $\phi(x) = (x, x^2, x^3)$ and $\rho(u, v, w) = uv - w + \tfrac{3}{2}(u^2 - v)$, $\rho(\phi(x_1) + \phi(x_2)) = x_1 x_2 (x_1 + x_2 + 3)$.
2. With $\phi(x) = (1, x)$ and $\rho(u, v) = v/u$, $\rho(\sum_x \phi(x))$ is the mean of the set.
3. With $\phi(x) = (e^{\alpha x}, x e^{\alpha x})$ and $\rho(u, v) = v/u$, $\rho(\sum_x \phi(x)) = \sum_x x\, e^{\alpha x} / \sum_x e^{\alpha x}$, which tends to $\max X$ as $\alpha \to \infty$.
4. The expression $\rho(u, v) = \big(v - (v/u)e^{\alpha v/u}\big) / \big(u - e^{\alpha v/u}\big)$ equals $v/u$ wherever $u \neq 0$ and $u \neq e^{\alpha v/u}$. With the $\phi$ of item 3 it therefore tends to the largest element, not the second largest.

### Assumptions

1. In item 3 and 4, the largest element of the finite set $X$ is unique. Used in step 3.

### Steps

1. Let $p_k = x_1^k + x_2^k$. Then $u v - w = p_1 p_2 - p_3 = x_1 x_2^2 + x_1^2 x_2 = x_1 x_2 (x_1 + x_2)$, and $\tfrac{3}{2}(p_1^2 - p_2) = \tfrac{3}{2} \cdot 2 x_1 x_2 = 3 x_1 x_2$. The sum is $x_1 x_2 (x_1 + x_2 + 3)$.
2. $\sum_x \phi(x) = (|X|, \sum_x x)$, and the ratio is the mean.
3. Dividing numerator and denominator by $e^{\alpha \max X}$, every other term carries a factor $e^{-\alpha(\max X - x)} \to 0$, so the ratio tends to $\max X$.
4. Write $t = v/u$ and $E = e^{\alpha t}$. Then $v - tE = ut - tE = t(u - E)$, so the expression is $t(u - E)/(u - E) = t = v/u$ wherever $u - E \neq 0$. By step 3 it tends to the maximum.

### Result and check

$$ \frac{v - (v/u) e^{\alpha v/u}}{u - e^{\alpha v/u}} = \frac{v}{u} . $$

Item 4 is a slip in the appendix, not a flaw of the theory: the second largest element of a set of fixed size is a continuous symmetric function, so it has decompositions, for example through the power sums of the derivation `power-sums`. The check `decompositions.check.py` evaluates every item on random sets, and evaluates the expression of item 4 on $\{0.3, 0.9, 0.5, 0.1\}$ in decimal arithmetic with 600 digits for $\alpha$ from 1 to 1000, since in floating point the denominator cancels; it approaches 0.9, the largest element, while the second largest is 0.5.
