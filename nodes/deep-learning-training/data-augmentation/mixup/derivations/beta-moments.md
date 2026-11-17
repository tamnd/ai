---
id: beta-moments
derives: [c-beta-moments, eq-beta-moments]
kind: identity
assumes: ["λ ∼ Beta(α, α) with α > 0"]
uses: [beta-distribution@m2, variance@m2]
status: full
check: beta-moments.check.py
lean: null
sources:
  - {work: zhang-2018-mixup, locator: "Section 2"}
  - {work: zhang-2018-mixup, locator: "p. 5"}
---

### Statement

Let $\lambda \sim \mathrm{Beta}(\alpha, \alpha)$ with $\alpha > 0$. Then

1. $\mathbb E[\lambda] = 1/2$ and $\operatorname{Var}[\lambda] = 1/(4(2\alpha + 1))$;
2. as $\alpha \to 0$, $\lambda$ converges in distribution to the law that puts mass $1/2$ on each of 0 and 1, so the mixed example $\lambda x_i + (1 - \lambda) x_j$ tends to one of the two training examples with its own label, as in ERM;
3. as $\alpha \to \infty$, $\lambda \to 1/2$ in probability, so every mixed example tends to the even blend of its pair;
4. $\alpha = 1$ gives the uniform distribution on $[0, 1]$, with variance $1/12$; $\alpha = 0.2$ gives $1/5.6 \approx 0.178571$.

### Assumptions

1. $\alpha > 0$, so that the density $\lambda^{\alpha - 1}(1 - \lambda)^{\alpha - 1}/B(\alpha, \alpha)$ is integrable.
2. In items 2 and 3, "tends to ERM" and "tends to the even blend" refer to the distribution of $\lambda$ only; the pair is drawn the same way for every $\alpha$.

### Steps

1. The density is symmetric under $\lambda \mapsto 1 - \lambda$, so $\lambda$ and $1 - \lambda$ have the same law and $\mathbb E[\lambda] = \mathbb E[1 - \lambda]$, which gives $1/2$. For $\mathrm{Beta}(a, b)$, $\mathbb E[\lambda^2] = a(a + 1)/((a + b)(a + b + 1))$; with $a = b = \alpha$ this is $(\alpha + 1)/(2(2\alpha + 1))$. Then $\operatorname{Var}[\lambda] = (\alpha + 1)/(2(2\alpha + 1)) - 1/4 = (2\alpha + 2 - 2\alpha - 1)/(4(2\alpha + 1)) = 1/(4(2\alpha + 1))$.
2. Any law on $[0, 1]$ with mean $1/2$ has $\operatorname{Var}[\lambda] = \mathbb E[\lambda^2] - 1/4 \le \mathbb E[\lambda] - 1/4 = 1/4$, with equality only if $\lambda^2 = \lambda$ almost surely, that is $\lambda \in \{0, 1\}$. As $\alpha \to 0$ the variance tends to $1/4$, so $\mathbb E[\lambda(1 - \lambda)] = 1/4 - \operatorname{Var}[\lambda] \to 0$. For $\delta \in (0, 1/2)$, $P(\delta < \lambda < 1 - \delta) \le \mathbb E[\lambda(1 - \lambda)]/(\delta(1 - \delta)) \to 0$, and by symmetry the remaining mass splits equally between $[0, \delta]$ and $[1 - \delta, 1]$.
3. By Chebyshev, $P(|\lambda - 1/2| > \delta) \le 1/(4\delta^2(2\alpha + 1)) \to 0$ as $\alpha \to \infty$.
4. With $\alpha = 1$ the density is $\lambda^0(1 - \lambda)^0/B(1, 1) = 1$ on $[0, 1]$. The variance formula gives $1/(4 \cdot 3) = 1/12$ at $\alpha = 1$ and $1/(4(2 \cdot 0.2 + 1)) = 1/5.6 \approx 0.178571$ at $\alpha = 0.2$.

### Result and check

$$ \mathbb E[\lambda] = \tfrac12 , \qquad \operatorname{Var}[\lambda] = \frac{1}{4(2\alpha + 1)} . $$

The check `beta-moments.check.py` draws $10^6$ samples of $\lambda$ for $\alpha \in \{0.01, 0.1, 0.2, 0.4, 1, 2, 8, 32, 1000\}$ and compares the sample mean and variance with item 1 to Monte Carlo accuracy.
For $\alpha = 0.01$ it checks that at least 90% of the draws lie within 0.01 of 0 or 1, split about evenly, and for $\alpha = 1000$ that all lie within 0.07 of $1/2$.
It checks the closed form of $\mathbb E[\lambda^2]$ by numerical integration of the density, and that the sample distribution at $\alpha = 1$ is uniform by a Kolmogorov-Smirnov distance.
