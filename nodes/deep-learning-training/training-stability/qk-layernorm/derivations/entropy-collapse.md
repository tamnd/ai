---
id: entropy-collapse
derives: [c-entropy-monotone, c-entropy-collapse, c-entropy-ties]
kind: limit
assumes: ["the logits z are a fixed vector of n >= 2 real numbers", "the scale c is at least 0"]
uses: [shannon-entropy@m2]
status: full
check: entropy-collapse.check.py
lean: null
sources:
  - {work: dehghani-2023-scaling-vision-transformers-22b, locator: "p. 2"}
---

### Statement

Let $\mathbf{z} \in \mathbb{R}^n$, $c \ge 0$, $\mathbf{p}(c) = \mathrm{softmax}(c\mathbf{z})$ and $H(c) = -\sum_i p_i \log p_i$, the entropy in nats as in eq-entropy.

1. $H(0) = \log n$ and $\frac{dH}{dc} = -c\, \mathrm{Var}_{\mathbf{p}}(\mathbf{z}) \le 0$, so the entropy never grows as the logits are scaled up.
2. If $\mathbf{z}$ has a unique largest entry $z_m$ with gap $\Delta = z_m - \max_{i \ne m} z_i > 0$, then $1 - p_m(c) \le (n - 1)e^{-c\Delta}$, so $\mathbf{p}(c)$ tends to the one-hot vector $\mathbf{e}_m$ and $H(c) \to 0$ as $c \to \infty$.
3. If $k$ entries tie for the largest value, $\mathbf{p}(c)$ tends to the uniform distribution on them and $H(c) \to \log k$.

### Assumptions

1. $\mathbf{z}$ is fixed and $n \ge 2$. Used in every step.
2. $c \ge 0$. Used in step 3 for the sign of the derivative.

### Steps

1. Let $Z(c) = \sum_j e^{c z_j}$. Then $\log p_i = c z_i - \log Z(c)$, so $H(c) = -\sum_i p_i (c z_i - \log Z) = \log Z(c) - c\, \mathbb{E}_{\mathbf{p}}[\mathbf{z}]$, because $\sum_i p_i = 1$. At $c = 0$, $Z = n$ and $H = \log n$.
2. Differentiating, $\frac{d}{dc} \log Z = \frac{1}{Z}\sum_j z_j e^{c z_j} = \mathbb{E}_{\mathbf{p}}[\mathbf{z}]$, and $\frac{dp_i}{dc} = p_i (z_i - \mathbb{E}_{\mathbf{p}}[\mathbf{z}])$ by the quotient rule, so $\frac{d}{dc}\mathbb{E}_{\mathbf{p}}[\mathbf{z}] = \sum_i z_i p_i (z_i - \mathbb{E}_{\mathbf{p}}[\mathbf{z}]) = \mathrm{Var}_{\mathbf{p}}(\mathbf{z})$.
3. By the product rule on step 1, $\frac{dH}{dc} = \mathbb{E}_{\mathbf{p}}[\mathbf{z}] - \mathbb{E}_{\mathbf{p}}[\mathbf{z}] - c\, \mathrm{Var}_{\mathbf{p}}(\mathbf{z}) = -c\, \mathrm{Var}_{\mathbf{p}}(\mathbf{z})$. A variance is at least 0 and $c \ge 0$ by assumption 2, so the derivative is at most 0. This is part 1.
4. Divide the numerator and the denominator of $p_m$ by $e^{c z_m}$: $p_m = 1 / (1 + \sum_{i \ne m} e^{-c(z_m - z_i)})$. Each $z_m - z_i \ge \Delta$, so each term of the sum is at most $e^{-c\Delta}$, and $1 - p_m = \frac{S}{1 + S} \le S \le (n - 1)e^{-c\Delta}$, where $S$ is the sum. As $c \to \infty$, $e^{-c\Delta} \to 0$, so $p_m \to 1$ and every other $p_i \to 0$.
5. Write $\varepsilon = 1 - p_m$. By the grouping rule of entropy, $H = \phi(\varepsilon) + \varepsilon\, H(\mathbf{r})$, where $\phi(\varepsilon) = -\varepsilon\log\varepsilon - (1 - \varepsilon)\log(1 - \varepsilon)$ and $\mathbf{r}$ is $\mathbf{p}$ without entry $m$, renormalized. The entropy of $n - 1$ outcomes is at most $\log(n - 1)$, so $H \le \phi(\varepsilon) + \varepsilon \log(n - 1)$. Both terms tend to 0 as $\varepsilon \to 0$, by step 4 and because $\varepsilon \log \varepsilon \to 0$. This is part 2.
6. With $k$ tied maxima, step 4 applies to each of them against the others: the tied entries have equal weights, the rest tends to 0, so each tied weight tends to $1/k$ and $H \to \log k$. This is part 3.

### Result and check

Scaling the logits of one attention row by $c$ lowers its entropy, from $\log n$ at $c = 0$ toward 0, and the row tends to a one-hot vector on its largest logit.
This is the attention entropy collapse that large logits cause.
For $\mathbf{z} = (2, 0, -2)$, our own run of the check gives $H = 1.0986$ nats at $c = 0$, $0.4411$ at $c = 1$, $0.0930$ at $c = 2$, $0.0030$ at $c = 4$ and $1.9 \times 10^{-6}$ at $c = 8$.

`entropy-collapse.check.py` draws 20 logit vectors, checks that $H$ falls on a grid of 61 values of $c$ from 0 to $10^3$, compares $dH/dc$ by central differences with $-c\,\mathrm{Var}_{\mathbf{p}}(\mathbf{z})$, checks the bounds of steps 4 and 5, and checks $H \to \log 3$ for three tied maxima.
The test `test_entropy_collapses_as_logits_scale` of `impl/numpy` checks the same fall on the code of the page.
