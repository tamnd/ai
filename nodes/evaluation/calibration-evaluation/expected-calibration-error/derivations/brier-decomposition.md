---
id: brier-decomposition
derives: [c-brier-population, c-brier-sample, c-ece-le-rms, eq-brier]
kind: identity
assumes: ["the confidence C is in [0, 1] and the correctness A is 0 or 1", "r(c) = P(A = 1 | C = c)", "bins are disjoint and cover [0, 1]"]
uses: [conditional-expectation@m2, law-of-total-expectation@m2, cauchy-schwarz-inequality-for-expectations@m2]
status: full
check: brier-decomposition.check.py
lean: null
sources:
  - {work: guo-2017-on-calibration-of-modern-neural-networks, locator: "equation 1"}
  - {work: kadavath-2022-language-models-mostly-know-what-they-know, locator: "p. 5"}
---

### Statement

Let $C$ be the confidence of the top-label prediction, $A = 1$ when it is correct and $0$ otherwise, and $r(c) = P(A = 1 \mid C = c)$. Perfect calibration (eq-perfect) is $r(C) = C$.

1. Population: $\mathbb{E}[(C - A)^2] = \mathbb{E}[(C - r(C))^2] + \mathbb{E}[r(C)(1 - r(C))]$. The first term is zero exactly when the model is calibrated; the second does not change when $C$ is replaced by $r(C)$.
2. Sample: for $n$ predictions in bins $B_1, \dots, B_M$ with $n_m$ members, mean confidence $\bar c_m$ and accuracy $\bar a_m$,
$$ \frac{1}{n}\sum_{i=1}^n (c_i - a_i)^2 = \sum_m \frac{n_m}{n}\Big[(\bar c_m - \bar a_m)^2 + \bar a_m(1 - \bar a_m) + s^2_m(c) - 2 s_m(c, a)\Big], $$
where $s^2_m(c)$ is the within-bin variance of the confidences and $s_m(c, a)$ their within-bin covariance with the correctness. When every bin holds a single confidence value, the last two terms vanish.
3. With the same bins, the ECE is at most the root-mean-square calibration error: $\sum_m \frac{n_m}{n}|\bar a_m - \bar c_m| \le \big(\sum_m \frac{n_m}{n}(\bar a_m - \bar c_m)^2\big)^{1/2}$.

### Assumptions

1. $C \in [0, 1]$ and $A \in \{0, 1\}$. Used in steps 1 and 3, where $A^2 = A$.
2. $r(c)$ is the conditional probability of a correct prediction given the confidence. Used in step 1.
3. The bins are disjoint and every prediction falls in one. Used in steps 2 and 4.

### Steps

1. Write $C - A = (C - r(C)) + (r(C) - A)$ and square. By the law of total expectation, $\mathbb{E}[(C - r(C))(r(C) - A)] = \mathbb{E}\big[(C - r(C))\,\mathbb{E}[r(C) - A \mid C]\big] = 0$, because $\mathbb{E}[A \mid C] = r(C)$.
2. Also $\mathbb{E}[(r(C) - A)^2 \mid C] = \mathrm{Var}(A \mid C) = r(C)(1 - r(C))$, since $A$ is Bernoulli given $C$. Taking expectations gives part 1. The term $\mathbb{E}[(C - r(C))^2]$ is the squared calibration error; $\mathbb{E}[r(C)(1 - r(C))]$ depends only on how well the confidence separates right from wrong answers.
3. In bin $m$ write $c_i - a_i = (\bar c_m - \bar a_m) + (c_i - \bar c_m) - (a_i - \bar a_m)$. Squaring and averaging over the bin, the cross terms with $\bar c_m - \bar a_m$ vanish because deviations from a mean sum to zero, which leaves $(\bar c_m - \bar a_m)^2 + s^2_m(c) + \frac{1}{n_m}\sum (a_i - \bar a_m)^2 - 2 s_m(c, a)$. Since $a_i^2 = a_i$, $\frac{1}{n_m}\sum (a_i - \bar a_m)^2 = \bar a_m - \bar a_m^2 = \bar a_m(1 - \bar a_m)$.
4. Weighting each bin by $n_m/n$ and summing over the bins gives part 2. When all members of a bin share one confidence, $c_i = \bar c_m$, so $s^2_m(c) = s_m(c, a) = 0$.
5. The weights $n_m/n$ are non-negative and sum to 1, so by the Cauchy-Schwarz inequality (or Jensen's inequality for the square) $\big(\sum_m w_m |g_m|\big)^2 \le \sum_m w_m g_m^2$ for the gaps $g_m = \bar a_m - \bar c_m$. Taking square roots gives part 3.

### Result and check

The Brier score of the top-label prediction splits into a calibration term, which is the squared binned gap of the reliability diagram, and a refinement term, plus within-bin terms that vanish when the bins are fine enough to hold one confidence each. The ECE is never above the RMS calibration error computed with the same bins.
The check `brier-decomposition.check.py` verifies part 1 in exact rationals for 300 random discrete distributions of $(C, r(C))$, part 2 and part 3 for 300 random samples with up to 40 predictions and 15 bins, and the vanishing of the within-bin terms when the confidences are rounded up to the bin edges; it prints ok.
