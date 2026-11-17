---
id: temperature-scaling
derives: [c-ts-argmax, c-ts-confidence, c-ts-convex, c-ts-stationary, eq-temperature, eq-stationary]
kind: identity
assumes: ["the logits z_i are finite real vectors of length K >= 2", "T > 0, written beta = 1/T", "the predicted class is the first index of the largest logit"]
uses: [cross-entropy@m2, convex-optimization@m2, maximum-likelihood-estimation@m2]
status: full
check: temperature-scaling.check.py
lean: null
sources:
  - {work: guo-2017-on-calibration-of-modern-neural-networks, locator: "equation 9"}
  - {work: guo-2017-on-calibration-of-modern-neural-networks, locator: "p. 8"}
---

### Statement

Let $z \in \mathbb{R}^K$ be the logits of one example and $q(\beta) = \mathrm{softmax}(\beta z)$ with $\beta = 1/T > 0$, so that $q_k(\beta) = e^{\beta z_k} / \sum_j e^{\beta z_j}$.
For $n$ examples with logits $z_i$ and labels $y_i$, let $L(\beta) = \frac{1}{n} \sum_i \big(\log \sum_k e^{\beta z_{ik}} - \beta z_{i y_i}\big)$ be the mean negative log-likelihood.

1. For every $T > 0$, $q_j > q_k$ if and only if $z_j > z_k$. So the predicted class, and with it the accuracy on any labelled set, is the same for every $T$.
2. The confidence $\max_k q_k$ does not increase in $T$; it tends to $1/K$ as $T \to \infty$, and to $1$ as $T \to 0$ when the largest logit is unique.
3. $L$ is convex in $\beta$, with $L'(\beta) = \frac{1}{n} \sum_i \big(\mathbb{E}_{q_i(\beta)}[z_i] - z_{i y_i}\big)$ and $L''(\beta) = \frac{1}{n} \sum_i \mathrm{Var}_{q_i(\beta)}[z_i] \ge 0$.
4. At an interior minimizer $\beta^*$, the mean logit of the true class equals the mean of the logits weighted by the fitted probabilities: $\frac{1}{n}\sum_i z_{i y_i} = \frac{1}{n}\sum_i \mathbb{E}_{q_i(\beta^*)}[z_i]$.

### Assumptions

1. The logits are finite. Used in every step.
2. $T > 0$, so $\beta > 0$. Used in steps 1 and 2.
3. Ties in the largest logit are broken by a fixed rule that looks only at the order of the entries, such as the first index. Used in step 1.

### Steps

1. The map $x \mapsto e^{\beta x}$ is strictly increasing when $\beta > 0$, and every $q_k$ shares the same positive denominator. So $q_j > q_k \iff e^{\beta z_j} > e^{\beta z_k} \iff z_j > z_k$, and equal logits give equal probabilities. The order of the entries, ties included, is the order of the logits, so a rule that picks the class from that order picks the same class for $q$ and for $z$. The predicted class does not depend on $T$, and neither does the fraction of examples where it equals the label. This is part 1.
2. Let $z_{\max}$ be the largest logit. Then $\max_k q_k = 1 / \sum_j e^{\beta (z_j - z_{\max})}$. Each exponent $\beta (z_j - z_{\max})$ is at most 0, so each term does not increase as $\beta$ grows, nor does the sum; the confidence does not decrease in $\beta$, that is, it does not increase in $T$. As $\beta \to 0$ every term tends to 1 and the confidence to $1/K$. As $\beta \to \infty$ the terms with $z_j < z_{\max}$ tend to 0 and the term $j = \arg\max$ stays 1, so the confidence tends to 1 when the maximum is unique. This is part 2.
3. Write $A_i(\beta) = \log \sum_k e^{\beta z_{ik}}$. Differentiating, $A_i'(\beta) = \sum_k z_{ik} e^{\beta z_{ik}} / \sum_j e^{\beta z_{ij}} = \mathbb{E}_{q_i(\beta)}[z_i]$, and differentiating again, $A_i''(\beta) = \mathbb{E}_{q_i}[z_i^2] - \mathbb{E}_{q_i}[z_i]^2 = \mathrm{Var}_{q_i}[z_i] \ge 0$. The term $-\beta z_{i y_i}$ is linear in $\beta$. Averaging over $i$ gives $L'$ and $L''$ of part 3, and $L'' \ge 0$ makes $L$ convex in $\beta$. The search for the temperature is a one-dimensional convex problem, and $L'$ does not decrease in $\beta$, so bisection on the sign of $L'$ finds the minimizer.
4. At an interior minimizer of a differentiable convex function the derivative is zero: $L'(\beta^*) = 0$. Writing this out with step 3 gives part 4, which is eq-stationary.

### Result and check

Temperature scaling changes the confidence of every prediction but never its class, so the accuracy is fixed and only the calibration moves; fitting $T$ by validation NLL is a convex problem in $1/T$ whose optimum matches the mean true-class logit to its expectation under the model.
The check `temperature-scaling.check.py` tests part 1 on 500 random logit matrices and six temperatures each, part 2 on the same, the slope and curvature of part 3 against central differences and convexity on a grid, the limits of part 2, and part 4 at a bisection optimum on 2,000 examples whose labels were drawn at $T = 2$; it prints ok.
The tests `test_temperature_keeps_argmax_and_accuracy`, `test_temperature_limits` and `test_fit_temperature_minimizes_nll` of `impl/` repeat the parts on the implementations.
