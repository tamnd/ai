---
id: optimal-discriminator
derives: [c-dstar]
kind: closed-form
assumes: ["p_data and p_g have densities on the same space", "D ranges over all measurable functions into [0, 1]"]
uses: [cross-entropy@m2, logistic-regression@m2]
status: full
check: optimal-discriminator.check.py
lean: null
sources:
  - {work: goodfellow-2014-generative-adversarial-nets, locator: "p. 4", note: "Proposition 1 and its proof, and the reading of the discriminator's objective as a conditional log-likelihood"}
---

### Statement

For a fixed generator with density $p_g$, the discriminator that maximizes

$$ V(G, D) = \int p_{\text{data}}(\mathbf{x}) \log D(\mathbf{x}) + p_g(\mathbf{x}) \log\big(1 - D(\mathbf{x})\big)\, d\mathbf{x} $$

is $D^*(\mathbf{x}) = p_{\text{data}}(\mathbf{x}) / \big(p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x})\big)$ wherever $p_{\text{data}} + p_g > 0$.
Its logit is $\log p_{\text{data}}(\mathbf{x}) - \log p_g(\mathbf{x})$, and $V(G, D)$ is twice the expected log-likelihood of $D$ as a classifier of the label $y = 1$ for data and $y = 0$ for fakes on an equal mixture of the two.

### Assumptions

1. Both distributions have densities, so $V$ is one integral over $\mathbf{x}$. Used in step 1.
2. $D$ can take any value in $[0, 1]$ at each point independently of the others. Used in step 1.

### Steps

1. Because $D$ is free at every point, $V$ is maximized by maximizing the integrand at each $\mathbf{x}$ separately: the function $y \mapsto a \log y + b \log(1 - y)$ with $a = p_{\text{data}}(\mathbf{x})$ and $b = p_g(\mathbf{x})$.
2. For $a, b \ge 0$ with $a + b > 0$ its derivative $a / y - b / (1 - y)$ vanishes at $y = a / (a + b)$, and its second derivative $-a / y^2 - b / (1 - y)^2$ is negative, so that point is the maximum on $[0, 1]$. This is $D^*$.
3. $\operatorname{logit} D^* = \log\big(D^* / (1 - D^*)\big) = \log(a / b)$.
4. On the mixture $\tfrac12 p_{\text{data}} + \tfrac12 p_g$ with the label of the component, the expected log-likelihood of the classifier $P(y = 1 \mid \mathbf{x}) = D(\mathbf{x})$ is $\tfrac12 \mathbb{E}_{p_{\text{data}}} \log D + \tfrac12 \mathbb{E}_{p_g} \log(1 - D) = \tfrac12 V(G, D)$; the Bayes classifier of the mixture is $D^*$, by Bayes' rule.

### Result and check

$$ D^*_G(\mathbf{x}) = \frac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x})}, \qquad \operatorname{logit} D^*_G(\mathbf{x}) = \log \frac{p_{\text{data}}(\mathbf{x})}{p_g(\mathbf{x})} . $$

The check `optimal-discriminator.check.py` maximizes the integrand on a grid of values of $y$ for random $a$ and $b$, compares $V$ at $D^*$ with $V$ at perturbed discriminators for two Gaussians on a grid, and fits a logistic regression on samples of the mixture whose logit approaches the log-ratio of the densities.
