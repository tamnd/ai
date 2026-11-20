---
id: gaussian-kl
derives: [c-kl, c-kl-grad, c-kl-full]
kind: closed-form
assumes: ["q(z|x) = N(mu, diag(sigma^2)) over J dimensions", "the prior p(z) = N(0, I)"]
uses: [kullback-leibler-divergence@m2, multivariate-normal-distribution@m2, linearity-of-expectation@m2]
status: full
check: gaussian-kl.check.py
lean: null
sources:
  - {work: kingma-2013-auto-encoding-variational-bayes, locator: "pp. 5, 10 to 11", note: "eq. 10 and appendix B, the KL term of a diagonal Gaussian encoder and a standard normal prior"}
  - {work: rezende-2014-stochastic-backpropagation-approximate, locator: "p. 4", note: "the KL term with a full covariance C, (Tr C - log|C| + mu^T mu - D) / 2"}
---

### Statement

1. $\mathrm{KL}\big(\mathcal{N}(\mu, \operatorname{diag}\sigma^2) \,\Vert\, \mathcal{N}(0, I)\big) = \frac12 \sum_{j=1}^{J} \big(\mu_j^2 + \sigma_j^2 - 1 - \log \sigma_j^2\big)$, which is zero exactly when $\mu = 0$ and $\sigma = 1$.
2. Its gradients are $\partial / \partial \mu_j = \mu_j$ and $\partial / \partial \log\sigma_j^2 = \frac12(\sigma_j^2 - 1)$, so in the log-variance the term pulls every $\sigma_j^2$ toward 1 with a gradient that stays finite as $\sigma_j \to 0$.
3. With a full covariance $C$ the same divergence is $\frac12\big(\operatorname{tr} C - \log\lvert C \rvert + \mu^\top\mu - J\big)$, of which statement 1 is the diagonal case.

### Assumptions

1. The encoder outputs a mean and a positive variance for each of the $J$ dimensions, independent given $x$. Used in steps 1 and 2.
2. The prior is the standard normal. Used in step 1.

### Steps

1. $\log q(z) = -\frac12 \sum_j \big(\log 2\pi + \log \sigma_j^2 + (z_j - \mu_j)^2 / \sigma_j^2\big)$ and $\log p(z) = -\frac12 \sum_j (\log 2\pi + z_j^2)$. Under $q$, $\mathbb{E}[(z_j - \mu_j)^2] = \sigma_j^2$ and $\mathbb{E}[z_j^2] = \mu_j^2 + \sigma_j^2$, so by linearity $\mathbb{E}_q[\log q - \log p] = \frac12 \sum_j (-\log\sigma_j^2 - 1 + \mu_j^2 + \sigma_j^2)$. Each term $t - 1 - \log t$ with $t = \sigma_j^2$ is at least 0, with equality only at $t = 1$, and $\mu_j^2 \ge 0$. This is statement 1; the two halves of the sum are eq. 10 of Kingma and Welling with the sign of the bound.
2. Differentiate statement 1 in $\mu_j$, and write $\sigma_j^2 = e^{v_j}$ to differentiate in $v_j = \log\sigma_j^2$: $\frac{d}{dv}\frac12(e^v - 1 - v) = \frac12(e^v - 1)$, which is $-\frac12$ at $\sigma_j \to 0$. This is statement 2.
3. For a full covariance, $\mathbb{E}_q[(z - \mu)^\top C^{-1} (z - \mu)] = \operatorname{tr}(C^{-1} C) = J$ and $\mathbb{E}_q[z^\top z] = \mu^\top \mu + \operatorname{tr} C$; the log-determinants replace the sum of $\log\sigma_j^2$. This is statement 3.

### Result and check

$$ \mathrm{KL}\big(q(z \mid x) \,\Vert\, p(z)\big) = \frac12 \sum_{j=1}^{J} \big(\mu_j^2 + \sigma_j^2 - 1 - \log \sigma_j^2\big) . $$

The check `gaussian-kl.check.py` compares statement 1 with a numerical integral in one dimension, with a Monte Carlo average of $\log q - \log p$ in several, and with statement 3 at a diagonal $C$; it checks statement 3 against the general Gaussian KL, and the gradients of statement 2 against finite differences of the reference's `kl`.
