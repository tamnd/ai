---
id: reparameterization-gradient
derives: [c-reparam, c-reparam-variance]
kind: closed-form
assumes: ["q = N(mu, sigma^2) in one dimension, the case of each coordinate of a diagonal encoder", "f differentiable with E|f'(z)| finite", "f(z) = z^2 in statement 3"]
uses: [linearity-of-expectation@m2, multivariate-normal-distribution@m2, reinforce@al]
status: full
check: reparameterization-gradient.check.py
lean: null
sources:
  - {work: kingma-2013-auto-encoding-variational-bayes, locator: "pp. 3 to 5", note: "the score-function estimator and its very high variance, z = g(eps, x), and the location-scale case z = mu + sigma eps"}
  - {work: rezende-2014-stochastic-backpropagation-approximate, locator: "pp. 3, 8", note: "the Gaussian gradient identities, the coordinate transformation y = mu + R eps, and the variance of the two estimators"}
---

### Statement

1. With $z = \mu + \sigma\varepsilon$ and $\varepsilon \sim \mathcal{N}(0, 1)$, $\frac{\partial}{\partial\mu}\mathbb{E}_{q}[f(z)] = \mathbb{E}_\varepsilon[f'(\mu + \sigma\varepsilon)]$ and $\frac{\partial}{\partial\sigma}\mathbb{E}_{q}[f(z)] = \mathbb{E}_\varepsilon[\varepsilon f'(\mu + \sigma\varepsilon)]$, so $f'(\mu + \sigma\varepsilon)$ and $\varepsilon f'(\mu + \sigma\varepsilon)$ for one draw are unbiased estimates of the two gradients: the reparameterization gradient.
2. The score-function estimate $f(z)\,\partial_\mu \log q(z) = f(z)(z - \mu)/\sigma^2$, with $z \sim q$, is also unbiased for the gradient in $\mu$.
3. For $f(z) = z^2$ the gradient in $\mu$ is $2\mu$; one draw of the reparameterization estimate has variance $4\sigma^2$, and one draw of the score-function estimate has variance $\mu^4/\sigma^2 + 14\mu^2 + 15\sigma^2$. At $\mu = 0$, $\sigma = 1$ the two are 4 and 15, and as $\sigma \to 0$ with $\mu \ne 0$ the first tends to 0 while the second grows without bound.

### Assumptions

1. A location-scale family, here the Gaussian. Used in steps 1 and 3.
2. Differentiation under the expectation is allowed, which holds when $f'$ is integrable under $q$. Used in steps 1 and 2.

### Steps

1. $\mathbb{E}_q[f(z)] = \mathbb{E}_\varepsilon[f(\mu + \sigma\varepsilon)]$, and the distribution of $\varepsilon$ does not depend on $\mu$ or $\sigma$, so the derivative moves inside: $\partial_\mu f(\mu + \sigma\varepsilon) = f'(\mu + \sigma\varepsilon)$ and $\partial_\sigma f(\mu + \sigma\varepsilon) = \varepsilon f'(\mu + \sigma\varepsilon)$. This is statement 1.
2. $\partial_\mu \int q(z) f(z)\, dz = \int q(z)\, \partial_\mu \log q(z)\, f(z)\, dz$ and $\partial_\mu \log q(z) = (z - \mu)/\sigma^2$. This is statement 2, the estimator of REINFORCE.
3. Reparameterization: $f'(z) = 2(\mu + \sigma\varepsilon)$ has variance $4\sigma^2$. Score function: with $z = \mu + \sigma\varepsilon$, $g = (\mu + \sigma\varepsilon)^2 \varepsilon / \sigma = (\mu^2\varepsilon + 2\mu\sigma\varepsilon^2 + \sigma^2\varepsilon^3)/\sigma$. The odd moments of $\varepsilon$ vanish and $\mathbb{E}\varepsilon^2 = 1$, $\mathbb{E}\varepsilon^4 = 3$, $\mathbb{E}\varepsilon^6 = 15$, so $\mathbb{E}g = 2\mu$ and $\mathbb{E}g^2 = (\mu^4 + 4\mu^2\sigma^2 \cdot 3 + \sigma^4 \cdot 15 + 2\mu^2\sigma^2 \cdot 3)/\sigma^2 = \mu^4/\sigma^2 + 18\mu^2 + 15\sigma^2$. Subtracting $(2\mu)^2$ gives statement 3.

### Result and check

$$ \nabla_{\mu,\sigma}\, \mathbb{E}_{q}[f(z)] = \mathbb{E}_{\varepsilon \sim \mathcal{N}(0, 1)}\big[\nabla_{\mu,\sigma}\, f(\mu + \sigma\varepsilon)\big] . $$

The check `reparameterization-gradient.check.py` computes both estimators from samples for several $\mu$ and $\sigma$, checks that their means agree with the exact gradient, checks the variances of statement 3, and checks statement 1 for $f = \sin$, where $\mathbb{E}_q[\sin z] = \sin\mu\, e^{-\sigma^2/2}$.
