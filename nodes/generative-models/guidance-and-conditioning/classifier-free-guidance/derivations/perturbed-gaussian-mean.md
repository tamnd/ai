---
id: perturbed-gaussian-mean
derives: [c-perturbed-mean]
kind: closed-form
assumes: ["the reverse step is Gaussian, N(mu, Sigma) with Sigma positive definite", "log p(y | x_t) is replaced by its first-order Taylor expansion at mu"]
uses: [multivariate-normal-distribution@m2, bayes-theorem@m2]
status: full
check: perturbed-gaussian-mean.check.py
lean: null
sources:
  - {work: dhariwal-2021-diffusion-models-beat-gans, locator: "pp. 6 to 7, Eqs. 2 to 10", note: "the perturbed transition, the Taylor expansion of the classifier at mu and the completed square with mean mu + Sigma g"}
---

### Statement

Let the unguided reverse step be $p_\theta(x_t \mid x_{t+1}) = \mathcal{N}(\mu, \Sigma)$ and let $g = \nabla_{x_t} \log p_\phi(y \mid x_t)\big|_{x_t = \mu}$. If $\log p_\phi(y \mid x_t) \approx \log p_\phi(y \mid \mu) + (x_t - \mu)^\top g$, then

$$ p_\theta(x_t \mid x_{t+1})\, p_\phi(y \mid x_t)^s \;\propto\; \mathcal{N}\big(x_t;\ \mu + s\,\Sigma g,\ \Sigma\big) , $$

which is Dhariwal and Nichol's Algorithm 1 with gradient scale $s$. With a linear log-classifier the statement is exact.

### Assumptions

1. A Gaussian reverse step with positive definite $\Sigma$. Used in step 1.
2. The first-order expansion of the log-classifier, accurate when $\Sigma$ is small, that is at small noise steps, or exact when the log-classifier is linear. Used in step 1.

### Steps

1. Up to constants in $x_t$, $\log\big(p_\theta\, p_\phi^s\big) = -\tfrac12 (x_t - \mu)^\top\Sigma^{-1}(x_t - \mu) + s(x_t - \mu)^\top g$.
2. Completing the square with $\Sigma^{-1}\Sigma g = g$: this equals $-\tfrac12 (x_t - \mu - s\Sigma g)^\top \Sigma^{-1}(x_t - \mu - s\Sigma g) + \tfrac{s^2}{2} g^\top\Sigma g$, and the last term does not depend on $x_t$.
3. A density proportional to the exponential of that quadratic is $\mathcal{N}(\mu + s\Sigma g, \Sigma)$.

### Result and check

Guidance moves the mean of every reverse step along $\Sigma g$, the classifier gradient seen through the step's covariance, and leaves the covariance alone; a larger $s$ moves it further.
The sampled distribution at scale $s$ is, step by step, proportional to $p\,p(y \mid x)^s$, which for $s > 1$ is sharper than $p(x \mid y)$ (Dhariwal and Nichol, p. 8).
The check `perturbed-gaussian-mean.check.py` verifies with `perturbed_mean` that $\log\mathcal{N}(x; \mu, \Sigma) + s(x - \mu)^\top g - \log\mathcal{N}(x; \mu + s\Sigma g, \Sigma)$ is the same constant $\tfrac{s^2}{2}g^\top\Sigma g$ at many random points for a random positive definite $\Sigma$ and $s$ of 0.5, 1 and 4, and that for $s \le 1$ importance weights $p_\phi(y \mid x)^s$ with an exactly linear log-classifier move the mean of Gaussian samples to $\mu + s\Sigma g$.
