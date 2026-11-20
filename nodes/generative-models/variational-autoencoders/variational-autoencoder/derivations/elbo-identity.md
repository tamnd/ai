---
id: elbo-identity
derives: [c-elbo, c-elbo-split, c-elbo-linear]
kind: identity
assumes: ["a joint density p(x, z) = p(z) p(x|z) with p(x) > 0", "an encoder q(z|x) that is positive wherever p(z|x) is"]
uses: [kullback-leibler-divergence@m2, bayes-theorem@m2, multivariate-normal-distribution@m2, evidence-lower-bound@al]
status: full
check: elbo-identity.check.py
lean: null
sources:
  - {work: kingma-2013-auto-encoding-variational-bayes, locator: "p. 3", note: "eq. 1, the log marginal likelihood as the KL to the true posterior plus the bound, and eqs. 2 and 3"}
  - {work: rezende-2014-stochastic-backpropagation-approximate, locator: "p. 4", note: "eq. 11, the free energy as a KL term and a reconstruction term"}
---

### Statement

1. For every $x$, $\log p(x) = \mathcal{L}(x) + \mathrm{KL}\big(q(z \mid x) \,\Vert\, p(z \mid x)\big)$ with $\mathcal{L}(x) = \mathbb{E}_{q(z \mid x)}\big[\log p(x, z) - \log q(z \mid x)\big]$. Since the KL divergence is nonnegative, $\mathcal{L}(x) \le \log p(x)$, with equality exactly when $q(z \mid x) = p(z \mid x)$.
2. The same bound is $\mathcal{L}(x) = \mathbb{E}_{q(z \mid x)}[\log p(x \mid z)] - \mathrm{KL}\big(q(z \mid x) \,\Vert\, p(z)\big)$, an expected reconstruction log-likelihood minus a KL term to the prior.
3. For the linear Gaussian model $p(z) = \mathcal{N}(0, I)$, $p(x \mid z) = \mathcal{N}(Wz + b, s^2 I)$, the posterior is $\mathcal{N}\big(M^{-1} W^\top (x - b) / s^2, M^{-1}\big)$ with $M = I + W^\top W / s^2$, every term of statements 1 and 2 has a closed form, and an encoder that outputs exactly that mean and covariance makes the bound equal to $\log p(x) = \log \mathcal{N}(x; b, W W^\top + s^2 I)$.

### Assumptions

1. $p(x) > 0$, so that $p(z \mid x) = p(x, z) / p(x)$ is defined. Used in step 1.
2. $q(z \mid x) > 0$ where $p(z \mid x) > 0$, so the KL divergence is finite. Used in step 1.
3. The linear Gaussian model in step 3.

### Steps

1. By Bayes' theorem $\log p(z \mid x) = \log p(x, z) - \log p(x)$. Then $\mathrm{KL}(q \Vert p(z \mid x)) = \mathbb{E}_q[\log q - \log p(x, z)] + \log p(x) = -\mathcal{L}(x) + \log p(x)$, because $\log p(x)$ does not depend on $z$. Rearranging gives statement 1; the KL divergence is nonnegative and zero only when the two densities agree.
2. Write $\log p(x, z) = \log p(x \mid z) + \log p(z)$ inside $\mathcal{L}$: $\mathbb{E}_q[\log p(z) - \log q]$ is $-\mathrm{KL}(q \Vert p(z))$. This is statement 2.
3. $\log p(x, z)$ is quadratic in $z$ with the quadratic part $-\frac12 z^\top M z$ and the linear part $z^\top W^\top (x - b) / s^2$, so completing the square gives the Gaussian posterior. Integrating $z$ out of a linear Gaussian model gives $x \sim \mathcal{N}(b, W W^\top + s^2 I)$. For $q = \mathcal{N}(m, C)$, $\mathbb{E}_q[\lVert x - Wz - b \rVert^2] = \lVert x - Wm - b \rVert^2 + \operatorname{tr}(W C W^\top)$, so the expected reconstruction term is $-\frac{D}{2}\log(2\pi s^2) - \frac{1}{2 s^2}\big(\lVert x - Wm - b \rVert^2 + \operatorname{tr}(W C W^\top)\big)$, and the KL divergence between two Gaussians is in closed form. With $m$ and $C$ the posterior mean and covariance the KL to the posterior is zero, so statement 1 gives $\mathcal{L} = \log p(x)$.

### Result and check

$$ \log p(x) = \mathbb{E}_{q}[\log p(x \mid z)] - \mathrm{KL}\big(q(z \mid x) \,\Vert\, p(z)\big) + \mathrm{KL}\big(q(z \mid x) \,\Vert\, p(z \mid x)\big) . $$

The check `elbo-identity.check.py` builds random linear Gaussian models, evaluates the three terms in closed form for random Gaussian encoders, diagonal or not, and checks the identity, the bound and its equality at the posterior; it also checks the closed-form reconstruction term against a Monte Carlo average, and the bound of a diagonal encoder against the reference's estimator.
