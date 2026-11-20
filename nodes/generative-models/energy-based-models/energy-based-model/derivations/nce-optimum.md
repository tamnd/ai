---
id: nce-optimum
derives: [c-nce-optimum]
kind: proof
assumes: ["the expectations are exact, the infinite-sample objective", "the noise density p_n is known and positive wherever q is positive", "f ranges over all measurable functions"]
uses: [logistic-regression@m2, kullback-leibler-divergence@m2, consistent-estimator@m2, normal-distribution@m2, newtons-method-in-optimization@m2]
status: full
check: nce-optimum.check.py
lean: null
sources:
  - {work: gutmann-2010-noise-contrastive-estimation, locator: "pp. 2 to 3", note: "the objective of Eq. 3 as logistic regression, and Theorem 1, the maximum at f = ln p_d with no normalization constraint"}
---

### Statement

For a function $f$ in place of $\log p_m$, let $G_f = f - \log p_n$ and

$$ J(f) = \mathbb{E}_{\mathbf{x} \sim q} \log \sigma(G_f(\mathbf{x})) + \mathbb{E}_{\mathbf{y} \sim p_n} \log\big(1 - \sigma(G_f(\mathbf{y}))\big) . $$

$J$ is maximized by $f = \log q$ on the support of $q$, and if $p_n > 0$ wherever $q > 0$ this is the only maximum. The maximizer $e^f = q$ integrates to 1 although no constraint was imposed, so if $f = -E_\theta + c$ the fitted $c$ is $-\log Z(\theta)$ at the optimum.

### Assumptions

1. The objective is the infinite-sample one. Used in step 1.
2. $p_n > 0$ on the support of $q$. Used in step 3.
3. $f$ is unconstrained. Used in step 2.

### Steps

1. Write both expectations as integrals: $J(f) = \int \big[q \log \sigma(G) + p_n \log(1 - \sigma(G))\big] d\mathbf{x}$, with $\sigma(G) = e^f / (e^f + p_n)$ and $1 - \sigma(G) = p_n / (e^f + p_n)$.
2. Since $f$ is unconstrained, $J$ is maximized pointwise. At each $\mathbf{x}$, with $r = \sigma(G(\mathbf{x})) \in (0, 1)$, the integrand is $q \log r + p_n \log(1 - r)$, a strictly concave function of $r$ when $q, p_n > 0$, with its maximum at $r^* = q/(q + p_n)$.
3. $r^* = e^f/(e^f + p_n)$ gives $e^f = q$. Where $q > 0$ and $p_n > 0$ the maximizer is unique; where $p_n = 0$ the point does not constrain $f$, which is why positivity of the noise is needed.
4. The maximizer is $e^f = q$, a density, so it integrates to 1 without any constraint. With $f = -E_\theta + c$ and a well-specified model, $e^{-E_\theta + c} = q$ forces $e^c = 1/Z(\theta)$.
5. With maximum likelihood instead, the objective $\mathbb{E}_q[-E_\theta + c]$ increases without bound in $c$, so $c$ cannot be learned that way.

### Result and check

$$ \arg\max_f J(f) = \log q , \qquad c^\star = -\log Z(\theta^\star) . $$

The check `nce-optimum.check.py` evaluates $J$ on a grid for a one-dimensional data density, shows that adding a perturbation to $f = \log q$ always lowers it, and shows that the maximizing constant $c$ for a model of the right shape is $-\log Z$. It also fits `nce_fit` to Gaussian data and checks that the gradient vanishes and the fitted normalizer matches.
