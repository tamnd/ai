---
id: score-matching-integration-by-parts
derives: [c-score-matching]
kind: identity
assumes: ["the data density q and the model score are continuously differentiable", "q(x) psi(x) goes to 0 as |x| goes to infinity, and the expectations of psi^2 and of the data score squared are finite", "for the Gaussian part, the sample covariance is invertible"]
uses: [score@m2, expected-value@m2, multivariate-normal-distribution@m2, maximum-likelihood-estimation@m2, consistent-estimator@m2]
status: full
check: score-matching-integration-by-parts.check.py
lean: null
sources:
  - {work: hyvarinen-2005-estimation-non-normalized-score-matching, locator: "pp. 3, 5", note: "Theorem 1 with its partial integration, and the Gaussian example where score matching gives the maximum likelihood estimator"}
---

### Statement

Let $\psi_\theta(\mathbf{x}) = \nabla_\mathbf{x} \log p_\theta(\mathbf{x}) = -\nabla_\mathbf{x} E_\theta(\mathbf{x})$ and $\psi_q = \nabla_\mathbf{x} \log q$. Then

$$ \tfrac{1}{2} \mathbb{E}_q \|\psi_\theta - \psi_q\|^2 = \mathbb{E}_q \sum_{i=1}^{D} \Big[\partial_i \psi_{\theta,i} + \tfrac{1}{2} \psi_{\theta,i}^2\Big] + \tfrac{1}{2} \mathbb{E}_q \|\psi_q\|^2 , $$

and the last term does not depend on $\theta$. For $E(\mathbf{x}) = \tfrac{1}{2}(\mathbf{x} - \boldsymbol{\mu}) M (\mathbf{x} - \boldsymbol{\mu})^\top$ and a sample $\mathbf{x}_1, \ldots, \mathbf{x}_T$, the sample objective $\tilde J(\boldsymbol{\mu}, M) = \frac{1}{T} \sum_t \big[-\operatorname{tr} M + \tfrac{1}{2} \|(\mathbf{x}_t - \boldsymbol{\mu}) M\|^2\big]$ is minimized by $\boldsymbol{\mu} = \bar{\mathbf{x}}$ and $M = S^{-1}$, $S = \frac{1}{T}\sum_t (\mathbf{x}_t - \bar{\mathbf{x}})^\top (\mathbf{x}_t - \bar{\mathbf{x}})$, the maximum likelihood estimate.

### Assumptions

1. $q$ and $\psi_\theta$ are differentiable. Used in step 2.
2. $q \psi_\theta \to 0$ at infinity and the squared scores are integrable. Used in step 2.
3. $S$ is invertible. Used in step 4.

### Steps

1. Expand the square: $\tfrac{1}{2}\|\psi_\theta - \psi_q\|^2 = \tfrac{1}{2}\|\psi_\theta\|^2 - \psi_\theta \cdot \psi_q + \tfrac{1}{2}\|\psi_q\|^2$. Only the cross term involves the unknown data score.
2. For each $i$, $\mathbb{E}_q[\psi_{\theta,i} \psi_{q,i}] = \int q \, \psi_{\theta,i} \, \partial_i \log q \, d\mathbf{x} = \int \psi_{\theta,i} \, \partial_i q \, d\mathbf{x}$. Integrating by parts in $x_i$, the boundary term $[q \psi_{\theta,i}]$ vanishes by assumption 2, which leaves $-\int q \, \partial_i \psi_{\theta,i} \, d\mathbf{x} = -\mathbb{E}_q[\partial_i \psi_{\theta,i}]$.
3. Substituting gives the statement. With $\psi_\theta = -\nabla E$, $\sum_i \partial_i \psi_{\theta,i} = -\Delta E$, which is eq-score-matching of the page.
4. For the Gaussian energy, $\psi = -(\mathbf{x} - \boldsymbol{\mu}) M$ and $\partial_i \psi_i = -M_{ii}$, so $\tilde J = -\operatorname{tr} M + \frac{1}{2T}\sum_t (\mathbf{x}_t - \boldsymbol{\mu}) M^2 (\mathbf{x}_t - \boldsymbol{\mu})^\top$. The gradient in $\boldsymbol{\mu}$ is $-M^2(\bar{\mathbf{x}} - \boldsymbol{\mu})^\top$, zero at $\boldsymbol{\mu} = \bar{\mathbf{x}}$ since $M$ is invertible. There $\tilde J = -\operatorname{tr} M + \tfrac{1}{2}\operatorname{tr}(M S M)$, whose gradient in the symmetric $M$ is $-I + \tfrac{1}{2}(SM + MS)$, zero at $M = S^{-1}$. The objective is a convex quadratic in $M$ with positive-definite Hessian when $S$ is, so this is the minimum.
5. Maximum likelihood for the Gaussian gives the same $\bar{\mathbf{x}}$ and $S^{-1}$, for every sample.

### Result and check

$$ \tfrac{1}{2}\mathbb{E}_q\|\psi_\theta - \psi_q\|^2 = \mathbb{E}_q\big[-\Delta E_\theta + \tfrac{1}{2}\|\nabla E_\theta\|^2\big] + \text{const}, \qquad \arg\min \tilde J = (\bar{\mathbf{x}}, S^{-1}) . $$

The check `score-matching-integration-by-parts.check.py` computes both objectives by numerical integration on a fine grid for a one-dimensional mixture density and several models, and checks that they differ by the same constant. It also checks that `gaussian_sm_fit` returns the sample mean and $S^{-1}$, that the gradient of the sample objective vanishes there, and that the network objective `mlp_sm_objective` equals the mean of $-\Delta E + \tfrac{1}{2}\|\nabla E\|^2$ from finite differences.
