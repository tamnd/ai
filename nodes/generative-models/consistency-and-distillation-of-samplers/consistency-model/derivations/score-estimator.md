---
id: score-estimator
derives: [c-ct-pair]
kind: identity
assumes: ["x_t = x + t z with x drawn from the data and z standard normal, independent", "the data distribution has a finite mean, so the integrals below exist and differentiation passes under the integral"]
uses: [conditional-expectation@m2, bayes-theorem@m2, multivariate-normal-distribution@m2, score@m2]
status: full
check: score-estimator.check.py
lean: null
sources:
  - {work: song-2023-consistency-models, locator: "p. 4, Eq. 6; p. 5, Section 5 and Lemma 1", note: "the Euler update of the empirical ODE, and the unbiased estimator -(x_t - x)/t^2 of the score that replaces the pre-trained score model"}
---

### Statement

Let $x_t = x + t z$ with $x \sim p_{\text{data}}$ and $z \sim \mathcal N(0, I)$ independent, and let $p_t$ be the density of $x_t$. Then

$$ \nabla \log p_t(x_t) = -\,\mathbb E\!\left[\frac{x_t - x}{t^2} \,\middle|\, x_t\right] , $$ {#eq-score-est}

so $-(x_t - x)/t^2$ is an unbiased estimate of the score from a single data point. Moreover, one Euler step of the probability-flow ODE from $x_{t_{n+1}} = x + t_{n+1} z$ down to $t_n$, with this estimate in place of the score, lands exactly on $x + t_n z$: the pair of consistency training, $(x + t_{n+1} z,\, x + t_n z)$, is the pair of consistency distillation with an Euler teacher whose score is the one-sample estimate.

### Assumptions

1. The noise is independent Gaussian with variance $t^2$. Used in steps 1 and 3.
2. Differentiation under the integral is allowed. Used in step 1.

### Steps

1. $p_t(y) = \int p_{\text{data}}(x)\,\mathcal N(y; x, t^2 I)\,dx$, and $\nabla_y \mathcal N(y; x, t^2 I) = -\frac{y - x}{t^2}\,\mathcal N(y; x, t^2 I)$, so $\nabla p_t(y) = \int p_{\text{data}}(x)\,\mathcal N(y; x, t^2 I)\,\bigl(-\tfrac{y - x}{t^2}\bigr)\,dx$.
2. Dividing by $p_t(y)$, the weight $p_{\text{data}}(x)\,\mathcal N(y; x, t^2 I)/p_t(y)$ is the posterior $p(x \mid x_t = y)$ by Bayes' theorem, which gives eq-score-est.
3. The Euler step of $dx/dt = -t\,s(x, t)$ from $t_{n+1}$ to $t_n$ is $\hat x = x_{t_{n+1}} - (t_n - t_{n+1})\,t_{n+1}\,s$. With $s = -(x_{t_{n+1}} - x)/t_{n+1}^2 = -z/t_{n+1}$ it is $\hat x = x + t_{n+1} z + (t_n - t_{n+1}) z = x + t_n z$.

### Result and check

Consistency training needs no diffusion model: the teacher's step is replaced by moving the same data point and the same noise to the lower level.
The replacement is exact only in expectation over $x$ given $x_t$, which is why Song and colleagues' Theorem 2 relates the training loss to the distillation loss only up to $o(\Delta t)$, and why the training loss has more variance than distillation, more so as the levels get closer.
The check `score-estimator.check.py` computes the posterior mean of $-(x_t - x)/t^2$ by quadrature for a one-dimensional mixture of three Gaussians and compares it with the exact score at several levels, and checks the exact Euler identity at random levels in three dimensions.
