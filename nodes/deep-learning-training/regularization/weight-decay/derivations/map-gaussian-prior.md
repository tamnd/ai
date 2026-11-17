---
id: map-gaussian-prior
derives: [c-map-penalty, c-map-ridge, c-map-optimal, c-min-norm, eq-map]
kind: identity
assumes: ["a Gaussian prior N(0, sigma_p^2 I) on the weights", "N independent examples", "for items 2 and 3, a linear model with Gaussian noise of variance sigma^2"]
uses: [normal-distribution@m2, bayes-theorem@m2, maximum-a-posteriori-estimation@al, ridge-regression@m2]
status: full
check: map-gaussian-prior.check.py
lean: null
sources:
  - {work: krogh-1991-simple-weight-decay, locator: "Section 3"}
  - {work: krogh-1991-simple-weight-decay, locator: "Section 4"}
---

### Statement

1. Under the prior $\theta \sim \mathcal N(0, \sigma_p^2 I)$, the maximum a posteriori estimate minimizes the negative log likelihood of the data plus $\frac{1}{2\sigma_p^2}\lVert\theta\rVert^2$. This is L2 regularization with $\lambda = 1/\sigma_p^2$ on the summed loss, or $\lambda = 1/(N\sigma_p^2)$ on the loss averaged over $N$ examples.
2. For a linear model $y = X\theta + \varepsilon$ with $\varepsilon \sim \mathcal N(0, \sigma^2 I)$, the MAP estimate is ridge regression, $\hat\theta_\lambda = (X^\top X + \lambda I)^{-1}X^\top y$ with $\lambda = \sigma^2/\sigma_p^2$, which is also the posterior mean.
3. Averaged over the prior and the noise, the squared error $\mathbb E\lVert\hat\theta_\lambda - \theta\rVert^2$ is smallest at $\lambda = \sigma^2/\sigma_p^2$, the ratio of the noise variance to the prior variance. This is the optimal weight decay (17) of Krogh and Hertz, whose $\sigma_u^2$ plays the role of $\sigma_p^2$.
4. As $\lambda \to 0$, $\hat\theta_\lambda$ tends to the minimum-norm solution $X^+ y$, and gradient descent with decay $\lambda$ shrinks any component of $\theta$ in the null space of $X$ by the factor $1 - \eta\lambda$ per step.

### Assumptions

1. The prior is $\mathcal N(0, \sigma_p^2 I)$ and the examples are independent given $\theta$. Used in step 1.
2. The model is linear with Gaussian noise of variance $\sigma^2$, and $X$ is fixed. Used in steps 2 to 4.
3. In step 3 the true $\theta$ is drawn from the prior; Krogh and Hertz average over a random teacher $u$ in the same way.

### Steps

1. By Bayes' theorem $-\log p(\theta \mid D) = -\sum_{n=1}^N \log p(y_n \mid x_n, \theta) - \log p(\theta) + \text{const}$, and $-\log p(\theta) = \frac{1}{2\sigma_p^2}\lVert\theta\rVert^2 + \text{const}$. Dividing the whole objective by $N$ does not move its minimum and turns the coefficient into $1/(N\sigma_p^2)$.
2. With Gaussian noise the negative log likelihood is $\frac{1}{2\sigma^2}\lVert y - X\theta\rVert^2$. Multiplying the objective of step 1 by $\sigma^2$ gives $\frac12\lVert y - X\theta\rVert^2 + \frac{\sigma^2}{2\sigma_p^2}\lVert\theta\rVert^2$, whose gradient vanishes at $(X^\top X + \lambda I)\theta = X^\top y$ with $\lambda = \sigma^2/\sigma_p^2$. The posterior is Gaussian, so its mode is its mean.
3. Write $A = (X^\top X + \lambda I)^{-1}$. Then $\hat\theta_\lambda - \theta = -\lambda A\theta + AX^\top\varepsilon$, and since $\theta$ and $\varepsilon$ are independent with zero means, $\mathbb E\lVert\hat\theta_\lambda - \theta\rVert^2 = \lambda^2\sigma_p^2\operatorname{tr}(A^2) + \sigma^2\operatorname{tr}(AX^\top XA)$. In the eigenbasis of $X^\top X$, with eigenvalues $a_r$, this is $\sum_r (\lambda^2\sigma_p^2 + \sigma^2 a_r)/(a_r + \lambda)^2$. The derivative of each term with respect to $\lambda$ is $2a_r(\lambda\sigma_p^2 - \sigma^2)/(a_r + \lambda)^3$, which is negative below $\sigma^2/\sigma_p^2$ and positive above it, for every $a_r > 0$. So the minimum is at $\lambda = \sigma^2/\sigma_p^2$ whatever the spectrum, as Krogh and Hertz state for their (17).
4. With the singular value decomposition $X = U\operatorname{diag}(s_i)V^\top$, $\hat\theta_\lambda = V\operatorname{diag}\big(s_i/(s_i^2 + \lambda)\big)U^\top y$, which tends to $V\operatorname{diag}(1/s_i)U^\top y = X^+ y$ as $\lambda \to 0$; it has no component in the null space of $X$. The loss gradient $-X^\top(y - X\theta)$ lies in the row space of $X$, so for a unit vector $n$ in the null space the step $\theta \leftarrow \theta - \eta(-X^\top(y - X\theta) + \lambda\theta)$ gives $n^\top\theta \leftarrow (1 - \eta\lambda)\,n^\top\theta$. This is the decay of the components outside the pattern subspace in Section 3 of Krogh and Hertz.

### Result and check

$$ \hat\theta_{\text{MAP}} = \arg\min_\theta \Big[ -\sum_{n=1}^N \log p(y_n \mid x_n, \theta) + \frac{1}{2\sigma_p^2}\lVert\theta\rVert^2 \Big], \qquad \lambda^\star = \frac{\sigma^2}{\sigma_p^2} . $$

The check `map-gaussian-prior.check.py` runs gradient descent on the negative log posterior of a linear model with 30 examples and 8 weights and finds the ridge solution with $\lambda = \sigma^2/\sigma_p^2$ to $10^{-10}$, and the same point from the averaged loss with coefficient $\lambda/N$.
It evaluates the exact expected error on a grid of 4001 values of $\lambda$, finds the minimum within 1% of $\sigma^2/\sigma_p^2$, and confirms the formula with 20000 Monte Carlo draws to 3%.
With 4 examples and 8 weights it checks that the ridge solution approaches $X^+ y$ as $\lambda$ falls from $10^{-2}$ to $10^{-8}$, and that a null-space component shrinks by exactly $1 - \eta\lambda$ per step.
