---
id: bayes-optimal-in-context-regression
derives: [c-bayes-ridge, c-bayes-noise-free, c-ls-curve, c-garg-lambda, eq-ridge-icl]
kind: proof
assumes: ["w ~ N(0, tau^2 I_d), independent of the inputs and of the noise", "y_i = w^T x_i + eps_i with eps_i ~ N(0, sigma^2) independent, or sigma = 0 for the noiseless case", "squared-error loss on the query output"]
uses: [bayes-estimator@m2, conditional-expectation@m2, multivariate-normal-distribution@m2, ridge-regression@m2, moore-penrose-inverse@m2, orthogonal-projection@m2, trace-of-a-matrix@m2]
status: full
check: bayes-optimal-in-context-regression.check.py
lean: null
sources:
  - {work: garg-2022-what-can-transformers-learn-in-context, locator: "p. 4"}
  - {work: garg-2022-what-can-transformers-learn-in-context, locator: "p. 5"}
  - {work: garg-2022-what-can-transformers-learn-in-context, locator: "p. 8"}
---

### Statement

Let $X \in \mathbb{R}^{k \times d}$ have the in-context inputs $x_1, \dots, x_k$ as rows and let $y = (y_1, \dots, y_k)$.
Suppose $w \sim \mathcal{N}(0, \tau^2 I_d)$ and $y = X w + \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, \sigma^2 I_k)$, independent of $w$, and let the query output be $y_q = w^\top x_q + \varepsilon_q$ with fresh noise.

1. **Ridge is Bayes-optimal.** If $\sigma > 0$, the prediction of $y_q$ with the least expected squared error given $X$, $y$ and $x_q$ is $\hat w_\lambda^\top x_q$ with $\hat w_\lambda = (X^\top X + \lambda I)^{-1} X^\top y$ and $\lambda = \sigma^2 / \tau^2$ (eq-ridge-icl).
2. **Noiseless limit.** If $\sigma = 0$, the posterior mean of $w$ is $X^+ y$, the minimum-norm least-squares solution, which equals $\Pi w$, where $\Pi = X^+ X$ is the orthogonal projection onto the span of the inputs; and $X^+ y = \lim_{\lambda \to 0} \hat w_\lambda$.
3. **Error curve.** If $x_1, \dots, x_k, x_q, w$ are independent $\mathcal{N}(0, I_d)$ and $\sigma = 0$, the expected error $\mathbb{E}[(x_q^\top X^+ y - w^\top x_q)^2] / d$ is $(d - k)/d$ for $k \le d$ and $0$ for $k \ge d$.
4. **The noisy task of Garg et al.** With $\tau^2 = 1$ and $\sigma^2 = 1$, $\lambda = 1$.

### Proof

**Part 1.** Under squared-error loss the best prediction of $y_q$ from the observed data is its conditional expectation, $\mathbb{E}[y_q \mid X, y, x_q] = \mathbb{E}[w \mid X, y]^\top x_q$, since $\varepsilon_q$ has mean zero and is independent of everything else.
Conditional on $X$, the log posterior density of $w$ is, up to a constant,

$$ -\frac{1}{2\sigma^2} \lVert y - X w \rVert^2 - \frac{1}{2\tau^2} \lVert w \rVert^2 = -\frac{1}{2\sigma^2} \Big( w^\top (X^\top X + \lambda I) w - 2 w^\top X^\top y + y^\top y \Big), \qquad \lambda = \frac{\sigma^2}{\tau^2}. $$

This is a quadratic in $w$ with positive definite Hessian $(X^\top X + \lambda I)/\sigma^2$, so the posterior is Gaussian with mean $(X^\top X + \lambda I)^{-1} X^\top y = \hat w_\lambda$, the maximizer of the quadratic.
That mean is the ridge regression solution with penalty $\lambda$, which proves part 1.

**Part 2.** Let $\Pi$ be the orthogonal projection onto the row space of $X$ and write $w = \Pi w + (I - \Pi) w$.
Because the prior is isotropic, $\Pi w$ and $(I - \Pi) w$ are jointly Gaussian with cross-covariance $\tau^2 \Pi (I - \Pi) = 0$, so they are independent.
With $\sigma = 0$ the data $y = X w = X \Pi w$ determine $\Pi w$ exactly (because $X$ is injective on its row space) and carry no information about $(I - \Pi) w$, whose mean is $0$.
So $\mathbb{E}[w \mid y] = \Pi w = X^+ X w = X^+ y$.
For the limit, with the singular value decomposition $X = U S V^\top$, $\hat w_\lambda = V \operatorname{diag}\big(s_j / (s_j^2 + \lambda)\big) U^\top y$, and each factor tends to $1/s_j$ when $s_j > 0$ and stays $0$ when $s_j = 0$, which is $X^+ y$.

**Part 3.** By part 2 the error is $x_q^\top (\Pi - I) w$.
Conditional on $X$ and $w$, it has mean $0$ and variance $\lVert (I - \Pi) w \rVert^2$ because $x_q \sim \mathcal{N}(0, I_d)$ is independent.
Taking the expectation over $w$ gives $\mathbb{E} \lVert (I - \Pi) w \rVert^2 = \operatorname{tr}(I - \Pi) = d - \operatorname{rank} X$.
For Gaussian inputs $\operatorname{rank} X = \min(k, d)$ with probability one, so the normalized error is $(d - \min(k, d))/d$.

**Part 4.** Garg et al. draw $w \sim \mathcal{N}(0, I_d)$ and add $\varepsilon_i \sim \mathcal{N}(0, 1)$ to each output in the noisy task, so part 1 applies with $\tau^2 = \sigma^2 = 1$. $\square$

### Check

`bayes-optimal-in-context-regression.check.py` compares the ridge formula with the posterior mean from Gaussian conditioning, $\tau^2 X^\top (\tau^2 X X^\top + \sigma^2 I)^{-1} y$; checks by Monte Carlo that $\lambda = \sigma^2/\tau^2$ has a lower average error than $\lambda/4$, $4\lambda$ and plain least squares; checks the limit $\lambda \to 0$ and the projection identity for $k < d$; and estimates the error curve at $d = 20$ for $k = 0, 5, 10, 15, 19, 20, 30$ against $(d - k)/d$.
