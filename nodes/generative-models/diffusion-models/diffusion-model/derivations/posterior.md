---
id: posterior
derives: [c-posterior]
kind: closed-form
assumes: ["the forward process of forward-marginal", "t > 1, so that alpha_bar_{t-1} < 1"]
uses: [bayes-theorem@m2, multivariate-normal-distribution@m2]
status: full
check: posterior.check.py
lean: null
sources:
  - {work: ho-2020-denoising-diffusion-probabilistic-models, locator: "p. 3", note: "eqs. (6) and (7): the forward process posterior conditioned on x_0, its mean and its variance"}
---

### Statement

Conditioned on $\mathbf{x}_0$, the step back from $\mathbf{x}_t$ is Gaussian:

$$ q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}\big(\tilde{\boldsymbol\mu}_t, \tilde\beta_t I\big), \quad \tilde{\boldsymbol\mu}_t = \frac{\sqrt{\bar\alpha_{t-1}}\, \beta_t}{1 - \bar\alpha_t} \mathbf{x}_0 + \frac{\sqrt{\alpha_t}\, (1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t} \mathbf{x}_t, \quad \tilde\beta_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t} \beta_t . $$

### Assumptions

1. $q(\mathbf{x}_t \mid \mathbf{x}_{t-1}, \mathbf{x}_0) = q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$, the Markov property. Used in step 1.
2. $q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)$ is given by forward-marginal. Used in step 1.

### Steps

1. By Bayes' rule, $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \propto q(\mathbf{x}_t \mid \mathbf{x}_{t-1})\, q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)$ as a function of $\mathbf{x}_{t-1}$.
2. Per coordinate the log of the right side is $-\frac{(x_t - \sqrt{\alpha_t} x_{t-1})^2}{2\beta_t} - \frac{(x_{t-1} - \sqrt{\bar\alpha_{t-1}} x_0)^2}{2(1 - \bar\alpha_{t-1})}$, a quadratic in $x_{t-1}$, so the posterior is Gaussian.
3. The coefficient of $-x_{t-1}^2 / 2$ is the precision $\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar\alpha_{t-1}} = \frac{\alpha_t (1 - \bar\alpha_{t-1}) + \beta_t}{\beta_t (1 - \bar\alpha_{t-1})} = \frac{1 - \bar\alpha_t}{\beta_t (1 - \bar\alpha_{t-1})}$, whose inverse is $\tilde\beta_t$.
4. The coefficient of $x_{t-1}$ is $\frac{\sqrt{\alpha_t}}{\beta_t} x_t + \frac{\sqrt{\bar\alpha_{t-1}}}{1 - \bar\alpha_{t-1}} x_0$; multiplying by $\tilde\beta_t$ gives $\tilde{\boldsymbol\mu}_t$.
5. Since $\tilde\beta_t / \beta_t = (1 - \bar\alpha_{t-1}) / (1 - \bar\alpha_t) < 1$, the posterior is narrower than a forward step; at $t = 1$, $\tilde\beta_1 = 0$ and the posterior is the point $\mathbf{x}_0$.

### Result and check

$$ \tilde{\boldsymbol\mu}_t = \frac{\sqrt{\bar\alpha_{t-1}}\, \beta_t}{1 - \bar\alpha_t} \mathbf{x}_0 + \frac{\sqrt{\alpha_t}\, (1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t} \mathbf{x}_t, \qquad \tilde\beta_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t} \beta_t . $$

The check `posterior.check.py` evaluates the log of the product of the two Gaussians on a grid of $x_{t-1}$ for every step of two schedules and verifies that it differs from the log-density of the NumPy reference's `posterior` by a constant, and that $\tilde\beta_t < \beta_t$.
