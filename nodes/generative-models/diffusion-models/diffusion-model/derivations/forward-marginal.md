---
id: forward-marginal
derives: [c-forward-marginal]
kind: closed-form
assumes: ["the forward process is the Markov chain q(x_t | x_{t-1}) = N(sqrt(1 - beta_t) x_{t-1}, beta_t I) with 0 < beta_t < 1", "the noise of each step is independent of x_0 and of the other steps"]
uses: [multivariate-normal-distribution@m2, independence-of-random-variables@m2, linearity-of-expectation@m2, variance@m2]
status: full
check: forward-marginal.check.py
lean: null
sources:
  - {work: ho-2020-denoising-diffusion-probabilistic-models, locator: "p. 2", note: "eqs. (2) and (4): the forward process and its marginal at an arbitrary step in closed form, with alpha_t = 1 - beta_t and alpha_bar_t the running product"}
---

### Statement

With $\alpha_t = 1 - \beta_t$ and $\bar\alpha_t = \prod_{s=1}^{t} \alpha_s$, the forward chain started at $\mathbf{x}_0$ has the marginal

$$ q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\big(\mathbf{x}_t; \sqrt{\bar\alpha_t}\, \mathbf{x}_0, (1 - \bar\alpha_t) I\big), $$

so $\mathbf{x}_t = \sqrt{\bar\alpha_t}\, \mathbf{x}_0 + \sqrt{1 - \bar\alpha_t}\, \boldsymbol\epsilon$ with a single $\boldsymbol\epsilon \sim \mathcal{N}(0, I)$.
As $t$ grows, $\bar\alpha_t \to 0$ and the marginal tends to $\mathcal{N}(0, I)$ whatever $\mathbf{x}_0$ is.

### Assumptions

1. $\mathbf{x}_t = \sqrt{\alpha_t}\, \mathbf{x}_{t-1} + \sqrt{\beta_t}\, \mathbf{z}_t$ with $\mathbf{z}_t \sim \mathcal{N}(0, I)$ independent of everything before. Used in steps 1 and 2.
2. Sums of independent Gaussians are Gaussian, with added means and added covariances. Used in step 2.

### Steps

1. For $t = 1$ the claim is assumption 1 with $\bar\alpha_1 = \alpha_1$ and $1 - \bar\alpha_1 = \beta_1$.
2. Suppose $\mathbf{x}_{t-1} = \sqrt{\bar\alpha_{t-1}}\, \mathbf{x}_0 + \sqrt{1 - \bar\alpha_{t-1}}\, \boldsymbol\epsilon'$. Then $\mathbf{x}_t = \sqrt{\alpha_t \bar\alpha_{t-1}}\, \mathbf{x}_0 + \sqrt{\alpha_t (1 - \bar\alpha_{t-1})}\, \boldsymbol\epsilon' + \sqrt{1 - \alpha_t}\, \mathbf{z}_t$. The two noise terms are independent Gaussians with total variance $\alpha_t - \alpha_t \bar\alpha_{t-1} + 1 - \alpha_t = 1 - \bar\alpha_t$ per coordinate, and $\alpha_t \bar\alpha_{t-1} = \bar\alpha_t$.
3. By induction the claim holds for every $t$. If $\sum_t \beta_t$ diverges, then $\bar\alpha_t \le \exp(-\sum_{s \le t} \beta_s) \to 0$, because $1 - \beta \le e^{-\beta}$.

### Result and check

$$ \mathbf{x}_t = \sqrt{\bar\alpha_t}\, \mathbf{x}_0 + \sqrt{1 - \bar\alpha_t}\, \boldsymbol\epsilon, \qquad \bar\alpha_t = \prod_{s \le t} (1 - \beta_s) . $$

The check `forward-marginal.check.py` pushes the mean and variance of one coordinate exactly through the steps of the NumPy reference's `forward_step`, which is affine, for the example schedule and the schedule of Ho et al. ($T = 1000$, $\beta$ from $10^{-4}$ to $0.02$), compares them with `q_sample` at every step, and bounds $\bar\alpha_t$ by $\exp(-\sum \beta)$.
