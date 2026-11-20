---
id: ddim-marginals
derives: [c-ddim-marginals, c-ddim-eta-one]
kind: closed-form
assumes: ["q(x_t | x_0) = N(sqrt(abar_t) x_0, (1 - abar_t) I) for every t, with abar_0 = 1", "0 <= sigma^2 <= 1 - abar_s for a step from t to s < t"]
uses: [multivariate-normal-distribution@m2, independence-of-random-variables@m2]
status: full
check: ddim-marginals.check.py
lean: null
sources:
  - {work: song-2020-denoising-diffusion-implicit-models, locator: "pp. 3 to 6", note: "eq. (7) and Lemma 1: the non-Markovian forward process with the DDPM marginals; eq. (12): the sampling step; eq. (16): sigma(eta), with eta = 1 the DDPM and eta = 0 the DDIM"}
---

### Statement

For $s < t$ and any $\sigma$ with $0 \le \sigma^2 \le 1 - \bar\alpha_s$, define

$$ q_\sigma(\mathbf{x}_s \mid \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}\Big(\sqrt{\bar\alpha_s}\, \mathbf{x}_0 + \sqrt{1 - \bar\alpha_s - \sigma^2}\; \frac{\mathbf{x}_t - \sqrt{\bar\alpha_t}\, \mathbf{x}_0}{\sqrt{1 - \bar\alpha_t}},\; \sigma^2 I\Big) . $$

1. If $\mathbf{x}_t \sim q(\mathbf{x}_t \mid \mathbf{x}_0)$, then $\mathbf{x}_s \sim q(\mathbf{x}_s \mid \mathbf{x}_0)$: every $\sigma$ keeps the forward marginals, which are all the simple loss depends on.
2. With $s = t - 1$ and $\sigma^2 = \tilde\beta_t$, which is $\eta = 1$ in $\sigma_{t,s}(\eta) = \eta \sqrt{\frac{1 - \bar\alpha_s}{1 - \bar\alpha_t}} \sqrt{1 - \frac{\bar\alpha_t}{\bar\alpha_s}}$, $q_\sigma$ is the DDPM posterior $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$, so the DDIM step with $\eta = 1$ is the ancestral step with $\sigma_t^2 = \tilde\beta_t$.

### Assumptions

1. $(\mathbf{x}_t - \sqrt{\bar\alpha_t}\mathbf{x}_0) / \sqrt{1 - \bar\alpha_t} = \boldsymbol\epsilon \sim \mathcal{N}(0, I)$ given $\mathbf{x}_0$. Used in step 1.
2. The fresh noise of $q_\sigma$ is independent of $\boldsymbol\epsilon$. Used in step 1.

### Steps

1. Write $\mathbf{x}_s = \sqrt{\bar\alpha_s}\mathbf{x}_0 + \sqrt{1 - \bar\alpha_s - \sigma^2}\,\boldsymbol\epsilon + \sigma \mathbf{z}$. Given $\mathbf{x}_0$ it is Gaussian with mean $\sqrt{\bar\alpha_s}\mathbf{x}_0$ and variance $(1 - \bar\alpha_s - \sigma^2) + \sigma^2 = 1 - \bar\alpha_s$ per coordinate, which is $q(\mathbf{x}_s \mid \mathbf{x}_0)$.
2. For $s = t - 1$ and $\eta = 1$, $\sigma^2 = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}\big(1 - \alpha_t\big) = \tilde\beta_t$.
3. Then $1 - \bar\alpha_{t-1} - \tilde\beta_t = (1 - \bar\alpha_{t-1})\frac{1 - \bar\alpha_t - \beta_t}{1 - \bar\alpha_t} = \frac{(1 - \bar\alpha_{t-1})^2 \alpha_t}{1 - \bar\alpha_t}$, so the coefficient of $\mathbf{x}_t$ in the mean is $\frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t}$, and that of $\mathbf{x}_0$ is $\sqrt{\bar\alpha_{t-1}} - \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})\sqrt{\bar\alpha_t}}{1 - \bar\alpha_t} = \frac{\sqrt{\bar\alpha_{t-1}}\,(1 - \bar\alpha_t - \alpha_t + \bar\alpha_t)}{1 - \bar\alpha_t} = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1 - \bar\alpha_t}$: both are those of $\tilde{\boldsymbol\mu}_t$ in the derivation posterior.
4. In the sampler $\mathbf{x}_0$ is replaced by the prediction $(\mathbf{x}_t - \sqrt{1 - \bar\alpha_t}\boldsymbol\epsilon_\theta) / \sqrt{\bar\alpha_t}$ and $\boldsymbol\epsilon$ by $\boldsymbol\epsilon_\theta$; with $\sigma = 0$ the step is a deterministic function of $\mathbf{x}_t$, and nothing requires $s = t - 1$.

### Result and check

$$ \mathbf{x}_s = \sqrt{\bar\alpha_s}\, \hat{\mathbf{x}}_0 + \sqrt{1 - \bar\alpha_s - \sigma^2}\, \boldsymbol\epsilon_\theta + \sigma \mathbf{z}, \qquad \hat{\mathbf{x}}_0 = \frac{\mathbf{x}_t - \sqrt{1 - \bar\alpha_t}\, \boldsymbol\epsilon_\theta}{\sqrt{\bar\alpha_t}} . $$

The check `ddim-marginals.check.py` pushes the moments of one coordinate exactly through the NumPy reference's `ddim_step` with the true noise for every pair $s < t$ of the example schedule and $\eta \in \{0, 0.5, 1\}$, and compares them with $q(\mathbf{x}_s \mid \mathbf{x}_0)$; it then checks that `ddim_sigma` with $\eta = 1$ and $s = t - 1$ is $\sqrt{\tilde\beta_t}$ and that `ddim_step` equals `ddpm_step` with $\sigma_t^2 = \tilde\beta_t$.
