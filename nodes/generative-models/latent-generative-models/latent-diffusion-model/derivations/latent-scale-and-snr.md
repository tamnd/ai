---
id: latent-scale-and-snr
derives: [c-scale-snr]
kind: closed-form
assumes: ["the latent has variance v in every coordinate", "the linear schedule beta_t from 1e-4 to 2e-2 over 1000 steps, indexed from 0"]
uses: [variance@m2, normal-distribution@m2]
status: full
check: latent-scale-and-snr.check.py
lean: null
sources:
  - {work: rombach-2021-high-resolution-image-synthesis-latent, locator: "p. 20", note: "the signal-to-noise ratio Var(z)/sigma_t^2 induced by the latent space"}
  - {work: ho-2020-denoising-diffusion-probabilistic-models, locator: "p. 5", note: "the linear schedule from 1e-4 to 0.02 over T = 1000 steps"}
---

### Statement

Under $z_t = \sqrt{\bar\alpha_t}\, z_0 + \sqrt{1 - \bar\alpha_t}\,\epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$ independent of $z_0$, a coordinate of $z_0$ with variance $v$ has signal-to-noise ratio $\mathrm{SNR}_v(t) = \bar\alpha_t v/(1 - \bar\alpha_t)$, and for the latent $k z_0$

$$ \mathrm{SNR}_{k^2 v}(t) = k^2\, \mathrm{SNR}_v(t) \quad \text{for every } t . $$

For the linear schedule with $\beta_t$ from $10^{-4}$ to $2 \times 10^{-2}$ over 1000 steps and $\bar\alpha_t = \prod_{s \le t}(1 - \beta_s)$, the first step with $\mathrm{SNR} < 1$ is $t = 259$ for $v = 1$ and $t = 620$ for $v = 49$.

### Assumptions

1. $\epsilon$ is independent of $z_0$, so the variances of signal and noise add. Used in step 1.
2. The linear schedule. Used in step 3.

### Steps

1. The signal part $\sqrt{\bar\alpha_t}\, z_0$ has variance $\bar\alpha_t v$ and the noise part $1 - \bar\alpha_t$, so the ratio is $\bar\alpha_t v/(1 - \bar\alpha_t)$.
2. Replacing $z_0$ by $k z_0$ replaces $v$ by $k^2 v$, and the ratio, linear in $v$, is multiplied by $k^2$ at every step.
3. $\bar\alpha_t$ is strictly decreasing, since each $1 - \beta_s < 1$, so $\mathrm{SNR}_v$ is strictly decreasing and crosses 1 once, at the first $t$ with $\bar\alpha_t < 1/(1 + v)$. Computing the product gives $t = 259$ for $v = 1$ and $t = 620$ for $v = 49$; for $v = 1/49$ it is $t = 40$.

### Result and check

Scaling the latent is the same as changing the schedule: a latent that is too large keeps a clean signal deep into the forward process, the model spends its steps where the data are almost noise-free, and the large-$t$ steps the sampler starts from were never seen at those noise levels in proportion. This is why LDM divides the KL-regularized latent by its standard deviation.
The check `latent-scale-and-snr.check.py` compares `snr` with the empirical ratio of the variances of the two parts of `q_sample`, checks the scaling by $k^2$ for several $k$, and finds the crossings with `alpha_bars`.
