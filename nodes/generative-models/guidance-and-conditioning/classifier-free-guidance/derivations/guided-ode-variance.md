---
id: guided-ode-variance
derives: [c-ode-variance]
kind: closed-form
assumes: ["the conditional data are N(m_1, v_1) and the unconditional data N(m_0, v_0) in one coordinate", "the noise predictors are exact and the sampler is the deterministic DDIM, or its limit, the probability-flow ODE", "the sampler starts from z_T ~ N(0, 1)"]
uses: [normal-distribution@m2, variance@m2, stochastic-differential-equation@m2]
status: full
check: guided-ode-variance.check.py
lean: null
sources:
  - {work: karras-2024-guiding-diffusion-model-bad, locator: "pp. 2 to 3", note: "the guided density at each noise level is not a valid heat diffusion of the guided density at sigma = 0, which distorts the sampling trajectories"}
  - {work: song-2020-score-based-generative-modeling-sdes, locator: "p. 9", note: "sampling a conditional model by adding the gradient of a noisy classifier to the score of the reverse process"}
---

### Statement

Let $x = z_t/\sqrt{\bar\alpha_t}$ and $\sigma = \sqrt{(1 - \bar\alpha_t)/\bar\alpha_t}$. Guide the exact noise predictors of $\mathcal{N}(m_1, v_1)$ and $\mathcal{N}(m_0, v_0)$ with scale $s$ and run the probability-flow ODE from $\sigma_T$ to 0 starting from $z_T \sim \mathcal{N}(0, 1)$. The variance of the output is

$$ \mathrm{Var}(x_0) = (1 + \sigma_T^2)\Big(\frac{v_1}{v_1 + \sigma_T^2}\Big)^{s}\Big(\frac{v_0}{v_0 + \sigma_T^2}\Big)^{1-s} \;\xrightarrow[\sigma_T \to \infty]{}\; v_1^{\,s}\, v_0^{\,1 - s} . $$

For $s > 1$ this is smaller than the variance $v_s = 1/(s/v_1 + (1 - s)/v_0)$ of the tilted density $p_1^s p_0^{1 - s}$ of derivation tilted-gaussian, for $0 < s < 1$ larger, and equal at $s \in \{0, 1\}$. Deterministic DDIM is the Euler method for this ODE.

### Assumptions

1. Gaussian data per coordinate. Used in step 2.
2. Exact predictors and the deterministic sampler. Used in steps 1 and 3.
3. The start $z_T \sim \mathcal{N}(0, 1)$, so $\mathrm{Var}(x_T) = 1/\bar\alpha_T = 1 + \sigma_T^2$. Used in step 4.

### Steps

1. Write $\hat x_0 = (z_t - \sqrt{1 - \bar\alpha_t}\,\hat\epsilon)/\sqrt{\bar\alpha_t}$, so $x = \hat x_0 + \sigma\hat\epsilon$. The DDIM update $z' = \sqrt{\bar\alpha'}\,\hat x_0 + \sqrt{1 - \bar\alpha'}\,\hat\epsilon$ divided by $\sqrt{\bar\alpha'}$ is $x' = \hat x_0 + \sigma'\hat\epsilon = x + (\sigma' - \sigma)\hat\epsilon$: an Euler step of $dx/d\sigma = \hat\epsilon(x, \sigma)$.
2. For $x_0 \sim \mathcal{N}(m, v)$ and $x = x_0 + \sigma\epsilon$, $\mathrm{Cov}(\epsilon, x) = \sigma$ and $\mathrm{Var}(x) = v + \sigma^2$, so the exact predictor is $\mathbb{E}[\epsilon \mid x] = \sigma(x - m)/(v + \sigma^2)$. The guided predictor is then affine in $x$, $\hat\epsilon = a(\sigma)x + b(\sigma)$, with $a(\sigma) = s\sigma/(v_1 + \sigma^2) + (1 - s)\sigma/(v_0 + \sigma^2)$.
3. The ODE is linear, so the flow map from $\sigma_T$ to 0 is affine with slope $J$ solving $dJ/d\sigma = aJ$, $J(\sigma_T) = 1$; hence $\log J(0) = -\int_0^{\sigma_T} a\, d\sigma = -\tfrac{s}{2}\log\frac{v_1 + \sigma_T^2}{v_1} - \tfrac{1 - s}{2}\log\frac{v_0 + \sigma_T^2}{v_0}$.
4. $\mathrm{Var}(x_0) = J(0)^2\,\mathrm{Var}(x_T) = (1 + \sigma_T^2)\, J(0)^2$, which is the formula; as $\sigma_T \to \infty$, $(1 + \sigma_T^2)/\big((v_1 + \sigma_T^2)^s (v_0 + \sigma_T^2)^{1-s}\big) \to 1$.
5. In precisions $u_i = 1/v_i$, the ODE's output precision is $u_1^s u_0^{1-s}$ and the tilted one $s u_1 + (1 - s)u_0$. Since $\log$ is concave, it lies above its chords between $u_0$ and $u_1$, which is $0 < s < 1$, and below their extensions outside, which is $s > 1$ or $s < 0$: $\log(s u_1 + (1 - s)u_0) \le s\log u_1 + (1 - s)\log u_0$ there, with equality only if $u_1 = u_0$. So for $s > 1$ the ODE's precision is the larger and its variance the smaller.

### Result and check

The guided sampler does not sample the tilted density it is built from. With $m_1 = 1$, $v_1 = 0.25$, $m_0 = 0$, $v_0 = 1.25$ and $s = 2$, the ODE ends with variance $0.05$ and mean $1.32$, against $0.139$ and $1.11$ for the tilted density; at $s = 3$ with $0.01$ and $1.40$ against $0.096$ and $1.15$. The output is much narrower and further from the unconditional mean than $p_1^s p_0^{1-s}$, the exaggerated truncation and overshoot that Karras and colleagues attribute to the guided densities not being a heat diffusion of one distribution. The variance $v_1^s v_0^{1-s}$ decays geometrically in $s$, which is why large scales give saturated, low-diversity samples.
The check `guided-ode-variance.check.py` verifies that DDIM is Euler in $(x, \sigma)$, compares a fine RK4 solution in $\log\sigma$ with the finite-$\sigma_T$ formula to $10^{-6}$ and 1000-step DDIM with it to 2%, recovers the two models at $s = 0$ and $s = 1$, checks the means 1.32 and 1.40 and the comparison with `tilted_gaussian`, and checks the concavity inequality at random precisions.
