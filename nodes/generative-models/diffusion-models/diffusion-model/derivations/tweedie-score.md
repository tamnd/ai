---
id: tweedie-score
derives: [c-tweedie]
kind: closed-form
assumes: ["x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps with eps ~ N(0, I) independent of x_0", "q_t, the density of x_t, is differentiable and the derivative may pass under the integral over x_0"]
uses: [conditional-expectation@m2, multivariate-normal-distribution@m2, mean-squared-error@m2]
status: full
check: tweedie-score.check.py
lean: null
sources:
  - {work: ho-2020-denoising-diffusion-probabilistic-models, locator: "p. 4", note: "the eps-prediction sampler resembles Langevin dynamics with eps_theta as a learned gradient of the data density, and the loss resembles denoising score matching"}
  - {work: song-2019-generative-modeling-gradients-data-distribution, locator: "p. 3", note: "denoising score matching: the optimal score network of the perturbed data equals the score of the perturbed density"}
---

### Statement

The minimizer over all functions of $\mathbb{E}\lVert \boldsymbol\epsilon - \boldsymbol\epsilon_\theta(\mathbf{x}_t, t)\rVert^2$ is the conditional mean $\boldsymbol\epsilon^*(\mathbf{x}_t, t) = \mathbb{E}[\boldsymbol\epsilon \mid \mathbf{x}_t]$, and

$$ \boldsymbol\epsilon^*(\mathbf{x}_t, t) = -\sqrt{1 - \bar\alpha_t}\; \nabla_{\mathbf{x}_t} \log q_t(\mathbf{x}_t) . $$

So a trained noise predictor is a scaled estimate of the score of the noisy data density, the quantity that score-based models learn and Langevin dynamics follows.
Its minimum loss is $\mathbb{E}\lVert \boldsymbol\epsilon - \boldsymbol\epsilon^* \rVert^2$, which is not 0 unless $\mathbf{x}_t$ determines $\boldsymbol\epsilon$.

### Assumptions

1. Squared error is minimized by the conditional mean. Used in step 1.
2. $q_t(\mathbf{x}_t) = \int q(\mathbf{x}_0)\, \mathcal{N}(\mathbf{x}_t; \sqrt{\bar\alpha_t}\mathbf{x}_0, (1 - \bar\alpha_t) I)\, d\mathbf{x}_0$, by forward-marginal. Used in step 2.

### Steps

1. For each $\mathbf{x}_t$, $\mathbb{E}[\lVert \boldsymbol\epsilon - \mathbf{c} \rVert^2 \mid \mathbf{x}_t]$ is minimized at $\mathbf{c} = \mathbb{E}[\boldsymbol\epsilon \mid \mathbf{x}_t]$, and the minimum is the conditional variance, summed over coordinates.
2. Differentiate under the integral: $\nabla q_t(\mathbf{x}_t) = \int q(\mathbf{x}_0)\, \mathcal{N}(\cdot)\, \frac{\sqrt{\bar\alpha_t}\mathbf{x}_0 - \mathbf{x}_t}{1 - \bar\alpha_t}\, d\mathbf{x}_0$. Dividing by $q_t(\mathbf{x}_t)$ turns the integral into an expectation under $q(\mathbf{x}_0 \mid \mathbf{x}_t)$: $\nabla \log q_t(\mathbf{x}_t) = \mathbb{E}\big[\frac{\sqrt{\bar\alpha_t}\mathbf{x}_0 - \mathbf{x}_t}{1 - \bar\alpha_t} \,\big|\, \mathbf{x}_t\big]$.
3. On the event $\mathbf{x}_t$, $\sqrt{\bar\alpha_t}\mathbf{x}_0 - \mathbf{x}_t = -\sqrt{1 - \bar\alpha_t}\,\boldsymbol\epsilon$, so $\nabla \log q_t(\mathbf{x}_t) = -\mathbb{E}[\boldsymbol\epsilon \mid \mathbf{x}_t] / \sqrt{1 - \bar\alpha_t}$, which rearranges to the claim. This is Tweedie's formula, $\mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t] = \big(\mathbf{x}_t + (1 - \bar\alpha_t)\nabla \log q_t(\mathbf{x}_t)\big) / \sqrt{\bar\alpha_t}$, in the variables of the noise.

### Result and check

$$ \boldsymbol\epsilon^*(\mathbf{x}_t, t) = \mathbb{E}[\boldsymbol\epsilon \mid \mathbf{x}_t] = -\sqrt{1 - \bar\alpha_t}\; \nabla \log q_t(\mathbf{x}_t) . $$

The check `tweedie-score.check.py` takes a two-component Gaussian mixture in the plane as the data, estimates $\mathbb{E}[\boldsymbol\epsilon \mid \mathbf{x}_t]$ at points drawn from $q_t$ by Monte Carlo, weighting draws of $\mathbf{x}_0$ by $q(\mathbf{x}_t \mid \mathbf{x}_0)$, compares it with the NumPy reference's closed form `mixture_eps` and with finite differences of `mixture_log_density`, and checks that no constant shift of the prediction lowers the loss.
