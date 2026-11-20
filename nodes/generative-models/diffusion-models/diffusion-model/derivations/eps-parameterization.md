---
id: eps-parameterization
derives: [c-eps-weight, c-reverse-mean]
kind: closed-form
assumes: ["the posterior of the derivation posterior", "a reverse step p(x_{t-1} | x_t) = N(mu_theta(x_t, t), sigma_t^2 I) with a fixed sigma_t^2", "t > 1"]
uses: [kullback-leibler-divergence@m2, multivariate-normal-distribution@m2]
status: full
check: eps-parameterization.check.py
lean: null
sources:
  - {work: ho-2020-denoising-diffusion-probabilistic-models, locator: "pp. 3 and 4", note: "eqs. (8) to (12): the KL term as a squared error of means, the reparameterization of x_t, the eps-prediction mean and the weighted noise error"}
---

### Statement

Write $\mathbf{x}_t = \sqrt{\bar\alpha_t}\, \mathbf{x}_0 + \sqrt{1 - \bar\alpha_t}\, \boldsymbol\epsilon$ and let the reverse mean be

$$ \boldsymbol\mu_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\, \boldsymbol\epsilon_\theta(\mathbf{x}_t, t) \Big) . $$

1. If $\boldsymbol\epsilon_\theta(\mathbf{x}_t, t) = \boldsymbol\epsilon$, then $\boldsymbol\mu_\theta = \tilde{\boldsymbol\mu}_t(\mathbf{x}_t, \mathbf{x}_0)$.
2. For any prediction, with $D$ coordinates,
$$ D_{\mathrm{KL}}\big(q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \,\|\, p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)\big) = \frac{\beta_t^2}{2 \sigma_t^2 \alpha_t (1 - \bar\alpha_t)} \lVert \boldsymbol\epsilon - \boldsymbol\epsilon_\theta(\mathbf{x}_t, t) \rVert^2 + \frac{D}{2}\Big(\frac{\tilde\beta_t}{\sigma_t^2} - 1 - \log \frac{\tilde\beta_t}{\sigma_t^2}\Big) . $$

### Assumptions

1. The KL divergence between $\mathcal{N}(\mathbf{m}_1, v_1 I)$ and $\mathcal{N}(\mathbf{m}_2, v_2 I)$ in $D$ dimensions is $\frac{D}{2}\big(\frac{v_1}{v_2} - 1 - \log \frac{v_1}{v_2}\big) + \frac{\lVert \mathbf{m}_1 - \mathbf{m}_2 \rVert^2}{2 v_2}$. Used in step 3.
2. $\sigma_t^2$ does not depend on $\theta$. Used in step 3.

### Steps

1. Solve the reparameterization for $\mathbf{x}_0 = (\mathbf{x}_t - \sqrt{1 - \bar\alpha_t}\, \boldsymbol\epsilon) / \sqrt{\bar\alpha_t}$ and substitute into $\tilde{\boldsymbol\mu}_t$. The coefficient of $\mathbf{x}_t$ becomes $\frac{\beta_t}{\sqrt{\alpha_t} (1 - \bar\alpha_t)} + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t} = \frac{\beta_t + \alpha_t - \bar\alpha_t}{\sqrt{\alpha_t}(1 - \bar\alpha_t)} = \frac{1}{\sqrt{\alpha_t}}$, using $\sqrt{\bar\alpha_{t-1} / \bar\alpha_t} = 1 / \sqrt{\alpha_t}$. The coefficient of $\boldsymbol\epsilon$ becomes $-\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1 - \bar\alpha_t}}$. So $\tilde{\boldsymbol\mu}_t = \frac{1}{\sqrt{\alpha_t}}\big(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}} \boldsymbol\epsilon\big)$, which is $\boldsymbol\mu_\theta$ with $\boldsymbol\epsilon_\theta = \boldsymbol\epsilon$; this is claim 1.
2. Subtracting, $\tilde{\boldsymbol\mu}_t - \boldsymbol\mu_\theta = \frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1 - \bar\alpha_t}} (\boldsymbol\epsilon_\theta - \boldsymbol\epsilon)$.
3. Apply assumption 1 with $v_1 = \tilde\beta_t$, $v_2 = \sigma_t^2$: the mean term is $\frac{\beta_t^2}{2 \sigma_t^2 \alpha_t (1 - \bar\alpha_t)} \lVert \boldsymbol\epsilon - \boldsymbol\epsilon_\theta \rVert^2$ and the variance term does not depend on $\theta$.
4. Dropping the weight in front of the squared error and averaging over $t$ gives the simple loss of Ho et al.; it is a reweighted bound, not the bound itself.

### Result and check

$$ L_{t-1} = \frac{\beta_t^2}{2 \sigma_t^2 \alpha_t (1 - \bar\alpha_t)}\, \mathbb{E}\, \lVert \boldsymbol\epsilon - \boldsymbol\epsilon_\theta(\mathbf{x}_t, t) \rVert^2 + C_t . $$

The check `eps-parameterization.check.py` draws $\mathbf{x}_0$, $\boldsymbol\epsilon$ and an arbitrary prediction at every step $t > 1$ of two schedules, compares the NumPy reference's `reverse_mean` at the true noise with `posterior`, and its `gauss_kl` with the right side for both $\sigma_t^2 = \beta_t$ and $\sigma_t^2 = \tilde\beta_t$; it also prints the weight at $t = 2$ and $t = T$ of the schedule of Ho et al., which falls from 0.27 to 0.010 with $\sigma_t^2 = \beta_t$, a factor of about 27, so the simple loss gives the large-$t$ terms relatively more weight than the bound does.
