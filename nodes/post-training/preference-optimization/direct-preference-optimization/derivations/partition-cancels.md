---
id: partition-cancels
derives: [c-dpo-is-nll, c-shift-invariance, c-we-bt, eq-dpo-loss]
kind: reparameterization
assumes: ["preferences follow the Bradley-Terry model with a latent reward r*(x, y)", "pi* is the optimal policy of the KL-constrained objective for r*, with the assumptions of kl-constrained-optimum", "beta > 0"]
uses: [logistic-regression@m2, cross-entropy@m2]
status: full
check: partition-cancels.check.py
lean: null
sources:
  - {work: rafailov-2023-direct-preference-optimization, locator: "p. 4"}
  - {work: rafailov-2023-direct-preference-optimization, locator: "p. 5"}
  - {work: rafailov-2023-direct-preference-optimization, locator: "p. 16"}
---

### Statement

Under the Bradley-Terry model $p^*(y_1\succ y_2\mid x) = \sigma\big(r^*(x, y_1) - r^*(x, y_2)\big)$, with $\pi^*$ the optimal policy of the KL-constrained objective for $r^*$:

1. $p^*(y_1\succ y_2\mid x) = \sigma\Big(\beta\log\frac{\pi^*(y_1\mid x)}{\pi_{\mathrm{ref}}(y_1\mid x)} - \beta\log\frac{\pi^*(y_2\mid x)}{\pi_{\mathrm{ref}}(y_2\mid x)}\Big)$, with no partition function;
2. the negative log-likelihood of a dataset of preferences $(x, y_w, y_l)$ under this model, with $\pi^*$ replaced by a parametrized $\pi_\theta$, is the DPO loss;
3. replacing $r^*$ by $r^* + f(x)$ for any function $f$ of the prompt changes neither the preference probabilities nor $\pi^*$.

### Assumptions

1. Preferences follow the Bradley-Terry model with a finite reward. Used in step 1.
2. The assumptions of `kl-constrained-optimum`, so that $r^* = \beta\log(\pi^*/\pi_{\mathrm{ref}}) + \beta\log Z(x)$ holds. Used in step 2.
3. $\beta > 0$.

### Steps

1. The Bradley-Terry probability $\frac{e^{a}}{e^{a} + e^{b}} = \frac{1}{1 + e^{-(a - b)}} = \sigma(a - b)$ depends only on the difference of the rewards.
2. By `kl-constrained-optimum`, $r^*(x, y) = \beta\log\frac{\pi^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)} + \beta\log Z(x)$. In the difference $r^*(x, y_1) - r^*(x, y_2)$ the two terms $\beta\log Z(x)$ are equal, because both responses answer the same prompt, and cancel. This gives item 1, equation 6 of Rafailov et al.
3. For a dataset of independent comparisons, the log-likelihood of a model $p_\theta$ is $\sum\log p_\theta(y_w\succ y_l\mid x)$. Writing $p_\theta$ by item 1 with $\pi_\theta$ in place of $\pi^*$ and negating the mean gives $-\mathbb E\log\sigma\big(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\big)$, the DPO loss. This is item 2.
4. Adding $f(x)$ to both rewards leaves their difference unchanged, so by step 1 the preferences are unchanged. In $\pi^*$ it multiplies every $e^{r(x, y)/\beta}$ by $e^{f(x)/\beta}$, and $Z(x)$ by the same factor, which cancels. This is item 3, Lemmas 1 and 2 of Rafailov et al. for the Bradley-Terry case.

### Result and check

$$ \mathcal L_{\mathrm{DPO}}(\pi_\theta; \pi_{\mathrm{ref}}) = -\,\mathbb E_{(x, y_w, y_l)}\Big[\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\Big)\Big] . $$

The check `partition-cancels.check.py` draws 2000 problems with 2 to 9 responses and $\beta$ from 0.05 to 5.
It checks item 1 against the Bradley-Terry probability computed from the rewards to $10^{-10}$, item 3 with a random offset of scale 50, and that the DPO per-pair loss at $\pi^*$ equals $-\log p^*(y_i\succ y_j)$.
For the worked example of the page it gives $p^*(y_1\succ y_2) = \sigma(1) = 0.731059$ from the policy.
