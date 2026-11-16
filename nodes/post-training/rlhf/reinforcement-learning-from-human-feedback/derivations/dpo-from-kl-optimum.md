---
id: dpo-from-kl-optimum
derives: [c-kl-gap, c-kl-optimum, c-implicit-reward, c-dpo-cancel, c-dpo-gradient, c-kl-shaped-unbiased, c-per-token-kl, eq-optimal, eq-implicit-reward, eq-dpo]
kind: identity
assumes: ["a finite set of responses for each prompt", "pi_ref(y|x) > 0 for every response", "beta > 0", "the Bradley-Terry model of preferences"]
uses: [kullback-leibler-divergence@m2, gibbs-inequality@m2, exponential-family@m2]
status: full
check: dpo-from-kl-optimum.check.py
lean: null
sources:
  - {work: rafailov-2023-direct-preference-optimization, locator: "Section 4"}
  - {work: rafailov-2023-direct-preference-optimization, locator: "Section 3"}
  - {work: stiennon-2020-learning-summarize-human-feedback, locator: "Section 3.4"}
---

### Statement

Fix a prompt $x$, a reference policy $\pi_{\text{ref}}(\cdot \mid x)$ that is positive on every response, a reward $r(x, y)$ and $\beta > 0$. Let $J(\pi) = \mathbb{E}_{y \sim \pi}[r(x, y)] - \beta\, \mathrm{KL}(\pi \,\|\, \pi_{\text{ref}})$ (eq-objective), $Z(x) = \sum_y \pi_{\text{ref}}(y \mid x)\, e^{r(x, y)/\beta}$, and $\pi^*(y \mid x) = \pi_{\text{ref}}(y \mid x)\, e^{r(x, y)/\beta} / Z(x)$ (eq-optimal). Then

1. $J(\pi) = \beta \log Z(x) - \beta\, \mathrm{KL}(\pi \,\|\, \pi^*)$ for every policy $\pi$;
2. so $\pi^*$ is the unique maximizer of $J$, and the maximum is $\beta \log Z(x)$;
3. $r(x, y) = \beta \log \dfrac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$ (eq-implicit-reward);
4. under the Bradley–Terry model, $p(y_w \succ y_l \mid x) = \sigma\!\Big(\beta \log \dfrac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \dfrac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\Big)$: $Z(x)$ cancels, so the maximum likelihood of a policy $\pi_\theta$ on the preferences is the DPO loss (eq-dpo), and at $\pi_\theta = \pi^*$ it equals the Bradley–Terry loss of $r$;
5. with $\hat r_\theta = \beta \log(\pi_\theta / \pi_{\text{ref}})$, the gradient of the DPO loss of one pair is $-\beta\, \sigma(\hat r_\theta(x, y_l) - \hat r_\theta(x, y_w))\,\big(\nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x)\big)$;
6. the KL-shaped reward $R(x, y) = r(x, y) - \beta \log(\pi(y \mid x)/\pi_{\text{ref}}(y \mid x))$ of eq-kl-reward has $\mathbb{E}_{y \sim \pi}[R] = J(\pi)$, so RL on $R$ maximizes $J$;
7. for an autoregressive policy, the per-token log-ratios $\log \pi(y_t \mid x, y_{<t}) - \log \pi_{\text{ref}}(y_t \mid x, y_{<t})$ add up to $\log(\pi(y \mid x)/\pi_{\text{ref}}(y \mid x))$, so a penalty at every token is the same as one on the whole response.

### Assumptions

1. Finitely many responses per prompt, so sums replace integrals; the same steps hold with integrals.
2. $\pi_{\text{ref}} > 0$ everywhere, so $\pi^*$ and the log-ratios are defined. Used in items 1 to 4.
3. $\beta > 0$. Used in item 2 (dividing by $\beta$) and item 3.
4. The Bradley–Terry model, eq-bt. Used in item 4.

### Steps

1. Write $\mathrm{KL}(\pi \| \pi_{\text{ref}}) = \sum_y \pi(y) \log(\pi(y)/\pi_{\text{ref}}(y))$. By the definition of $\pi^*$, $\log \pi_{\text{ref}}(y) = \log \pi^*(y) - r(y)/\beta + \log Z$. Substituting, $J(\pi) = \sum_y \pi(y) r(y) - \beta \sum_y \pi(y)\big(\log \pi(y) - \log \pi^*(y) + r(y)/\beta - \log Z\big) = \beta \log Z - \beta\, \mathrm{KL}(\pi \| \pi^*)$, using $\sum_y \pi(y) = 1$. This is item 1.
2. By Gibbs' inequality, $\mathrm{KL}(\pi \| \pi^*) \ge 0$ with equality only at $\pi = \pi^*$, and $\beta > 0$. This is item 2, eq. 4 of Rafailov et al., whose Appendix A.1 gives the same argument.
3. Take logarithms of eq-optimal: $\log \pi^*(y) = \log \pi_{\text{ref}}(y) + r(y)/\beta - \log Z$, and multiply by $\beta$. This is item 3, eq. 5 of Rafailov et al.
4. In eq-bt the probability depends on the rewards only through $r(y_w) - r(y_l)$ (the derivation `bradley-terry-gradient`). Substituting item 3 for both, the two terms $\beta \log Z(x)$ are equal, because they depend only on $x$, and cancel. This is item 4, eq. 6 of Rafailov et al. The negative log-likelihood of $\pi_\theta$ in place of $\pi^*$ is eq. 7, which is eq-dpo. At $\pi_\theta = \pi^*$ the argument of $\sigma$ is $r(y_w) - r(y_l)$ by item 3.
5. The DPO loss is the Bradley–Terry loss of $\hat r_\theta(y_w) - \hat r_\theta(y_l)$, whose derivative in that difference is $-\sigma(\hat r_\theta(y_l) - \hat r_\theta(y_w))$ by item 2 of the derivation `bradley-terry-gradient`. The difference has the gradient $\beta(\nabla \log \pi_\theta(y_w) - \nabla \log \pi_\theta(y_l))$, since $\pi_{\text{ref}}$ does not depend on $\theta$. This is item 5, the gradient that Rafailov et al. write in Section 4.
6. $\mathbb{E}_{y \sim \pi}[r(y) - \beta \log(\pi(y)/\pi_{\text{ref}}(y))] = \mathbb{E}_\pi[r] - \beta \sum_y \pi(y) \log(\pi(y)/\pi_{\text{ref}}(y))$, which is $J(\pi)$. This is item 6: the KL term of Stiennon et al. and Ziegler et al. is a one-sample estimate of the penalty.
7. By the chain rule of probability, $\log \pi(y \mid x) = \sum_t \log \pi(y_t \mid x, y_{<t})$, and the same for $\pi_{\text{ref}}$; subtract. This is item 7.

### Result and check

$$ \pi^*(y \mid x) = \frac{\pi_{\text{ref}}(y \mid x)\, e^{r(x,y)/\beta}}{Z(x)}, \qquad r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x), $$

$$ \mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}\Big[\log \sigma\Big(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\Big)\Big] . $$

The check `dpo-from-kl-optimum.check.py` tests item 1 on random policies over 3 to 20 responses with a fixed seed, recovers $r$ from $\pi^*$, shows that a constant added to $r$ changes $Z$ and not $\pi^*$, compares the DPO loss at $\pi^*$ with the Bradley–Terry loss of $r$, compares the gradient of item 5 with central differences in the logits of a softmax policy, checks items 6 and 7, and the worked example of the page.
