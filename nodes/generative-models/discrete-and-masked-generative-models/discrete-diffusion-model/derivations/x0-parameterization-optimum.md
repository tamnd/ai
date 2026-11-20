---
id: x0-parameterization-optimum
derives: [c-x0-optimum]
kind: proof
assumes: ["one token, so the prediction can depend only on x_t", "the prior p(x_T) is the marginal q(x_T) of the forward chain", "every marginal q(x_t) that is divided by is positive"]
uses: [markov-chain@m2, bayes-theorem@m2, joint-probability-distribution@m2, kullback-leibler-divergence@m2, evidence-lower-bound@al]
status: full
check: x0-parameterization-optimum.check.py
lean: null
sources:
  - {work: austin-2021-structured-denoising-diffusion-discrete, locator: "p. 5, Eqs. 4 and 5", note: "the x0-parameterization of the reverse step and the auxiliary loss"}
---

### Statement

Let $q(x_0)$ be a distribution over the states of one token and $q(x_{1:T} \mid x_0)$ a forward chain with matrices $Q_1, \dots, Q_T$, and let the model be $p(x_T) = q(x_T)$ and the x0-parameterized steps $p(x_{t-1} \mid x_t) \propto \sum_{\tilde x_0} q(x_{t-1}, x_t \mid \tilde x_0)\, \tilde p(\tilde x_0 \mid x_t)$, with $q(x_{t-1}, x_t \mid \tilde x_0) = [\bar Q_{t-1}]_{\tilde x_0 x_{t-1}} [Q_t]_{x_{t-1} x_t}$ and $\bar Q_0 = I$.

1. For any prediction, $\mathbb{E}_{q(x_0)}[L_{\text{vb}}(x_0)] = H(q) + \mathrm{KL}\big(q(x_{0:T}) \,\|\, p(x_{0:T})\big) \ge H(q)$.
2. With $\tilde p(\cdot \mid x_t) = q(x_0)$ for every $x_t$, every reverse step is the true reverse step $q(x_{t-1} \mid x_t)$, so $p(x_{0:T}) = q(x_{0:T})$ and the expected bound is exactly $H(q)$, its minimum.
3. With the posterior $\tilde p(\cdot \mid x_t) = q(x_0 \mid x_t)$, the reverse step is proportional to $\sum_{\tilde x_0} q(x_{t-1}, x_t \mid \tilde x_0)\, q(\tilde x_0)\, [\bar Q_t]_{\tilde x_0 x_t}$. Under absorbing noise $[\bar Q_t]_{\tilde x_0 x_t}$ is the same for every $\tilde x_0$ with a nonzero term, so the step is again the true one; under uniform noise it is not, and the bound exceeds $H(q)$ unless $q$ is uniform.

### Assumptions

1. One token. Used in step 4: the prediction depends on nothing but $x_t$.
2. $p(x_T) = q(x_T)$. Used in step 3.
3. The marginals divided by are positive. Used in step 3.

### Steps

1. By definition $L_{\text{vb}}(x_0) = \mathbb{E}_{q(x_{1:T} \mid x_0)}\big[\log q(x_{1:T} \mid x_0) - \log p(x_{0:T})\big]$, which eq-vlb splits into its KL terms. Averaging over $q(x_0)$ and adding and subtracting $\log q(x_0)$ inside gives $\mathbb{E}_q[\log q(x_{0:T}) - \log p(x_{0:T})] - \mathbb{E}_q[\log q(x_0)] = \mathrm{KL}(q(x_{0:T}) \,\|\, p(x_{0:T})) + H(q)$. The KL divergence is non-negative and zero only when the joints agree, which proves 1.
2. With $\tilde p = q(x_0)$, the numerator is $\sum_{\tilde x_0} q(\tilde x_0)\, q(x_{t-1} \mid \tilde x_0)\, q(x_t \mid x_{t-1}) = q(x_{t-1}, x_t)$, by the Markov property. Normalizing over $x_{t-1}$ gives $q(x_{t-1} \mid x_t)$. At $t = 1$, $\bar Q_0 = I$ and the same sum is $q(x_0)[Q_1]_{x_0 x_1} = q(x_0, x_1)$, so the last step is $q(x_0 \mid x_1)$.
3. The model is then $q(x_T) \prod_t q(x_{t-1} \mid x_t)$, which by the chain rule of probability, written backwards, is $q(x_{0:T})$. The KL of step 1 is zero, which proves 2, and since no model does better than $H(q)$ the prediction is optimal.
4. The posterior is $q(\tilde x_0 \mid x_t) = q(\tilde x_0)[\bar Q_t]_{\tilde x_0 x_t} / q(x_t)$. Inserting it, the factor $1/q(x_t)$ cancels in the normalization, and the numerator is $\sum_{\tilde x_0} q(\tilde x_0)[\bar Q_{t-1}]_{\tilde x_0 x_{t-1}}[Q_t]_{x_{t-1} x_t}[\bar Q_t]_{\tilde x_0 x_t}$: the term $q(x_t \mid \tilde x_0)$ appears twice, once inside $q(x_{t-1}, x_t \mid \tilde x_0)$ and once in the posterior.
5. Under absorbing noise with mask $m$, if $x_t \ne m$ then $x_t = \tilde x_0$ is the only term, and both steps put all their mass on $x_{t-1} = x_t$. If $x_t = m$ then $[\bar Q_t]_{\tilde x_0 m} = 1 - \bar\alpha_t$ for every data state $\tilde x_0$, a constant that cancels. In both cases the step is that of step 2.
6. Under uniform noise $[\bar Q_t]_{\tilde x_0 x_t} = \bar\alpha_t[\tilde x_0 = x_t] + (1 - \bar\alpha_t)/K$ depends on $\tilde x_0$ when $\bar\alpha_t > 0$, so the weights $q(\tilde x_0)$ of step 2 are tilted towards $\tilde x_0 = x_t$, and the step differs from $q(x_{t-1} \mid x_t)$ whenever $q$ is not uniform. The joints then differ and, by step 1, the expected bound exceeds $H(q)$.

### Result and check

$$ \min_{\tilde p}\ \mathbb{E}_{q(x_0)}[L_{\text{vb}}] = H(q) , \quad \text{attained by } \tilde p(\cdot \mid x_t) = q(x_0) . $$

In a sequence, the same argument with the other tokens as context makes the best prediction the distribution of the clean token given the corrupted others, not the posterior given the token's own noisy value.
The check `x0-parameterization-optimum.check.py` computes the exact bound of `d3pm_bound` for a random data distribution under uniform and absorbing noise: with the data distribution as the prediction it equals $H(q)$ to $10^{-12}$; with `posterior_mean` it equals $H(q)$ under absorbing noise and exceeds it under uniform noise; random predictions always exceed it; and the excess equals the KL divergence between the joint distributions of the forward chain and the model, computed by enumerating every trajectory.
