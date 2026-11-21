---
id: multinomial-posterior
derives: [c-multinomial-is-uniform]
kind: identity
assumes: ["the transition matrices are uniform, Q_t = (1 - beta_t) I + beta_t 11^T / K", "x_t is a one-hot row vector and x_0 is a probability row vector"]
uses: [transition-matrix@m2, markov-chain@m2, bayes-theorem@m2, conditional-probability@m2]
status: full
check: multinomial-posterior.check.py
lean: null
sources:
  - {work: austin-2021-structured-denoising-diffusion-discrete, locator: "p. 3", note: "the posterior of a D3PM in terms of Q_t and its products"}
  - {work: hoogeboom-2021-argmax-flows-multinomial-diffusion, locator: "p. 5, Eqs. 11 to 13", note: "the multinomial diffusion process, its marginal and its posterior"}
---

### Statement

Let $Q_t = \alpha_t I + (1 - \alpha_t)\mathbf{1}\mathbf{1}^\top/K$ with $\alpha_t = 1 - \beta_t$, and $\bar{Q}_{t-1} = Q_1 \cdots Q_{t-1}$.
For a one-hot $\mathbf{x}_t$ and any probability vector $\mathbf{x}_0$,

$$ \frac{\mathbf{x}_t Q_t^\top \odot \mathbf{x}_0 \bar{Q}_{t-1}}{\mathbf{x}_0 \bar{Q}_t \mathbf{x}_t^\top} = \frac{\tilde{\boldsymbol\theta}}{\sum_k \tilde\theta_k} , \qquad \tilde{\boldsymbol\theta} = \Big[\alpha_t \mathbf{x}_t + \frac{1 - \alpha_t}{K}\Big] \odot \Big[\bar\alpha_{t-1} \mathbf{x}_0 + \frac{1 - \bar\alpha_{t-1}}{K}\Big] , $$

so the D3PM posterior with uniform matrices is the multinomial diffusion posterior, and the x0-parameterized reverse steps of the two models agree.
For a one-hot $\mathbf{x}_0$ both are $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$ by Bayes' rule.

### Assumptions

1. The matrices are uniform. Used in steps 2 and 3.
2. $\mathbf{x}_t$ is one-hot and the entries of $\mathbf{x}_0$ sum to 1. Used in steps 2 and 3.

### Steps

1. By Bayes' rule and the Markov property, $q(x_{t-1} \mid x_t, x_0) \propto q(x_t \mid x_{t-1})\, q(x_{t-1} \mid x_0) = [Q_t]_{x_{t-1} x_t} [\bar{Q}_{t-1}]_{x_0 x_{t-1}}$. As vectors over $x_{t-1}$, the first factor is the column $x_t$ of $Q_t$, which is $\mathbf{x}_t Q_t^\top$, and the second is $\mathbf{x}_0 \bar{Q}_{t-1}$; the normalizer is their sum, $\mathbf{x}_0 \bar{Q}_t \mathbf{x}_t^\top$. This is the D3PM posterior, and replacing a one-hot $\mathbf{x}_0$ with a predicted distribution gives the reverse step of eq-x0-param.
2. $Q_t$ is symmetric and $\mathbf{x}_t \mathbf{1} = 1$, so $\mathbf{x}_t Q_t^\top = \alpha_t \mathbf{x}_t + (1 - \alpha_t)\mathbf{1}^\top/K$.
3. Uniform matrices commute and $(\mathbf{1}\mathbf{1}^\top/K)^2 = \mathbf{1}\mathbf{1}^\top/K$, so a product of them is $\bar\alpha I + (1 - \bar\alpha)\mathbf{1}\mathbf{1}^\top/K$ with $\bar\alpha$ the product of the $\alpha$'s: expanding $(a I + (1 - a)J)(b I + (1 - b)J)$ with $J = \mathbf{1}\mathbf{1}^\top/K$ gives $ab I + (a(1 - b) + (1 - a)b + (1 - a)(1 - b))J = ab I + (1 - ab)J$. With $\mathbf{x}_0 \mathbf{1} = 1$, $\mathbf{x}_0 \bar{Q}_{t-1} = \bar\alpha_{t-1}\mathbf{x}_0 + (1 - \bar\alpha_{t-1})\mathbf{1}^\top/K$.
4. The numerator of step 1 is therefore $\tilde{\boldsymbol\theta}$, and both sides are normalized to sum to 1, so they are equal.

### Result and check

$$ q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \operatorname{Cat}\big(\mathbf{x}_{t-1} ;\, \tilde{\boldsymbol\theta} / \textstyle\sum_k \tilde\theta_k\big) . $$

The check `multinomial-posterior.check.py` builds the products of uniform matrices for several $K$ and schedules, compares them with the closed form of step 3, compares `posterior` with the joint $q(x_{t-1}, x_t \mid x_0)$ normalized, and compares `posterior` with `multinomial_posterior` for one-hot and random probability vectors $\mathbf{x}_0$ at every step.
