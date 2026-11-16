---
id: ppo-clip
derives: [c-clip-lower-bound, c-clip-flat, c-clip-first-step]
kind: identity
assumes: ["rho = pi_theta(a|s) / pi_theta_old(a|s) > 0", "0 <= eps < 1", "the advantage A does not depend on theta"]
uses: [ppo@al, policy-gradient-theorem@al]
status: full
check: ppo-clip.check.py
lean: null
sources:
  - {work: schulman-2017-proximal-policy-optimization, locator: "Section 3"}
---

### Statement

For one sample, let $\rho = \pi_\theta(a \mid s)/\pi_{\theta_{\text{old}}}(a \mid s) > 0$, an advantage $A$ and $0 \le \epsilon < 1$, and $\ell(\rho) = \min\big(\rho A,\ \mathrm{clip}(\rho, 1 - \epsilon, 1 + \epsilon) A\big)$, the term of eq-ppo-clip. Then

1. $\ell(\rho) \le \rho A$, with equality when $1 - \epsilon \le \rho \le 1 + \epsilon$;
2. if $A > 0$, then $\ell(\rho) = (1 + \epsilon) A$ for $\rho > 1 + \epsilon$; if $A < 0$, then $\ell(\rho) = (1 - \epsilon) A$ for $\rho < 1 - \epsilon$; there the derivative in $\theta$ is 0. On the other side of the interval, $\ell(\rho) = \rho A$;
3. at $\theta = \theta_{\text{old}}$, $\rho = 1$, and $\nabla_\theta \ell = A\, \nabla_\theta \log \pi_\theta(a \mid s)$, the term of the policy gradient.

### Assumptions

1. $\rho > 0$, since both policies give the sampled action a positive probability.
2. $0 \le \epsilon < 1$, so $1 - \epsilon \ge 0$ and the interval contains 1.
3. $A$ is computed from the old policy and the value function and is held fixed, so it has no gradient in $\theta$.

### Steps

1. Inside the interval, $\mathrm{clip}(\rho) = \rho$ and both terms of the minimum are $\rho A$. Outside it, the minimum is at most its first term $\rho A$. This is item 1, the pessimistic bound that Schulman et al. describe.
2. Take $A > 0$ and $\rho > 1 + \epsilon$: the terms are $\rho A$ and $(1 + \epsilon) A$, and the second is smaller, so $\ell = (1 + \epsilon) A$, which does not depend on $\theta$. For $\rho < 1 - \epsilon$ the terms are $\rho A$ and $(1 - \epsilon) A$, and the first is smaller. Multiplying by $A < 0$ reverses the inequalities, which gives the two cases with $A < 0$. This is item 2: the clipping removes the incentive to move $\rho$ further from 1 only in the direction that would raise the objective.
3. At $\rho = 1$, which is inside the interval, $\ell = \rho A$ near $\theta_{\text{old}}$, and $\nabla_\theta \rho = \rho\, \nabla_\theta \log \pi_\theta$, which is $\nabla_\theta \log \pi_\theta$ at $\rho = 1$. This is item 3.

### Result and check

$$ \min\big(\rho A,\ \mathrm{clip}(\rho, 1 - \epsilon, 1 + \epsilon) A\big) = \begin{cases} (1 + \epsilon) A & A > 0,\ \rho > 1 + \epsilon, \\ (1 - \epsilon) A & A < 0,\ \rho < 1 - \epsilon, \\ \rho A & \text{otherwise.} \end{cases} $$

The check `ppo-clip.check.py` tests the bound and the three cases on 20,000 random pairs of $\rho$ and $A$ with a fixed seed for four values of $\epsilon$, the derivative in $\log \pi$ by central differences, the gradient at $\rho = 1$, and the worked example of the page.
