---
id: kl-constrained-optimum
derives: [c-optimal-policy, c-reward-reparam, c-we-optimal, eq-optimal-policy, eq-reward-reparam]
kind: closed-form
assumes: ["for each prompt x the responses y range over a finite set, or the sums are replaced by integrals that converge", "pi_ref(y|x) > 0 for every y considered", "the reward r(x, y) is finite", "beta > 0", "the policy class is unrestricted: any distribution over responses is allowed"]
uses: [kullback-leibler-divergence@m2]
status: full
check: kl-constrained-optimum.check.py
lean: null
sources:
  - {work: rafailov-2023-direct-preference-optimization, locator: "p. 4"}
  - {work: rafailov-2023-direct-preference-optimization, locator: "p. 15"}
  - {work: rafailov-2023-direct-preference-optimization, locator: "p. 16"}
---

### Statement

Fix a prompt $x$ and drop it from the notation. Let $\pi_{\mathrm{ref}}$ be a distribution over a finite set of responses with $\pi_{\mathrm{ref}}(y) > 0$, let $r$ be a finite reward and $\beta > 0$. Define

$$ J(\pi) = \sum_y \pi(y)\,r(y) - \beta\sum_y \pi(y)\log\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)}, \qquad Z = \sum_y \pi_{\mathrm{ref}}(y)\,e^{r(y)/\beta}, \qquad \pi^*(y) = \frac{\pi_{\mathrm{ref}}(y)\,e^{r(y)/\beta}}{Z} . $$

Then

1. $J(\pi) = \beta\log Z - \beta\,\mathbb D_{\mathrm{KL}}(\pi\,\Vert\,\pi^*)$ for every distribution $\pi$;
2. $\pi^*$ is the unique maximizer of $J$, and $\max_\pi J = \beta\log Z$;
3. $r(y) = \beta\log\big(\pi^*(y)/\pi_{\mathrm{ref}}(y)\big) + \beta\log Z$ for every $y$.

With an expectation over prompts the same holds prompt by prompt, since $J$ is a sum of independent per-prompt terms.

### Assumptions

1. The response set is finite, so all sums are finite. For sequences of bounded length this holds; otherwise the sums must converge. Used in all steps.
2. $\pi_{\mathrm{ref}}(y) > 0$ for every $y$, so the logarithms in step 1 are defined. Responses with $\pi_{\mathrm{ref}}(y) = 0$ get $\pi^*(y) = 0$ and can be removed.
3. $r$ is finite and $\beta > 0$, so $0 < Z < \infty$ and $\pi^*$ is a distribution with full support.
4. $\pi$ ranges over all distributions. A parametric family that cannot express $\pi^*$ has a different optimum.

### Steps

1. Divide by $\beta$ and write $r(y)/\beta = \log\big(e^{r(y)/\beta}\big)$. Then $J(\pi)/\beta = -\sum_y \pi(y)\log\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)e^{r(y)/\beta}}$.
2. Multiply and divide the argument of the logarithm by $Z$: $\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)e^{r(y)/\beta}} = \frac{\pi(y)}{\pi^*(y)}\cdot\frac1Z$. Since $\sum_y\pi(y) = 1$, $J(\pi)/\beta = -\sum_y \pi(y)\log\frac{\pi(y)}{\pi^*(y)} + \log Z = \log Z - \mathbb D_{\mathrm{KL}}(\pi\Vert\pi^*)$. This is item 1, and the rewriting in equations 12 to 14 of Rafailov et al.
3. $Z$ does not depend on $\pi$. By Gibbs' inequality $\mathbb D_{\mathrm{KL}}(\pi\Vert\pi^*) \ge 0$ with equality if and only if $\pi = \pi^*$, so $J$ is maximized exactly at $\pi^*$, with value $\beta\log Z$. This is item 2.
4. Take logarithms of the definition of $\pi^*$: $\log\pi^*(y) = \log\pi_{\mathrm{ref}}(y) + r(y)/\beta - \log Z$. Multiply by $\beta$ and rearrange to get item 3.

### Result and check

$$ \pi^*(y\mid x) = \frac{1}{Z(x)}\,\pi_{\mathrm{ref}}(y\mid x)\exp\!\Big(\frac{r(x, y)}{\beta}\Big), \qquad r(x, y) = \beta\log\frac{\pi^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)} + \beta\log Z(x) . $$

The check `kl-constrained-optimum.check.py` draws 2000 problems with 2 to 11 responses, reward scales from 0.1 to 5 and $\beta$ from 0.05 to 20.
On each it checks that $\pi^*$ is a distribution, that $J(\pi^*) = \beta\log Z$, item 3 to $10^{-8}$, that five random policies fall short of $J(\pi^*)$ by $\beta\,\mathbb D_{\mathrm{KL}}(\pi\Vert\pi^*)$, and that small perturbations of $\pi^*$ do not improve $J$.
It evaluates the worked example of the page, $\pi_{\mathrm{ref}} = (0.5, 0.3, 0.2)$, $r = (1, 0, -1)$, which at $\beta = 0.5$ gives $Z = 4.021595$ and $\pi^* = (0.918672, 0.074597, 0.006730)$, and at $\beta = 10$ gives $(0.534647, 0.290261, 0.175093)$.
