---
id: dual-norm-perturbation
derives: [c-eps-dual, c-eps-p2, c-eps-scale, c-first-order-penalty]
kind: closed-form
assumes: ["the loss is differentiable at w with gradient g != 0", "1 < p <= infinity and 1/p + 1/q = 1", "rho > 0"]
uses: []
status: full
check: dual-norm-perturbation.check.py
lean: null
sources:
  - {work: foret-2021-sharpness-aware-minimization, locator: "p. 3"}
  - {work: foret-2021-sharpness-aware-minimization, locator: "p. 4"}
---

### Statement

Let $g = \nabla_w L_S(w) \ne 0$, $\rho > 0$, $1 < p \le \infty$ and $1/p + 1/q = 1$, and define

$$ \hat\epsilon(w) = \rho \, \operatorname{sign}(g) \, |g|^{q-1} \big/ \big( \lVert g \rVert_q^q \big)^{1/p} , $$

with the sign, absolute value and power taken entry by entry, read as $\rho\,\operatorname{sign}(g)$ at $p = \infty$.
Then (i) $\lVert \hat\epsilon \rVert_p = \rho$ and $\hat\epsilon^\top g = \rho \lVert g \rVert_q$, which is the largest value of $\epsilon^\top g$ over $\lVert \epsilon \rVert_p \le \rho$, so $\hat\epsilon$ solves the linearized inner problem $\max_{\lVert \epsilon \rVert_p \le \rho} L_S(w) + \epsilon^\top g$;
(ii) at $p = 2$ it is $\rho\, g / \lVert g \rVert_2$ and at $p = \infty$ it is $\rho\,\operatorname{sign}(g)$;
(iii) it does not change when $g$ is multiplied by any $c > 0$;
(iv) the linearized objective is $L_S(w) + \rho \lVert g \rVert_q$, and for a loss with a bounded Hessian near $w$ it differs from $\max_{\lVert \epsilon \rVert_p \le \rho} L_S(w + \epsilon)$ by $O(\rho^2)$.

### Assumptions

1. $L_S$ is differentiable at $w$ and $g \ne 0$; for (iv) it is twice differentiable with a bounded Hessian on the ball.
2. $1 < p \le \infty$, $q$ is the dual exponent, and $\rho > 0$.

### Steps

1. For $p < \infty$, $(q - 1)p = q$, so $\lVert \hat\epsilon \rVert_p^p = \rho^p \sum_i |g_i|^{(q-1)p} / \lVert g \rVert_q^q = \rho^p$.
2. $\hat\epsilon^\top g = \rho \sum_i |g_i|^q / \lVert g \rVert_q^{q/p} = \rho \lVert g \rVert_q^{q - q/p} = \rho \lVert g \rVert_q$, since $q - q/p = 1$.
3. By Hölder's inequality $\epsilon^\top g \le \lVert \epsilon \rVert_p \lVert g \rVert_q \le \rho \lVert g \rVert_q$ for every $\epsilon$ in the ball, so $\hat\epsilon$ attains the maximum. At $p = \infty$, $q = 1$ and $\rho\,\operatorname{sign}(g)$ gives $\rho \sum_i |g_i| = \rho \lVert g \rVert_1$ with $\lVert \cdot \rVert_\infty = \rho$.
4. At $p = q = 2$ the formula is $\rho\, g / (\lVert g \rVert_2^2)^{1/2}$.
5. Replacing $g$ by $cg$ multiplies the numerator by $c^{q-1}$ and the denominator by $(c^q)^{1/p} = c^{q-1}$.
6. Adding $L_S(w)$ to step 3 gives the linearized objective. Taylor's theorem bounds $|L_S(w + \epsilon) - L_S(w) - \epsilon^\top g|$ by $\tfrac12 \sup \lVert \nabla^2 L_S \rVert\, \lVert \epsilon \rVert_2^2$, which is $O(\rho^2)$ on the ball, so the two maxima differ by $O(\rho^2)$.

### Result and check

The check confirms the norm and the value $\rho\lVert g\rVert_q$ for $p \in \{1.25, 1.5, 2, 3, 4, \infty\}$, that no point sampled on the $p$-sphere beats $\hat\epsilon$, the closed forms at $p = 2$ and $p = \infty$, the concentration on the largest entry as $p \to 1$, the scale invariance, and that the gap between the linearized objective and the exact inner maximum on a smooth loss shrinks about fourfold when $\rho$ is halved.
