---
id: reflow-transport-cost
derives: [c-rf-cost]
kind: closed-form
assumes: ["(X_0, X_1) is any coupling with finite second moments and X_t = t X_1 + (1 - t) X_0", "the ODE dZ_t = v(Z_t, t) dt with v(x, t) = E[X_1 - X_0 | X_t = x] has a unique solution", "c is convex and the expectations are finite"]
uses: [conditional-expectation@m2, coupling@m2, convex-optimization@m2]
status: full
check: reflow-transport-cost.check.py
lean: null
sources:
  - {work: liu-2022-flow-straight-fast-rectified, locator: "pp. 5, 6", note: "Theorems 3.3 and 3.5: the rectified flow is marginal preserving, and E[c(Z_1 - Z_0)] <= E[c(X_1 - X_0)] for any convex cost c"}
---

### Statement

Let $Z$ be the rectified flow of the coupling $(X_0, X_1)$: $Z_0 = X_0$ and $\dot Z_t = v(Z_t, t)$, where $v(\mathbf{x}, t) = \mathbb{E}[X_1 - X_0 \mid X_t = \mathbf{x}]$. Then:

1. $Z_t$ has the law of $X_t$ for every $t$, so $(Z_0, Z_1)$ is again a coupling of the noise and the data;
2. $\mathbb{E}\, c(Z_1 - Z_0) \le \mathbb{E}\, c(X_1 - X_0)$ for every convex $c$.

Reflow repeats the construction on the new coupling $(Z_0, Z_1)$. The cost cannot rise from one round to the next, and the paths straighten.

### Assumptions

1. Marginal preservation is marginal-field applied to the Dirac conditional paths $p_t(\cdot \mid \mathbf{x}_0, \mathbf{x}_1) = \delta_{t\mathbf{x}_1 + (1 - t)\mathbf{x}_0}$. Their conditional fields are the constant $\mathbf{x}_1 - \mathbf{x}_0$, and the marginal field is $v$. Used in steps 1 and 3.
2. Jensen's inequality for the convex $c$, applied to an integral over $t$ and to a conditional expectation. Used in steps 2 and 3.

### Steps

1. By assumption 1, the flow of $v$ carries $\mathrm{Law}(X_0)$ along $\mathrm{Law}(X_t)$. Since $Z_0 = X_0$ and the ODE has a unique solution, $Z_t \sim X_t$ for all $t$.
2. Write the displacement as an integral, $Z_1 - Z_0 = \int_0^1 v(Z_t, t)\, dt$, which is an average over $t \in [0, 1]$. Jensen's inequality gives $c(Z_1 - Z_0) \le \int_0^1 c(v(Z_t, t))\, dt$.
3. Take expectations and use step 1 to replace $Z_t$ by $X_t$: $\mathbb{E}\, c(v(Z_t, t)) = \mathbb{E}\, c(v(X_t, t)) = \mathbb{E}\, c\big(\mathbb{E}[X_1 - X_0 \mid X_t]\big)$. Conditional Jensen bounds this by $\mathbb{E}\, c(X_1 - X_0)$, which does not depend on $t$.

### Result and check

$$ \mathbb{E}\, c(Z_1 - Z_0) \le \int_0^1 \mathbb{E}\, c(v(X_t, t))\, dt \le \mathbb{E}\, c(X_1 - X_0) . $$

The first inequality is an equality exactly when $v(Z_t, t)$ is constant along each path, that is, when the paths are straight. The second is an equality exactly when $X_t$ determines $X_1 - X_0$, that is, when the paths of the interpolation do not cross. The inequality says nothing about optimality: in more than one dimension, a fixed point of reflow need not be the optimal transport plan.

The check `reflow-transport-cost.check.py` uses independent pairs of noise and a two-mode mixture. It runs the exact `mixture_field` with `midpoint_sample` and:

- checks that the means and covariances of $Z_{1/2}$ and $Z_1$ match those of $X_{1/2}$ and $X_1$;
- checks that the rectified coupling has the lower cost for the squared distance, the distance, and the cube of the first coordinate's displacement;
- checks both inequalities of the chain for the squared distance.
