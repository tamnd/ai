---
id: marginal-field
derives: [c-marginal]
kind: closed-form
assumes: ["p_t(x) > 0 for all x and t in [0, 1)", "each conditional field u_t(x | x_1) generates its conditional path p_t(x | x_1) through the continuity equation", "differentiation in t and x may pass under the integral over x_1"]
uses: [conditional-expectation@m2, mixture-model@m2, multivariate-normal-distribution@m2]
status: full
check: marginal-field.check.py
lean: null
sources:
  - {work: lipman-2022-flow-matching-generative-modeling, locator: "p. 3", note: "Theorem 1: the marginal vector field, eq. 8, generates the marginal probability path, eq. 6, given conditional fields that generate conditional paths"}
---

### Statement

Let $q$ be the data distribution. Define the marginal path and the marginal field by

$$ p_t(\mathbf{x}) = \int p_t(\mathbf{x} \mid \mathbf{x}_1)\, q(\mathbf{x}_1)\, d\mathbf{x}_1 , \qquad u_t(\mathbf{x}) = \int u_t(\mathbf{x} \mid \mathbf{x}_1)\, \frac{p_t(\mathbf{x} \mid \mathbf{x}_1)\, q(\mathbf{x}_1)}{p_t(\mathbf{x})}\, d\mathbf{x}_1 . $$

If each conditional field generates its conditional path, then $u_t$ generates $p_t$:

$$ \partial_t p_t + \nabla\cdot(p_t u_t) = 0 . $$

The weight $p_t(\mathbf{x} \mid \mathbf{x}_1)\, q(\mathbf{x}_1)/p_t(\mathbf{x})$ is the posterior of $\mathbf{x}_1$ given that the path is at $\mathbf{x}$ at time $t$. So $u_t(\mathbf{x}) = \mathbb{E}[u_t(\mathbf{x}_t \mid \mathbf{x}_1) \mid \mathbf{x}_t = \mathbf{x}]$. On the path of conditional-ot-field this is $\mathbb{E}[\mathbf{x}_1 - (1 - \sigma_{\min})\mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x}]$.

### Assumptions

1. Each conditional path solves $\partial_t p_t(\mathbf{x} \mid \mathbf{x}_1) + \nabla\cdot\big(p_t(\mathbf{x} \mid \mathbf{x}_1)\, u_t(\mathbf{x} \mid \mathbf{x}_1)\big) = 0$. Used in step 2.
2. $\partial_t$ and $\nabla\cdot$ commute with $\int \cdot\, q(\mathbf{x}_1)\, d\mathbf{x}_1$. Used in steps 1 and 3.

### Steps

1. Pass the time derivative under the integral: $\partial_t p_t(\mathbf{x}) = \int \partial_t p_t(\mathbf{x} \mid \mathbf{x}_1)\, q(\mathbf{x}_1)\, d\mathbf{x}_1$.
2. Replace each $\partial_t p_t(\mathbf{x} \mid \mathbf{x}_1)$ using the conditional continuity equation. This gives $\partial_t p_t(\mathbf{x}) = -\int \nabla\cdot\big(p_t(\mathbf{x} \mid \mathbf{x}_1)\, u_t(\mathbf{x} \mid \mathbf{x}_1)\big)\, q(\mathbf{x}_1)\, d\mathbf{x}_1$.
3. Take the divergence outside the integral. By the definition of $u_t$, the integral that remains is $p_t(\mathbf{x})\, u_t(\mathbf{x})$, so $\partial_t p_t(\mathbf{x}) = -\nabla\cdot\big(p_t(\mathbf{x})\, u_t(\mathbf{x})\big)$.
4. For data that is a mixture $\sum_k w_k \mathcal{N}(\mathbf{m}_k, s^2 I)$, the posterior splits by component. Component $k$ makes $\mathbf{x}_t$ normal with mean $t\mathbf{m}_k$ and variance $a^2 + t^2 s^2$, where $a = 1 - (1 - \sigma_{\min})t$. Within component $k$, the target $\mathbf{x}_1 - (1 - \sigma_{\min})\mathbf{x}_0$ and $\mathbf{x}_t$ are jointly normal. The linear regression of the target on $\mathbf{x}_t$ is $\mathbf{m}_k + \kappa_t(\mathbf{x} - t\mathbf{m}_k)$, where
   $$\kappa_t = \frac{\operatorname{Cov}(\text{target}, \mathbf{x}_t)}{\operatorname{Var}(\mathbf{x}_t)} = \frac{t s^2 - (1 - \sigma_{\min})a}{a^2 + t^2 s^2} .$$
   The marginal field is the sum of these regressions, each weighted by the responsibility $r_k(\mathbf{x})$ of its component. This is the NumPy reference's `mixture_field`.

### Result and check

$$ \partial_t p_t(\mathbf{x}) + \nabla\cdot\big(p_t(\mathbf{x})\, u_t(\mathbf{x})\big) = 0, \qquad u_t(\mathbf{x}) = \sum_k r_k(\mathbf{x})\big(\mathbf{m}_k + \kappa_t(\mathbf{x} - t\mathbf{m}_k)\big) \text{ for the mixture.} $$

The check `marginal-field.check.py` uses a two-component mixture in the plane, with $\sigma_{\min} = 0$ and $\sigma_{\min} = 0.1$ and three values of $t$. It does two things at points drawn from $p_t$:

- it confirms the continuity equation by central differences of `mixture_log_density` and `mixture_field`;
- it estimates $\mathbb{E}[\mathbf{x}_1 - (1 - \sigma_{\min})\mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x}]$ by Monte Carlo, with 400 000 data draws weighted by $p_t(\mathbf{x} \mid \mathbf{x}_1)$, and compares the estimate with `mixture_field`.
