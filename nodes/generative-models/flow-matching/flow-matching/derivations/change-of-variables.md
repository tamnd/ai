---
id: change-of-variables
derives: [c-icov]
kind: closed-form
assumes: ["v_t is uniformly Lipschitz in x and continuous in t", "p_0 is a differentiable density", "p_t is the push-forward of p_0 by the flow of v_t"]
uses: [trace-of-a-matrix@m2, multivariate-normal-distribution@m2]
status: full
check: change-of-variables.check.py
lean: null
sources:
  - {work: chen-2018-neural-ordinary-differential-equations, locator: "p. 4", note: "Theorem 1, instantaneous change of variables: d log p(z(t)) / dt = -tr(df / dz(t))"}
  - {work: lipman-2022-flow-matching-generative-modeling, locator: "p. 2", note: "the continuity equation links a vector field and the probability path it generates"}
---

### Statement

Let $\mathbf{x}_t$ solve $\dot{\mathbf{x}}_t = v_t(\mathbf{x}_t)$, and let $p_t$ be its density. Then

$$ \frac{d}{dt}\log p_t(\mathbf{x}_t) = -\nabla\cdot v_t(\mathbf{x}_t) = -\operatorname{tr}\frac{\partial v_t}{\partial \mathbf{x}}(\mathbf{x}_t) . $$

Integrating from 0 to 1 gives

$$ \log p_1(\mathbf{x}_1) = \log p_0(\mathbf{x}_0) - \int_0^1 \nabla\cdot v_t(\mathbf{x}_t)\, dt . $$

This is the exact likelihood of a flow trained by flow matching. It needs one backward ODE solve, together with the divergence along the path, and no determinant.

### Assumptions

1. $p_t$ solves the continuity equation $\partial_t p_t + \nabla\cdot(p_t v_t) = 0$. Used in step 1.
2. The Lipschitz condition makes the flow unique and invertible, so the ODE can be run backwards from $\mathbf{x}_1$. Used in step 3.

### Steps

1. Expand the continuity equation: $\partial_t p_t = -\nabla p_t \cdot v_t - p_t\, \nabla\cdot v_t$.
2. Apply the chain rule along a path: $\frac{d}{dt}p_t(\mathbf{x}_t) = \partial_t p_t + \nabla p_t \cdot \dot{\mathbf{x}}_t = -p_t\, \nabla\cdot v_t$. Dividing by $p_t$ gives the claim.
3. Integrate in $t$. To get the log-likelihood of a data point, solve the ODE from $\mathbf{x}_1$ back to $\mathbf{x}_0$, accumulate the integral of the divergence on the way, and evaluate $\log p_0(\mathbf{x}_0)$ under the noise.
4. For the network $v_\theta = \tanh([\mathbf{x}, e(t)]W_1 + b_1)W_2 + b_2$, the Jacobian in $\mathbf{x}$ is $W_2^\top \operatorname{diag}(1 - h^2) W_{1,\mathbf{x}}^\top$. Its trace is $\sum_j (1 - h_j^2)\sum_{i \le D} W_1[i, j]\, W_2[j, i]$, which is exact and costs $O(DH)$ per point. For larger networks an unbiased stochastic estimate of the trace replaces it.

### Result and check

$$ \log p_1(\mathbf{x}_1) = \log p_0(\mathbf{x}_0) - \int_0^1 \operatorname{tr}\frac{\partial v_t}{\partial \mathbf{x}}(\mathbf{x}_t)\, dt . $$

The check `change-of-variables.check.py` drives `log_likelihood` (400 midpoint steps) with the exact `mixture_field` of a two-component mixture and a finite-difference divergence. It compares the log-likelihoods of six data points with the closed-form `mixture_log_density` at $t = 1$. It also compares the network's exact `divergence` with central differences of `velocity`.
