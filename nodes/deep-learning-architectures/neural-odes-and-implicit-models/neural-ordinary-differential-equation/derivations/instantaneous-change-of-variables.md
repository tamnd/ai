---
id: instantaneous-change-of-variables
derives: [c-cnf, c-trace-cost]
kind: identity
assumes: ["f is uniformly Lipschitz continuous in h and continuous in t", "f is continuously differentiable in h", "h(0) has a density p"]
uses: []
status: full
check: instantaneous-change-of-variables.check.py
lean: null
sources:
  - {work: chen-2018-neural-ordinary-differential-equations, locator: "Section 4"}
---

### Statement

1. If $h(t)$ follows eq-ode and $h(0)$ has the density $p$, the log density along a trajectory changes as $d \log p(h(t)) / dt = -\operatorname{tr}(\partial f / \partial h)$ (eq-cnf), so $\log p(h(T)) = \log p(h(0)) - \int_0^T \operatorname{tr}(\partial f / \partial h(t))\, dt$.
2. For the field eq-field, $\operatorname{tr}(\partial f / \partial h) = \sum_{j=1}^{H} (1 - s_j^2)\, (W_1 W_2)_{jj}$ with $s = \tanh(W_1 h + t u + b_1)$. With the $H$ numbers $(W_1 W_2)_{jj}$ computed once, in $D H$ multiply-adds, the trace costs $O(H)$ per example, against $O(D^2 H)$ to form the Jacobian and $O(D^3)$ for a determinant.

### Assumptions

1. The Lipschitz condition makes the flow map $\phi_t: h(0) \mapsto h(t)$ a bijection with a differentiable inverse. Used in step 1.
2. $J(t) = \partial h(t) / \partial h(0)$ solves $dJ/dt = (\partial f / \partial h) J$ with $J(0) = I$, the derivative of eq-ode with respect to the initial state. Used in step 2.

### Steps

1. By the change of variables for the bijection $\phi_t$, $p_t(h(t)) = p(h(0)) / \lvert \det J(t) \rvert$, so $\log p_t(h(t)) = \log p(h(0)) - \log \lvert \det J(t) \rvert$.
2. Jacobi's formula gives $d \det J / dt = \det J \operatorname{tr}(J^{-1} dJ/dt) = \det J \operatorname{tr}(J^{-1} (\partial f / \partial h) J) = \det J \operatorname{tr}(\partial f / \partial h)$. Since $\det J(0) = 1$ and $\det J$ is continuous and never zero, it stays positive, and $d \log \det J / dt = \operatorname{tr}(\partial f / \partial h)$. With step 1 this is statement 1.
3. For eq-field, $\partial f / \partial h = W_2 \operatorname{diag}(1 - s^2) W_1$. The trace is invariant under cyclic shifts, so $\operatorname{tr}(W_2 \operatorname{diag}(1 - s^2) W_1) = \operatorname{tr}(\operatorname{diag}(1 - s^2) W_1 W_2) = \sum_j (1 - s_j^2) (W_1 W_2)_{jj}$, and $(W_1 W_2)_{jj} = \sum_i (W_1)_{ji} (W_2)_{ij}$ costs $D$ multiply-adds for each $j$. This is statement 2.

### Result and check

$$ \frac{d \log p(h(t))}{dt} = -\operatorname{tr}\Big(\frac{\partial f}{\partial h(t)}\Big), \qquad \operatorname{tr}\Big(\frac{\partial f}{\partial h}\Big) = \sum_{j} (1 - s_j^2) (W_1 W_2)_{jj} $$

The check `instantaneous-change-of-variables.check.py` compares the trace of the NumPy reference with the trace of a finite-difference Jacobian, the integrated change of log density with $-\log \lvert \det \rvert$ of the Jacobian of the flow map, and, for a linear field, $\log \det e^{TA}$ with $T \operatorname{tr} A$.
