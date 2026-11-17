---
id: gradient-flow
derives: [c-unroll, c-gradient, c-gradient-bounds, c-scaled-shortcut, c-preact-form, c-postln-jacobian, eq-unroll, eq-grad, eq-jacobian, eq-backward, eq-scaled]
kind: gradient
assumes: ["x_{i+1} = λ_i x_i + F_i(x_i) for every unit i from l to L - 1, with λ_i = 1 unless stated", "each F_i is differentiable at x_i, with Jacobian J_i", "E is a scalar loss of x_L, and its gradient is a row vector"]
uses: [backpropagation@al, matrix-multiplication@m2]
status: full
check: gradient-flow.check.py
lean: null
sources:
  - {work: he-2016-identity-mappings, locator: "Section 2"}
  - {work: he-2016-identity-mappings, locator: "Section 3"}
  - {work: he-2016-identity-mappings, locator: "Section 4.1"}
  - {work: xiong-2020-layer-normalization-transformer, locator: "Section 3.3"}
---

### Statement

1. If $x_{i+1} = x_i + F_i(x_i)$ for $i = l, \dots, L-1$, then $x_L = x_l + \sum_{i=l}^{L-1} F_i(x_i)$.
2. Under the same form, $\partial x_L / \partial x_l = (I + J_{L-1}) \cdots (I + J_l)$, and for a scalar loss $E$
$$ \frac{\partial E}{\partial x_l} = \frac{\partial E}{\partial x_L} \Big( I + \frac{\partial}{\partial x_l} \sum_{i=l}^{L-1} F_i(x_i) \Big) . $$
The backward pass of one unit is $g_i = g_{i+1} + g_{i+1} J_i$, with $g_i = \partial E / \partial x_i$.
3. If $\lVert J_i \rVert_2 \le \varepsilon < 1$ for every unit, with $n = L - l$, then $(1 - \varepsilon)^n \lVert g_L \rVert \le \lVert g_l \rVert \le (1 + \varepsilon)^n \lVert g_L \rVert$. A plain stack $x_{i+1} = F_i(x_i)$ with Jacobians $J'_i$ along its own path has $\lVert g_l \rVert \le \prod_i \lVert J'_i \rVert_2\, \lVert g_L \rVert$.
4. If $x_{i+1} = \lambda_i x_i + F_i(x_i)$ with scalars $\lambda_i$, the term of $\partial x_L / \partial x_l$ that goes through no $J_i$ is $\big(\prod_{i=l}^{L-1} \lambda_i\big) I$.
5. The pre-activation unit $x_{i+1} = x_i + F_i(\hat f(x_i))$ and the pre-LN unit $x_{i+1} = x_i + F_i(\mathrm{LN}(x_i))$ have the form of statement 1, with the branch $F_i \circ \hat f$ or $F_i \circ \mathrm{LN}$. The post-LN unit $x_{i+1} = \mathrm{LN}(x_i + F_i(x_i))$ has $\partial x_{i+1} / \partial x_i = J_{\mathrm{LN}}(y_i)\,(I + J_i)$ with $y_i = x_i + F_i(x_i)$, so the term without any $J_i$ is $\prod_i J_{\mathrm{LN}}(y_i)$, not $I$.

### Assumptions

1. Each unit computes $x_{i+1} = \lambda_i x_i + F_i(x_i)$, with $\lambda_i = 1$ except in statement 4: an identity shortcut and no function after the addition, the unit of He et al. (2016, Identity Mappings, Section 2). Used in steps 1 to 5 and 7.
2. Each $F_i$ is differentiable at $x_i$, with Jacobian $J_i$. Used in steps 3 to 8.
3. $E$ is a scalar function of $x_L$; its gradient is a row vector. Used in steps 3, 4 and 6.
4. In step 8, $\mathrm{LN}$ is differentiable at the points where it is evaluated, which holds when the variance of its input is positive.

### Steps

1. By assumption 1, $x_{l+2} = x_{l+1} + F_{l+1}(x_{l+1}) = x_l + F_l(x_l) + F_{l+1}(x_{l+1})$.
2. By induction on $L$: if $x_L = x_l + \sum_{i=l}^{L-1} F_i(x_i)$, then $x_{L+1} = x_L + F_L(x_L) = x_l + \sum_{i=l}^{L} F_i(x_i)$. This proves statement 1, $x_L = x_l + \sum_{i=l}^{L-1} F_i(x_i)$.
3. One unit has $\partial x_{i+1} / \partial x_i = I + J_i$ (assumption 2). By the chain rule, $\partial x_L / \partial x_l = (I + J_{L-1}) \cdots (I + J_l)$, the factors in the order of the backward pass.
4. Differentiating statement 1 with respect to $x_l$ gives $\partial x_L / \partial x_l = I + \partial \big(\sum_i F_i(x_i)\big) / \partial x_l$, the same matrix as in step 3. Multiplying on the left by the row vector $\partial E / \partial x_L$ (assumption 3) gives the gradient of statement 2. The first term is $\partial E / \partial x_L$ itself; He et al. (Section 2) note that the second term is unlikely to be $-1$ for all the examples of a mini-batch, so the sum is unlikely to cancel.
5. Step 3 for one unit gives $g_i = g_{i+1} (I + J_i) = g_{i+1} + g_{i+1} J_i$: the backward pass of the unit adds the gradient through the branch to the gradient that the shortcut passes unchanged.
6. For statement 3, take a row vector $v$ and $\lVert J_i \rVert_2 \le \varepsilon$. By the triangle inequality, $\lVert v + v J_i \rVert \le \lVert v \rVert + \lVert v J_i \rVert \le (1 + \varepsilon) \lVert v \rVert$ and $\lVert v + v J_i \rVert \ge \lVert v \rVert - \lVert v J_i \rVert \ge (1 - \varepsilon) \lVert v \rVert$. Applying step 5 $n$ times from $g_L$ gives the two bounds. For the plain stack, $g_i = g_{i+1} J'_i$ and $\lVert g_{i+1} J'_i \rVert \le \lVert J'_i \rVert_2 \lVert g_{i+1} \rVert$, so $n$ applications give the product. With $\varepsilon < 1$ the residual bound stays away from zero for any fixed $n$, while the plain bound is at most $\varepsilon'^n$ when every $\lVert J'_i \rVert_2 \le \varepsilon'$.
7. For statement 4, one unit has $\partial x_{i+1} / \partial x_i = \lambda_i I + J_i$. Expand the product by picking $\lambda_i I$ or $J_i$ from each factor. The single term that picks no $J_i$ is $\prod_i \lambda_i\, I$. If every $\lambda_i > 1$ it grows exponentially in $L - l$; if every $\lambda_i < 1$ it shrinks exponentially, which is the argument of He et al. (Section 3).
8. For statement 5, write $G_i = F_i \circ \hat f$. Then $x_{i+1} = x_i + G_i(x_i)$, which is assumption 1 with branch $G_i$, so steps 1 to 6 hold with $J_i$ replaced by the Jacobian of $G_i$; He et al. (Section 4.1) make the same observation for their Eqn. 9. The pre-LN unit is the case $\hat f = \mathrm{LN}$. For the post-LN unit, the chain rule gives $\partial x_{i+1} / \partial x_i = J_{\mathrm{LN}}(y_i)\,(I + J_i)$. Expanding the product over the units as in step 7, the term that picks $I$ from every $(I + J_i)$ is the ordered product of the $J_{\mathrm{LN}}(y_i)$, which is not $I$ in general: the normalization stands on the shortcut path.

### Result and check

$$ \frac{\partial E}{\partial x_l} = \frac{\partial E}{\partial x_L} \prod_{i = L-1}^{l} (I + J_i) = \frac{\partial E}{\partial x_L} \Big( I + \frac{\partial}{\partial x_l} \sum_{i=l}^{L-1} F_i(x_i) \Big) . $$

The check `gradient-flow.check.py` builds random nonlinear residual stacks and compares the unrolled sum with the forward pass, the product of the $I + J_i$ with a central-difference Jacobian, the bounds of statement 3 with the measured norms, the scaled direct term for $\lambda = 0.9$ and $1.1$, and the pre-LN and post-LN Jacobians with finite differences.
