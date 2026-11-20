---
id: gradient-bound
derives: [c-jacobian, c-product, c-bound, c-vanish-sufficient, c-explode-necessary, c-gamma]
kind: bound
assumes: ["one simple recurrent layer h_t = f(x_t W + b_i + h_{t-1} U + b_h)", "the slope of the nonlinearity is bounded, |f'| <= gamma, with gamma = 1 for tanh and relu and 1/4 for the logistic sigmoid", "norms are spectral norms, the largest singular value of a matrix"]
uses: [backpropagation-through-time@al, matrix-norm@m2, singular-value-decomposition@m2]
status: full
check: gradient-bound.check.py
lean: null
sources:
  - {work: pascanu-2013-difficulty-training-recurrent, locator: "Section 2.1"}
---

### Statement

Write $J_t = \partial \mathbf{h}_t / \partial \mathbf{h}_{t-1}$ for one sequence, with $\mathbf{h}_t$ as a column vector, and $\sigma_1(U)$ for the largest singular value of $U$.

1. $J_t = \operatorname{diag}(f'(\mathbf{a}_t))\, U^\top$, where $\mathbf{a}_t$ is the pre-activation of step $t$.
2. For $k < t$, $\partial \mathbf{h}_t / \partial \mathbf{h}_k = J_t J_{t-1} \cdots J_{k+1}$, a product of $t - k$ factors.
3. If $|f'| \le \gamma$ everywhere, then $\| \partial \mathbf{h}_t / \partial \mathbf{h}_k \|_2 \le (\gamma\, \sigma_1(U))^{t-k}$.
4. Hence $\sigma_1(U) < 1 / \gamma$ is sufficient for the gradient that reaches $\mathbf{h}_k$ from step $t$ to vanish exponentially in $t - k$, and $\sigma_1(U) > 1/\gamma$ is necessary for it to explode.
5. $\gamma = 1$ for tanh and for relu, and $\gamma = 1/4$ for the logistic sigmoid.

### Assumptions

1. $\mathbf{h}_t = f(W^\top \mathbf{x}_t + U^\top \mathbf{h}_{t-1} + \mathbf{b})$ in column form, with $\mathbf{b} = \mathbf{b}_i + \mathbf{b}_h$; this is eq-step for one sequence. Used in step 1.
2. $|f'(a)| \le \gamma$ for every $a$; for relu, $f'$ is taken as 0 or 1. Used in steps 3 and 5.
3. The spectral norm is submultiplicative, $\|AB\|_2 \le \|A\|_2 \|B\|_2$, and $\|\operatorname{diag}(\mathbf{v})\|_2 = \max_i |v_i|$, $\|U^\top\|_2 = \sigma_1(U)$. Used in step 3.

### Steps

1. Coordinate $i$ of $\mathbf{h}_t$ is $f(a_{t,i})$ with $a_{t,i} = \dots + \sum_j U_{ji} h_{t-1,j}$, so $\partial h_{t,i} / \partial h_{t-1,j} = f'(a_{t,i})\, U_{ji}$. As a matrix, row $i$ is $f'(a_{t,i})$ times row $i$ of $U^\top$, which is $\operatorname{diag}(f'(\mathbf{a}_t))\, U^\top$. This is statement 1; it is eq. 5 of Pascanu et al. in the notation of the page.
2. $\mathbf{h}_t$ depends on $\mathbf{h}_k$ only through $\mathbf{h}_{t-1}$, which depends on it only through $\mathbf{h}_{t-2}$, and so on. The chain rule of [[al:backpropagation-through-time|backpropagation through time]] multiplies the step Jacobians along this single path. This is statement 2.
3. By assumption 3, $\|J_t\|_2 \le \|\operatorname{diag}(f'(\mathbf{a}_t))\|_2 \|U^\top\|_2 \le \gamma\, \sigma_1(U)$, and the norm of the product of statement 2 is at most the product of the norms. This is statement 3.
4. If $\gamma \sigma_1(U) = \eta < 1$, the bound $\eta^{t-k}$ goes to 0 exponentially, whatever the inputs and whatever the states. Conversely, if $\sigma_1(U) \le 1/\gamma$ the bound is at most 1 for every distance, so no term can grow; growth without bound therefore needs $\sigma_1(U) > 1/\gamma$. The condition is not sufficient for explosion: saturated units give $f' \approx 0$ and shrink the product however large $U$ is. This is statement 4.
5. $\tanh' = 1 - \tanh^2 \in (0, 1]$, with the maximum 1 at 0. The slope of relu is 0 or 1. The logistic sigmoid has $\sigma' = \sigma(1 - \sigma) \le 1/4$, with the maximum at 0. This is statement 5.

Pascanu et al. state the conditions with $\lambda_1$, the absolute value of the largest eigenvalue of the recurrent matrix, and their proof bounds the 2-norm of each Jacobian by the product of the norms. For a symmetric or, more generally, normal $U$, the largest singular value and the largest absolute eigenvalue are equal. For a non-normal $U$ they differ: the singular value is the one that bounds every product, while the eigenvalue only governs the powers $U^k$ in the limit of large $k$.

### Result and check

$$ \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} = \prod_{j=t}^{k+1} \operatorname{diag}(f'(\mathbf{a}_j))\, U^\top, \qquad \Big\| \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} \Big\|_2 \le \big(\gamma\, \sigma_1(U)\big)^{t-k} . $$

In the worked example, one unit with $U = 0.9$, $\gamma \sigma_1(U) = 0.9$ and the gradient over nine steps is at most $0.9^9 = 0.387$; with tanh it is $0.042$, with relu it reaches the bound. The check `gradient-bound.check.py` compares the product of statement 2 with finite differences of `forward`, tests the bound of statement 3 on random layers, both nonlinearities and a range of scales, shows a non-normal $U$ with largest eigenvalue $0.5$ whose gradient first grows above 1, and checks the slopes of statement 5.
