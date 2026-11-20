---
id: identity-initialization
derives: [c-irnn-state, c-irnn-grad, c-irnn-scaled, c-tanh-identity]
kind: identity
assumes: ["one simple recurrent layer with relu, the recurrent matrix U = c I, zero biases and zero input", "for statements 1 and 2, c = 1 and an initial state with nonnegative coordinates", "for statement 4, tanh in place of relu"]
uses: [relu@al, backpropagation-through-time@al]
status: full
check: identity-initialization.check.py
lean: null
sources:
  - {work: le-2015-simple-way-initialize-recurrent, locator: "Section 2"}
---

### Statement

Take eq-step with relu, $U = cI$, $\mathbf{b}_i = \mathbf{b}_h = 0$ and $\mathbf{x}_t = 0$ for every step.

1. If $c = 1$ and $\mathbf{h}_0 \ge 0$, the state never changes: $\mathbf{h}_t = \mathbf{h}_0$ for every $t$.
2. If moreover every coordinate of $\mathbf{h}_0$ is positive, $\partial \mathbf{h}_t / \partial \mathbf{h}_k = I$ for every $k < t$: the gradient is carried back unchanged over any number of steps.
3. If $0 < c < 1$, $\mathbf{h}_t = c^t \mathbf{h}_0$ and $\partial \mathbf{h}_t / \partial \mathbf{h}_k = c^{t-k} I$: the layer forgets at the rate $c$.
4. With tanh in place of relu and $c = 1$, the state is not kept: a coordinate that starts at $h_0 \in (0, 1]$ decreases at every step, and $h_t \sqrt{2t/3} \to 1$.

### Assumptions

1. $\operatorname{relu}(a) = \max(a, 0)$, with slope 1 for $a > 0$ and 0 for $a \le 0$. Used in steps 1 to 3.
2. $U = cI$, zero biases and zero inputs, so that $\mathbf{a}_t = c\, \mathbf{h}_{t-1}$. Used in every step.
3. The step Jacobian of the derivation gradient-bound statement 1, $J_t = \operatorname{diag}(f'(\mathbf{a}_t))\, U^\top$. Used in steps 2 and 3.

### Steps

1. With $c = 1$, $\mathbf{a}_t = \mathbf{h}_{t-1}$. If $\mathbf{h}_{t-1} \ge 0$, $\operatorname{relu}(\mathbf{h}_{t-1}) = \mathbf{h}_{t-1}$, so $\mathbf{h}_t = \mathbf{h}_{t-1} \ge 0$, and by induction from $\mathbf{h}_0 \ge 0$ the state is constant. This is statement 1, the observation of Le et al. that an IRNN with no input stays in the same state indefinitely.
2. If $\mathbf{h}_0 > 0$, every pre-activation equals $\mathbf{h}_0 > 0$, so $f'(\mathbf{a}_t) = \mathbf{1}$ and $J_t = I \cdot I^\top = I$. The product of statement 2 of the derivation gradient-bound is $I$. A coordinate with $h_{0,i} = 0$ has slope 0 and passes no gradient, which is why the statement asks for positive coordinates. This is statement 2: the error derivatives stay constant as they are backpropagated through time.
3. With $0 < c < 1$ and $\mathbf{h}_{t-1} \ge 0$, $\mathbf{a}_t = c\, \mathbf{h}_{t-1} \ge 0$ and $\mathbf{h}_t = c\, \mathbf{h}_{t-1}$, so $\mathbf{h}_t = c^t \mathbf{h}_0$. On positive coordinates $J_t = cI$, and the product over $t - k$ steps is $c^{t-k} I$. This is statement 3, the scaled-down identity that Le et al. use when long-range effects should be forgotten.
4. With tanh, $h_t = \tanh(h_{t-1}) < h_{t-1}$ for $h_{t-1} > 0$, and $h_t > 0$, so the sequence decreases to its only fixed point $0$. Near 0, $\tanh h = h - h^3/3 + O(h^5)$, so $1/h_t^2 = 1/h_{t-1}^2 + 2/3 + O(h_{t-1}^2)$. Summing, $1/h_t^2 = 2t/3 + o(t)$, which is $h_t \sqrt{2t/3} \to 1$. The state decays as $1/\sqrt{t}$, slowly but without bound, and the slope $1 - h_t^2$ is below 1 at every step. This is statement 4: the identity trick needs the linear part of relu.

### Result and check

$$ \text{relu},\ U = I:\ \mathbf{h}_t = \mathbf{h}_0,\ \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} = I; \qquad U = cI:\ \mathbf{h}_t = c^t \mathbf{h}_0; \qquad \text{tanh},\ U = I:\ h_t \sim \sqrt{3 / (2t)} . $$

The check `identity-initialization.check.py` runs `irnn_init` with zero input for 1000 steps and compares the state and the gradient of `backward` with $\mathbf{h}_0$ and the identity bit for bit, runs the scaled identity against $c^t \mathbf{h}_0$, and runs the tanh layer for $10^5$ steps against $\sqrt{3/(2t)}$.
