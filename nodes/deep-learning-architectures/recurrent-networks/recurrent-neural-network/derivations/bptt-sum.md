---
id: bptt-sum
derives: [c-bptt-recursion, c-bptt-sum, c-bptt-cost, c-bptt-memory]
kind: identity
assumes: ["one simple recurrent layer h_t = f(x_t W + b_i + h_{t-1} U + b_h), the same W, U, b_i and b_h at every step", "a loss L that depends on the states h_1, ..., h_T", "a multiply-add counts as 2 FLOPs and only the matrix products are counted"]
uses: [backpropagation@al, backpropagation-through-time@al, matrix-multiplication@m2]
status: full
check: bptt-sum.check.py
lean: null
sources:
  - {work: lipton-2015-critical-review-recurrent, locator: "Section 3"}
---

### Statement

Write $\mathbf{g}_t = \partial L / \partial \mathbf{h}_t$ for the whole gradient with respect to the state of step $t$, $\mathbf{e}_t$ for the part that comes from the loss at step $t$ directly, and $\boldsymbol{\delta}_t = \mathbf{g}_t \odot f'(\mathbf{a}_t)$, all as row vectors of one sequence.

1. $\mathbf{g}_T = \mathbf{e}_T$ and $\mathbf{g}_{t-1} = \mathbf{e}_{t-1} + \boldsymbol{\delta}_t U^\top$ for $t = T, \dots, 2$; the same recursion gives $\partial L / \partial \mathbf{h}_0 = \boldsymbol{\delta}_1 U^\top$.
2. The gradients of the shared parameters are sums over the steps: $\partial L / \partial U = \sum_t \mathbf{h}_{t-1}^\top \boldsymbol{\delta}_t$, $\partial L / \partial W = \sum_t \mathbf{x}_t^\top \boldsymbol{\delta}_t$ and $\partial L / \partial \mathbf{b}_i = \partial L / \partial \mathbf{b}_h = \sum_t \boldsymbol{\delta}_t$. Each term is the gradient with respect to the copy of the parameter used at step $t$ in the unrolled network.
3. Over a batch of $B$ sequences, the backward pass costs $4BTd_h(d_x + d_h)$ FLOPs in its products, twice the forward pass, or $2BTd_h(2d_h + d_x)$ when the gradient with respect to the inputs is not needed.
4. It needs the $T$ states, or the $T$ pre-activations, of the forward pass: $BTd_h$ numbers that grow with the length of the sequence.

### Assumptions

1. The layer of eq-step, with parameters that do not depend on $t$. Used in step 2.
2. $L$ depends on the parameters only through the states. Used in steps 1 and 2.
3. Products cost $2mkn$ FLOPs, as in the derivation parameter-and-flop-count. Used in step 3.

### Steps

1. $L$ depends on $\mathbf{h}_{t-1}$ directly, through $\mathbf{e}_{t-1}$, and through $\mathbf{h}_t$, which depends on it through $\mathbf{a}_t = \dots + \mathbf{h}_{t-1} U$. By the chain rule of [[al:backpropagation|backpropagation]], $\mathbf{g}_{t-1} = \mathbf{e}_{t-1} + (\mathbf{g}_t \odot f'(\mathbf{a}_t))\, U^\top$. The last state has no later step, so $\mathbf{g}_T = \mathbf{e}_T$. The initial state only enters step 1. This is statement 1, the backward recursion of [[al:backpropagation-through-time|backpropagation through time]].
2. Unroll the layer: give step $t$ its own copies $U_t$, $W_t$, $\mathbf{b}_{i,t}$ and $\mathbf{b}_{h,t}$, and recover the layer by setting all the copies equal. The unrolled network is a feedforward network of depth $T$, and the layer is its composition with the map $U \mapsto (U, \dots, U)$. The Jacobian of that map is a stack of identities, so the chain rule gives $\partial L / \partial U = \sum_t \partial L / \partial U_t$. In the unrolled network $U_t$ enters only $\mathbf{a}_t = \dots + \mathbf{h}_{t-1} U_t$, so $\partial L / \partial U_t = \mathbf{h}_{t-1}^\top \boldsymbol{\delta}_t$. The same argument gives the sums for $W$ and the biases. This is statement 2.
3. At each step, the backward pass computes $\boldsymbol{\delta}_t U^\top$ ($2Bd_h^2$), $\boldsymbol{\delta}_t W^\top$ for the inputs ($2Bd_hd_x$), $\mathbf{h}_{t-1}^\top \boldsymbol{\delta}_t$ ($2Bd_h^2$) and $\mathbf{x}_t^\top \boldsymbol{\delta}_t$ ($2Bd_xd_h$). The sum is $4Bd_h(d_x + d_h)$ per step, twice the $2Bd_h(d_x + d_h)$ of the forward step; without the input gradient it is $2Bd_h(2d_h + d_x)$. This is statement 3.
4. Statement 2 needs $\mathbf{h}_{t-1}$ at every step, and $\boldsymbol{\delta}_t$ needs $f'(\mathbf{a}_t)$, which can be computed from $\mathbf{h}_t$ for tanh and relu. Keeping one of the two for every step stores $BTd_h$ numbers. Truncated backpropagation through time stops the recursion of statement 1 after a fixed number of steps and keeps only that many states, at the price of dropping the terms of the sum that connect steps further apart. This is statement 4.

### Result and check

$$ \mathbf{g}_{t-1} = \mathbf{e}_{t-1} + (\mathbf{g}_t \odot f'(\mathbf{a}_t))\, U^\top, \qquad \frac{\partial L}{\partial U} = \sum_{t=1}^{T} \mathbf{h}_{t-1}^\top \big(\mathbf{g}_t \odot f'(\mathbf{a}_t)\big) . $$

The check `bptt-sum.check.py` compares `backward` with finite differences on every parameter, input and initial-state entry, perturbs the copy of $U$ used at one step at a time and checks that the per-step gradients add up to $\partial L / \partial U$, and replays the products of `backward` to count its FLOPs.
