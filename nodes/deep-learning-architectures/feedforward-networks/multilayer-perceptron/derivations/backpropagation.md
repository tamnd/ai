---
id: backpropagation
derives: [c-delta, c-grads, c-backward-cost]
kind: gradient
assumes: ["the activation is differentiable at every pre-activation that occurs, or the gradient is taken as a subgradient (relu'(0) = 0)", "the loss is a sum over the batch of terms that depend on one row of y each"]
uses: []
status: full
check: backpropagation.check.py
lean: null
sources:
  - {work: schmidhuber-2014-deep-learning-overview, locator: "Section 5.5.1"}
---

### Statement

Write $\delta_l = \partial E / \partial z_l$ for the error of the pre-activations of layer $l$, with the output layer as layer $L$, $z_L = y$.

1. The errors satisfy the backward recursion $\delta_{l-1} = (\delta_l W_l) \odot \sigma'(z_{l-1})$, row by row over the batch, starting from $\delta_L = \partial E / \partial y$.
2. The gradients of the parameters are $\partial E / \partial W_l = \delta_l^\top h_l$ and $\partial E / \partial b_l = \mathbf{1}^\top \delta_l$, where $h_l$ is the input of layer $l$ and $h_0 = x$.
3. The backward pass costs $2 C_\text{fwd}$ FLOPs, counting only the matrix products and including the gradient of the input $x$: each layer makes two products of the size of its forward product.

### Assumptions

1. The forward pass is $z_l = h_l W_l^\top + b_l$ for each layer, and $h_{l+1} = \sigma(z_l)$ for the hidden layers (eq-layer, eq-output). Used in all steps.
2. The derivative of $\sigma$ acts element-wise. Used in step 1.

### Steps

1. Entry $(b, j)$ of $z_{l-1}$ enters the loss only through $h_l[b, j] = \sigma(z_{l-1}[b, j])$, and $h_l[b, j]$ enters only through row $b$ of $z_l$, with $\partial z_l[b, i] / \partial h_l[b, j] = W_l[i, j]$. The chain rule gives $\partial E / \partial h_l[b, j] = \sum_i \delta_l[b, i]\, W_l[i, j]$, which is the row-matrix product $\delta_l W_l$, and multiplying by $\sigma'(z_{l-1}[b, j])$ gives statement 1.
2. $W_l[i, j]$ enters every row: $\partial z_l[b, i] / \partial W_l[i, j] = h_l[b, j]$. Summing over the batch, $\partial E / \partial W_l[i, j] = \sum_b \delta_l[b, i]\, h_l[b, j] = (\delta_l^\top h_l)[i, j]$. The bias enters each row with derivative 1, so its gradient is the column sum of $\delta_l$. This is statement 2.
3. The forward product of layer $l$ is $B \times n_l$ by $n_l \times n_{l+1}$, $B n_l n_{l+1}$ multiply-adds. The product $\delta_l^\top h_l$ is $n_{l+1} \times B$ by $B \times n_l$, and $\delta_l W_l$ is $B \times n_{l+1}$ by $n_{l+1} \times n_l$: $B n_l n_{l+1}$ multiply-adds each. Two per layer is twice the forward pass. This is statement 3; leaving out $\partial E / \partial x$ saves $B n_0 n_1$ multiply-adds.

### Result and check

$$ \delta_{l-1} = (\delta_l W_l) \odot \sigma'(z_{l-1}), \qquad \frac{\partial E}{\partial W_l} = \delta_l^\top h_l, \qquad \frac{\partial E}{\partial b_l} = \mathbf{1}^\top \delta_l $$

The check `backpropagation.check.py` compares the backward pass of the NumPy reference with central finite differences for every activation, and counts the multiply-adds of the backward pass against the forward pass.
