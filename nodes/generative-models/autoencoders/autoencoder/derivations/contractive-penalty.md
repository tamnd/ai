---
id: contractive-penalty
derives: [c-cae-sigmoid, c-cae-linear, c-cae-cost]
kind: closed-form
assumes: ["an encoder of one layer, h = s(x W + b) with W of shape (d_x, d_h) and an elementwise activation s", "the contractive penalty is the squared Frobenius norm of the Jacobian of h in x"]
uses: [frobenius-norm@m2, matrix-multiplication@m2]
status: full
check: contractive-penalty.check.py
lean: null
sources:
  - {work: rifai-2011-contractive-auto-encoders-explicit, locator: "pp. 3 to 4", note: "the penalty, its equality with weight decay for a linear encoder, the sigmoid formula and its O(d_x d_h) cost"}
---

### Statement

1. The Jacobian of $h = s(xW + b)$ is $J_{ji} = \partial h_j / \partial x_i = s'(a_j)\, W_{ij}$, so $\lVert J_f(x) \rVert_F^2 = \sum_{j=1}^{d_h} s'(a_j)^2 \sum_{i=1}^{d_x} W_{ij}^2$.
2. For the sigmoid, $s'(a_j) = h_j (1 - h_j)$ and the penalty is $\sum_j \big(h_j (1 - h_j)\big)^2 \sum_i W_{ij}^2$, the expression of Rifai et al.
3. For a linear encoder, $s'(a) = 1$, the penalty is $\lVert W \rVert_F^2$ at every $x$: the contractive autoencoder is then the autoencoder with weight decay.
4. Over a batch of $B$ the penalty and its gradient cost $O(B\, d_x d_h)$ operations, the order of the encoder itself. With $g_j = h_j (1 - h_j)$, $c_j = \sum_i W_{ij}^2$ and the penalty $\Omega$ averaged over the batch, its gradients are $\partial \Omega / \partial W_{ij} = \frac{1}{B} \sum_b \big(2 W_{ij} g_{bj}^2 + x_{bi}\, r_{bj}\big)$ and $\partial \Omega / \partial b_j = \frac1B \sum_b r_{bj}$, with $r_{bj} = 2 c_j g_{bj}^2 (1 - 2 h_{bj})$.

### Assumptions

1. One layer with an elementwise activation. Used in step 1.
2. The sigmoid in step 2, the identity in step 3.

### Steps

1. $a_j = \sum_i x_i W_{ij} + b_j$ and $h_j = s(a_j)$, so $\partial h_j / \partial x_i = s'(a_j) W_{ij}$ by the chain rule. Summing the squares over $i$ and $j$ factors $s'(a_j)^2$ out of the sum over $i$. This is statement 1.
2. $\sigma'(a) = \sigma(a)(1 - \sigma(a))$. This is statement 2.
3. With $s' = 1$ the sum is $\sum_{ij} W_{ij}^2$. This is statement 3, and with it the remark of Rifai et al. that in the linear case keeping the weights small is the only way to contract; a sigmoid encoder can also contract by saturating, $h_j (1 - h_j) \to 0$.
4. The column norms $c_j$ cost $d_x d_h$; for each example, $g_j$ costs $O(d_h)$ once $h$ is known. For the gradient, $\Omega = \frac1B \sum_{b,j} g_{bj}^2 c_j$; differentiating $c_j$ gives $2 W_{ij} g_{bj}^2$, and differentiating $g_{bj}^2$ through $a_{bj}$ gives $2 g_{bj} \cdot g_{bj}(1 - 2h_{bj}) \cdot c_j$ times $\partial a_{bj} / \partial W_{ij} = x_{bi}$ or $\partial a_{bj} / \partial b_j = 1$, since $\frac{d}{da} h(1-h) = h(1-h)(1-2h)$. The term $x^\top r$ is one $d_x \times B$ by $B \times d_h$ product, $2 B d_x d_h$ FLOPs, the cost of the encoder's own forward product. This is statement 4.

### Result and check

$$ \lVert J_f(x) \rVert_F^2 = \sum_{j=1}^{d_h} \big(h_j (1 - h_j)\big)^2 \sum_{i=1}^{d_x} W_{ij}^2 . $$

The check `contractive-penalty.check.py` compares the closed form with the Jacobian built by finite differences and by the chain rule of the NumPy reference, for sigmoid, tanh and linear encoders, checks the gradients of statement 4 against finite differences, and checks that a linear encoder's penalty equals the weight decay at random points.
