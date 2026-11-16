---
id: lora-gradients
derives: [c-grads, c-grad-projection, c-zero-init-start, c-first-step, eq-lora-grads]
kind: gradient
assumes: ["the layer is Y = X W0^T + 1 b0^T + s X A^T B^T", "the loss L is a differentiable scalar function of Y", "W0 and b0 are constants"]
uses: [matrix-multiplication@m2, trace-of-a-matrix-definition@m2, trace-of-a-product-is-symmetric@m2, trace-is-linear@m2, backpropagation@al]
status: full
check: lora-gradients.check.py
lean: null
sources:
  - {work: hu-2021-lora-low-rank-adaptation, locator: "Section 4.1"}
---

### Statement

Let $X \in \mathbb{R}^{n \times k}$ be a batch of $n$ inputs, let $W_0 \in \mathbb{R}^{d \times k}$ and $b_0 \in \mathbb{R}^d$ be the frozen weight and bias, and let $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ be the adapter, with the scale $s > 0$.
The layer gives $Y = X W_0^\top + \mathbf{1} b_0^\top + s\, X A^\top B^\top \in \mathbb{R}^{n \times d}$.
Let $G = \partial L / \partial Y \in \mathbb{R}^{n \times d}$, and let $G_W = G^\top X \in \mathbb{R}^{d \times k}$ be the gradient that full fine-tuning would give for the weight. Then

1. $\dfrac{\partial L}{\partial B} = s\, G^\top X A^\top = s\, G_W A^\top$,
2. $\dfrac{\partial L}{\partial A} = s\, B^\top G^\top X = s\, B^\top G_W$,
3. $\dfrac{\partial L}{\partial X} = G\,(W_0 + s B A)$,
4. at the start, where $B = 0$: $\partial L / \partial A = 0$ and $\partial L / \partial B = s\, G_W A^\top$.

### Assumptions

1. The layer is the one of the statement. Used in step 1.
2. $L$ is a differentiable function of $Y$, so that $dL = \operatorname{tr}(G^\top dY)$ for every small change $dY$. Used in step 2.
3. $W_0$ and $b_0$ are constants: they get no gradient and their differentials are zero. Used in step 1.
4. At the start of training $B = 0$ (the initialization of Hu et al. 2021, Section 4.1). Used in step 7.

### Steps

1. Take the differential of $Y$ with respect to $A$, $B$ and $X$. By assumption 3 the terms in $W_0$ and $b_0$ have no differential, and by the product rule for matrix products,
   $dY = dX\, W_0^\top + s\, dX\, A^\top B^\top + s\, X\, dA^\top B^\top + s\, X A^\top dB^\top$.
2. By assumption 2, $dL = \operatorname{tr}(G^\top dY)$. By linearity of the trace, $dL$ is the sum of four terms, one for each term of step 1.
3. The term in $dB$ is $s \operatorname{tr}(G^\top X A^\top dB^\top)$. The trace of a product does not change when its factors are moved in a cycle, so it equals $s \operatorname{tr}(dB^\top\, G^\top X A^\top)$. For matrices of the same shape, $\operatorname{tr}(U^\top V)$ is the entrywise inner product $\langle U, V \rangle$, so the term is $\langle dB,\; s\, G^\top X A^\top \rangle$, and $\partial L / \partial B = s\, G^\top X A^\top$. This is item 1, since $G^\top X = G_W$.
4. The term in $dA$ is $s \operatorname{tr}(G^\top X\, dA^\top B^\top)$. Moving $B^\top$ to the front by the cyclic rule gives $s \operatorname{tr}(B^\top G^\top X\, dA^\top)$. Since $\operatorname{tr}(V\, U^\top) = \langle V, U \rangle$, the term is $\langle s\, B^\top G^\top X,\; dA \rangle$, and $\partial L / \partial A = s\, B^\top G^\top X = s\, B^\top G_W$. This is item 2.
5. The two terms in $dX$ add to $\operatorname{tr}(G^\top dX\, (W_0 + sBA)^\top)$, by the rule $(BA)^\top = A^\top B^\top$. The cyclic rule gives $\operatorname{tr}((W_0 + sBA)^\top G^\top dX) = \langle G (W_0 + sBA),\; dX \rangle$. So $\partial L / \partial X = G\,(W_0 + sBA)$, item 3: the layer passes back the same gradient as a plain linear layer with the merged weight.
6. Items 1 and 2 say that the adapter sees the full gradient $G_W$ only through two products: $G_W$ times $A^\top$ on the right for $B$, and $B^\top$ times $G_W$ on the left for $A$.
7. By assumption 4, $B = 0$ at the start. Item 2 then gives $\partial L / \partial A = s \cdot 0 \cdot G_W = 0$, and item 1 gives $\partial L / \partial B = s\, G_W A^\top$, which is not zero in general. So the first gradient step changes $B$ only, and $A$ starts to move from the second step on. This is item 4.

### Result and check

$$ \frac{\partial L}{\partial B} = s\, G^\top X A^\top, \qquad \frac{\partial L}{\partial A} = s\, B^\top G^\top X, \qquad \frac{\partial L}{\partial X} = G\,(W_0 + sBA). $$

The check `lora-gradients.check.py` draws $X$, $W_0$, $b_0$, $A$, $B$ and $G$ with a fixed seed, takes the loss $L = \langle G, Y \rangle$ (whose gradient in $Y$ is $G$), and compares the three formulas with central finite differences, to a relative error under $10^{-6}$.
It also checks item 4 with $B = 0$.
