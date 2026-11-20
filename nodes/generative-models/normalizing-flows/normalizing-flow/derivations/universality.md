---
id: universality
derives: [c-universal]
kind: closed-form
assumes: ["p_x(x) > 0 for every x in R^D", "the conditional cumulative distribution functions are differentiable in all their arguments"]
uses: [probability-density-function@m2, determinant-of-a-triangular-matrix@m2]
status: full
check: universality.check.py
lean: null
sources:
  - {work: papamakarios-2021-normalizing-flows-probabilistic-modeling, locator: "p. 5", note: "eqs. (7) to (12): the map of conditional cumulative distribution functions, its triangular Jacobian with determinant p_x(x), and the uniform result"}
  - {work: papamakarios-2021-normalizing-flows-probabilistic-modeling, locator: "p. 13", note: "autoregressive flows are universal approximators under these conditions"}
---

### Statement

Let $p_x > 0$ on $\mathbb{R}^D$ with differentiable conditional distribution functions, and define $F : \mathbb{R}^D \to (0, 1)^D$ by

$$ z_i = F_i(x_i, \mathbf{x}_{<i}) = \Pr(x_i' \le x_i \mid \mathbf{x}_{<i}) . $$

Then $F$ is a diffeomorphism onto $(0, 1)^D$ with a lower-triangular Jacobian, $\det J_F(\mathbf{x}) = p_x(\mathbf{x})$, and $\mathbf{z} = F(\mathbf{x})$ is uniform on $(0, 1)^D$.
Composing such maps, $T = F^{-1} \circ G$ turns any base density $p_u$ of the same kind into $p_x$, so an autoregressive flow with flexible enough transformers and conditioners can represent any such density.

### Assumptions

1. $p_x > 0$ everywhere, so every conditional density is positive. Used in step 2.
2. The conditional distribution functions are differentiable. Used in steps 2 and 3.

### Steps

1. The chain rule of probability gives $p_x(\mathbf{x}) = \prod_{i=1}^{D} p(x_i \mid \mathbf{x}_{<i})$.
2. $z_i$ does not depend on $x_j$ for $j > i$, so $J_F$ is lower triangular, and its diagonal is $\partial F_i / \partial x_i = p(x_i \mid \mathbf{x}_{<i}) > 0$. Each $F_i(\cdot, \mathbf{x}_{<i})$ is strictly increasing from $\mathbb{R}$ onto $(0, 1)$, so $F$ is inverted coordinate by coordinate, $x_i = F_i(\cdot, \mathbf{x}_{<i})^{-1}(z_i)$ for $i = 1, \dots, D$.
3. The determinant of a triangular matrix is the product of its diagonal: $\det J_F = \prod_i p(x_i \mid \mathbf{x}_{<i}) = p_x(\mathbf{x})$, which is nonzero, so $F$ is a diffeomorphism.
4. By the change of variables, $p_z(\mathbf{z}) = p_x(\mathbf{x})\, |\det J_F(\mathbf{x})|^{-1} = 1$ on $(0, 1)^D$. The same construction $G$ for $p_u$ makes $G(\mathbf{u})$ uniform, so $F^{-1}(G(\mathbf{u}))$ has the density $p_x$.

### Result and check

$$ \det J_F(\mathbf{x}) = \prod_{i=1}^{D} p(x_i \mid \mathbf{x}_{<i}) = p_x(\mathbf{x}), \qquad F(\mathbf{x}) \sim U(0, 1)^D . $$

The check `universality.check.py` builds $F$ for the banana density of the page in the order $(x_2, x_1)$, where $x_2 \sim \mathcal{N}(0, 4)$ and $x_1 \mid x_2 \sim \mathcal{N}(x_2^2 / 4, 1)$, compares the absolute determinant of its finite-difference Jacobian, taken in the coordinates $(x_1, x_2)$, with $p_x(\mathbf{x})$, confirms that $F$ of samples is uniform by binned counts, and maps standard normal $\mathbf{u}$ through $F^{-1} \circ \Phi$ to samples whose moments match the banana's.
The statement is about what can be represented; it says nothing about how fast training finds such a map.
