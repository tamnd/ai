---
id: invariances
derives: [c-matrix-invariance, c-case-invariance, c-not-vector-invariant, c-not-data-recentering, c-weight-gradient]
kind: identity
assumes: ["epsilon is 0", "the summed inputs are a = W x, with no bias added before the normalization", "delta > 0"]
uses: [sample-mean@m2, standard-deviation@m2, frobenius-norm@m2, stochastic-gradient-descent@m2]
status: full
check: invariances.check.py
lean: null
sources:
  - {work: ba-2016-layer-normalization, locator: "Section 5.1"}
  - {work: ba-2016-layer-normalization, locator: "Table 1"}
  - {work: ba-2016-layer-normalization, locator: "Section 5.2.2"}
---

### Statement

Let $\mathrm{LN}(\mathbf{a}) = \mathbf{g} \odot (\mathbf{a} - \mu(\mathbf{a})\mathbf{1}) / \sigma(\mathbf{a}) + \mathbf{b}$ with $\epsilon = 0$, $\mathbf{a} = W\mathbf{x} \in \mathbb{R}^H$ and $W \in \mathbb{R}^{H \times D}$, and let $f(W) = \ell(\mathrm{LN}(W\mathbf{x}))$ for a differentiable loss $\ell$, summed over a set of examples. For every $\delta > 0$ and $\boldsymbol{\gamma} \in \mathbb{R}^D$:

1. weight matrix re-scaling and re-centering: $\mathrm{LN}((\delta W + \mathbf{1}\boldsymbol{\gamma}^\top)\mathbf{x}) = \mathrm{LN}(W\mathbf{x})$ for every $\mathbf{x}$;
2. single training case re-scaling: $\mathrm{LN}(W(\delta\mathbf{x})) = \mathrm{LN}(W\mathbf{x})$;
3. weight vector re-scaling is not an invariance: scaling one row of $W$ can change the output;
4. dataset re-centering is not an invariance: $\mathbf{x} \mapsto \mathbf{x} + \mathbf{c}$ for a fixed $\mathbf{c}$ can change the output;
5. the gradient: $\nabla f(\delta W + \mathbf{1}\boldsymbol{\gamma}^\top) = \frac{1}{\delta}\nabla f(W)$, the columns of $\nabla f(W)$ sum to zero ($\mathbf{1}^\top \nabla f(W) = \mathbf{0}^\top$), and $\langle W, \nabla f(W) \rangle_F = 0$. So a step of plain gradient descent $W' = W - \eta\nabla f(W)$ keeps $\mathbf{1}^\top W$ and gives $\lVert W' \rVert_F^2 = \lVert W \rVert_F^2 + \eta^2 \lVert \nabla f(W) \rVert_F^2$.

### Assumptions

1. $\epsilon = 0$. With $\epsilon > 0$, $\sigma$ becomes $\sqrt{v + \epsilon}$, and the scaling in items 1 and 2 changes the output by a factor of $\sqrt{v + \epsilon} / \sqrt{\delta^2 v + \epsilon}$, which is near 1 only when $v \gg \epsilon$. Used in steps 1, 2 and 6.
2. No bias is added before the normalization: a bias $\mathbf{c}$ in $\mathbf{a} = W\mathbf{x} + \mathbf{c}$ is not scaled with $W$ or $\mathbf{x}$. Used in steps 2 and 6.
3. $\delta > 0$; a negative $\delta$ flips the sign of $\hat{\mathbf{a}}$. Used in steps 1 and 2.
4. $\sigma(W\mathbf{x}) > 0$. Used in every step.

### Steps

1. Let $\mathbf{a}' = (\delta W + \mathbf{1}\boldsymbol{\gamma}^\top)\mathbf{x} = \delta\mathbf{a} + (\boldsymbol{\gamma}^\top\mathbf{x})\mathbf{1}$. Adding the same scalar to every entry adds it to the mean and leaves the deviations alone, and scaling by $\delta$ scales the deviations and $\sigma$ by $\delta$. So $\mu(\mathbf{a}') = \delta\mu(\mathbf{a}) + \boldsymbol{\gamma}^\top\mathbf{x}$, $\mathbf{a}' - \mu(\mathbf{a}')\mathbf{1} = \delta(\mathbf{a} - \mu(\mathbf{a})\mathbf{1})$ and $\sigma(\mathbf{a}') = \delta\sigma(\mathbf{a})$. The ratio is unchanged, which is item 1.
2. $W(\delta\mathbf{x}) = \delta\mathbf{a}$, which is step 1 with $\boldsymbol{\gamma} = \mathbf{0}$. This is item 2. Each example is scaled by its own $\delta$, since the statistics are those of one example.
3. Item 3, by a counterexample: $W = I_3$ and $\mathbf{x} = (1, 2, 3)$ give $\mathbf{a} = (1, 2, 3)$ and $\hat{\mathbf{a}} = (-1.2247, 0, 1.2247)$. Doubling the first row of $W$ gives $\mathbf{a} = (2, 2, 3)$ and $\hat{\mathbf{a}} = (-0.7071, -0.7071, 1.4142)$. The scaled row changes the deviations of one entry and not the others, so step 1 does not apply.
4. Item 4, by the same numbers: $\mathbf{c} = (1, 0, 0)$ moves $\mathbf{x} = (1, 2, 3)$ to $(2, 2, 3)$, so $\mathbf{a}$ changes from $(1, 2, 3)$ to $(2, 2, 3)$ as in step 3. A shift of the input is a shift $W\mathbf{c}$ of $\mathbf{a}$, which is a multiple of $\mathbf{1}$ only when $W\mathbf{c}$ is.
5. The first part of item 5. Write $\mathbf{d} = \partial f / \partial \mathbf{a}$ for one example. Then $\nabla f(W) = \sum \mathbf{d}\mathbf{x}^\top$ over the examples. By step 1, $\hat{\mathbf{a}}' = \hat{\mathbf{a}}$, so the upstream gradient $\mathbf{u}$ of the derivation `ln-backward` is the same. Also $s' = \delta s$, so the gradient formula of that derivation gives $\mathbf{d}' = \mathbf{d}/\delta$. Summing $\mathbf{d}'\mathbf{x}^\top$ gives $\frac{1}{\delta}\nabla f(W)$.
6. The rest of item 5. By `ln-backward`, $\mathbf{1}^\top\mathbf{d} = 0$, so $\mathbf{1}^\top\nabla f(W) = \sum (\mathbf{1}^\top\mathbf{d})\mathbf{x}^\top = \mathbf{0}^\top$. With $\epsilon = 0$, $\hat{\mathbf{a}}^\top\mathbf{d} = 0$ as well, so $\langle W, \nabla f(W) \rangle_F = \sum \mathbf{d}^\top W\mathbf{x} = \sum \mathbf{d}^\top\mathbf{a} = \sum \mathbf{d}^\top(\sigma\hat{\mathbf{a}} + \mu\mathbf{1}) = 0$. This uses $\mathbf{a} = W\mathbf{x}$, which is assumption 2. Then $\mathbf{1}^\top W' = \mathbf{1}^\top W - \eta\mathbf{1}^\top\nabla f(W) = \mathbf{1}^\top W$, and $\lVert W' \rVert_F^2 = \lVert W \rVert_F^2 - 2\eta\langle W, \nabla f(W)\rangle_F + \eta^2\lVert\nabla f(W)\rVert_F^2$, whose middle term is zero.

### Result and check

$$ \mathrm{LN}\big((\delta W + \mathbf{1}\boldsymbol{\gamma}^\top)\mathbf{x}\big) = \mathrm{LN}(W\mathbf{x}), \qquad \nabla f(\delta W + \mathbf{1}\boldsymbol{\gamma}^\top) = \tfrac{1}{\delta}\nabla f(W), \qquad \langle W, \nabla f(W)\rangle_F = 0 . $$

So the gradient shrinks as the weights grow, and plain gradient descent only makes the weights grow. This is the sense in which the norm of the weights controls the effective learning rate (Section 5.2.2 of Ba et al., 2016).
The check `invariances.check.py` tests items 1 and 2 for random $W$, $\mathbf{x}$, $\delta$ and $\boldsymbol{\gamma}$ with a fixed seed, and items 3 and 4 with the numbers of steps 3 and 4.
It compares the gradient of item 5 with central finite differences, and runs 50 steps of gradient descent to check that $\mathbf{1}^\top W$ is kept and that $\lVert W \rVert_F^2$ grows by exactly $\eta^2\lVert\nabla f\rVert_F^2$ at each step.
It also checks that $\epsilon > 0$ breaks item 1 by the factor of assumption 1.
