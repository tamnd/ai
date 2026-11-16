---
id: ln-backward
derives: [c-ln-backward, c-ln-grad-sum-orthogonal, c-ln-jacobian-norm, c-ln-sphere, eq-ln-backward]
kind: gradient
assumes: ["the mean and the variance are computed from the same vector as the output", "the gradient flows through the mean and the variance", "H >= 2 and the variance plus epsilon is positive", "H >= 3 for the spectral norm"]
uses: [backpropagation@al, sample-mean@m2, variance@m2, orthogonal-projection@m2, matrix-norm@m2]
status: full
check: ln-backward.check.py
lean: null
sources:
  - {work: ba-2016-layer-normalization, locator: "Section 3"}
  - {work: ba-2016-layer-normalization, locator: "p. 13"}
  - {work: xiong-2020-layer-normalization-transformer, locator: "Section 3.3"}
---

### Statement

Let $\mathbf{a} \in \mathbb{R}^H$ be the summed inputs of one example, with $H \ge 2$.
Let $\mu = \frac{1}{H} \sum_k a_k$, $v = \frac{1}{H} \sum_k (a_k - \mu)^2$, $s = \sqrt{v + \epsilon} > 0$, $\hat{\mathbf{a}} = (\mathbf{a} - \mu \mathbf{1}) / s$ and $\mathbf{y} = \mathbf{g} \odot \hat{\mathbf{a}} + \mathbf{b}$.
Let $\boldsymbol{\delta} = \partial \ell / \partial \mathbf{y}$ for a differentiable loss $\ell$, and $\mathbf{u} = \mathbf{g} \odot \boldsymbol{\delta}$. Then

1. $\partial \ell / \partial \mathbf{g} = \boldsymbol{\delta} \odot \hat{\mathbf{a}}$ and $\partial \ell / \partial \mathbf{b} = \boldsymbol{\delta}$, summed over the examples of a batch;
2. the Jacobian of $\hat{\mathbf{a}}$ is $J = \frac{1}{s} \big( I - \frac{1}{H} \mathbf{1}\mathbf{1}^\top - \frac{1}{H} \hat{\mathbf{a}}\hat{\mathbf{a}}^\top \big)$, a symmetric matrix, so $\partial \ell / \partial \mathbf{a} = \frac{1}{H s} \big( H \mathbf{u} - (\mathbf{1}^\top \mathbf{u}) \mathbf{1} - (\hat{\mathbf{a}}^\top \mathbf{u}) \hat{\mathbf{a}} \big)$;
3. $\mathbf{1}^\top \partial \ell / \partial \mathbf{a} = 0$ for every $\epsilon \ge 0$, and $\hat{\mathbf{a}}^\top \partial \ell / \partial \mathbf{a} = 0$ when $\epsilon = 0$;
4. when $\epsilon = 0$, $\lVert \hat{\mathbf{a}} \rVert_2 = \sqrt{H}$, and $J = \frac{1}{\sigma} P$ where $P$ is the orthogonal projection onto the complement of the span of $\mathbf{1}$ and $\hat{\mathbf{a}}$; for $H \ge 3$, $\lVert J \rVert_2 = 1/\sigma = \sqrt{H} / \lVert \mathbf{a} - \mu \mathbf{1} \rVert_2$. For $\epsilon > 0$ and $H \ge 3$, $\lVert J \rVert_2 = 1/s$.

### Assumptions

1. $\mu$ and $v$ are functions of the same $\mathbf{a}$ that is normalized, and the backward pass differentiates through them. Used in step 2.
2. $s > 0$: either $\epsilon > 0$, or the entries of $\mathbf{a}$ are not all equal. Used in steps 2 and 4.
3. $H \ge 3$ for the norms in item 4, so that the complement of the span of $\mathbf{1}$ and $\hat{\mathbf{a}}$ is not zero. With $H = 2$ that complement is zero, and $J$ has only the eigenvalues of step 5 on $\mathbf{1}$ and $\hat{\mathbf{a}}$. Used in step 5.

### Steps

1. $y_i = g_i \hat{a}_i + b_i$, so $\partial \ell / \partial g_i = \delta_i \hat{a}_i$ and $\partial \ell / \partial b_i = \delta_i$ for one example; the gain and the bias are shared by the examples of a batch, so their gradients add up over the batch. This is item 1. The upstream gradient of $\hat{\mathbf{a}}$ is $\mathbf{u} = \mathbf{g} \odot \boldsymbol{\delta}$.
2. $\partial \mu / \partial a_k = 1/H$. With $\sum_j (a_j - \mu) = 0$, $\partial v / \partial a_k = \frac{2}{H}(a_k - \mu)$, so $\partial s / \partial a_k = (a_k - \mu)/(H s) = \hat{a}_k / H$. Then $\partial \hat{a}_i / \partial a_k = \frac{1}{s}(\delta_{ik} - \frac{1}{H}) - \frac{a_i - \mu}{s^2} \frac{\hat{a}_k}{H} = \frac{1}{s}\big(\delta_{ik} - \frac{1}{H} - \frac{\hat{a}_i \hat{a}_k}{H}\big)$. This is $J$, and it is symmetric. The chain rule gives $\partial \ell / \partial \mathbf{a} = J^\top \mathbf{u} = J \mathbf{u}$, which is item 2 and eq-ln-backward. The step uses no assumption on $\epsilon$.
3. $\mathbf{1}^\top \hat{\mathbf{a}} = 0$, so $\mathbf{1}^\top J = \frac{1}{s}(\mathbf{1}^\top - \mathbf{1}^\top - 0) = 0$. For $\epsilon = 0$, $\hat{\mathbf{a}}^\top \hat{\mathbf{a}} = H v / v = H$, so $\hat{\mathbf{a}}^\top J = \frac{1}{s}(\hat{\mathbf{a}}^\top - 0 - \hat{\mathbf{a}}^\top) = 0$. This is item 3.
4. For $\epsilon = 0$, $\mathbf{1}$ and $\hat{\mathbf{a}}$ are orthogonal and both have the squared norm $H$. So $\frac{1}{H}\mathbf{1}\mathbf{1}^\top + \frac{1}{H}\hat{\mathbf{a}}\hat{\mathbf{a}}^\top$ is the orthogonal projection onto their span, and $P = I$ minus it is the projection onto the complement.
5. A nonzero orthogonal projection has the spectral norm 1, so $\lVert J \rVert_2 = 1/\sigma$, and $\sigma = \lVert \mathbf{a} - \mu\mathbf{1} \rVert_2 / \sqrt{H}$ by the definition of $v$. For $\epsilon > 0$, $J$ has the eigenvalue 0 on $\mathbf{1}$, $\frac{1}{s} \cdot \frac{\epsilon}{v + \epsilon}$ on $\hat{\mathbf{a}}$, since $\hat{\mathbf{a}}^\top\hat{\mathbf{a}} = H v/(v + \epsilon)$, and $1/s$ on the complement, so $\lVert J \rVert_2 = 1/s$ when the complement is not zero. This is item 4.
6. When $\mathbf{a}$ has mean zero, $\lVert \mathbf{a} - \mu\mathbf{1} \rVert_2 = \lVert \mathbf{a} \rVert_2$, and item 4 is the bound $\lVert J_{\mathrm{LN}}(\mathbf{a}) \rVert_2 = O(\sqrt{H} / \lVert \mathbf{a} \rVert_2)$ of Lemma 3 of Xiong et al. (2020), as an equality. In general $\lVert \mathbf{a} - \mu \mathbf{1} \rVert_2 \le \lVert \mathbf{a} \rVert_2$, so the exact norm can be larger than $\sqrt{H} / \lVert \mathbf{a} \rVert_2$ when the mean is large.

### Result and check

$$ \frac{\partial \ell}{\partial \mathbf{a}} = \frac{1}{s}\Big( I - \tfrac{1}{H}\mathbf{1}\mathbf{1}^\top - \tfrac{1}{H}\hat{\mathbf{a}}\hat{\mathbf{a}}^\top \Big) (\mathbf{g} \odot \boldsymbol{\delta}), \qquad \Big\lVert \frac{\partial \hat{\mathbf{a}}}{\partial \mathbf{a}} \Big\rVert_2 = \frac{1}{\sqrt{v + \epsilon}} . $$

The check `ln-backward.check.py` draws inputs, gains and upstream gradients with a fixed seed for several $H$ and $\epsilon$.
It compares items 1 and 2 with central finite differences to a relative error under $10^{-6}$, checks item 3 to $10^{-12}$, and checks the norms of item 4 and the eigenvalues of step 5 to $10^{-10}$.
