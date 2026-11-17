---
id: gradient
derives: [c-jacobian, c-grad-orthogonal, c-grad-eps-leak, c-gain-grad, c-grad-inverse-scale, c-prms-grad]
kind: gradient
assumes: ["the root mean square is computed from the same vector that is normalized, and the backward pass differentiates through it", "epsilon is 0 for the exact orthogonality and the 1/delta scaling"]
uses: [backpropagation@al, orthogonal-projection@m2]
status: full
check: gradient.check.py
lean: null
sources:
  - {work: zhang-2019-root-mean-square-layer-normalization, locator: "Section 4.2"}
  - {work: zhang-2019-root-mean-square-layer-normalization, locator: "p. 5"}
---

### Statement

Let $\mathbf{a} \in \mathbb{R}^n$, $1 \le k \le n$, $m_k = \frac{1}{k}\sum_{i=1}^k a_i^2$, $r = \sqrt{m_k + \epsilon} > 0$, and $\mathbf{y} = \mathbf{g}\odot\mathbf{a}/r$. Let $\mathbf{s}$ be the vector with $s_i = 1$ for $i \le k$ and $s_i = 0$ otherwise, $\boldsymbol{\delta} = \partial\ell/\partial\mathbf{y}$ and $\mathbf{u} = \mathbf{g}\odot\boldsymbol{\delta}$. Then

1. the Jacobian is $\dfrac{\partial\mathbf{y}}{\partial\mathbf{a}} = \dfrac{\mathrm{diag}(\mathbf{g})}{r}\Big(I - \dfrac{\mathbf{a}(\mathbf{s}\odot\mathbf{a})^\top}{k\,r^2}\Big)$; for RMSNorm ($k = n$, $\mathbf{s} = \mathbf{1}$) it is $\dfrac{\mathrm{diag}(\mathbf{g})}{r}\Big(I - \dfrac{\mathbf{a}\mathbf{a}^\top}{n\,r^2}\Big)$;
2. the input gradient is $\dfrac{\partial\ell}{\partial\mathbf{a}} = \dfrac{1}{r}\Big(\mathbf{u} - \dfrac{(\mathbf{a}^\top\mathbf{u})}{k\,r^2}\,\mathbf{s}\odot\mathbf{a}\Big)$, and the gain gradient is $\boldsymbol{\delta}\odot\mathbf{a}/r$;
3. $\mathbf{a}^\top\dfrac{\partial\ell}{\partial\mathbf{a}} = \dfrac{\mathbf{a}^\top\mathbf{u}}{r}\cdot\dfrac{\epsilon}{m_k + \epsilon}$; so with $\epsilon = 0$ the input gradient is orthogonal to $\mathbf{a}$, and the Jacobian maps $\mathbf{a}$ to $\mathbf{0}$;
4. with $\epsilon = 0$ and $\delta > 0$, the input gradient at $\delta\mathbf{a}$ is $1/\delta$ times the input gradient at $\mathbf{a}$.

### Assumptions

1. $r$ depends on $\mathbf{a}$, and the backward pass differentiates through it. A backward pass that treats $r$ as a constant gives $\mathbf{u}/r$ instead. Used in step 1.
2. $r > 0$. Used in every step.
3. $\epsilon = 0$ for the exact orthogonality and for item 4. Used in steps 3 and 4.

### Steps

1. $\partial m_k/\partial a_j = 2 s_j a_j / k$, so $\partial r/\partial a_j = s_j a_j/(k\,r)$. The quotient rule on $y_i = g_i a_i / r$ gives $\partial y_i/\partial a_j = \frac{g_i}{r}\big(\delta_{ij} - \frac{a_i s_j a_j}{k\,r^2}\big)$. This is item 1. With $\mathbf{s} = \mathbf{1}$ and $n r^2 = \lVert\mathbf{a}\rVert^2$ when $\epsilon = 0$, it is $\frac{g}{\mathrm{RMS}}\big(I - \frac{\mathbf{a}\mathbf{a}^\top}{n\,\mathrm{RMS}^2}\big)$, the matrix $R$ of Eq. 9 of Zhang and Sennrich with the gain in front.
2. The chain rule gives $\partial\ell/\partial\mathbf{a} = J^\top\boldsymbol{\delta}$. With $J^\top = \frac{1}{r}\big(I - \frac{(\mathbf{s}\odot\mathbf{a})\mathbf{a}^\top}{k r^2}\big)\mathrm{diag}(\mathbf{g})$, this is $\frac{1}{r}\big(\mathbf{u} - \frac{(\mathbf{a}^\top\mathbf{u})}{k r^2}\mathbf{s}\odot\mathbf{a}\big)$. The gain enters as $y_i = g_i (a_i/r)$, so $\partial\ell/\partial g_i = \delta_i a_i / r$.
3. Take the dot product of item 2 with $\mathbf{a}$. Since $\mathbf{a}^\top(\mathbf{s}\odot\mathbf{a}) = k\,m_k$, it is $\frac{\mathbf{a}^\top\mathbf{u}}{r}\big(1 - \frac{m_k}{r^2}\big) = \frac{\mathbf{a}^\top\mathbf{u}}{r}\cdot\frac{\epsilon}{m_k + \epsilon}$. With $\epsilon = 0$ it is zero for every $\mathbf{u}$. In the same way $J\mathbf{a} = \frac{\mathrm{diag}(\mathbf{g})}{r}\big(\mathbf{a} - \mathbf{a}\frac{k m_k}{k r^2}\big) = \mathbf{0}$. So a change of $\mathbf{a}$ along itself does not change the output, which matches the scaling invariance, and gradient descent on $\mathbf{a}$ moves only across the direction of $\mathbf{a}$.
4. With $\epsilon = 0$, $r(\delta\mathbf{a}) = \delta r(\mathbf{a})$, and the matrix in brackets in item 1 is the same at $\mathbf{a}$ and at $\delta\mathbf{a}$. So the Jacobian, and with it the gradient, is divided by $\delta$. This is Eq. 10 of Zhang and Sennrich, $R' = R/\delta$. If $\mathbf{a} = W\mathbf{x}$, the weight gradient is $\frac{\partial\ell}{\partial\mathbf{a}}\mathbf{x}^\top$; scaling $\mathbf{x}$ by $\delta$ divides the first factor by $\delta$ and multiplies the second by $\delta$, so the weight gradient does not change, while scaling $W$ by $\delta$ divides it by $\delta$.

### Result and check

$$ \frac{\partial\ell}{\partial\mathbf{a}} = \frac{1}{r}\Big(I - \frac{\mathbf{a}\mathbf{a}^\top}{n\,r^2}\Big)(\mathbf{g}\odot\boldsymbol{\delta}), \qquad \mathbf{a}^\top\frac{\partial\ell}{\partial\mathbf{a}} = 0 \ \ (\epsilon = 0) . $$

Item 4 is the implicit learning-rate adaptation that Zhang and Sennrich describe: a weight matrix with a large scale gets a small gradient.
The check `gradient.check.py` compares the Jacobian of item 1 with central finite differences for $n \in \{2, 3, 8, 33\}$, three values of $\epsilon$ and three values of $k$, with a relative tolerance of $10^{-6}$. It checks the identity of item 3 for each case, the gain gradient against finite differences, and item 4 for three values of $\delta$. It also checks the gradient of the worked example, $(5, 0, 2, -4)/27$.
