---
id: rmsnorm
derives: [c-rms-equals-ln, c-rms-invariance, c-rms-not-recentering, c-rms-backward, c-rms-sphere, eq-rms-backward]
kind: gradient
assumes: ["the root mean square is computed from the same vector as the output", "epsilon is 0 for the invariances and the sphere", "n >= 2"]
uses: [sample-mean@m2, variance@m2, orthogonal-projection@m2, matrix-norm@m2]
status: full
check: rmsnorm.check.py
lean: null
sources:
  - {work: zhang-2019-root-mean-square-layer-normalization, locator: "Section 4"}
  - {work: zhang-2019-root-mean-square-layer-normalization, locator: "Section 4.1"}
  - {work: zhang-2019-root-mean-square-layer-normalization, locator: "Section 4.2"}
---

### Statement

Let $\mathbf{a} \in \mathbb{R}^n$ with $n \ge 2$, $r = \sqrt{\frac{1}{n}\sum_k a_k^2 + \epsilon} > 0$, $\hat{\mathbf{a}} = \mathbf{a}/r$ and $\mathrm{RN}(\mathbf{a}) = \mathbf{g} \odot \hat{\mathbf{a}}$. Let $\mathrm{LN}$ be layer normalization with the same $\epsilon$ and gain and no bias, $\boldsymbol{\delta} = \partial\ell/\partial\mathrm{RN}(\mathbf{a})$ and $\mathbf{u} = \mathbf{g} \odot \boldsymbol{\delta}$. Then

1. $\mathrm{LN}(\mathbf{a}) = \mathrm{RN}(\mathbf{a} - \mu\mathbf{1})$; so when the mean $\mu$ is zero, RMSNorm equals layer normalization;
2. with $\epsilon = 0$ and $\delta > 0$, $\mathrm{RN}(\delta W\mathbf{x}) = \mathrm{RN}(W\mathbf{x})$ and $\mathrm{RN}(W(\delta\mathbf{x})) = \mathrm{RN}(W\mathbf{x})$;
3. RMSNorm is not invariant to re-centering: adding the same scalar to every entry can change the output;
4. the Jacobian of $\hat{\mathbf{a}}$ is $J = \frac{1}{r}\big(I - \frac{1}{n}\hat{\mathbf{a}}\hat{\mathbf{a}}^\top\big)$, so $\partial\ell/\partial\mathbf{a} = \frac{1}{r}\big(\mathbf{u} - \frac{1}{n}\hat{\mathbf{a}}(\hat{\mathbf{a}}^\top\mathbf{u})\big)$ and $\partial\ell/\partial\mathbf{g} = \boldsymbol{\delta}\odot\hat{\mathbf{a}}$; $\lVert J \rVert_2 = 1/r$; with $\epsilon = 0$ and $\mathbf{a} = W\mathbf{x}$, $J$ is the matrix $R$ of Eq. 9 of Zhang and Sennrich (2019), and the gradient for $\delta W$ is $1/\delta$ times the gradient for $W$;
5. with $\epsilon = 0$, $\lVert\hat{\mathbf{a}}\rVert_2 = \sqrt{n}$: the normalized vector lies on the sphere of radius $\sqrt{n}$.

### Assumptions

1. $r$ is computed from the same $\mathbf{a}$ that is normalized, and the backward pass differentiates through it. Used in step 4.
2. $\epsilon = 0$ for items 2 and 5 and for the match with Eq. 9, which has no $\epsilon$. With $\epsilon > 0$, $\mathrm{RN}(\delta\mathbf{a}) \ne \mathrm{RN}(\mathbf{a})$ unless $\delta = 1$. Used in steps 2, 5 and 6.
3. $n \ge 2$, so that the complement of $\hat{\mathbf{a}}$ is not zero. Used in step 5.
4. $r > 0$. Used in every step.

### Steps

1. $\mathbf{a} - \mu\mathbf{1}$ has mean zero, so its mean square is $\frac{1}{n}\sum_k(a_k - \mu)^2 = v$, the biased variance of $\mathbf{a}$. So $\mathrm{RN}(\mathbf{a} - \mu\mathbf{1}) = \mathbf{g}\odot(\mathbf{a} - \mu\mathbf{1})/\sqrt{v + \epsilon} = \mathrm{LN}(\mathbf{a})$. With $\mu = 0$ the shift is zero, which is item 1.
2. The mean square of $\delta\mathbf{a}$ is $\delta^2$ times that of $\mathbf{a}$, so with $\epsilon = 0$, $r(\delta\mathbf{a}) = \delta r(\mathbf{a})$ for $\delta > 0$, and the ratio is unchanged. $\delta W\mathbf{x}$ and $W(\delta\mathbf{x})$ are both $\delta\mathbf{a}$. This is item 2.
3. Item 3, by a counterexample with $\mathbf{g} = \mathbf{1}$: $\mathbf{a} = (1, -1)$ has $r = 1$ and $\mathrm{RN}(\mathbf{a}) = (1, -1)$. Adding 1 to both entries gives $(2, 0)$, with $r = \sqrt{2}$ and $\mathrm{RN} = (\sqrt{2}, 0)$. Layer normalization gives $(1, -1)$ for both.
4. $\partial r/\partial a_k = a_k/(n r) = \hat{a}_k/n$, so $\partial\hat{a}_i/\partial a_k = \frac{\delta_{ik}}{r} - \frac{a_i}{r^2}\cdot\frac{\hat{a}_k}{n} = \frac{1}{r}\big(\delta_{ik} - \frac{\hat{a}_i\hat{a}_k}{n}\big)$. This is $J$, which is symmetric, so $\partial\ell/\partial\mathbf{a} = J\mathbf{u}$, which is eq-rms-backward. The gain gradient follows from $\mathrm{RN}_i = g_i\hat{a}_i$.
5. $\hat{\mathbf{a}}^\top\hat{\mathbf{a}} = n m/(m + \epsilon)$, where $m$ is the mean square of $\mathbf{a}$. So $J$ has the eigenvalue $\frac{1}{r}\cdot\frac{\epsilon}{m + \epsilon}$ on $\hat{\mathbf{a}}$ and $\frac{1}{r}$ on its complement, which is not zero for $n \ge 2$. So $\lVert J\rVert_2 = 1/r$. With $\epsilon = 0$, $\hat{\mathbf{a}}^\top\hat{\mathbf{a}} = n$, which is item 5, and $J\hat{\mathbf{a}} = 0$.
6. With $\epsilon = 0$ and $\mathbf{a} = W\mathbf{x}$, $\hat{\mathbf{a}}\hat{\mathbf{a}}^\top/n = (W\mathbf{x})(W\mathbf{x})^\top/(n\,r^2)$, so $J = \frac{1}{r}\big(I - \frac{(W\mathbf{x})(W\mathbf{x})^\top}{n\,r^2}\big)$, the $R$ of Eq. 9. For $\delta W$, $\hat{\mathbf{a}}$ is unchanged by step 2 and $r$ is multiplied by $\delta$, so $J$ and the gradient for the weights, $J\mathbf{u}\,\mathbf{x}^\top$, are divided by $\delta$. This is Eq. 10.

### Result and check

$$ \frac{\partial\ell}{\partial\mathbf{a}} = \frac{1}{r}\Big(I - \tfrac{1}{n}\hat{\mathbf{a}}\hat{\mathbf{a}}^\top\Big)(\mathbf{g}\odot\boldsymbol{\delta}), \qquad \mathrm{LN}(\mathbf{a}) = \mathrm{RN}(\mathbf{a} - \mu\mathbf{1}) . $$

Unlike layer normalization, the gradient of RMSNorm need not sum to zero. Only the radial direction $\hat{\mathbf{a}}$ is removed, and the mean direction $\mathbf{1}$ is not.
The check `rmsnorm.check.py` draws inputs with a fixed seed for several $n$ and $\epsilon$. It checks items 1, 2 and 5 to $10^{-12}$, and checks item 4 against central finite differences, against the eigenvalues of step 5 and against the $R$ of Eq. 9 with its $1/\delta$ scaling.
It checks item 3 with the numbers of step 3, and checks that the gradient of RMSNorm sums to a nonzero value for a generic input.
