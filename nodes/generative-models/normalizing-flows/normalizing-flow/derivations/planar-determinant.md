---
id: planar-determinant
derives: [c-planar-det, c-planar-fix]
kind: closed-form
assumes: ["h = tanh", "w is not the zero vector"]
uses: [matrix-determinant-lemma@m2, determinant-of-the-identity-matrix@m2]
status: full
check: planar-determinant.check.py
lean: null
sources:
  - {work: rezende-2015-variational-inference-normalizing-flows, locator: "p. 5", note: "eqs. (10) to (12): the planar flow and its determinant by the matrix determinant lemma"}
  - {work: rezende-2015-variational-inference-normalizing-flows, locator: "p. 10", note: "appendix A.1: w^T u >= -1 suffices for invertibility, and the modified u-hat"}
---

### Statement

The planar map $f(\mathbf{z}) = \mathbf{z} + \mathbf{u}\, h(\mathbf{w}^\top \mathbf{z} + b)$ has

$$ \det \frac{\partial f}{\partial \mathbf{z}} = 1 + \mathbf{u}^\top \boldsymbol{\psi}(\mathbf{z}), \qquad \boldsymbol{\psi}(\mathbf{z}) = h'(\mathbf{w}^\top \mathbf{z} + b)\, \mathbf{w}, $$

computed in $O(D)$ time.
With $h = \tanh$ it is invertible when $\mathbf{w}^\top \mathbf{u} \ge -1$, and for any $\mathbf{u}$ the vector $\hat{\mathbf{u}} = \mathbf{u} + \big(m(\mathbf{w}^\top \mathbf{u}) - \mathbf{w}^\top \mathbf{u}\big)\, \mathbf{w} / \lVert \mathbf{w} \rVert^2$ with $m(a) = -1 + \log(1 + e^a)$ satisfies $\mathbf{w}^\top \hat{\mathbf{u}} = m(\mathbf{w}^\top \mathbf{u}) > -1$.

### Assumptions

1. $h$ is differentiable with $0 < h' \le 1$, as $\tanh$ is. Used in steps 2 and 3.
2. $\mathbf{w} \ne \mathbf{0}$. Used in steps 3 and 4.

### Steps

1. The Jacobian is $I + \mathbf{u}\, \boldsymbol{\psi}(\mathbf{z})^\top$, a rank-one update of the identity, and the matrix determinant lemma gives $\det(I + \mathbf{u}\mathbf{v}^\top) = 1 + \mathbf{v}^\top \mathbf{u}$. Computing $\boldsymbol{\psi}$ and the inner product takes $O(D)$ operations, against $O(D^3)$ for a general determinant.
2. Write $\mathbf{z} = \mathbf{z}_\perp + \alpha\, \mathbf{w} / \lVert \mathbf{w} \rVert^2$ with $\mathbf{w}^\top \mathbf{z}_\perp = 0$, so $\mathbf{w}^\top \mathbf{z} = \alpha$. Then $\mathbf{w}^\top f(\mathbf{z}) = \alpha + (\mathbf{w}^\top \mathbf{u})\, h(\alpha + b)$, a scalar function of $\alpha$ alone, with derivative $1 + (\mathbf{w}^\top \mathbf{u})\, h'(\alpha + b)$.
3. If $\mathbf{w}^\top \mathbf{u} \ge -1$, that derivative is at least $1 - h'(\alpha + b) \ge 0$, and it is zero at most at the single point where $h' = 1$, so the scalar map is strictly increasing and a bijection of $\mathbb{R}$. Given $\mathbf{y} = f(\mathbf{z})$, solve it for $\alpha$, then $\mathbf{z}_\perp = \mathbf{y}_\perp - \mathbf{u}_\perp h(\alpha + b)$, so $f$ is invertible. If instead $\mathbf{w}^\top \mathbf{u} < -1$, the derivative $1 + (\mathbf{w}^\top \mathbf{u})\, h'$ is negative near $\alpha = -b$, where $h' = 1$, and positive for large $|\alpha|$, so the scalar map is not monotone and $f$ is not injective.
4. $\mathbf{w}^\top \hat{\mathbf{u}} = \mathbf{w}^\top \mathbf{u} + m(\mathbf{w}^\top \mathbf{u}) - \mathbf{w}^\top \mathbf{u} = m(\mathbf{w}^\top \mathbf{u})$, and $m(a) > -1$ because $\log(1 + e^a) > 0$. The fix moves $\mathbf{u}$ along $\mathbf{w}$ only and keeps its component orthogonal to $\mathbf{w}$.

### Result and check

$$ \log\left|\det \frac{\partial f}{\partial \mathbf{z}}\right| = \log\left|1 + h'(\mathbf{w}^\top \mathbf{z} + b)\, \mathbf{w}^\top \hat{\mathbf{u}}\right|, \qquad \mathbf{w}^\top \hat{\mathbf{u}} = -1 + \log\big(1 + e^{\mathbf{w}^\top \mathbf{u}}\big) . $$

The check `planar-determinant.check.py` compares the formula with the determinant of the finite-difference Jacobian in dimensions 2 to 8, confirms that $\mathbf{w}^\top \hat{\mathbf{u}} = m(\mathbf{w}^\top \mathbf{u})$, that the scalar map of step 2 is increasing with $\hat{\mathbf{u}}$ and not with a $\mathbf{u}$ for which $\mathbf{w}^\top \mathbf{u} = -3$, and inverts the fixed map by bisection on $\alpha$.
