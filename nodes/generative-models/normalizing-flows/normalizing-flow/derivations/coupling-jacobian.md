---
id: coupling-jacobian
derives: [c-coupling-jac, c-coupling-inverse]
kind: closed-form
assumes: ["the mask splits the D coordinates into a kept set K and a transformed set T", "s and t are any differentiable functions of the kept coordinates"]
uses: [determinant-of-a-triangular-matrix@m2, determinant-of-a-block-triangular-matrix@m2]
status: full
check: coupling-jacobian.check.py
lean: null
sources:
  - {work: dinh-2016-density-estimation-real-nvp, locator: "pp. 4 and 5", note: "eqs. (4) to (8): the affine coupling layer, its triangular Jacobian, its determinant and its inverse"}
  - {work: dinh-2014-nice, locator: "p. 2", note: "the additive coupling layer and its unit Jacobian determinant"}
---

### Statement

The affine coupling layer $y_K = x_K$, $y_T = x_T \odot \exp\big(s(x_K)\big) + t(x_K)$ has, after ordering the coordinates as $(K, T)$, the block lower-triangular Jacobian

$$ \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \begin{pmatrix} I & 0 \\ \partial y_T / \partial x_K & \operatorname{diag}\big(\exp s(x_K)\big) \end{pmatrix}, \qquad \log\left|\det \frac{\partial \mathbf{y}}{\partial \mathbf{x}}\right| = \sum_{j \in T} s_j(x_K) . $$

Its inverse is $x_K = y_K$, $x_T = \big(y_T - t(y_K)\big) \odot \exp\big(-s(y_K)\big)$, which evaluates $s$ and $t$ forward and never inverts them.
The additive coupling of NICE is the case $s = 0$, with determinant 1.

### Assumptions

1. $s$ and $t$ depend on $x_K$ only. Used in steps 1 and 3.
2. Permuting the coordinates changes neither $|\det|$ nor the map. Used in step 2.

### Steps

1. $y_K = x_K$ gives the blocks $I$ and $0$ of the first block row. In the second block row, $y_j$ for $j \in T$ depends on $x_j$ only through $x_j \exp(s_j(x_K))$, so $\partial y_j / \partial x_i = 0$ for $i \in T$, $i \ne j$, and $\partial y_j / \partial x_j = \exp(s_j(x_K))$. The block $\partial y_T / \partial x_K$ may be anything; it does not enter the determinant.
2. The determinant of a block-triangular matrix is the product of the determinants of its diagonal blocks: $1 \cdot \prod_{j \in T} \exp s_j = \exp \sum_{j \in T} s_j$, which is positive, so the logarithm of its absolute value is $\sum_{j \in T} s_j$.
3. Given $\mathbf{y}$, $x_K = y_K$ is known, so $s(x_K) = s(y_K)$ and $t(x_K) = t(y_K)$ can be computed, and solving $y_T = x_T \odot e^{s} + t$ for $x_T$ gives the inverse.
4. A stack whose masks alternate transforms every coordinate: the coordinates kept by one layer are transformed by the next.

### Result and check

$$ \log\left|\det \frac{\partial \mathbf{y}}{\partial \mathbf{x}}\right| = \sum_{j \in T} s_j(x_K), \qquad x_T = \big(y_T - t(y_K)\big) \odot e^{-s(y_K)} . $$

The check `coupling-jacobian.check.py` computes the finite-difference Jacobian of single coupling layers of the NumPy reference with random weights, verifies the zero block and the diagonal block, compares $\log|\det|$ with the sum of the scales, and runs the inverse, for $D$ from 2 to 6 and hidden widths from 3 to 16; with $s = 0$ it finds the unit determinant of additive coupling.
