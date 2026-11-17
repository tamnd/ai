---
id: relative-position
derives: [c-orthogonal, c-norm-preserved, c-position-zero, c-composition, c-relative-score, c-complex-real, c-efficient-form, c-efficient-count, c-half-pairing, eq-score, eq-rope-complex, eq-rope-efficient]
kind: identity
assumes: ["d is even", "the angles are theta_j = b^(-2j/d) for j = 0, ..., d/2 - 1, or any real numbers", "the positions m and n are real numbers"]
uses: [block-diagonal-matrix@m2, linear-isometry@m2, permutation-matrix@m2, dot-product@m2]
status: full
check: relative-position.check.py
lean: null
sources:
  - {work: su-2024-roformer, locator: "Section 3.2.2"}
  - {work: su-2024-roformer, locator: "Section 3.4.2"}
  - {work: chen-2023-position-interpolation, locator: "Section 2.1"}
---

### Statement

Let $d$ be even, let $\theta_0, \dots, \theta_{d/2-1}$ be real, and let $\mathbf{R}_m$ be the block-diagonal matrix of eq-rope-matrix, whose block $j$ is the $2 \times 2$ rotation $\mathbf{B}(m\theta_j)$ with $\mathbf{B}(\phi) = \begin{pmatrix} \cos\phi & -\sin\phi \\ \sin\phi & \cos\phi \end{pmatrix}$.
For all real $m$, $n$ and all $\mathbf{q}, \mathbf{k}, \mathbf{x} \in \mathbb{R}^d$:

1. $\mathbf{R}_m^\top \mathbf{R}_m = \mathbf{I}$, so $\lVert \mathbf{R}_m \mathbf{x} \rVert = \lVert \mathbf{x} \rVert$.
2. $\mathbf{R}_0 = \mathbf{I}$.
3. $\mathbf{R}_m \mathbf{R}_n = \mathbf{R}_{m+n}$ and $\mathbf{R}_m^\top = \mathbf{R}_{-m}$.
4. $(\mathbf{R}_m \mathbf{q})^\top (\mathbf{R}_n \mathbf{k}) = \mathbf{q}^\top \mathbf{R}_{n-m} \mathbf{k}$, which depends on $m$ and $n$ only through $n - m$ (eq-score).
5. With $z_j = x_{2j} + i x_{2j+1}$, the pair $j$ of $\mathbf{R}_m \mathbf{x}$ is the real and the imaginary part of $z_j e^{i m \theta_j}$ (eq-rope-complex), and $(\mathbf{R}_m \mathbf{q})^\top (\mathbf{R}_n \mathbf{k}) = \mathrm{Re} \sum_j \hat{q}_j \overline{\hat{k}_j} e^{i(m-n)\theta_j}$, with $\hat{q}_j = q_{2j} + i q_{2j+1}$ and $\hat{k}_j = k_{2j} + i k_{2j+1}$.
6. $\mathbf{R}_m \mathbf{x} = \mathbf{x} \odot \mathbf{c}_m + \tilde{\mathbf{x}} \odot \mathbf{s}_m$ (eq-rope-efficient), and given the tables $\mathbf{c}_m$ and $\mathbf{s}_m$ this takes $2d$ multiplications and $d$ additions, while $\mathbf{R}_m$ has $2d$ non-zero entries out of $d^2$.
7. For a permutation matrix $\mathbf{P}$, $(\mathbf{P}^\top \mathbf{R}_m \mathbf{P} \mathbf{q})^\top (\mathbf{P}^\top \mathbf{R}_n \mathbf{P} \mathbf{k}) = (\mathbf{R}_m \mathbf{P}\mathbf{q})^\top (\mathbf{R}_n \mathbf{P}\mathbf{k})$. So a code base that rotates dimension $j$ with dimension $j + d/2$ computes the scores of the paired form on permuted queries and keys.

### Assumptions

1. $d$ is even, so the $d$ dimensions split into $d/2$ pairs. It is used in every step.
2. The angles $\theta_j$ are real. The value $b^{-2j/d}$ of eq-theta is not used, so the result holds for position interpolation and for the NTK-aware base as well.
3. The positions $m$ and $n$ are real numbers, not only integers. It is used in step 4 for the fractional positions of position interpolation.

### Steps

1. For one block, $\mathbf{B}(\phi)^\top \mathbf{B}(\phi)$ has the diagonal entries $\cos^2\phi + \sin^2\phi = 1$ and the off-diagonal entries $-\cos\phi\sin\phi + \sin\phi\cos\phi = 0$, by the Pythagorean identity. A block-diagonal matrix is multiplied block by block, so $\mathbf{R}_m^\top \mathbf{R}_m = \mathbf{I}$. Then $\lVert \mathbf{R}_m \mathbf{x} \rVert^2 = \mathbf{x}^\top \mathbf{R}_m^\top \mathbf{R}_m \mathbf{x} = \lVert \mathbf{x} \rVert^2$. This is part 1.
2. At $m = 0$ every angle is $0$, and $\mathbf{B}(0)$ has $\cos 0 = 1$ on the diagonal and $\sin 0 = 0$ off it. So $\mathbf{R}_0 = \mathbf{I}$. This is part 2.
3. The angle-sum formulas $\cos(\alpha + \beta) = \cos\alpha\cos\beta - \sin\alpha\sin\beta$ and $\sin(\alpha + \beta) = \sin\alpha\cos\beta + \cos\alpha\sin\beta$ give $\mathbf{B}(\alpha)\mathbf{B}(\beta) = \mathbf{B}(\alpha + \beta)$ by multiplying the two matrices out. With $\alpha = m\theta_j$ and $\beta = n\theta_j$, $\alpha + \beta = (m + n)\theta_j$, so block by block $\mathbf{R}_m \mathbf{R}_n = \mathbf{R}_{m+n}$. Since $\cos$ is even and $\sin$ is odd, $\mathbf{B}(\phi)^\top = \mathbf{B}(-\phi)$, so $\mathbf{R}_m^\top = \mathbf{R}_{-m}$. This is part 3.
4. By the rule $(\mathbf{A}\mathbf{u})^\top = \mathbf{u}^\top \mathbf{A}^\top$, $(\mathbf{R}_m \mathbf{q})^\top (\mathbf{R}_n \mathbf{k}) = \mathbf{q}^\top \mathbf{R}_m^\top \mathbf{R}_n \mathbf{k}$. By step 3 this is $\mathbf{q}^\top \mathbf{R}_{-m} \mathbf{R}_n \mathbf{k} = \mathbf{q}^\top \mathbf{R}_{n-m} \mathbf{k}$. The matrix $\mathbf{R}_{n-m}$ is a function of $n - m$ alone, so replacing $(m, n)$ by $(m + t, n + t)$ for any real $t$ leaves the score unchanged. This is part 4 and eq-score.
5. Multiplying $z = x_{2j} + i x_{2j+1}$ by $e^{i\phi} = \cos\phi + i\sin\phi$ gives the real part $x_{2j}\cos\phi - x_{2j+1}\sin\phi$ and the imaginary part $x_{2j}\sin\phi + x_{2j+1}\cos\phi$, by the product rule of complex numbers and $i^2 = -1$. These are the two rows of $\mathbf{B}(\phi)$ applied to $(x_{2j}, x_{2j+1})$, so the complex form and the real form agree. For two pairs $u = a + ib$ and $w = c + ie$, $\mathrm{Re}(u\overline{w}) = ac + be$, their real dot product. The score is the sum over the pairs of $\mathrm{Re}\big(\hat{q}_j e^{im\theta_j}\, \overline{\hat{k}_j e^{in\theta_j}}\big) = \mathrm{Re}\big(\hat{q}_j \overline{\hat{k}_j} e^{i(m-n)\theta_j}\big)$, because $\overline{e^{i\phi}} = e^{-i\phi}$. This is part 5.
6. Write $c_j = \cos m\theta_j$ and $s_j = \sin m\theta_j$. By step 5, pair $j$ of $\mathbf{R}_m \mathbf{x}$ is $(x_{2j}c_j - x_{2j+1}s_j,\ x_{2j+1}c_j + x_{2j}s_j)$. Let $\mathbf{c}_m$ hold $c_j$ in places $2j$ and $2j+1$, let $\mathbf{s}_m$ hold $s_j$ in the same places, and let $\tilde{\mathbf{x}} = (-x_1, x_0, -x_3, x_2, \dots, -x_{d-1}, x_{d-2})$. Then entry $2j$ of $\mathbf{x} \odot \mathbf{c}_m + \tilde{\mathbf{x}} \odot \mathbf{s}_m$ is $x_{2j}c_j - x_{2j+1}s_j$ and entry $2j+1$ is $x_{2j+1}c_j + x_{2j}s_j$, the same numbers. The two products take $d$ multiplications each and the sum takes $d$ additions; the signs of $\tilde{\mathbf{x}}$ are free in floating point. Each of the $d/2$ blocks has 4 entries, so $\mathbf{R}_m$ has $2d$ non-zero entries, and a dense product would take $d^2$ multiplications. This is part 6.
7. A permutation matrix is orthogonal, so $\mathbf{P}\mathbf{P}^\top = \mathbf{I}$. Then $(\mathbf{P}^\top \mathbf{R}_m \mathbf{P}\mathbf{q})^\top (\mathbf{P}^\top \mathbf{R}_n \mathbf{P}\mathbf{k}) = \mathbf{q}^\top \mathbf{P}^\top \mathbf{R}_m^\top \mathbf{P} \mathbf{P}^\top \mathbf{R}_n \mathbf{P}\mathbf{k} = (\mathbf{R}_m\mathbf{P}\mathbf{q})^\top(\mathbf{R}_n\mathbf{P}\mathbf{k})$. Rotating dimension $j$ with $j + d/2$ is $\mathbf{P}^\top \mathbf{R}_m \mathbf{P}$ for the $\mathbf{P}$ that moves dimension $j$ to place $2j$ and dimension $j + d/2$ to place $2j + 1$. So it has every property of parts 1 to 4. Weights trained with one pairing need their rows of $\mathbf{W}_q$ and $\mathbf{W}_k$ permuted by $\mathbf{P}$ to give the same scores under the other. This is part 7.

### Result and check

$\mathbf{R}_m$ is orthogonal, $\mathbf{R}_0 = \mathbf{I}$, and the score of a rotated query and a rotated key is $\mathbf{q}^\top \mathbf{R}_{n-m} \mathbf{k}$, a function of the content and of $n - m$ only.
Su et al. (2021, Equation 16) write the same identity as $\mathbf{R}^d_{\Theta, n-m} = (\mathbf{R}^d_{\Theta, m})^\top \mathbf{R}^d_{\Theta, n}$, and Chen et al. (2023, Equation 2) write the complex form of part 5.
The real, the complex and the elementwise forms are three ways to compute the same map.

The check `relative-position.check.py` draws random $\mathbf{q}$, $\mathbf{k}$ and positions with a fixed seed for $d = 2, 8, 64$, and asserts parts 1 to 7 within $10^{-10}$ relative error, including fractional positions.
The tests of `impl/` check the same parts in each framework: `test_norm_preserved`, `test_score_depends_on_m_minus_n`, `test_position_zero_is_identity`, `test_shared_fixture_forms_agree`, `test_matrix_is_orthogonal_and_composes` and `test_half_pairing_is_a_permutation`.
