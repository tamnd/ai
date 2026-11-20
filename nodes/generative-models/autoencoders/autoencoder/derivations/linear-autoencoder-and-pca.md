---
id: linear-autoencoder-and-pca
derives: [c-linear-pca, c-gl-symmetry, c-transpose, c-shrink]
kind: proof
assumes: ["a linear autoencoder with biases: code y = x W_1 + b_1 of C < D units and reconstruction z = y W_2 + b_2", "the squared error summed over the D components and averaged over the N examples of a data set", "the sample covariance S of the data has eigenvalues s_1 >= ... >= s_D with s_C > s_{C+1}"]
uses: [singular-value-decomposition@m2, orthogonal-projection@m2, frobenius-norm@m2, matrix-multiplication@m2]
status: full
check: linear-autoencoder-and-pca.check.py
lean: null
sources:
  - {work: kunin-2019-loss-landscapes-regularized-linear, locator: "pp. 1 to 4", note: "the GL symmetry, the Transpose Theorem 2.1 and its proof, and the minimum of the regularized loss"}
  - {work: plaut-2018-principal-subspaces-principal-components, locator: "p. 3", note: "the optimal decoder bias, the encoder as the pseudoinverse of the decoder, and the principal subspace"}
---

### Statement

Write the encoder matrix as $A = W_1$ ($D \times C$) and the decoder matrix as $B = W_2$ ($C \times D$), with rows as examples, and $E(A, B) = \frac{1}{N} \sum_n \lVert x_n - (x_n A + b_1) B - b_2 \rVert^2$ at the best biases.

1. The minimum of $E$ is $s_{C+1} + \dots + s_D$, the variance that PCA with $C$ components discards, and at every minimum the rows of $B$ span the principal subspace, the span of the first $C$ eigenvectors of $S$, and $AB$ is the orthogonal projection onto it.
2. For every invertible $C \times C$ matrix $G$, $(A G, G^{-1} B)$ with biases $(b_1 G, b_2)$ computes the same reconstruction as $(A, B)$. So the rows of $B$ at a minimum can be any basis of the principal subspace, and its singular vectors need not be the principal directions.
3. With weight decay, $E_\lambda = E + \lambda (\lVert A \rVert_F^2 + \lVert B \rVert_F^2)$ for $\lambda > 0$, every critical point has $A = B^\top$.
4. At the minimum of $E_\lambda$, $B^\top = U_C (I - \lambda \Sigma_C^{-1})_+^{1/2} O$, where $U_C$ holds the first $C$ eigenvectors of $S$, $\Sigma_C = \operatorname{diag}(s_1, \dots, s_C)$, $O$ is orthogonal and $(\cdot)_+$ sets negative entries to 0: the left singular vectors of the decoder are the principal directions, shrunk by $1 - \lambda / s_i$, and a direction with $s_i \le \lambda$ is dropped.

### Assumptions

1. The model and loss of the assumptions above. Used in all steps.
2. $s_C > s_{C+1}$, so that the principal subspace is unique. Used in step 2.
3. $\lambda > 0$. Used in step 4.

### Steps

1. **The biases centre the data.** For fixed $A$ and $B$, $E$ is a convex quadratic in $b_2$, minimised by $b_2 = \bar x - (\bar x A + b_1) B$ with $\bar x$ the mean, which turns the error into $\frac1N \sum_n \lVert (x_n - \bar x) - (x_n - \bar x) A B \rVert^2$, independent of $b_1$ (Plaut, p. 3; Kunin et al., Appendix A). From here on the data are centred and $S = \frac1N X^\top X$.
2. **Eckart and Young.** $AB$ has rank at most $C$, and $E = \operatorname{tr}\big((I - AB)^\top S (I - AB)\big)$ is the squared error of a rank-$C$ approximation of $X / \sqrt N$, which is at least the sum of its discarded squared singular values, $s_{C+1} + \dots + s_D$, with equality exactly when $X AB$ is the projection of $X$ onto the top-$C$ eigenvectors (unique by assumption 2). Since the data span every eigendirection of positive variance, $AB = P = U_C U_C^\top$ on them, so the row space of $B$ is the principal subspace and $A$ is the pseudoinverse of $B$. This is statement 1.
3. **The symmetry.** $(x A G + b_1 G) G^{-1} B = (x A + b_1) B$. If $B$ is a minimum, so is $G^{-1} B$; its rows are another basis of the same subspace, and for $G^{-1} = \begin{pmatrix} 2 & 1 \\ 0 & 1 \end{pmatrix}$ applied to the two principal directions the rows are neither orthogonal nor of unit length. This is statement 2, and it is why Plaut's hypothesis, that the first left singular vectors of a minimising decoder are the loading vectors, fails without regularization: the singular vectors of $G^{-1} U_C^\top$ are those of $G^{-1}$ rotated into the subspace.
4. **Kunin's Transpose Theorem.** In the column notation of Kunin et al., the encoder is $W_1 = A^\top$, the decoder $W_2 = B^\top$, and with the centred data the critical points of $E_\lambda$ satisfy $-2 W_2^\top K + 2 \lambda W_1 = 0$ and $-2 K W_1^\top + 2 \lambda W_2 = 0$, where $K = (I - W_2 W_1) S$. Kunin et al. show that $K$ is symmetric and positive semi-definite at a critical point (p. 3). Subtracting the transpose of the second condition from the first gives $2 (W_1 - W_2^\top)(K + \lambda I) = 0$, and $K + \lambda I$ is positive definite for $\lambda > 0$, so $W_1 = W_2^\top$, that is $A = B^\top$. This is statement 3.
5. **The shrinkage.** With $W_1 = V^\top$ and $W_2 = V$, the second condition reads $S V - V V^\top S V = \lambda V$. Put $V = U_C\, \operatorname{diag}(m_1, \dots, m_C)\, O$ with $S U_C = U_C \Sigma_C$ and $O$ orthogonal: then $V V^\top = U_C \operatorname{diag}(m_i^2) U_C^\top$ and the condition becomes $s_i m_i - s_i m_i^3 = \lambda m_i$ for each $i$, so $m_i = 0$ or $m_i^2 = 1 - \lambda / s_i$, which needs $s_i > \lambda$. Kunin et al. show that the global minimum is the one that uses the largest eigenvalues (p. 4); the scalar case, $w_2 w_1 = 1 - \lambda x^{-2}$ with $w_1 = w_2$ (p. 3), is the same equation. This is statement 4.

### Result and check

$$ \min_{A, B, b_1, b_2} E = \sum_{i > C} s_i , \qquad A = B^\top , \quad B^\top = U_C \big(I - \lambda \Sigma_C^{-1}\big)_+^{1/2} O \ \text{ at the minimum of } E_\lambda . $$

The check `linear-autoencoder-and-pca.check.py` builds data with an exact covariance, compares the error of the PCA reconstruction with the discarded variance and with the error of random rank-$C$ maps, applies a $G$ to a minimum and measures the change in the reconstruction and the angles of the decoder's singular vectors, and runs gradient descent on $E_\lambda$ from random starts through the NumPy reference to find $A = B^\top$ and the squared singular values $1 - \lambda / s_i$, including a direction with $s_i < \lambda$ that collapses.
