---
id: hippo-nplr
derives: [c-hippo-nplr, c-hippo-eigs, c-hippo-conditioning, eq-hippo, eq-nplr]
kind: identity
assumes: ["A is the N by N HiPPO matrix of S4, equation (2): A_nk = -(2n+1)^{1/2}(2k+1)^{1/2} for n > k, -(n+1) for n = k, 0 for n < k, with n, k = 0, ..., N-1"]
uses: [skew-symmetric-matrix@m2, spectral-theorem-for-normal-matrices@m2, eigenvalue@m2]
status: full
check: hippo-nplr.check.py
lean: null
sources:
  - {work: gu-2021-structured-state-spaces-s4, locator: "Section 2.2"}
  - {work: gu-2021-structured-state-spaces-s4, locator: "Section 3.2"}
  - {work: gu-2020-hippo, locator: "Section 3"}
---

### Statement

Let $A$ be the HiPPO matrix of S4 and $P_n = (n + \tfrac12)^{1/2}$. Then

1. $A = S - P P^\top$, where $S = -\tfrac12 I + J$ and $J$ is skew-symmetric. So $S$ is normal, and every eigenvalue of $S$ has real part $-\tfrac12$: $A$ is normal plus rank one.
2. $A$ is lower triangular, with eigenvalues $-1, -2, \dots, -N$.
3. The eigenvector matrix of $A$ is badly conditioned: in our computation with unit-norm eigenvectors, its condition number is about $8 \times 10^{4}$ at $N = 8$ and $8 \times 10^{10}$ at $N = 16$.

### Assumptions

1. $A$ is the matrix of equation (2) of S4, which is minus the LegS matrix of Theorem 2 of HiPPO. Used in all steps.

### Steps

1. The entries of $P P^\top$ are $P_n P_k = \tfrac12 (2n+1)^{1/2}(2k+1)^{1/2}$.
2. Form $S = A + P P^\top$. For $n > k$: $S_{nk} = -(2n+1)^{1/2}(2k+1)^{1/2} + \tfrac12 (2n+1)^{1/2}(2k+1)^{1/2} = -\tfrac12 (2n+1)^{1/2}(2k+1)^{1/2}$. For $n < k$: $S_{nk} = 0 + \tfrac12 (2n+1)^{1/2}(2k+1)^{1/2}$. For $n = k$: $S_{nn} = -(n+1) + \tfrac12 (2n+1) = -\tfrac12$.
3. So $S = -\tfrac12 I + J$ with $J_{nk} = -\tfrac12 \operatorname{sign}(n - k)(2n+1)^{1/2}(2k+1)^{1/2}$, and $J^\top = -J$.
4. Since $J^\top = -J$, $S^\top = -\tfrac12 I - J$ and $S S^\top = (-\tfrac12 I + J)(-\tfrac12 I - J) = \tfrac14 I - J^2 = S^\top S$. So $S$ is normal, and by the spectral theorem it is unitarily diagonalizable. A real skew-symmetric $J$ has purely imaginary eigenvalues, so the eigenvalues of $S$ are $-\tfrac12 + i\omega$.
5. Thus $A = S - P P^\top$ is normal plus low rank with rank $r = 1$, as Theorem 1 of S4 states for this matrix; conjugating by the unitary eigenvector matrix of $S$ gives the diagonal plus low-rank form $\Lambda - \tilde P \tilde P^{*}$.
6. $A$ is lower triangular, so its eigenvalues are its diagonal entries $-(n+1)$, which are distinct; $A$ is diagonalizable, but S4 (Lemma 3.2) shows that its eigenvectors have entries that grow exponentially in $N$, so the diagonal form of $A$ itself is not usable in floating point. Step 4 shows that the normal part $S$ has no such problem: its eigenvectors are orthonormal.
7. Statement 3 is our own measurement with `numpy.linalg.eig`, not a figure of the paper; it agrees with the exponential growth of Lemma 3.2.

### Result and check

$$ A = \underbrace{\big(-\tfrac12 I + J\big)}_{\text{normal}} - P P^\top, \qquad P_n = \sqrt{n + \tfrac12} . $$

The check `hippo-nplr.check.py` builds $A$ for $N$ from 1 to 40, verifies the decomposition, the normality, the real parts $-\tfrac12$ and the eigenvalues of $A$, and computes the condition numbers of the eigenvector matrices.
