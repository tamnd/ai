---
id: gcn-renormalization
derives: [c-gcn-first-order, c-norm-adj-spectrum, c-smoothing-limit]
kind: bound
assumes: ["A is the symmetric 0/1 adjacency matrix of an undirected graph", "for the first-order filter, every node has at least one neighbour so that D^(-1/2) exists", "for the limit, the graph is connected"]
uses: [eigenvalue@m2, eigenvector@m2, quadratic-form-defined-by-a-symmetric-matrix@m2]
status: full
check: gcn-renormalization.check.py
lean: null
sources:
  - {work: kipf-2017-semi-supervised-classification, locator: "Section 2.1"}
  - {work: kipf-2017-semi-supervised-classification, locator: "Section 2.2"}
  - {work: li-2018-deeper-insights-gcn, locator: "Theorem 1"}
---

### Statement

1. With $K = 1$ and $\lambda_{\max} \approx 2$, the Chebyshev filter of Kipf and Welling (2017, Eq. 5) becomes $\theta'_0 x - \theta'_1 D^{-1/2} A D^{-1/2} x$, and with one parameter $\theta = \theta'_0 = -\theta'_1$ it becomes $\theta (I_N + D^{-1/2} A D^{-1/2}) x$ (their Eqs. 6 and 7).
2. The eigenvalues of $I_N + D^{-1/2} A D^{-1/2}$ lie in $[0, 2]$, while those of the renormalized $\hat A = \tilde D^{-1/2} \tilde A \tilde D^{-1/2}$, with $\tilde A = A + I_N$, lie in $[-1, 1]$; $\hat A$ is symmetric, and $\tilde D^{1/2} \mathbf 1$ is an eigenvector with eigenvalue $1$.
3. On a connected graph, $\hat A^k x$ converges as $k \to \infty$ to $s\, (s^\top x) / (s^\top s)$ with $s = \tilde D^{1/2} \mathbf 1$: every node's value becomes proportional to $\sqrt{\tilde d_v}$, whatever $x$ was. This is the connected case of Theorem 1 of Li, Han and Wu (2018).

### Assumptions

1. $A$ is symmetric with entries in $\{0, 1\}$ and a zero diagonal; $D = \operatorname{diag}(A\mathbf 1)$. Used in steps 1 to 5.
2. For statement 1 and the first half of statement 2, every node has degree at least 1, so that $D^{-1/2}$ exists. Used in steps 1 and 3.
3. For statement 3, the graph is connected. Used in step 5.

### Steps

1. Kipf and Welling (2017, Section 2.1) approximate a spectral filter by $\sum_{k=0}^{K} \theta'_k T_k(\tilde L) x$ with $\tilde L = \frac{2}{\lambda_{\max}} L - I_N$ and $L = I_N - D^{-1/2} A D^{-1/2}$, where $T_0(y) = 1$ and $T_1(y) = y$ are Chebyshev polynomials. With $K = 1$ and $\lambda_{\max} = 2$, $\tilde L = L - I_N = -D^{-1/2} A D^{-1/2}$, so the filter is $\theta'_0 x + \theta'_1 \tilde L x = \theta'_0 x - \theta'_1 D^{-1/2} A D^{-1/2} x$. Setting $\theta'_0 = \theta$ and $\theta'_1 = -\theta$ gives $\theta (I_N + D^{-1/2} A D^{-1/2}) x$. That is statement 1.
2. Let $M$ be a symmetric matrix with non-negative entries and positive row sums $r = M\mathbf 1$, and let $R = \operatorname{diag}(r)$. Then $S = R^{-1/2} M R^{-1/2}$ is symmetric, so its eigenvalues are real, and $S = R^{1/2} (R^{-1} M) R^{-1/2}$ is similar to $R^{-1} M$, whose rows are non-negative and sum to 1. For an eigenvector $u$ of $R^{-1}M$ with eigenvalue $\lambda$, take the entry $i$ of largest modulus: $|\lambda| |u_i| = |\sum_j (R^{-1}M)_{ij} u_j| \le \sum_j (R^{-1}M)_{ij} |u_i| = |u_i|$, so $|\lambda| \le 1$. And $S\, R^{1/2}\mathbf 1 = R^{-1/2} M \mathbf 1 = R^{-1/2} r = R^{1/2}\mathbf 1$, so $1$ is an eigenvalue with eigenvector $R^{1/2}\mathbf 1$.
3. With $M = A$ (assumption 2 makes $r = d > 0$), the eigenvalues of $D^{-1/2} A D^{-1/2}$ lie in $[-1, 1]$, so those of $I_N + D^{-1/2} A D^{-1/2}$ lie in $[0, 2]$. Kipf and Welling (2017, Section 2.2) note this range and that repeating the operator can therefore lead to numerical instabilities and exploding or vanishing gradients.
4. With $M = \tilde A = A + I_N$, whose row sums are $\tilde d_v = d_v + 1 \ge 1$ even for isolated nodes, step 2 gives that $\hat A$ is symmetric with eigenvalues in $[-1, 1]$ and eigenvector $s = \tilde D^{1/2}\mathbf 1$ for the eigenvalue 1. That is statement 2.
5. On a connected graph, $\tilde A$ is irreducible and has a positive diagonal, so $\tilde D^{-1}\tilde A$ is the transition matrix of an irreducible aperiodic random walk; by the Perron-Frobenius theorem its eigenvalue 1 is simple and every other eigenvalue has modulus below 1. $\hat A$ is similar to it (step 2) and symmetric, so $\hat A = \frac{s s^\top}{s^\top s} + \sum_{i \ge 2} \lambda_i u_i u_i^\top$ with $|\lambda_i| < 1$, and $\hat A^k x \to s\, s^\top x / s^\top s$. Entry $v$ of the limit is $\sqrt{\tilde d_v}$ times a number shared by all nodes. That is statement 3; Li, Han and Wu (2018, Theorem 1) state it for graphs without bipartite components, one indicator per component.

### Result and check

$$ \operatorname{spec}(I_N + D^{-1/2} A D^{-1/2}) \subset [0, 2], \quad \operatorname{spec}(\hat A) \subset [-1, 1], \quad \hat A^k x \to \frac{s\, s^\top x}{s^\top s},\ s = \tilde D^{1/2}\mathbf 1 . $$

The check `gcn-renormalization.check.py` evaluates the filter of step 1 with a Chebyshev recursion and compares it with Eq. 7, computes the two spectra on random graphs (with and without isolated nodes), and iterates $\hat A$ on connected graphs until the iterate matches the limit of step 5.
