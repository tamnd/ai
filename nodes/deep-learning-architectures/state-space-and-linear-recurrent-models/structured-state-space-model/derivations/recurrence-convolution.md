---
id: recurrence-convolution
derives: [c-convolution-view, c-causal, c-impulse, c-diagonal-kernel, c-conjugation, c-mode-costs, eq-recurrence, eq-kernel, eq-diagonal-kernel]
kind: identity
assumes: ["the discrete SSM x_k = Ā x_{k-1} + B̄ u_k, y_k = C x_k + D u_k with x_{-1} = 0", "Ā, B̄, C, D do not depend on k or on u"]
uses: [matrix-multiplication@m2, diagonal-matrix@m2, eigendecomposition@m2]
status: full
check: recurrence-convolution.check.py
lean: null
sources:
  - {work: gu-2021-structured-state-spaces-s4, locator: "Section 2.4"}
  - {work: gu-2021-structured-state-spaces-s4, locator: "Section 3.1"}
  - {work: gu-2022-diagonal-state-space-models-s4d, locator: "equation 7"}
---

### Statement

For the discrete SSM $x_k = \bar A x_{k-1} + \bar B u_k$, $y_k = C x_k + D u_k$, $x_{-1} = 0$, on a sequence of length $L$:

1. $y_k = \sum_{j=0}^{k} K_{k-j}\, u_j + D u_k$ with $K = (C\bar B, C\bar A\bar B, \dots, C\bar A^{L-1}\bar B)$: the recurrence and the causal convolution $y = K * u + D u$ give the same outputs.
2. $y_k$ does not depend on $u_j$ for $j > k$.
3. The response to the unit impulse $u = (1, 0, \dots, 0)$ is $K + D e_0$.
4. If $\bar A = \operatorname{diag}(\bar a_1, \dots, \bar a_N)$, then $K_\ell = \sum_{n} C_n \bar B_n \bar a_n^{\ell}$: a Vandermonde product.
5. The change of state $x = V \tilde x$, which replaces $(A, B, C)$ by $(V^{-1} A V, V^{-1} B, C V)$, leaves $K$ unchanged, under both discretization rules.
6. For a diagonal $\bar A$, the recurrence costs $O(N)$ per step per channel and keeps $O(N)$ numbers of state; the kernel by statement 4 costs $O(NL)$, and the convolution of length $L$ by the FFT costs $O(L \log L)$.

### Assumptions

1. Time invariance: $\bar A, \bar B, C, D$ are the same at every step and do not depend on the input. Used in steps 1 and 2.
2. Zero initial state $x_{-1} = 0$. Used in step 1.
3. $V$ is invertible. Used in step 5.

### Steps

1. Unroll the recurrence from $x_{-1} = 0$: $x_0 = \bar B u_0$, $x_1 = \bar A \bar B u_0 + \bar B u_1$, and by induction $x_k = \sum_{j=0}^{k} \bar A^{k-j} \bar B u_j$: if it holds for $k - 1$, then $x_k = \bar A \sum_{j<k} \bar A^{k-1-j}\bar B u_j + \bar B u_k$. This is the unrolling of S4 (Section 2.4).
2. Apply $C$ and add $D u_k$: $y_k = \sum_{j=0}^{k} C \bar A^{k-j} \bar B\, u_j + D u_k = \sum_{j=0}^{k} K_{k-j} u_j + D u_k$. This is statement 1, equation (4) and (5) of S4. Time invariance makes the weight of $u_j$ depend only on $k - j$.
3. The sum in step 2 stops at $j = k$, so no $u_j$ with $j > k$ enters $y_k$: statement 2.
4. With $u_0 = 1$ and $u_j = 0$ otherwise, step 2 gives $y_k = K_k + D[k = 0]$: statement 3.
5. If $\bar A$ is diagonal, $\bar A^\ell$ is the diagonal of the powers $\bar a_n^\ell$, and $C \bar A^\ell \bar B = \sum_n C_n \bar a_n^\ell \bar B_n$. Stacking $\ell = 0, \dots, L-1$ gives $K = (C \circ \bar B)^\top \mathcal V$ with $\mathcal V_{n\ell} = \bar a_n^\ell$, equation (7) of S4D (Section 3.2). When $A$ is real, its complex eigenvalues come in conjugate pairs, so the sum can keep one entry of each pair and take twice the real part.
6. Under $x = V\tilde x$: both rules give $\bar A$ as a function of $A$ that commutes with conjugation, $f(V^{-1}AV) = V^{-1} f(A) V$, and $\bar B$ as such a function times $B$. So $\tilde{\bar A}^{\ell} \tilde{\bar B} = V^{-1} \bar A^\ell V V^{-1} \bar B = V^{-1}\bar A^\ell \bar B$ and $C V\, V^{-1}\bar A^\ell \bar B = C\bar A^\ell\bar B$. This is Lemma 3.1 of S4 in discrete form. It is why a diagonalizable $A$ can be replaced by its diagonal, if the conjugating matrix is well conditioned.
7. Costs: one step of the diagonal recurrence is $N$ multiplies and adds for the state and $N$ for the output. Statement 4 forms $L$ powers of $N$ entries. The causal convolution pads $u$ and $K$ to length $2L$, so the circular product of the FFT does not wrap, and costs $O(L \log L)$ per channel. The recurrence does $L$ dependent steps; the convolution does no sequential loop over $L$.

### Result and check

$$ y = K * u + D u, \qquad K_\ell = C \bar A^{\ell} \bar B = \sum_{n=1}^{N} C_n \bar B_n \bar a_n^{\ell} \ (\bar A \text{ diagonal}). $$

The check `recurrence-convolution.check.py` runs the recurrence, the direct convolution and the FFT convolution of random dense and diagonal SSMs under both rules, perturbs future inputs, feeds an impulse, conjugates by a random $V$, and verifies the conjugate-pair form.
