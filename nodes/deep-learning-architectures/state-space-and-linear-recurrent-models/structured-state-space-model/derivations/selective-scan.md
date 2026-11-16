---
id: selective-scan
derives: [c-selective-gate, c-selective-no-kernel, c-scan-associative]
kind: identity
assumes: ["a diagonal SSM whose step size, input matrix and output matrix may change at each position k", "zero-order hold at each position: Ā_k = exp(Δ_k A), B̄_k = (Δ_k A)^{-1}(exp(Δ_k A) - I) Δ_k B_k"]
uses: [diagonal-matrix@m2, matrix-exponential@m2]
status: full
check: selective-scan.check.py
lean: null
sources:
  - {work: gu-2023-mamba, locator: "Section 3.2"}
  - {work: gu-2023-mamba, locator: "Section 3.5.1"}
  - {work: gu-2023-mamba, locator: "Section 3.3.2"}
---

### Statement

For the time-varying recurrence $x_k = \bar A_k x_{k-1} + \bar B_k u_k$, $y_k = C_k x_k$ of a selective SSM:

1. With $N = 1$, $A = -1$, $B = 1$ and $\Delta_k = \operatorname{softplus}(z_k)$, the recurrence is $x_k = (1 - g_k) x_{k-1} + g_k u_k$ with $g_k = \sigma(z_k)$, the logistic sigmoid.
2. When $\Delta_k$ depends on $u$, the map $u \mapsto y$ is not linear, so no kernel $K$ gives $y = K * u$.
3. The pairs $(\bar A_k, \bar B_k u_k)$ compose under $(a_1, b_1) \bullet (a_2, b_2) = (a_2 a_1, a_2 b_1 + b_2)$, which is associative, and the prefix products give the states $x_k$. A parallel prefix scan therefore computes the states in $O(\log L)$ depth.

### Assumptions

1. $A$ and every $\bar A_k$ are diagonal, so the products commute entrywise. Used in step 4.
2. Zero-order hold at each position (Mamba, equation 4). Used in step 1.
3. $x_{-1} = 0$. Used in step 4.

### Steps

1. With $A = -1$ and $B = 1$, zero-order hold gives $\bar a_k = e^{-\Delta_k}$ and $\bar b_k = (e^{-\Delta_k} - 1)/(-1) = 1 - e^{-\Delta_k}$.
2. With $\Delta_k = \operatorname{softplus}(z_k) = \log(1 + e^{z_k})$: $e^{-\Delta_k} = 1/(1 + e^{z_k}) = 1 - \sigma(z_k)$. So $\bar a_k = 1 - g_k$ and $\bar b_k = g_k$, and the recurrence is the gated update of statement 1, Theorem 1 of Mamba (Section 3.5.1). A large $\Delta_k$ resets the state toward the current input; a small one keeps the state and ignores the input.
3. Statement 2: take $u = s e_0$ with $\Delta_k = \operatorname{softplus}(w u_k)$. Then $y_0 = C_0 \bar B_0 s$ with $\bar B_0 = 1 - e^{-\operatorname{softplus}(w s)}$ for the scalar case above, which is not proportional to $s$. A convolution $K * u$ is linear in $u$, so it cannot equal this map. This is why Mamba loses the convolutional view and computes the model as a scan (Section 3.2, Algorithm 2).
4. Statement 3: one step is the affine map $x \mapsto a x + b$ with $a = \bar A_k$ and $b = \bar B_k u_k$. Applying $(a_1, b_1)$ then $(a_2, b_2)$ gives $x \mapsto a_2 (a_1 x + b_1) + b_2 = a_2 a_1 x + (a_2 b_1 + b_2)$. Composition of maps is associative, so $\bullet$ is associative. From $x_{-1} = 0$, the composition of the first $k + 1$ maps applied to 0 is $x_k$, which is its second component. A Blelloch prefix scan evaluates all $L$ prefixes in $O(\log L)$ parallel depth and $O(L)$ work; Mamba uses a parallel scan, fused so that the expanded states stay in fast memory (Section 3.3.2).

### Result and check

$$ \Delta_k = \operatorname{softplus}(z_k) \ \Rightarrow \ x_k = (1 - \sigma(z_k))\, x_{k-1} + \sigma(z_k)\, u_k ; \qquad (a_1, b_1) \bullet (a_2, b_2) = (a_2 a_1,\ a_2 b_1 + b_2) . $$

The check `selective-scan.check.py` compares the selective recurrence with the gated update, tests additivity on random inputs, checks the associativity of $\bullet$ on random triples, and computes the states with a recursive-doubling prefix scan.
