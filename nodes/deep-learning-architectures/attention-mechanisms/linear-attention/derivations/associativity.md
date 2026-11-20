---
id: associativity
derives: [c-assoc, c-softmax-kernel, c-causal-recurrence, c-chunk, c-state-size]
kind: proof
assumes: ["a similarity of the form sim(q, k) = phi(q)^T phi(k) with phi into R^C and every denominator positive"]
uses: [matrix-multiplication@m2]
status: full
check: associativity.check.py
lean: null
sources:
  - {work: katharopoulos-2020-transformers-are-rnns, locator: "Section 3.1", note: "Eq. 2 and 3, softmax attention and the generalized attention"}
  - {work: katharopoulos-2020-transformers-are-rnns, locator: "Section 3.2", note: "Eq. 4 to 6, the kernel form and associativity"}
  - {work: katharopoulos-2020-transformers-are-rnns, locator: "Section 3.3", note: "Eq. 8 to 12, causal masking with S_i and z_i"}
  - {work: katharopoulos-2020-transformers-are-rnns, locator: "Section 3.4", note: "Eq. 16 to 20, the layer as an RNN"}
---

### Statement

Write $\phi(Q) \in \mathbb{R}^{N \times C}$ and $\phi(K) \in \mathbb{R}^{N \times C}$ for the mapped queries and keys, row by row, and $V \in \mathbb{R}^{N \times M}$ for the values.

1. The generalized attention $V'_i = \sum_j \operatorname{sim}(Q_i, K_j) V_j / \sum_j \operatorname{sim}(Q_i, K_j)$ with $\operatorname{sim}(q, k) = \phi(q)^\top \phi(k)$ equals $\phi(Q_i)^\top (\phi(K)^\top V) / \phi(Q_i)^\top \sum_j \phi(K_j)$ for every $i$.
2. Softmax attention is the generalized attention with $\operatorname{sim}(q, k) = \exp(q^\top k / \sqrt{D})$.
3. With the sums restricted to $j \le i$, the states $S_i = S_{i-1} + \phi(K_i) V_i^\top$ and $z_i = z_{i-1} + \phi(K_i)$ from $S_0 = 0$, $z_0 = 0$ give $V'_i = \phi(Q_i)^\top S_i / \phi(Q_i)^\top z_i$, the attention with the future masked.
4. Split the positions into consecutive chunks of length $T$. For position $i$ in chunk $c$, $V'_i$ is $(\phi(Q_i)^\top S_{\text{before}} + \sum_{j \in c, j \le i} \phi(Q_i)^\top \phi(K_j) V_j)$ over $(\phi(Q_i)^\top z_{\text{before}} + \sum_{j \in c, j \le i} \phi(Q_i)^\top \phi(K_j))$, where $S_{\text{before}}$ and $z_{\text{before}}$ are the states after the chunks before $c$. $T = 1$ is item 3 and $T = N$ is the masked $N \times N$ form.
5. The state $(S_i, z_i)$ of one head has $C M + C$ numbers at every position, while a cache of the keys and values that softmax attention needs has $i (D + M)$ at position $i$.

### Assumptions

1. The similarity factorizes through a feature map: $\operatorname{sim}(q, k) = \phi(q)^\top \phi(k)$ with $\phi : \mathbb{R}^D \to \mathbb{R}^C$. Used in items 1, 3, 4 and 5.
2. Every denominator is positive, which holds when $\phi$ is positive, as $\operatorname{elu}(x) + 1$ is. Used in items 1, 3 and 4.

### Steps

1. The numerator is $\sum_j (\phi(Q_i)^\top \phi(K_j)) V_j^\top = \phi(Q_i)^\top \sum_j \phi(K_j) V_j^\top = \phi(Q_i)^\top (\phi(K)^\top V)$, by linearity of the product in its second factor; the denominator is the same with $V_j$ replaced by 1. In matrix form, $(\phi(Q) \phi(K)^\top) V = \phi(Q) (\phi(K)^\top V)$ by associativity, and the $C \times M$ matrix $\phi(K)^\top V$ is computed once for all the queries.
2. Substitute $\exp(q^\top k / \sqrt D)$ for sim: the numerator over the denominator is the softmax of the scaled scores of row $i$ times $V$. The exponential kernel has no finite feature map, which is why softmax attention itself is not linearized by item 1.
3. By induction, $S_i = \sum_{j \le i} \phi(K_j) V_j^\top$ and $z_i = \sum_{j \le i} \phi(K_j)$; then apply item 1 to the sums over $j \le i$.
4. Split $\sum_{j \le i}$ into the sum over the earlier chunks, which is $S_{\text{before}}$ and $z_{\text{before}}$ by item 3, and the sum over $j \le i$ inside the chunk, which is the masked $T \times T$ product of the chunk.
5. $S_i$ is $C \times M$ and $z_i$ has $C$ entries whatever $i$; the cache holds $i$ keys of size $D$ and $i$ values of size $M$.

### Result and check

Linear attention is the generalized attention of a factorized kernel computed in another order; it is exact for that kernel, and an approximation of softmax attention only to the degree that $\phi(q)^\top \phi(k)$ approximates $\exp(q^\top k / \sqrt D)$. The causal layer is an RNN with a matrix-valued state of fixed size, and the chunked form trades the two: $N / T$ sequential steps, each with $T^2$ work inside the chunk. The check `associativity.check.py` verifies the five items on random inputs, for the elu + 1, ReLU, degree-2 polynomial and DPFP maps and for every chunk length that divides $N$.
