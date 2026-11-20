---
id: capacity
derives: [c-capacity-rank, c-relu-zero, c-delta-overwrite]
kind: proof
assumes: ["a memory read as y = S^T phi(q) from a state S in R^{C x M}"]
uses: [matrix-multiplication@m2]
status: full
check: capacity.check.py
lean: null
sources:
  - {work: schlag-2021-linear-transformers-fast-weight, locator: "Section 4.1", note: "the capacity limitation, at most d_dot orthogonal keys"}
  - {work: schlag-2021-linear-transformers-fast-weight, locator: "Section 4.2", note: "the delta rule, Eq. 20 to 25"}
  - {work: katharopoulos-2020-transformers-are-rnns, locator: "Section 3.2", note: "sim only needs to be non-negative"}
---

### Statement

1. No state $S \in \mathbb{R}^{C \times M}$ maps $L > C$ keys to $L$ linearly independent values: if $S^\top \phi(k_j) = v_j$ for $j = 1, \dots, L$, the values span at most $C$ dimensions. With orthonormal values, at least one retrieval is wrong for every $S$, so a linear attention head of feature size $C$ cannot recall more than $C$ arbitrary associations exactly, whatever it learns.
2. With $C$ orthonormal mapped keys the sum rule recalls each of $C$ values exactly; more generally, it returns $v_j$ plus the crosstalk $\sum_{l \ne j} (\phi(k_j)^\top \phi(k_l)) v_l$.
3. With $\phi = \operatorname{ReLU}$, a query with no positive coordinate has $\phi(q) = 0$, and its numerator and denominator are both 0; with $\phi = \operatorname{elu} + 1$ every coordinate is positive and the denominator is a sum of positive terms.
4. Write $(k, v_1)$ and then $(k, v_2)$ with $\|\phi(k)\| = 1$. The sum rule then reads $v_1 + v_2$ under $k$, and the delta rule $S \leftarrow S + \beta \phi(k)(v - S^\top \phi(k))^\top$ with $\beta = 1$ reads $v_2$.

### Assumptions

1. The memory is read linearly, $y = S^\top \phi(q)$, without the normalizer; dividing by the positive number $\phi(q)^\top z$ rescales each read and does not change items 1 and 2 for exact recall up to that scale. Used in items 1 and 2.
2. In item 4, the key is written with the same feature vector both times. Used in item 4.

### Steps

1. Stack the mapped keys as the columns of $\Phi \in \mathbb{R}^{C \times L}$ and the values as the columns of $W \in \mathbb{R}^{M \times L}$. Exact recall is $S^\top \Phi = W$, and $\operatorname{rank}(S^\top \Phi) \le \operatorname{rank}(\Phi) \le C < L = \operatorname{rank}(W)$ when the values are independent.
2. $S = \sum_l \phi(k_l) v_l^\top$, so $S^\top \phi(k_j) = \sum_l (\phi(k_l)^\top \phi(k_j)) v_l$, which is $v_j$ when the keys are orthonormal.
3. $\operatorname{ReLU}(x) = 0$ for $x \le 0$, and both the numerator and the denominator are linear in $\phi(q)$. $\operatorname{elu}(x) + 1$ is $x + 1 > 1$ for $x > 0$ and $e^x > 0$ otherwise.
4. The sum rule gives $S = \phi(k)(v_1 + v_2)^\top$. The delta rule writes $S_1 = \phi(k) v_1^\top$, retrieves $\bar v = S_1^\top \phi(k) = v_1$ and writes $S_2 = S_1 + \phi(k)(v_2 - v_1)^\top = \phi(k) v_2^\top$.

### Result and check

The state of linear attention is an associative memory of fixed size, and item 1 is a hard limit on what it can hold: Schlag, Irie and Schmidhuber's "storing more than $d_{\text{dot}}$ associations will result in a retrieval error". Enlarging $C$ (DPFP, random features) raises the bound, and the delta rule lets the memory replace an association instead of piling a new one onto it; softmax attention has no such bound, because its cache grows with the sequence. The check `capacity.check.py` solves the least-squares recall problem for $L \le C$ and $L > C$, measures the crosstalk, and runs both update rules.
