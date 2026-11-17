---
id: cost
derives: [c-flops, c-crossover, c-poly-exact]
kind: calculation
assumes: ["one head of non-causal self-attention on N positions, Q, K, V and the feature maps given", "a multiply-add counted as 2 FLOPs, only the matrix and vector products counted, the sums as products with a vector of ones"]
uses: [matrix-multiplication@m2]
status: full
check: cost.check.py
lean: null
sources:
  - {work: katharopoulos-2020-transformers-are-rnns, locator: "Section 3.2.1", note: "the costs O(N^2 max(D, M)) and O(N C M), the polynomial kernel and N > D^2"}
---

### Statement

1. Softmax attention takes $2N^2 D + 2N^2 M + 2N^2$ FLOPs: $QK^\top$, the weights times $V$ and the row sums of the weights. Linear attention with a feature map of size $C$ takes $4NCM + 4NC$: $\phi(K)^\top V$, $z = \phi(K)^\top \mathbf 1$, and $\phi(Q)$ times both.
2. Linear attention takes fewer FLOPs exactly when $N > 2C(M + 1) / (D + M + 1)$. With $C = D = M$ that is $N > 2D(D+1)/(2D+1)$, just under $N > D$; at $D = M = 64$ the first such $N$ is 65.
3. The map $\phi(x) = (c, \sqrt{2c}\, x, (x_a x_b)_{a, b})$ of size $C = 1 + D + D^2$ satisfies $\phi(q)^\top \phi(k) = (q^\top k + c)^2$, so attention with the degree-2 polynomial kernel is linear attention with this map; it is cheaper than softmax attention when $N > 2(1 + D + D^2)(M + 1)/(D + M + 1)$, which at $D = M = 64$ is $N \ge 4194$, between $D^2 = 4096$ and $D^2 + 2D + 2 = 4226$.

### Assumptions

1. One head, no causal mask, and the feature maps already applied; the exponentials, the divisions and the feature maps are left out, being $O(N^2)$, $O(NM)$ and $O(NC)$ at most. Used in items 1 to 3.
2. A multiply-add is 2 FLOPs and an $a \times b$ by $b \times c$ product costs $2abc$. Used in items 1 to 3.

### Steps

1. $QK^\top$ is $(N \times D)(D \times N)$, $2N^2D$; the weights times $V$ are $(N \times N)(N \times M)$, $2N^2M$; the row sums are $(N \times N)(N \times 1)$, $2N^2$. On the linear side $\phi(K)^\top V$ and $\phi(Q)(\phi(K)^\top V)$ are $2NCM$ each, and $\phi(K)^\top \mathbf 1$ and $\phi(Q) z$ are $2NC$ each.
2. $4NC(M + 1) < 2N^2(D + M + 1)$ is, dividing by $2N$, $N > 2C(M+1)/(D+M+1)$. At $C = D = M = 64$ the bound is $2 \cdot 64 \cdot 65 / 129 = 64.5$.
3. $(q^\top k + c)^2 = c^2 + 2c\, q^\top k + (q^\top k)^2$, and $(q^\top k)^2 = \sum_{a,b} q_a q_b k_a k_b$ is the dot product of the vectors of all products $x_a x_b$. The three blocks of $\phi$ give the three terms. With $C = 1 + D + D^2 = 4161$ at $D = 64$, the bound of item 2 is $2 \cdot 4161 \cdot 65 / 129 = 4193.2$.

### Result and check

The quadratic term of softmax attention is $N^2(D + M)$ against $NCM$ for linear attention, so linear attention pays when the sequence is longer than about the size of the feature map; for the elu + 1 map that is about the head width, and for an exact polynomial kernel about its square, which is Katharopoulos et al.'s "favorable when $N > D^2$". The check `cost.check.py` counts the products of the NumPy functions on random sizes with a counting array and compares them with item 1, finds the crossovers of items 2 and 3 by search, and verifies the polynomial map.
