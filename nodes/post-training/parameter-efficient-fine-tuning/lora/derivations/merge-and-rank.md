---
id: merge-and-rank
derives: [c-merge, c-rank-bound, eq-lora-merge]
kind: identity
assumes: ["the adapted layer is linear in its input", "B is d by r and A is r by k", "exact arithmetic"]
uses: [matrix-multiplication@m2, rank-of-a-matrix-definition@m2, rank-of-a-product-of-matrices@m2]
status: full
check: merge-and-rank.check.py
lean: null
sources:
  - {work: hu-2021-lora-low-rank-adaptation, locator: "Section 4.1"}
---

### Statement

Let $W_0 \in \mathbb{R}^{d \times k}$, $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$ and $s \in \mathbb{R}$. Then

1. for every $x \in \mathbb{R}^k$, $W_0 x + s\, B (A x) = (W_0 + s B A)\, x$, so the adapted layer equals the plain linear layer with the merged weight $W = W_0 + s B A \in \mathbb{R}^{d \times k}$;
2. $W - s B A = W_0$, so the merge can be undone;
3. $\operatorname{rank}(s B A) \le r$.

### Assumptions

1. The adapted layer is $x \mapsto W_0 x + s\, B(Ax)$, linear in $x$. Used in step 1.
2. $B$ has $r$ columns and $A$ has $r$ rows. Used in steps 1 and 3.
3. The arithmetic is exact. In floating point, items 1 and 2 hold up to rounding. Used in steps 1 and 2.

### Steps

1. Matrix multiplication is associative and distributes over addition, and scalars commute with matrices. So $s\, B(Ax) = (s\, BA)\, x$, and $W_0 x + (sBA)\, x = (W_0 + sBA)\, x$. The product $BA$ is $d \times k$ by assumption 2, the shape of $W_0$, so the sum is defined. This is item 1.
2. $(W_0 + sBA) - sBA = W_0$ by the rules of matrix addition. This is item 2.
3. The rank of a product is at most the rank of each factor: $\operatorname{rank}(BA) \le \min(\operatorname{rank} B, \operatorname{rank} A)$. The rank of $B$ is at most its number of columns, $r$. So $\operatorname{rank}(BA) \le r$, and multiplying by $s$ does not raise the rank. This is item 3.
4. Item 1 means that at inference the merged layer does one matrix-vector product with a $d \times k$ matrix, the same work as the pretrained layer, so the adapter adds no latency once merged.

### Result and check

$$ W_0 x + s\, B A x = (W_0 + s B A)\, x, \qquad \operatorname{rank}(s B A) \le r . $$

The check `merge-and-rank.check.py` draws $W_0$, $B$, $A$ and a batch of inputs with a fixed seed for several shapes and ranks, compares the two sides of item 1 and of item 2 to a relative error under $10^{-12}$, and computes the numerical rank of $sBA$, which must not exceed $r$.
