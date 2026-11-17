---
id: properties
derives: [c-head-sum, c-convex, c-equivariant, c-cross-invariant, c-causal-independence, c-padding-truncation, c-param-count, c-flops, c-kv-cache, eq-heads, eq-multihead]
kind: identity
assumes: ["no positional information is added inside the layer", "masked entries get weight exactly 0", "d_k = d_v = d_model / h for the counts that use them", "a multiply-add counts as 2 FLOPs, and the softmax and the mask are not counted"]
uses: [matrix-multiplication@m2, permutation-matrix@m2, dot-product@m2]
status: full
check: properties.check.py
lean: null
sources:
  - {work: vaswani-2017-attention, locator: "Section 3.2.2"}
  - {work: vaswani-2017-attention, locator: "Section 3.2.3"}
  - {work: shazeer-2019-fast-transformer-decoding, locator: "Section 3"}
  - {work: ainslie-2023-gqa, locator: "Section 2.2"}
---

### Statement

Let $X \in \mathbb{R}^{T \times d}$ give the queries and $M \in \mathbb{R}^{S \times d}$ the keys and values, with $d = d_{\text{model}}$, and let $\mathrm{MHA}(X, M)$ be eq-multihead with $h$ heads. Then

1. $\mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h)\, W^O = \sum_{r=1}^h \mathrm{head}_r W^O_r$, where $W^O_r$ is the block of $d_v$ rows of $W^O$ that multiplies head $r$.
2. Each row of a head output is a convex combination of the rows of $V_r = M W^V_r$ that the mask allows.
3. For every $T \times T$ permutation matrix $P$, $\mathrm{MHA}(PX, PX) = P\, \mathrm{MHA}(X, X)$.
4. For every $S \times S$ permutation matrix $P$, $\mathrm{MHA}(X, PM) = \mathrm{MHA}(X, M)$.
5. With the causal mask ($s$ allowed for $t$ when $s \le t$), row $t$ of $\mathrm{MHA}(X, X)$ depends only on rows $1, \dots, t$ of $X$.
6. If a padding mask allows exactly the first $L$ keys, the output equals $\mathrm{MHA}(X, M_{1:L})$ with no mask.
7. The layer has $2 d h (d_k + d_v)$ weights, which is $4 d^2$ for every $h$ when $d_k = d_v = d/h$.
8. A forward pass over a batch of $B$ costs $2BTdhd_k + 2BSdh(d_k + d_v) + 2BTdhd_v + 2BhTS(d_k + d_v)$ FLOPs; with $S = T$ and $d_k = d_v = d/h$ this is $8BTd^2 + 4BT^2 d$. The weights $A$ have $BhTS$ entries.
9. In incremental decoding, each cached position of a layer stores $2 h d_k$ numbers for the keys and values of multi-head attention, $2 d_k$ for multi-query attention and $2 G d_k$ for grouped-query attention with $G$ groups, the ratios $h$ and $h/G$ to multi-head attention.

### Assumptions

1. No position enters inside the layer: no positional encoding is added between the inputs and the projections, and no position-dependent bias is added to the scores. Used in steps 3 and 4.
2. Masked entries get weight 0 (the softmax of $-\infty$). Used in steps 2, 5 and 6.
3. The counting conventions of item 8: a multiply-add counts as 2 FLOPs; the softmax, the scaling and the mask are left out. Used in step 8.

### Steps

1. Write $C = [\mathrm{head}_1 \cdots \mathrm{head}_h]$ and split $W^O$ into row blocks $W^O_1, \dots, W^O_h$. Block matrix multiplication gives $C W^O = \sum_r \mathrm{head}_r W^O_r$. This is item 1: the concatenation followed by $W^O$ is a sum of $h$ per-head outputs.
2. The weights of a row are nonnegative and sum to 1 over the allowed keys, so $\mathbf{a}^\top V$ is a convex combination. This is item 2.
3. Replace $X$ by $PX$. Then $Q_r \mapsto PQ_r$, $K_r \mapsto PK_r$, $V_r \mapsto PV_r$, the scores become $P Q_r K_r^\top P^\top / \sqrt{d_k}$, and the row softmax commutes with permuting rows and with permuting the columns of every row in the same way. So $A_r \mapsto P A_r P^\top$ and $A_r V_r \mapsto P A_r P^\top P V_r = P A_r V_r$, since $P^\top P = I$. Item 1 then gives item 3.
4. Replace $M$ by $PM$. The queries do not change, the scores become $Q_r K_r^\top P^\top$, so $A_r \mapsto A_r P^\top$, and $A_r P^\top P V_r = A_r V_r$. This is item 4.
5. Row $t$ of a head output is $\sum_{s \le t} a_{ts} \mathbf{v}_s$, where $a_{ts}$ is a softmax over the scores $\mathbf{q}_t^\top \mathbf{k}_s$ with $s \le t$ only. All of $\mathbf{q}_t, \mathbf{k}_s, \mathbf{v}_s$ for $s \le t$ are functions of rows $1, \dots, t$ of $X$. This is item 5.
6. The masked keys have weight 0 and the softmax over the remaining $L$ scores is the softmax of the truncated problem, so every head output and the sum of item 1 are the same. This is item 6.
7. $W^Q$ and $W^K$ have $d \cdot h d_k$ entries each, and $W^V$ and $W^O$ have $d \cdot h d_v$ each. With $h d_k = h d_v = d$ the sum is $4d^2$. This is item 7.
8. Each projection of a row is a vector-matrix product with $d \cdot h d_k$ or $d \cdot h d_v$ multiply-adds: $T$ query rows, $S$ key rows, $S$ value rows and $T$ output rows. The scores are $h T S$ dot products of length $d_k$, and the weighted sums are $h T$ sums of $S$ vectors of length $d_v$. Multiplying by $2B$ gives item 8; the substitution gives $2BTd^2 + 4BTd^2 + 2BTd^2 + 4BT^2d$.
9. Multi-head attention keeps a key and a value of size $d_k$ (taking $d_v = d_k$) per head and per position. Multi-query attention keeps one key and one value shared by all heads; grouped-query attention keeps one per group. This is item 9.

### Result and check

$$ \mathrm{MHA}(PX, PX) = P\,\mathrm{MHA}(X, X), \qquad \#\text{weights} = 4d^2, \qquad \text{FLOPs} = 8BTd^2 + 4BT^2d . $$

The check `properties.check.py` builds the layer from the definition with a fixed seed, checks items 1 to 6 on random inputs to $10^{-12}$, counts the weights of item 7 from the arrays, counts the multiply-adds of item 8 by running the loops and counting, and counts the cached numbers of item 9 for $h = 8$ and $G \in \{1, 2, 4, 8\}$.
