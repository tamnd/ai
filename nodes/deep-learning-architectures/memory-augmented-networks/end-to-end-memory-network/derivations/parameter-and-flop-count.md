---
id: parameter-and-flop-count
derives: [c-params-adjacent, c-params-tying-halves, c-params-lm-ratio, c-flops, c-flops-embedding]
kind: count
assumes: ["adjacent tying of the embedding and the temporal matrices, the position encoding, a story of N sentences of J words", "a multiply-add counts as 2 FLOPs and only products are counted"]
uses: []
status: full
check: parameter-and-flop-count.check.py
lean: null
sources:
  - {work: sukhbaatar-2015-end-to-end-memory-networks, locator: "Section 2.2"}
  - {work: sukhbaatar-2015-end-to-end-memory-networks, locator: "Section 5.2"}
---

### Statement

1. With adjacent tying and the temporal encoding, the model has $P_{\text{adj}} = (K + 1)(Vd + Nd)$ parameters, and none of them depends on the sentence length $J$.
2. With no tying it would have $2K(Vd + Nd) + 2Vd$. The ratio of the two is $(K + 1)(V + N) / (2K(V + N) + 2V)$, close to one half when $N \ll V$; at the bAbI sizes it is $18{,}160$ against $34{,}320$.
3. The language model of Section 5, with layer-wise tying and a constant question, has $3Vd + d^2 + 2Nd$ parameters. A simple recurrent network with $d$ hidden units on the same vocabulary has $2Vd + d^2$ weights. At the Penn Treebank sizes, $V = 10{,}000$, $d = 150$, $N = 100$, the ratio is about $1.5$.
4. One forward pass costs $C_{\text{fwd}} = B\,[\,2JD((K + 1)N + 1) + 4KND + 2VD\,]$ FLOPs, and the embedding of the story, $2BJD(K + 1)N$, is the largest term when $J(K + 1) N > V$.

### Assumptions

1. Adjacent tying: $A^{k+1} = C^k$, $W^\top = C^K$, $B = A^1$, and the temporal matrices are tied the same way, as Sukhbaatar et al. require of $T_A$ and $T_C$. Used in steps 1 and 4.
2. A product's multiply-add counts as 2 FLOPs; lookups, additions and the softmax are not counted. Used in step 4.

### Steps

1. Under adjacent tying the distinct matrices are $A^1 = B$, $C^1 = A^2$, ..., $C^{K-1} = A^K$, $C^K = W^\top$: $K + 1$ matrices of shape $V \times d$, and with them $K + 1$ temporal matrices of $N \times d$. That is $(K + 1)(Vd + Nd)$, eq-params. $J$ enters only the position encoding, which has no parameters.
2. Without tying each hop has its own $A^k$, $C^k$, $T_A^k$, $T_C^k$, $2K(Vd + Nd)$, and $B$ and $W$ add $2Vd$. The ratio of step 1 to this is $(K + 1)(V + N) / (2K(V + N) + 2V)$. With $N \ll V$ it is close to $(K + 1)/(2K + 2) = 1/2$. At the bAbI sizes, $18{,}160$ against $34{,}320$.
3. Layer-wise tying keeps one $A$ and one $C$ for all hops, a $W$ and the $d \times d$ map $H$, and one $T_A$ and one $T_C$; a constant question needs no $B$. So $3Vd + d^2 + 2Nd$. The recurrent network has an input matrix $V \times d$, a recurrent $d \times d$ and an output $d \times V$, $2Vd + d^2$. At the Penn Treebank sizes these are $4{,}552{,}500$ and $3{,}022{,}500$, a ratio of $1.506$, the "∼1.5x" of Section 5.2.
4. With the position encoding each sentence costs $J$ multiply-adds per dimension, $JD$; the story is embedded once per distinct matrix, $K + 1$ times, and the question once: $JD((K + 1)N + 1)$. Each hop has $N$ inner products of length $D$ for eq-match and a weighted sum of $N$ vectors for eq-out, $2ND$. The answer is a $V \times D$ matrix times a vector, $VD$. Doubling and multiplying by $B$ gives eq-flops. The story term exceeds the answer term $2BVD$ when $J(K+1)N > V$; at the bAbI sizes $7 \cdot 4 \cdot 50 = 1{,}400 > 177$.

### Result and check

$$ P_{\text{adj}} = (K + 1)(Vd + Nd), \qquad C_{\text{fwd}} = B\,[\,2JD((K + 1)N + 1) + 4KND + 2VD\,]. $$

The check `parameter-and-flop-count.check.py` counts the arrays of the NumPy reference for both tyings and several configs, counts the multiply-adds of every einsum and matrix product of one forward pass, and compares them with the formulas and with the numbers above.
