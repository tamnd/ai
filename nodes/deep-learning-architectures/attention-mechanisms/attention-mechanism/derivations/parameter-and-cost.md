---
id: parameter-and-cost
derives: [c-params, c-precompute, c-share]
kind: calculation
assumes: ["the model of Bahdanau et al.'s Appendix A.2 without biases, with source and target vocabularies of 30,000 words", "a multiply-add counted as 2 FLOPs, only the matrix and vector products counted"]
uses: [matrix-multiplication@m2]
status: full
check: parameter-and-cost.check.py
lean: null
sources:
  - {work: bahdanau-2015-neural-machine-translation, locator: "p. 12", note: "A.1.2, the shapes of W_a, U_a and v_a, and the precomputation of U_a h_j"}
  - {work: bahdanau-2015-neural-machine-translation, locator: "p. 13", note: "A.2.1 and A.2.2, the encoder and the decoder"}
  - {work: bahdanau-2015-neural-machine-translation, locator: "p. 14", note: "A.2.2 and A.2.3, the deep output and the sizes n = 1000, m = 620, l = 500, n' = 1000"}
  - {work: bahdanau-2015-neural-machine-translation, locator: "Section 4.1", note: "the shortlists of 30,000 words"}
---

### Statement

With $K_x = K_y = 30{,}000$, $m = 620$, $n = n' = 1000$ and $l = 500$, RNNsearch has
$$ P = (K_x + K_y) m + 2(3nm + 3n^2) + (3nm + 3n^2 + 6n^2 + n^2) + (nn' + 2nn' + n') + (2ln + 2lm + 4ln + lK_y) = 80{,}401{,}000 $$
weights, of which the alignment model has $3nn' + n' = 3{,}001{,}000$. For a pair of sentences of $T_x$ and $T_y$ words the alignment model and the contexts cost
$$ C_{\text{att}} = 4T_x n n' + 2T_y n n' + 2T_x T_y (n' + 2n) $$
FLOPs with $U_a h_j$ computed once per sentence, and $4T_xT_y n n' + 2T_y n n' + 2T_xT_y(n' + 2n)$ if it is recomputed at every target step. At $T_x = T_y = 30$ these are 185,400,000 and 3,665,400,000, and the whole forward pass is 2,539,400,000 FLOPs, of which attention is 7.3%.

### Assumptions

1. The matrices are those of Appendices A.2.1 and A.2.2 and the sizes those of A.2.3. Used in step 1.
2. The products are counted; the tanh, the softmax, the maxout and the gates' elementwise work are not. Used in steps 2 and 3.

### Steps

1. Embeddings: $(30{,}000 + 30{,}000) \cdot 620 = 37{,}200{,}000$. Each encoder GRU has $W, W_z, W_r$ of $n \times m$ and $U, U_z, U_r$ of $n \times n$: $3 \cdot 620{,}000 + 3 \cdot 10^6 = 4{,}860{,}000$, two of them 9,720,000. The decoder adds $C, C_z, C_r$ of $n \times 2n$ and $W_s$: $4{,}860{,}000 + 6{,}000{,}000 + 1{,}000{,}000 = 11{,}860{,}000$. The alignment model: $W_a$ of $n' \times n$, $U_a$ of $n' \times 2n$ and $v_a$: $3{,}001{,}000$. The deep output: $U_o$ ($2l \times n$), $V_o$ ($2l \times m$), $C_o$ ($2l \times 2n$) and $W_o$ ($K_y \times l$): $1{,}000{,}000 + 620{,}000 + 2{,}000{,}000 + 15{,}000{,}000 = 18{,}620{,}000$. The sum is 80,401,000, and $3{,}001{,}000 / 80{,}401{,}000 = 3.73\%$.
2. Attention: $U_a h_j$ for the $T_x$ annotations of width $2n$ is $2 \cdot T_x \cdot 2n \cdot n'$; $W_a s_{i-1}$ for the $T_y$ steps is $2 T_y n n'$; the dot products with $v_a$ are $2T_xT_y n'$ and the weighted sums $2T_xT_y \cdot 2n$. Recomputing $U_a h_j$ at each of the $T_y$ steps multiplies its term by $T_y$. At 30 words: $120{,}000{,}000 + 60{,}000{,}000 + 5{,}400{,}000 = 185{,}400{,}000$ against $3{,}600{,}000{,}000 + 65{,}400{,}000$.
3. The rest per sentence pair: the encoder, $2 \cdot T_x \cdot 2(3nm + 3n^2) = 583{,}200{,}000$; $s_0$, $2n^2 = 2{,}000{,}000$; the decoder GRU, $T_y \cdot 2(3nm + 3n^2 + 6n^2) = 651{,}600{,}000$; the deep output, $T_y \cdot 2(2ln + 2lm + 4ln + lK_y) = 1{,}117{,}200{,}000$. With attention the total is 2,539,400,000, and $185{,}400{,}000 / 2{,}539{,}400{,}000 = 7.30\%$.

### Result and check

Attention is cheap at the sentence lengths of translation, under a twentieth of the weights and under a tenth of the arithmetic, and the precomputation that the paper recommends saves a factor of about 20 on it. Its cost grows as $T_x T_y$, which is what limits it on long inputs. The check `parameter-and-cost.check.py` compares the formulas with the shapes of the NumPy reference and with a count of the products its forward pass performs on random small models, then evaluates them at the paper's sizes.
