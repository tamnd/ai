---
id: permutation
derives: [c-bag, c-two-layers, c-mask-position]
kind: proof
assumes: ["the network of the page with pos = none: no position embeddings", "the causal mask of eq-mask", "layer normalization, the MLP and the projections act on each token alone with shared weights"]
uses: [permutation-matrix@m2, matrix-multiplication@m2]
status: full
check: permutation.check.py
lean: null
sources:
  - {work: vaswani-2017-attention, locator: "pp. 3 and 5", note: "the masking of the decoder self-attention"}
  - {work: radford-2018-improving-language-understanding, locator: "p. 3", note: "eq. 2, where W_p adds the position"}
---

### Statement

1. With no position embeddings and one layer, the logits at position $i$ depend only on the token at $i$ and on the multiset of the tokens at positions $1, \dots, i$: reordering the tokens before $i$ does not change them, whatever the parameters.
2. With two layers this fails: there are parameters for which reordering the tokens before $i$ changes the logits at $i$.
3. Without position embeddings the causal mask still tells positions apart. If all the tokens are the same, every position gives the same logits; if the first token is $a$ and the $i - 1$ after it are $b \neq a$, a head of the first layer gives position $i \geq 2$ the output $w_i \mathbf{v}_a + (1 - w_i) \mathbf{v}_b$ with $w_i = e^{s_a} / (e^{s_a} + (i-1) e^{s_b})$, which is different at every position when $\mathbf{v}_a \neq \mathbf{v}_b$.

### Assumptions

1. No position embeddings, so the state of a token entering the first layer is its row of the token embedding. Used in steps 1, 4 and 5.
2. Query $i$ sees the keys $j \leq i$ and no others. Used in steps 2, 4 and 5.
3. Layer normalization, the MLP, the qkv and output projections, the final normalization and the unembedding act on each position alone with the same weights at every position. Used in steps 1, 3 and 4.

### Steps

1. Let $\pi$ reorder positions $1, \dots, i - 1$ and keep position $i$. Before the first layer, position $j$ holds $\mathbf{e}_{x_j}$, a function of its token alone, and so do the normalized states and the query, key and value of each position.
2. The attention output at $i$ is $\sum_{j \leq i} \operatorname{softmax}_j(\mathbf{q}_i \mathbf{k}_j^\top / \sqrt{d_{\text{head}}})\, \mathbf{v}_j$, a sum over the pairs $(\mathbf{k}_j, \mathbf{v}_j)$ of positions $1, \dots, i$ with weights that depend on each pair and on $\mathbf{q}_i$. The query is that of the token at $i$, which $\pi$ keeps, and the pairs of positions $1, \dots, i$ are those of the tokens at those positions, which $\pi$ only reorders. A sum does not depend on the order of its terms, so the output at $i$ is unchanged, in every head.
3. The output projection, the residual addition, the MLP, the final normalization and the unembedding act on position $i$ alone, so the logits at $i$ are unchanged. This is statement 1.
4. With two layers, the state of position $j < i$ after the first layer depends on the tokens at positions $1, \dots, j$ by step 2, and $\pi$ changes which tokens those are. The second layer at $i$ then sees different keys and values, and nothing forces its output to stay the same. The check finds random parameters for which the logits at the last position change by more than $10^{-3}$, which proves statement 2. It uses widths of 4 to 9: at widths 1 and 2 layer normalization leaves at most the sign of each state, and the reordering can then be invisible.
5. If every token is $b$, all the keys and values are equal, any weights summing to one give $\mathbf{v}_b$, and every position has the same state after every layer. If the first token is $a$ and the rest $b$, the query at every $i \geq 2$ is that of $b$, so the two scores $s_a$ and $s_b$ are the same at every position; position $i$ has one key of $a$ and $i - 1$ of $b$, and the softmax gives $w_i$ to $a$ and $1 - w_i$ to the $b$s together. $w_i$ is strictly decreasing in $i$, so $w_i \mathbf{v}_a + (1 - w_i)\mathbf{v}_b$ is different at every $i$ when $\mathbf{v}_a \neq \mathbf{v}_b$. This is statement 3.

### Result and check

$$ \text{one layer, no positions:}\quad \text{logits}_i = f\big(x_i, \{\!\{x_1, \dots, x_i\}\!\}\big), \qquad w_i = \frac{e^{s_a}}{e^{s_a} + (i - 1)\,e^{s_b}} . $$

The check `permutation.check.py` reorders the tokens before the last position for random networks without position embeddings, with one layer and with two, and with learned positions; it builds the sequence of statement 3 and compares the attention weights of the reference with $w_i$, and checks that the logits at positions 2 to $n$ are pairwise different.
