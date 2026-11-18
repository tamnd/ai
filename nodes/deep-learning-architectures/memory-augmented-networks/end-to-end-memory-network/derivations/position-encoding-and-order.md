---
id: position-encoding-and-order
derives: [c-pe-ramp, c-pe-swap]
kind: identity
assumes: ["the position encoding of Section 4.1 with 1-based word and dimension indices"]
uses: []
status: full
check: position-encoding-and-order.check.py
lean: null
sources:
  - {work: sukhbaatar-2015-end-to-end-memory-networks, locator: "Section 4.1"}
---

### Statement

Write $\alpha = j/J$ for the position of word $j$ in a sentence of $J$ words and $\beta = k/d$ for the dimension $k$ of $d$.

1. The weight of eq-pe is a bilinear interpolation, $l_{kj} = (1 - \alpha)(1 - \beta) + \alpha \beta$, so it lies in $[0, 1]$. In each dimension it is a linear ramp over the positions with slope $(2\beta - 1)/J$: the low dimensions weigh the first words most, the high dimensions the last words, and the word at $\alpha = 1/2$ weighs $1/2$ in every dimension.
2. Swapping the words at positions $j$ and $j'$ changes the sentence vector in dimension $k$ by $\frac{j - j'}{J}\,(2\beta - 1)\,(a_k - b_k)$, where $a$ and $b$ are the embeddings of the word first at $j'$ and the word first at $j$. The bag of words changes by zero. When $d$ is even, the dimension $k = d/2$ does not see the order at all.

### Assumptions

1. $l_{kj} = (1 - j/J) - (k/d)(1 - 2j/J)$ with $j = 1, \dots, J$ and $k = 1, \dots, d$, and $m = \sum_j l_j \circ A x_j$. Used in all the steps.

### Steps

1. Expand: $(1 - \alpha) - \beta(1 - 2\alpha) = 1 - \alpha - \beta + 2\alpha\beta$, and $(1 - \alpha)(1 - \beta) + \alpha\beta = 1 - \alpha - \beta + \alpha\beta + \alpha\beta$, the same. A convex combination of the corner values 1, 0, 0, 1 lies in $[0, 1]$. As a function of $j$ with $k$ fixed it is $1 - \beta - \alpha(1 - 2\beta)$, a line in $\alpha = j/J$ with slope $2\beta - 1$ per unit of $\alpha$, so $(2\beta - 1)/J$ per word. At $\alpha = 1/2$ it is $1/2 - \beta/2 + \beta/2 = 1/2$.
2. Before the swap, dimension $k$ holds $l_{kj} b_k + l_{kj'} a_k$ plus the other words; after it, $l_{kj} a_k + l_{kj'} b_k$. The difference is $(l_{kj} - l_{kj'})(a_k - b_k)$, and by step 1, $l_{kj} - l_{kj'} = (2\beta - 1)(j - j')/J$. At $\beta = 1/2$ this is zero. The bag of words has $l_{kj} = 1$ for all $j$, so the difference is zero for every $k$.

### Result and check

$$ l_{kj} = \Big(1 - \frac{j}{J}\Big)\Big(1 - \frac{k}{d}\Big) + \frac{j}{J}\,\frac{k}{d}, \qquad l_{kj} - l_{kj'} = \frac{j - j'}{J}\Big(\frac{2k}{d} - 1\Big). $$

The check `position-encoding-and-order.check.py` compares the two forms of the weight and the difference formula on a grid of $J$ and $d$, and swaps words in sentences embedded by the NumPy reference.
