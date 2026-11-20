---
id: causal-gradient
derives: [c-alg1, c-alg1-memory]
kind: proof
assumes: ["the causal numerator nbar_i = phi(Q_i)^T sum_{j <= i} phi(K_j) V_j^T and a scalar loss L with G_i = dL/dnbar_i given"]
uses: [matrix-multiplication@m2, backpropagation@al]
status: full
check: causal-gradient.check.py
lean: null
sources:
  - {work: katharopoulos-2020-transformers-are-rnns, locator: "Section 3.3.1", note: "Eq. 13 to 15 and the cost of the cumulative sums"}
  - {work: katharopoulos-2020-transformers-are-rnns, locator: "Algorithm 1", note: "the forward and backward pass of the numerator"}
  - {work: katharopoulos-2020-transformers-are-rnns, locator: "p. 11", note: "Supplementary Section A, Eq. 21 to 27, the derivation"}
---

### Statement

Write $q_i = \phi(Q_i)$, $k_j = \phi(K_j)$ in $\mathbb{R}^C$ and $v_j \in \mathbb{R}^M$, and $\bar n_i = \sum_{j \le i} (q_i^\top k_j) v_j$ for the causal numerator. Given $G_i = \partial L / \partial \bar n_i$,

1. $\partial L / \partial q_i = S_i G_i$ with $S_i = \sum_{j \le i} k_j v_j^\top$, the state of the forward pass;
2. $\partial L / \partial k_i = R_i v_i$ and $\partial L / \partial v_i = R_i^\top k_i$, with $R_i = \sum_{j \ge i} q_j G_j^\top$, a $C \times M$ sum run from the last position back;
3. so the gradient takes two passes over the positions, each keeping one $C \times M$ running sum, and needs only $q$, $k$, $v$ and $G$, $N(2C + 2M)$ numbers, instead of the $N C M$ numbers of all the $S_i$ that reverse-mode differentiation of the loop of the forward pass would store.

### Assumptions

1. The numerator is the masked sum above, and the denominator $q_i^\top z_i$ is differentiated separately, as Katharopoulos et al. leave it to automatic differentiation. Used in items 1 to 3.
2. Item 3 counts the stored numbers of one head and one sequence, not the working memory of the two running sums. Used in item 3.

### Steps

1. $\bar n_{ie} = \sum_{d} q_{id} \sum_{j \le i} k_{jd} v_{je}$. Only $\bar n_i$ depends on $q_i$, so $\partial L / \partial q_{it} = \sum_e G_{ie} \sum_{j \le i} k_{jt} v_{je} = (S_i G_i)_t$: the paper's Eq. 13 and 25.
2. $k_j$ enters every $\bar n_i$ with $i \ge j$: $\partial L / \partial k_{jt} = \sum_{i \ge j} \sum_e G_{ie} q_{it} v_{je} = (R_j v_j)_t$, Eq. 14 and 27. In the same way $\partial L / \partial v_{je} = \sum_{i \ge j} G_{ie} (q_i^\top k_j) = (R_j^\top k_j)_e$, Eq. 15.
3. $S_i = S_{i-1} + k_i v_i^\top$ forward and $R_i = R_{i+1} + q_i G_i^\top$ backward each cost $O(CM)$ per position, so both passes cost $O(NCM)$ and keep one matrix at a time. Algorithm 1 of the paper is these two loops.

### Result and check

The backward pass of causal linear attention is itself a pair of linear recurrences, one forward like the forward pass and one backward like backpropagation through time, so training needs neither the $N \times N$ weights of softmax attention nor the $N$ states of a naive recurrence. The check `causal-gradient.check.py` compares the two loops with the gradient of the masked $N \times N$ form and with finite differences, and counts the numbers each one stores.
