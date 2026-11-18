---
id: linear-start-is-linear
derives: [c-linear-start-linear]
kind: identity
assumes: ["the softmax of every memory layer is removed, as in linear start training", "the story is fixed"]
uses: []
status: full
check: linear-start-is-linear.check.py
lean: null
sources:
  - {work: sukhbaatar-2015-end-to-end-memory-networks, locator: "Section 4.2"}
---

### Statement

For a fixed story, a model in linear start maps the question vector $u^1$ to the input of the answer map by a matrix: $u^{K+1} = G_K \cdots G_1\, u^1$ with $G_k = H_k + \sum_i c^k_i\, (m^k_i)^\top$, where $H_k$ is the identity under adjacent tying and $H$ under layer-wise tying. So the logits $W u^{K+1}$ are linear in $u^1$, and the model is linear up to the final softmax, as Sukhbaatar et al. describe it.

### Assumptions

1. The softmax of eq-match is replaced by the identity, so $p_i = u^\top m_i$. Used in step 1.
2. The memory vectors $m^k_i$ and output vectors $c^k_i$ depend on the story and the weights but not on $u$. Used in step 1.

### Steps

1. With $p_i = (m^k_i)^\top u^k$, eq-out gives $o^k = \sum_i c^k_i (m^k_i)^\top u^k$, a matrix times $u^k$. eq-stack, or its layer-wise form, adds $H_k u^k$, so $u^{k+1} = G_k u^k$.
2. By induction over the hops, $u^{K+1} = G_K \cdots G_1 u^1$, and eq-answer applies the fixed matrix $W$.
3. With the softmax in place, $p$ depends on $u$ through a ratio of exponentials, so the map is not linear; doubling $u$ sharpens the attention instead of doubling the logits.

### Result and check

$$ u^{K+1} = \Big(\prod_{k = K}^{1} \big(H_k + \textstyle\sum_i c^k_i\, (m^k_i)^\top\big)\Big)\, u^1. $$

The check `linear-start-is-linear.check.py` builds the matrices $G_k$ from the memory and output vectors of the NumPy reference, compares their product with the forward pass in linear mode for both tyings, and checks that the logits scale with $u^1$ in linear mode and not with the softmax.
