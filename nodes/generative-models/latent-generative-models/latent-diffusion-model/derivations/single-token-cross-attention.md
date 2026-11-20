---
id: single-token-cross-attention
derives: [c-single-token]
kind: identity
assumes: ["the softmax is over the M context tokens"]
uses: [matrix-multiplication@m2]
status: full
check: single-token-cross-attention.check.py
lean: null
sources:
  - {work: rombach-2021-high-resolution-image-synthesis-latent, locator: "pp. 4 to 5, Eq. 3", note: "cross-attention with K and V from the condition tokens"}
---

### Statement

For a query $q \in \mathbb{R}^d$ and context tokens $c_1, \ldots, c_M$, cross-attention returns $\sum_m w_m\, c_m W_V$ with $w = \mathrm{softmax}_m(q \cdot c_m W_K/\sqrt d)$. If $M = 1$, then $w_1 = 1$ for every $q$, and the output $c_1 W_V$ does not depend on the query, so on the features of $z_t$: for a class token $c_1 = \mathrm{ctx}_y$, cross-attention adds the bias $b_y = \mathrm{ctx}_y W_V$.

### Assumptions

1. The softmax normalizes over the $M$ context tokens. Used in step 1.

### Steps

1. With one token, $w_1 = e^{s_1}/e^{s_1} = 1$ whatever the score $s_1$.
2. The output is then $1 \cdot c_1 W_V$, a function of the condition alone. $W_Q$ and $W_K$ have no effect on it, and their gradients are zero.
3. For the denoiser of this page, $\tanh(h + \mathrm{ctx}_y W_V)$ is $\tanh(h + b_y)$ with a learned bias $b_y$ per class, the same as a class embedding added to the hidden layer.

### Result and check

A class label needs no attention: one token gives a lookup table of biases, which is how class-conditional U-Nets and DiT add the class embedding. Attention matters when the condition has several tokens, as text does, so that different features can read different tokens.
The check `single-token-cross-attention.check.py` builds the denoiser with $M = 1$, checks that the attention weights returned by `_forward` are 1, that the output equals the class-bias network for every class, and that it does not change when $W_Q$ and $W_K$ are replaced by random matrices; with $M = 2$ the output does change.
