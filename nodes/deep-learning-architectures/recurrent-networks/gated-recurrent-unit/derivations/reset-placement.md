---
id: reset-placement
derives: [c-reset-agree, c-reset-cost, c-flip]
kind: identity
assumes: ["the two candidates eq-candidate and eq-candidate-cho with the same parameters", "the reset gate ranges over an open subset of (0, 1)^H as the input varies"]
uses: [hadamard-product@m2, matrix-multiplication@m2]
status: full
check: reset-placement.check.py
lean: null
sources:
  - {work: cho-2014-learning-phrase-representations-rnn, locator: "eq. (8) and Appendix A.1"}
  - {work: chung-2014-empirical-evaluation-gated-recurrent, locator: "Section 3.2 and footnote 1"}
  - {work: pytorch-torch-nn-gru, locator: "the note on the new gate"}
---

### Statement

1. The candidates $\tanh(a_n + r \odot (h U_n + b_{hn}))$ and $\tanh(a_n + (r \odot h) U_n + b_{hn})$ are equal for all $r \in (0, 1)^H$ and all $h \in \mathbb R^H$ if and only if $U_n$ is diagonal and $b_{hn} = 0$.
2. With the reset after the matrix, the recurrent products of the three blocks, $h_{t-1} U$, do not depend on $r_t$ and can be computed as one $(B, H) \times (H, 3H)$ product per step. With the reset before the matrix, the product of the candidate block needs $r_t$, which needs $h_{t-1} U_r$ first: two products per step, one after the other, with the same FLOPs.
3. The update of Chung et al., $h_t = (1 - z'_t) h_{t-1} + z'_t \tilde h_t$, is the update of eq-update with $z'_t = 1 - z_t$, and $z'_t = \sigma(-(\text{pre-activation of } z_t))$. Negating the weights and the bias of the update gate maps one convention onto the other exactly.

### Assumptions

1. The arguments $a_n$, $U_n$ and $b_{hn}$ are the same in both candidates. Used in step 1.
2. A product that depends on $r_t$ cannot start before $r_t$ is known. Used in step 2.
3. $1 - \sigma(u) = \sigma(-u)$ for every real $u$. Used in step 3.

### Steps

1. The tanh is injective, so the candidates agree exactly when $r \odot (h U_n) + r \odot b_{hn} = (r \odot h) U_n + b_{hn}$. Put $h = 0$: $r \odot b_{hn} = b_{hn}$ for all $r$ in an open set, so $b_{hn} = 0$. Then put $h = e_i$ and compare coordinate $j$: $r_j (U_n)_{ij} = r_i (U_n)_{ij}$ for all $r$, so $(U_n)_{ij} = 0$ whenever $i \ne j$. Conversely, if $U_n = \operatorname{diag}(u)$ and $b_{hn} = 0$, both sides are $r \odot u \odot h$. This is statement 1: in general the two placements are different functions with the same parameters, and a model trained with one does not compute the same thing when run with the other.
2. In eq-candidate the reset gate multiplies $h_{t-1} U_n + b_{hn}$ after the product, so $h_{t-1} [U_r\ U_z\ U_n]$ is one product of $h_{t-1}$ with $U$. In eq-candidate-cho the matrix multiplies $r_t \odot h_{t-1}$, and $r_t = \sigma(a_r + h_{t-1} U_r + b_{hr})$ must be computed first. The number of multiply-adds is $3BH^2$ in both cases. This is statement 2, and it is the reason the PyTorch documentation gives for its choice: the implementation differs on purpose for efficiency.
3. Substitute $z'_t = 1 - z_t$ in the update of Chung et al.: $(1 - z'_t) h_{t-1} + z'_t \tilde h_t = z_t h_{t-1} + (1 - z_t) \tilde h_t$, which is eq-update. If $z_t = \sigma(u)$ then $z'_t = \sigma(-u)$, and $-u$ is the pre-activation of a gate with the weights and the bias negated. This is statement 3.

### Result and check

$$ r \odot (h U_n + b_{hn}) \equiv (r \odot h) U_n + b_{hn} \iff U_n \text{ diagonal and } b_{hn} = 0, \qquad z' = 1 - z = \sigma(-u) . $$

The check `reset-placement.check.py` finds a difference of order 0.1 between the two placements for random parameters, finds agreement to rounding when $U_n$ is diagonal and $b_{hn} = 0$, and a difference again when either condition is broken; it counts the products of both orders, and runs the convention of Chung et al. against the layer with the update gate negated.
