---
id: alignment-gradient
derives: [c-grad-energy, c-zero-init]
kind: proof
assumes: ["the context is c_i = sum_j alpha_ij h_j with alpha_i the softmax of the energies over the real source positions"]
uses: [softmax-regression@al, backpropagation@al]
status: full
check: alignment-gradient.check.py
lean: null
sources:
  - {work: bahdanau-2015-neural-machine-translation, locator: "p. 3", note: "eq. 5 and 6, the context and the weights"}
  - {work: bahdanau-2015-neural-machine-translation, locator: "p. 4", note: "the soft alignment lets the gradient of the cost be backpropagated through"}
  - {work: bahdanau-2015-neural-machine-translation, locator: "p. 14", note: "B.1, v_a initialized to zero"}
---

### Statement

Let $g_i = \partial \mathcal{L} / \partial c_i$ and $\gamma_{ij} = g_i^\top h_j$. Then
$$ \frac{\partial \mathcal{L}}{\partial e_{ij}} = \alpha_{ij} \Big( \gamma_{ij} - \sum_k \alpha_{ik} \gamma_{ik} \Big), \qquad \frac{\partial \mathcal{L}}{\partial h_j} \ni \alpha_{ij} g_i . $$
The gradient at an energy is zero at a masked position, where $\alpha_{ij} = 0$, and zero at every position when all the real annotations are equal. The energies of one step get gradients that sum to zero.

### Assumptions

1. $\alpha_{ij} = \exp(e_{ij}) / \sum_k \exp(e_{ik})$ over the real positions $k$, and $\alpha_{ij} = 0$ at a masked one. Used in step 2.
2. $c_i$ reaches the loss only through $g_i$. Used in step 1.

### Steps

1. $c_i = \sum_j \alpha_{ij} h_j$ is linear in the weights, so $\partial \mathcal{L} / \partial \alpha_{ij} = g_i^\top h_j = \gamma_{ij}$, and it is linear in each $h_j$ with coefficient $\alpha_{ij}$, which gives the second formula: every annotation gets the gradient at the context in proportion to its weight.
2. The Jacobian of the softmax is $\partial \alpha_{ik} / \partial e_{ij} = \alpha_{ik} (\delta_{jk} - \alpha_{ij})$. Then $\partial \mathcal{L} / \partial e_{ij} = \sum_k \gamma_{ik} \alpha_{ik} (\delta_{jk} - \alpha_{ij}) = \alpha_{ij} \gamma_{ij} - \alpha_{ij} \sum_k \alpha_{ik} \gamma_{ik}$.
3. At a masked position $\alpha_{ij} = 0$, so the factor in front is 0. If all the real $h_j$ are equal, all the $\gamma_{ij}$ are equal, to their weighted mean, and the bracket is 0.
4. Summing step 2 over $j$ gives $\sum_j \alpha_{ij} \gamma_{ij} - (\sum_j \alpha_{ij}) \sum_k \alpha_{ik} \gamma_{ik} = 0$, since the weights sum to 1.

### Result and check

The soft alignment is a differentiable function of the energies, so the alignment model $a$ learns from the translation loss alone, with no alignment labels: an energy rises when its annotation agrees with the gradient direction at the context more than the current average does. The check `alignment-gradient.check.py` compares the formula with finite differences of a random loss through a masked softmax and context, verifies the zero sum and the zero gradient for equal annotations and at the pads, and checks in the backward pass of the NumPy reference that at Bahdanau et al.'s initialization, $v_a = 0$, the weights are uniform, $v_a$ gets a gradient and $W_a$ and $U_a$, whose gradients carry a factor $v_a$, get none until $v_a$ has moved.
