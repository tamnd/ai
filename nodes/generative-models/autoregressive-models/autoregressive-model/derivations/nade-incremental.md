---
id: nade-incremental
derives: [c-nade-cost]
kind: closed-form
assumes: ["the input weights W are shared by all the conditionals", "a hidden layer of H units and D binary variables in a fixed ordering"]
uses: [matrix-multiplication@m2, logistic-regression@m2, bernoulli-distribution@m2]
status: full
check: nade-incremental.check.py
lean: null
sources:
  - {work: larochelle-2011-neural-autoregressive-distribution-estimator, locator: "Section 4", note: "the conditionals of eqs. 9 and 10, the shared hidden pre-activations and the O(HD) cost"}
---

### Statement

NADE computes conditional $i$ from a hidden layer that reads the earlier variables:

$$ \mathbf{h}_i = \sigma(\mathbf{c} + W_{<i}^\top \mathbf{x}_{<i}), \qquad p(x_i = 1 \mid \mathbf{x}_{<i}) = \sigma(b_i + \mathbf{h}_i^\top V_{:, i}) , $$

where $W \in \mathbb{R}^{D \times H}$ and $V \in \mathbb{R}^{H \times D}$. Write $\mathbf{z}_i = \mathbf{c} + \sum_{j < i} x_j W_{j,:}$ for the pre-activation. Then $\mathbf{z}_1 = \mathbf{c}$ and $\mathbf{z}_{i+1} = \mathbf{z}_i + x_i W_{i,:}$. All $D$ conditionals of one vector therefore cost $O(HD)$, against $O(HD^2)$ for $D$ separate passes, and the values are identical.

### Assumptions

1. $W$ is the same matrix in every conditional. With a separate input matrix per conditional, the recursion fails and the cost is $O(HD^2)$. Used in step 1.
2. The output weights $V$ need not be tied to $W$. Untying them, as Larochelle and Murray did, changes no cost. Used in step 3.

### Steps

1. The pre-activation of conditional $i + 1$ is a sum over $j < i + 1$, which is the sum over $j < i$ plus the term $j = i$. So $\mathbf{z}_{i+1} = \mathbf{z}_i + x_i W_{i,:}$, one vector addition of length $H$.
2. Over $i = 1, \dots, D$ these updates cost $O(HD)$. Over a batch of $B$ vectors they form one exclusive cumulative sum of the terms $x_j W_{j,:}$ along $j$, which costs $O(BDH)$.
3. Each conditional then costs $O(H)$ for its sigmoid and the dot product with $V_{:,i}$, so $O(HD)$ in all.
4. A naive pass recomputes $\mathbf{z}_i$ from scratch in $O(H(i-1))$, giving $O(H D^2 / 2)$ in all. Both compute the same numbers, since they evaluate the same sums.
5. By the derivation `autoregressive-normalization`, the product of the conditionals is a normalized distribution over $\{0, 1\}^D$.

### Result and check

$$ \mathbf{z}_i = \mathbf{c} + \sum_{j < i} x_j W_{j,:} = \mathbf{c} + \mathrm{cumsum}(\mathbf{x} \odot W)_i - x_i W_{i,:}, \qquad \text{cost } O(HD) \text{ per vector.} $$

The check `nade-incremental.check.py` compares `nade_logits`, the cumulative sum, with `nade_logits_naive`, one pass per conditional, on random weights and all binary vectors of length 7. It sums the probabilities to 1 and counts the multiply-adds of the two schemes.
