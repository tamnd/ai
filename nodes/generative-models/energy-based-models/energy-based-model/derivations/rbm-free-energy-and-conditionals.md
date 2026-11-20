---
id: rbm-free-energy-and-conditionals
derives: [c-rbm-conditionals]
kind: closed-form
assumes: ["binary units with values 0 and 1", "no connections within the visible layer or within the hidden layer"]
uses: [bernoulli-distribution@m2, markov-random-field@m2, logistic-regression@m2, matrix-multiplication@m2]
status: full
check: rbm-free-energy-and-conditionals.check.py
lean: null
sources:
  - {work: hinton-2002-training-products-experts-contrastive-divergence, locator: "p. 7", note: "the RBM as a product of experts, one per hidden unit, and exact inference because the hidden states are conditionally independent given the data"}
---

### Statement

For $E(\mathbf{v}, \mathbf{h}) = -\mathbf{v} W \mathbf{h}^\top - \mathbf{b} \cdot \mathbf{v} - \mathbf{c} \cdot \mathbf{h}$ with $\mathbf{v} \in \{0,1\}^D$, $\mathbf{h} \in \{0,1\}^H$,

$$ \sum_{\mathbf{h}} e^{-E(\mathbf{v}, \mathbf{h})} = e^{\mathbf{b} \cdot \mathbf{v}} \prod_{j=1}^{H} \big(1 + e^{c_j + (\mathbf{v} W)_j}\big) = e^{-F(\mathbf{v})} , \qquad F(\mathbf{v}) = -\mathbf{b} \cdot \mathbf{v} - \sum_j \operatorname{softplus}\big(c_j + (\mathbf{v}W)_j\big) , $$

and

$$ p(\mathbf{h} \mid \mathbf{v}) = \prod_j \sigma\big(c_j + (\mathbf{v}W)_j\big)^{h_j} \big(1 - \sigma(c_j + (\mathbf{v}W)_j)\big)^{1 - h_j} , $$

with the same form for $p(\mathbf{v} \mid \mathbf{h})$ with $b_i + (W \mathbf{h}^\top)_i$.

### Assumptions

1. Units take the values 0 and 1. Used in steps 2 and 4.
2. The energy has no terms $v_i v_{i'}$ or $h_j h_{j'}$. Used in steps 1 and 5.

### Steps

1. With no hidden-hidden terms, $-E = \mathbf{b} \cdot \mathbf{v} + \sum_j h_j a_j$ with $a_j = c_j + (\mathbf{v}W)_j$, a sum of one term per hidden unit.
2. The sum over $\mathbf{h} \in \{0,1\}^H$ of $\prod_j e^{h_j a_j}$ factorizes into $\prod_j \sum_{h_j \in \{0,1\}} e^{h_j a_j} = \prod_j (1 + e^{a_j})$.
3. Taking $-\log$ gives $F(\mathbf{v})$, and $p(\mathbf{v}) = \sum_\mathbf{h} e^{-E}/Z = e^{-F(\mathbf{v})}/Z$. Computing the $H$ values $a_j$ costs one product $\mathbf{v}W$, $O(DH)$.
4. $p(\mathbf{h} \mid \mathbf{v}) = e^{-E(\mathbf{v}, \mathbf{h})} / e^{-F(\mathbf{v})} = \prod_j e^{h_j a_j} / (1 + e^{a_j})$, a product of Bernoulli distributions with $p(h_j = 1 \mid \mathbf{v}) = e^{a_j}/(1 + e^{a_j}) = \sigma(a_j)$.
5. The energy is symmetric in the roles of $\mathbf{v}$ and $\mathbf{h}$, with $W^\top$ in place of $W$, so the same steps give $p(\mathbf{v} \mid \mathbf{h})$.

### Result and check

$$ p(\mathbf{v}) = \frac{e^{-F(\mathbf{v})}}{Z}, \qquad p(h_j = 1 \mid \mathbf{v}) = \sigma\big(c_j + (\mathbf{v}W)_j\big), \qquad p(v_i = 1 \mid \mathbf{h}) = \sigma\big(b_i + (W\mathbf{h}^\top)_i\big) . $$

The check `rbm-free-energy-and-conditionals.check.py` sums $e^{-E}$ over all hidden vectors for random RBMs, compares the result and both conditionals with the formulas, and checks that $p(h_j \mid \mathbf{v})$ does not depend on the other hidden units.
