---
id: temporal-rows-and-empty-memories
derives: [c-temporal-common-shift, c-empty-memory]
kind: identity
assumes: ["the temporal encoding of Section 4.1", "the null word embeds to zero"]
uses: []
status: full
check: temporal-rows-and-empty-memories.check.py
lean: null
sources:
  - {work: sukhbaatar-2015-end-to-end-memory-networks, locator: "Section 4.1"}
  - {work: sukhbaatar-2015-end-to-end-memory-networks, locator: "Section 4.2"}
---

### Statement

1. Adding the same vector $t$ to every row of $T_A$ changes no attention weight: the attention depends on $T_A$ only through the differences of its rows.
2. A memory slot whose sentence is all null words has $m_i = T_A(i)$ and $c_i = T_C(i)$. It still takes the attention weight $p_i \propto \exp(u^\top T_A(i))$ and adds $p_i\, T_C(i)$ to the output. A model whose stories are padded therefore learns what an empty slot at each position is worth, and the empty memories that random noise inserts move the real sentences to other rows of $T_A$.

### Assumptions

1. $m_i = \sum_j l_j \circ A x_{ij} + T_A(i)$ and $c_i = \sum_j l_j \circ C x_{ij} + T_C(i)$, eq-temporal. Used in both steps.
2. The embedding of the null word is zero, as Section 4.2 constrains it. Used in step 2.

### Steps

1. With $T_A(i) + t$ in place of $T_A(i)$, every score $u^\top m_i$ grows by the same $u^\top t$. The softmax of eq-match is unchanged when the same number is added to all its inputs, since the factor $e^{u^\top t}$ cancels between the numerator and the denominator.
2. All the words of an empty slot are the null word, whose embedding is zero by assumption 2, so the sums are zero and $m_i = T_A(i)$, $c_i = T_C(i)$. eq-match gives it the weight $\exp(u^\top T_A(i)) / \sum_{i'} \exp(u^\top m_{i'})$, which is positive, and eq-out adds $p_i T_C(i)$. If an empty slot is inserted among the sentences, every older sentence moves down one row and is encoded with the next row of $T_A$ and $T_C$.

### Result and check

$$ \operatorname{softmax}_i\big(u^\top (m_i + t)\big) = \operatorname{softmax}_i\big(u^\top m_i\big), \qquad m_i = T_A(i) \text{ for an empty slot.} $$

The check `temporal-rows-and-empty-memories.check.py` shifts the temporal rows of the NumPy reference and compares the attention, and embeds an empty slot.
