---
id: backward
derives: [c-backward, c-score-grad-sum, eq-grad-out, eq-grad-v, eq-grad-weights, eq-grad-scores, eq-grad-qk]
kind: gradient
assumes: ["masked entries have weight exactly 0", "every row has at least one key that it may attend to, or its weights are all 0", "the loss is differentiable in the output"]
uses: [backpropagation@al, matrix-multiplication@m2, dot-product@m2]
status: full
check: backward.check.py
lean: null
sources:
  - {work: vaswani-2017-attention, locator: "Section 3.2.2"}
---

### Statement

Fix one head and drop its index. Let $Q \in \mathbb{R}^{T \times d_k}$, $K \in \mathbb{R}^{S \times d_k}$, $V \in \mathbb{R}^{S \times d_v}$, $Z = QK^\top/\sqrt{d_k}$ with the masked entries set to $-\infty$, $A = \mathrm{softmax}(Z)$ by rows, $O = AV$, and $Y = \mathrm{Concat}(O_1, \dots, O_h) W^O$ with $C = \mathrm{Concat}(O_1, \dots, O_h)$.
Let $\bar{Y} = \partial \ell / \partial Y$, and write $\bar{X}$ for the gradient of the loss with respect to $X$. Then, with $\bar{O}_r$ the columns of $\bar{Y} W^{O\top}$ that belong to head $r$,

1. $\bar{W}^O = C^\top \bar{Y}$;
2. $\bar{V} = A^\top \bar{O}$ and $\bar{A} = \bar{O} V^\top$;
3. $\bar{Z}_{ts} = A_{ts}\big(\bar{A}_{ts} - \sum_{u} A_{tu} \bar{A}_{tu}\big)$, and every row of $\bar{Z}$ sums to 0;
4. $\bar{Q} = \bar{Z} K / \sqrt{d_k}$ and $\bar{K} = \bar{Z}^\top Q / \sqrt{d_k}$;
5. $\bar{W}^Q_r = X^\top \bar{Q}_r$, $\bar{W}^K_r = M^\top \bar{K}_r$ and $\bar{W}^V_r = M^\top \bar{V}_r$, where $X$ gives the queries and $M$ the keys and values; the gradient of $X$ is $\sum_r \bar{Q}_r W^{Q\top}_r$, plus $\sum_r (\bar{K}_r W^{K\top}_r + \bar{V}_r W^{V\top}_r)$ when $M = X$.

### Assumptions

1. A masked entry has $A_{ts} = 0$ exactly, so it gets $\bar{Z}_{ts} = 0$ by item 3 and passes no gradient. Used in step 3.
2. A row with every key masked has all its weights 0 and passes no gradient. Used in step 3.

### Steps

1. $Y = C W^O$ is linear in $W^O$ and in $C$, so $\bar{W}^O = C^\top \bar{Y}$ and $\bar{C} = \bar{Y} W^{O\top}$. The columns of $C$ that hold head $r$ give $\bar{O}_r$. This is item 1.
2. $O = AV$ gives $\bar{V} = A^\top \bar{O}$ and $\bar{A} = \bar{O} V^\top$ in the same way. This is item 2.
3. Each row of $A$ is the softmax of the same row of $Z$, with Jacobian $\mathrm{diag}(\mathbf{a}) - \mathbf{a}\mathbf{a}^\top$ (derivation `scale`, step 2). It is symmetric, so $\bar{\mathbf{z}} = (\mathrm{diag}(\mathbf{a}) - \mathbf{a}\mathbf{a}^\top)\bar{\mathbf{a}} = \mathbf{a} \odot (\bar{\mathbf{a}} - (\mathbf{a}^\top\bar{\mathbf{a}})\mathbf{1})$. Summing over the row, $\sum_s \bar{z}_s = \mathbf{a}^\top\bar{\mathbf{a}} - (\mathbf{1}^\top\mathbf{a})(\mathbf{a}^\top\bar{\mathbf{a}}) = 0$, since $\mathbf{1}^\top\mathbf{a} = 1$. The zero sum is the gradient form of the fact that adding a constant to a row of scores does not change the softmax. Where $a_s = 0$, $\bar{z}_s = 0$; a row of zeros gives a zero row. This is item 3.
4. $Z = QK^\top / \sqrt{d_k}$ is bilinear, so $\bar{Q} = \bar{Z}K/\sqrt{d_k}$ and $\bar{K} = \bar{Z}^\top Q / \sqrt{d_k}$. This is item 4.
5. $Q_r = X W^Q_r$, $K_r = M W^K_r$, $V_r = M W^V_r$ are linear, which gives the weight gradients and the input gradients of item 5. In self-attention $X$ reaches the output along three paths, and the chain rule adds them.

### Result and check

$$ \bar{Z} = A \odot \big(\bar{A} - (A \odot \bar{A})\mathbf{1}\mathbf{1}^\top\big), \qquad \bar{Q} = \frac{\bar{Z}K}{\sqrt{d_k}}, \qquad \bar{K} = \frac{\bar{Z}^\top Q}{\sqrt{d_k}}, \qquad \bar{V} = A^\top \bar{O} . $$

The check `backward.check.py` builds random self-attention with a causal mask and cross-attention with a padding mask, for several $h$, $d_k$ and $d_v$, with a fixed seed.
It compares every gradient of items 1 to 5 with central finite differences to a relative error under $10^{-6}$, and checks that the rows of $\bar{Z}$ sum to 0 and that masked entries of $\bar{Z}$ are 0 to $10^{-12}$.
