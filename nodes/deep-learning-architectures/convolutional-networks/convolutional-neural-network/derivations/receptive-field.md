---
id: receptive-field
derives: [c-rf-recursion, c-rf-linear, c-rf-lenet]
kind: identity
assumes: ["each layer l is a window of size k_l with stride s_l, a convolution or a pooling, along one axis; the other axis is the same", "no dilation", "positions are counted in the coordinates of the padded input, so padding moves a receptive field but does not change its size"]
uses: []
status: full
check: receptive-field.check.py
lean: null
sources:
  - {work: simonyan-2015-very-deep-convolutional-networks, locator: "Section 2.3"}
  - {work: lecun-1998-gradient-based-learning, locator: "p. 7"}
---

### Statement

1. Let $r_l$ be the receptive field of a unit of layer $l$, the span of input positions that it depends on, and $j_l$ the jump, the distance in input positions between two adjacent units of layer $l$. With $r_0 = j_0 = 1$,
$$ r_l = r_{l-1} + (k_l - 1)\, j_{l-1}, \qquad j_l = j_{l-1}\, s_l , $$
so that $r_L = 1 + \sum_{l=1}^{L} (k_l - 1) \prod_{i < l} s_i$. If $s_l \le k_l$ at every layer, the receptive field has no gaps.
2. A stack of $m$ layers of size $k$ with stride 1 has $r = 1 + m(k - 1)$: the receptive field grows linearly with depth. For $k = 3$ it is $2m + 1$, so two such layers see $5 \times 5$ inputs and three see $7 \times 7$.
3. In LeNet-5 (C1, S2, C3, S4, C5 with $k = 5, 2, 5, 2, 5$ and $s = 1, 2, 1, 2, 1$), the receptive fields are 5, 6, 14, 16 and 32, and the jumps 1, 2, 2, 4 and 4: a unit of C5 sees the whole $32 \times 32$ input.

### Assumptions

1. Unit $i$ of layer $l$ reads units $s_l i, \dots, s_l i + k_l - 1$ of layer $l - 1$ (eq-conv, and the same index for pooling). Used in steps 2 and 3.
2. The receptive field of unit $a$ of layer $l - 1$ is the interval $[a\, j_{l-1} + o_{l-1},\; a\, j_{l-1} + o_{l-1} + r_{l-1} - 1]$ for an offset $o_{l-1}$ that does not depend on $a$. This holds for $l - 1 = 0$ with $o_0 = 0$, $r_0 = j_0 = 1$, and step 4 shows that it carries over to layer $l$. Used in steps 2 to 4.

### Steps

1. Proceed by induction on $l$; the base case is assumption 2 at $l = 0$, where each unit is one input.
2. By assumption 1, unit $i$ of layer $l$ depends on the units $a = s_l i, \dots, s_l i + k_l - 1$ of layer $l - 1$. By assumption 2, the first of them starts at input position $s_l i\, j_{l-1} + o_{l-1}$ and the last ends at $(s_l i + k_l - 1)\, j_{l-1} + o_{l-1} + r_{l-1} - 1$.
3. The span from the first to the last position is $(k_l - 1) j_{l-1} + r_{l-1}$. This is the recursion for $r_l$.
4. The start of unit $i$ is $i\,(s_l j_{l-1}) + o_{l-1}$, so adjacent units of layer $l$ are $j_l = s_l j_{l-1}$ apart and $o_l = o_{l-1}$; assumption 2 holds at layer $l$. This is the recursion for $j_l$. Unrolling it gives $j_{l-1} = \prod_{i < l} s_i$, and summing the increments of step 3 gives the closed form of statement 1.
5. The union of the receptive fields of adjacent units of layer $l - 1$ has no gap when $j_{l-1} \le r_{l-1}$. This holds at $l - 1 = 0$, and if $s_l \le k_l$ then $j_l = s_l j_{l-1} \le k_l j_{l-1} \le r_{l-1} + (k_l - 1) j_{l-1} = r_l$; so by induction the receptive field is an interval at every layer.
6. With $s_l = 1$ for all $l$, every $j$ is 1 and each layer adds $k - 1$: $r = 1 + m(k - 1)$. For $k = 3$, $r = 2m + 1$: 5 for $m = 2$ and 7 for $m = 3$, the effective receptive fields that Simonyan and Zisserman give for stacks of two and three $3 \times 3$ layers. This is statement 2.
7. LeNet-5: $r = 1 + 4 = 5$, $j = 1$; then $5 + 1 \cdot 1 = 6$, $j = 2$; then $6 + 4 \cdot 2 = 14$, $j = 2$; then $14 + 1 \cdot 2 = 16$, $j = 4$; then $16 + 4 \cdot 4 = 32$, $j = 4$. This is statement 3.

### Result and check

$$ r_l = r_{l-1} + (k_l - 1)\, j_{l-1}, \qquad j_l = j_{l-1}\, s_l $$

The check `receptive-field.check.py` builds random stacks of windows, measures the receptive field of a unit as the set of inputs where the gradient of that unit through kernels of ones is nonzero, and compares it with the recursion; it also evaluates statements 2 and 3.
