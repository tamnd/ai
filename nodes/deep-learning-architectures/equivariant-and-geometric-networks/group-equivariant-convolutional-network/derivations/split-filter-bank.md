---
id: split-filter-bank
derives: [c-filter-bank, c-bank-permutation, c-bank-cost, c-shared-gradient]
kind: identity
assumes: ["G splits as translations times the stabilizer S of the origin, g = t s", "L_g L_h = L_{gh}"]
uses: [group-correlation-equivariance]
status: full
check: split-filter-bank.check.py
lean: null
sources:
  - {work: cohen-2016-group-equivariant-convolutional-networks, locator: "Section 7, eq. 18, and Sections 7.1 and 7.2"}
---

### Statement

1. Writing $g = t s$, with $t$ a translation and $s$ in the stabilizer $S$ of the origin (the four rotations for p4), the correlation is $[f \star \psi](t s) = \sum_{h} \sum_k f_k(h)\, [L_t [L_s \psi_k]](h)$ (eq. 18 of Cohen and Welling): turn the filters by every $s \in S$, then run one planar correlation.
2. For p4 the augmented bank is $F^+[o, r, c, s] = L_r\, W[o, c, s - r]$, with $s - r$ taken mod 4 (eq-bank); reshaped to $(4 C_{\text{out}}, 4 C_{\text{in}}, k, k)$, it is an ordinary planar filter bank on $4 C_{\text{in}}$ input planes. In the lifting layer the input has one plane per channel and $F^+[o, r, c] = L_r W[o, c]$.
3. Each filter of $F^+$ is a permutation of the entries of a stored filter, so $F^+$ holds each stored weight exactly $|S| = 4$ times. Building it costs no arithmetic, and the layer costs as much as a planar correlation with $4 C_{\text{out}}$ outputs and $4 C_{\text{in}}$ inputs.
4. The gradient with respect to a stored weight is the sum of the gradients of its 4 copies in $F^+$, each turned back: $\partial \mathcal{L} / \partial W[o, c, q] = \sum_r L_{r^{-1}}\, \partial \mathcal{L} / \partial F^+[o, r, c, q + r]$.

### Assumptions

1. Every $g \in G$ is $t s$ for a unique translation $t$ and $s \in S$. For p4, $S$ is the four rotations about the origin. Used in steps 1 and 2.
2. $L$ is an action, $L_g L_h = L_{gh}$. Used in step 1.
3. Rotation by 90 degrees about the centre of a $k \times k$ array with $k$ odd maps the array onto itself. Used in step 3.

### Steps

1. In eq-gconv put $g = t s$, so $\psi(g^{-1} h) = \psi(s^{-1} t^{-1} h) = [L_t [L_s \psi]](h)$, by the action property applied twice. So $[f \star \psi](t s) = \sum_h \sum_k f_k(h)\, [L_s \psi_k](t^{-1} h)$: for each $s$ this is a correlation, over translations $t$ only, of $f$ with the fixed filter $L_s \psi$.
2. For p4, a filter on the group is an array $W[c, q]$ over rotations $q$ and offsets, and $L_r$ acts on it as on a p4 map, by turning each plane and moving plane $q$ to $q + r$ (eq-p4-action). So $[L_r W](c, s) = L_r\, W[c, s - r]$, and the filter of output plane $(o, r)$ on input plane $(c, s)$ is $F^+[o, r, c, s] = L_r W[o, c, s - r]$. The sum over $h = (y, s)$ is a sum over input channels $c$, input planes $s$ and positions $y$. A planar correlation sums over channels and positions, so merging $(c, s)$ into one axis of $4 C_{\text{in}}$ planes turns it into one planar call.
3. $L_r$ permutes the $k^2$ positions of a filter plane (assumption 3), and $s \mapsto s - r$ permutes the 4 planes. So each $F^+[o, r, c, \cdot]$ is a permutation of $W[o, c, \cdot]$, and over $r = 0, \dots, 3$ each weight appears once per $r$, 4 times in all. The FLOPs are those of the planar correlation: $2 B n^2 k^2 \cdot (4 C_{\text{out}})(4 C_{\text{in}})$ for the group layer and $2 B n^2 k^2 \cdot 4 C_{\text{out}} C_{\text{in}}$ for the lifting layer.
4. $F^+$ is a linear function of $W$ that copies entries. The gradient of a copy is read off $\partial \mathcal{L} / \partial F^+$, and the chain rule adds the gradients of all copies of the same weight. The copy in row $r$ sits at plane $q + r$, turned by $L_r$, so its gradient is turned back by $L_{r^{-1}}$.

### Result and check

$$ F^+[o, r, c, s] = L_r\, W[o, c, s - r], \qquad f \star \psi = \operatorname{corr2d}\big(f_{(B,\, 4 C_{\text{in}},\, n,\, n)},\ F^+_{(4 C_{\text{out}},\, 4 C_{\text{in}},\, k,\, k)}\big), \qquad \frac{\partial \mathcal{L}}{\partial W[o, c, q]} = \sum_{r} L_{r^{-1}} \frac{\partial \mathcal{L}}{\partial F^+[o, r, c, q + r]}. $$

The check `split-filter-bank.check.py` builds the bank of the NumPy reference from distinct numbers and checks that every weight appears 4 times, that each row is a permutation of a stored filter, that the planar call equals the sum over the group, and that the gradient of the stored filter matches finite differences.
