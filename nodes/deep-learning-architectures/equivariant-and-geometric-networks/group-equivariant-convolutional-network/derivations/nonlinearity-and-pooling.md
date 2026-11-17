---
id: nonlinearity-and-pooling
derives: [c-pointwise-commutes, c-bias-per-map, c-coset-pooling, c-model-invariant, c-stride-even-grid]
kind: identity
assumes: ["the nonlinearity acts on each value separately", "G acts on p4 maps by eq-p4-action", "the image is n by n and rotations turn it about its centre"]
uses: [group-correlation-equivariance]
status: full
check: nonlinearity-and-pooling.check.py
lean: null
sources:
  - {work: cohen-2016-group-equivariant-convolutional-networks, locator: "Sections 6.1 to 6.3"}
  - {work: bronstein-2021-geometric-deep-learning, locator: "Section 3.5"}
---

### Statement

1. A pointwise nonlinearity $\nu$ commutes with the action, $\nu \circ L_u = L_u \circ \nu$. So does a bias with one value per G-feature map, shared by its 4 rotation planes. A bias with one value per plane does not.
2. Coset pooling over the rotations, $[P F](t) = \max_r F(t, r)$ (eq-rotpool), maps a p4 map to a planar map that turns with the input: $P\, L_u = L_u\, P$, where on the left $u$ acts on p4 maps and on the right on the plane.
3. The mean over the plane of a map that turns with the input is invariant to rotations, and to cyclic translations when the correlations wrap around. So the model of the page, lift, ReLU, group correlation, ReLU, $P$, mean, linear head, has logits invariant to the 4 rotations.
4. Subsampling with stride 2, keeping the pixels with even indices, keeps rotation equivariance on a grid of odd side and breaks it on a grid of even side.

### Assumptions

1. $\nu$ acts on each value alone: $[\nu \circ F](g) = \nu(F(g))$. Used in step 1.
2. The action on p4 maps is $[L_u F](t, r) = F(R^{-1} t, r - 1)$ for the rotation $u$, from the derivation group-correlation-equivariance. Used in steps 1 and 2.
3. Rotation about the centre of an $n \times n$ grid maps index $(i, j)$ to $(n - 1 - j, i)$. Used in step 4.

### Steps

1. $[\nu \circ L_u F](g) = \nu(F(u^{-1} g)) = [L_u (\nu \circ F)](g)$. A bias $b$ that is constant on $G$ for each channel satisfies $L_u b = b$, so $L_u F + b = L_u (F + b)$. A bias $b(r)$ that depends on the plane gives $L_u F + b$, whose plane $r$ is $F(R^{-1} \cdot, r - 1) + b(r)$, while $L_u (F + b)$ has $b(r - 1)$ there; the two differ unless $b(r) = b(r - 1)$ for all $r$. Batch normalization is a per-channel affine map of the same kind, so it needs one scale and one shift per G-feature map.
2. $[P L_u F](t) = \max_r F(R^{-1} t, r - 1) = \max_{r'} F(R^{-1} t, r') = [P F](R^{-1} t) = [L_u (P F)](t)$, because $r \mapsto r - 1$ permutes the four rotations. The max over a coset $t R$ does not depend on which element of the coset names it, so $P F$ is a function on $p4 / R \cong \mathbb{Z}^2$ that transforms like the input image.
3. $\frac{1}{n^2} \sum_t [L_u (P F)](t) = \frac{1}{n^2} \sum_t [P F](R^{-1} t) = \frac{1}{n^2} \sum_{t'} [P F](t')$, because $t \mapsto R^{-1} t$ permutes the pixels of the grid (and so does a cyclic shift). By group-correlation-equivariance and steps 1 and 2, the pooled map of a rotated image is the rotated pooled map, so the mean, and the linear head after it, give the same logits.
4. Rotation maps $(i, j)$ to $(n - 1 - j, i)$. If $n = 2m + 1$, then $n - 1 - j = 2m - j$ is even exactly when $j$ is even, so the set of pixels with two even indices maps onto itself, and subsampling commutes with rotation. If $n$ is even, $n - 1 - j$ is odd for every even $j$, so the kept set maps onto pixels with an odd first index, and the subsampled rotated map reads a different set of pixels.

### Result and check

$$ \nu \circ L_u = L_u \circ \nu, \qquad P\, L_u = L_u\, P, \qquad \text{logits}(L_u x) = \text{logits}(x) \quad (u \in \{\text{rotations}\}). $$

The check `nonlinearity-and-pooling.check.py` confirms each statement on the NumPy reference: ReLU and a shared bias commute with the action, a bias per plane does not, coset pooling turns with the input, the logits are invariant to the 4 rotations (and to cyclic translations with wrapping), and stride 2 is equivariant on a grid of side 7 but not of side 6.
