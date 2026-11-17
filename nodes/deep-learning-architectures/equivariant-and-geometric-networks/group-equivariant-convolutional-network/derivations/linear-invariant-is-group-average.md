---
id: linear-invariant-is-group-average
derives: [c-linear-invariant-average, c-why-equivariant-layers]
kind: identity
assumes: ["G is a finite group acting linearly on the space of signals", "f is linear and G-invariant"]
uses: []
status: full
check: linear-invariant-is-group-average.check.py
lean: null
sources:
  - {work: bronstein-2021-geometric-deep-learning, locator: "Section 3.5"}
---

### Statement

1. If $f$ is linear and $f(g . x) = f(x)$ for every $g$ in a finite group $G$, then $f(x) = f(A x)$ with $A x = \frac{1}{|G|} \sum_{g \in G} g . x$, the group average: $f$ sees only the average of $x$ over the group.
2. For $G$ the cyclic shifts of an $n \times n$ image, $A x$ is the constant image at the mean of $x$, so a linear translation-invariant map of an image depends only on its mean value per channel. For the four rotations, $A x$ is the image averaged with its three rotated copies.
3. By contrast, a linear equivariant map $B$ followed by a pointwise nonlinearity $\sigma$ is equivariant, $\sigma(B(g . x)) = g . \sigma(B(x))$, and the group average of it, $A \circ \sigma \circ B$, is invariant without being a function of $A x$ alone.

### Assumptions

1. $G$ is finite and acts linearly: $g . (a x + b y) = a\, g . x + b\, g . y$. Used in step 1.
2. $f$ is linear and invariant. Used in step 1.

### Steps

1. Invariance gives $f(x) = f(g . x)$ for each $g$, so averaging the $|G|$ equal numbers, $f(x) = \frac{1}{|G|} \sum_g f(g . x)$. Linearity moves the sum inside: $\frac{1}{|G|} \sum_g f(g . x) = f(\frac{1}{|G|} \sum_g g . x) = f(A x)$.
2. The cyclic shifts move each pixel through every position of the grid once, so $(A x)(u) = \frac{1}{n^2} \sum_t x(u - t)$ is the mean of $x$, the same at every $u$. For the rotations, $A x = \frac14 (x + L_1 x + L_2 x + L_3 x)$ by definition.
3. $\sigma(B(g . x)) = \sigma(g . B(x)) = g . \sigma(B(x))$, because $\sigma$ acts on each value alone and $g$ only moves values. Then $A\, \sigma(B(g . x)) = A\, g . \sigma(B(x)) = A\, \sigma(B(x))$, since $A g = A$. Two images with the same average generally give different values of $\sigma(B(x))$, so the invariant is not a function of $A x$. The G-CNN is this construction: equivariant layers with nonlinearities, then an invariant pooling.

### Result and check

$$ f \text{ linear and } G\text{-invariant} \implies f(x) = f\!\left(\frac{1}{|G|} \sum_{g \in G} g . x\right). $$

The check `linear-invariant-is-group-average.check.py` makes a linear map invariant by averaging a random weight over the group, for the rotations and for the cyclic shifts, confirms that it gives the same value on an image and on its group average, and finds two images with the same rotation average that the G-CNN of the NumPy reference tells apart.
