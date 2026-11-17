---
id: l1-shrinkage
derives: [c-shrink-soft-threshold, c-shrink-relu-encoder, c-shrink-figure, c-norm-scale-symmetry]
kind: closed-form
assumes: ["one latent with a fixed decoder direction d of unit norm", "the input x already has the decoder bias subtracted", "the activation f is constrained to f >= 0", "lambda >= 0"]
uses: []
status: full
check: l1-shrinkage.check.py
lean: null
sources:
  - {work: rajamanoharan-2024-gated-sparse-autoencoders, locator: "p. 4"}
  - {work: rajamanoharan-2024-gated-sparse-autoencoders, locator: "p. 3"}
  - {work: gao-2024-scaling-sparse-autoencoders, locator: "p. 3"}
---

### Statement

For a unit vector $d \in \mathbb{R}^m$, an input $x \in \mathbb{R}^m$ and $\lambda \ge 0$, the activation that minimizes the loss of eq-l1-loss for one latent,

$$ g(f) = \lVert x - f d \rVert_2^2 + \lambda f, \qquad f \ge 0, $$

is

$$ f^\star = \max\!\left(0, \; \langle d, x \rangle - \tfrac{\lambda}{2}\right). $$

Without the penalty the minimizer is $\max(0, \langle d, x \rangle)$, so whenever the latent is active the penalty lowers it by exactly $\lambda/2$, and it switches the latent off when $\langle d, x \rangle \le \lambda/2$.
A ReLU encoder with weight row $d$ and bias $-\lambda/2$ computes $f^\star$ exactly.
Second, with no constraint on the decoder norm, the penalty can be made as small as wished without changing any reconstruction.

### Assumptions

1. One latent, or several latents whose decoder directions are orthonormal, in which case the problem separates into one such problem per latent.
2. $\lVert d \rVert_2 = 1$; the input is centred by the decoder bias, so $x$ here is $x - b_{\text{dec}}$.
3. $f \ge 0$, as the ReLU makes it.

### Steps

1. Expand: $g(f) = \lVert x \rVert^2 - 2 f \langle d, x \rangle + f^2 \lVert d \rVert^2 + \lambda f = \lVert x \rVert^2 - 2 f \langle d, x \rangle + f^2 + \lambda f$.
2. $g$ is a convex parabola in $f$ with derivative $g'(f) = 2f - 2\langle d, x \rangle + \lambda$, zero at $f = \langle d, x \rangle - \lambda/2$.
3. On $f \ge 0$ the minimizer of a convex parabola is its vertex if the vertex is feasible and $0$ otherwise, which gives $f^\star = \max(0, \langle d, x \rangle - \lambda/2)$. With $\lambda = 0$ it is $\max(0, \langle d, x \rangle)$, and the difference is $\lambda/2$ whenever $\langle d, x \rangle > \lambda/2$.
4. A ReLU encoder with weight row $d$ and bias $-\lambda/2$ outputs $\mathrm{ReLU}(\langle d, x \rangle - \lambda/2) = f^\star$, so the best encoder for this loss reproduces the shrunken value, not the true coefficient.
5. Example: $x = d$, so the true coefficient is 1 and perfect reconstruction is possible; with $\lambda = 1$, $f^\star = 1/2$ and the reconstruction loss at the optimum is $1/4$.
6. Scale symmetry: for $s > 0$ replace the decoder column $d_i$ by $s\,d_i$ and the encoder row $w_i$ and bias $b_i$ by $w_i/s$ and $b_i/s$. Since $\mathrm{ReLU}(a/s) = \mathrm{ReLU}(a)/s$ for $s > 0$, the activation becomes $f_i/s$ and the reconstruction $\sum_i (f_i/s)(s\,d_i)$ is unchanged, while the penalty $\lambda \sum_i f_i$ is divided by $s$. Letting $s \to \infty$ drives the penalty to zero with the same reconstruction and the same set of active latents, which is why the decoder columns are held at unit norm.

### Result and check

The check minimizes $g$ on a fine grid for random unit $d$, inputs and $\lambda$ and compares with $f^\star$; confirms the $\lambda/2$ gap, the ReLU-encoder form, and the one-latent example; and confirms that rescaling by $s$ leaves reconstructions unchanged and divides the penalty by $s$.
