---
id: small-noise-denoising
derives: [c-dae-small-noise, c-dae-linear-exact]
kind: identity
assumes: ["Gaussian corruption, x~ = x + e with e ~ N(0, sigma^2 I)", "the squared reconstruction error", "a reconstruction r with continuous fourth derivatives, bounded with its derivatives, for the order of the remainder"]
uses: [multivariate-normal-distribution@m2, linearity-of-expectation@m2, trace-of-a-matrix@m2]
status: full
check: small-noise-denoising.check.py
lean: null
sources:
  - {work: rifai-2011-contractive-auto-encoders-explicit, locator: "p. 4", note: "for small Gaussian noise the denoising error carries a penalty on the Frobenius norm of the Jacobian of the reconstruction, citing Bishop (1995)"}
  - {work: vincent-2010-stacked-denoising-autoencoders, locator: "p. 3381", note: "isotropic Gaussian noise as one of the corruptions of the denoising autoencoder"}
---

### Statement

Let $r : \mathbb{R}^D \to \mathbb{R}^D$ be the reconstruction, $e_0 = x - r(x)$ the error on the clean input, $J_r(x)$ the Jacobian of $r$ at $x$ and $\Delta r(x)$ the vector of the Laplacians of its components. With $\epsilon \sim \mathcal N(0, \sigma^2 I)$,

1. $\mathbb{E}\, \lVert x - r(x + \epsilon) \rVert^2 = \lVert e_0 \rVert^2 + \sigma^2 \big( \lVert J_r(x) \rVert_F^2 - e_0 \cdot \Delta r(x) \big) + O(\sigma^4)$.
2. For an affine reconstruction $r(x) = A x + c$ the identity is exact: $\mathbb{E}\, \lVert x - A(x + \epsilon) - c \rVert^2 = \lVert x - A x - c \rVert^2 + \sigma^2 \lVert A \rVert_F^2$.

Training on Gaussian corruption therefore adds, to first order in $\sigma^2$, a penalty on the Jacobian of the whole map $g \circ f$, where the contractive autoencoder penalises the Jacobian of the encoder $f$ alone. The term $e_0 \cdot \Delta r$ vanishes when the clean reconstruction is perfect.

### Assumptions

1. Gaussian noise, used in step 2 for the moments $\mathbb{E}\,\epsilon = 0$, $\mathbb{E}\,\epsilon \epsilon^\top = \sigma^2 I$ and the vanishing odd moments.
2. Smoothness of $r$, used in step 1 for the expansion and its remainder.

### Steps

1. Expand $r(x + \epsilon) = r(x) + J \epsilon + \tfrac12 H[\epsilon, \epsilon] + R_3(\epsilon)$, where $H[\epsilon, \epsilon]_k = \epsilon^\top \nabla^2 r_k(x)\, \epsilon$ and $R_3$ is cubic in $\epsilon$ plus $O(\lVert \epsilon \rVert^4)$. Then $x - r(x + \epsilon) = e_0 - J\epsilon - \tfrac12 H[\epsilon, \epsilon] - R_3$.
2. Square and take the expectation. $\lVert e_0 \rVert^2$ is constant. $-2\, e_0 \cdot J \epsilon$ has mean 0. $-e_0 \cdot H[\epsilon, \epsilon]$ has mean $-\sigma^2 \sum_k e_{0k} \operatorname{tr} \nabla^2 r_k = -\sigma^2\, e_0 \cdot \Delta r$. $\lVert J \epsilon \rVert^2$ has mean $\sigma^2 \operatorname{tr} J^\top J = \sigma^2 \lVert J \rVert_F^2$. The remaining products are either of odd degree in $\epsilon$, mean 0, or of degree at least 4, mean $O(\sigma^4)$. This is statement 1.
3. For $r(x) = A x + c$, $J = A$, $H = 0$ and $R_3 = 0$, so the expansion has no remainder: $\mathbb{E}\, \lVert e_0 - A \epsilon \rVert^2 = \lVert e_0 \rVert^2 + \sigma^2 \lVert A \rVert_F^2$. This is statement 2.

### Result and check

$$ \mathbb{E}\, \lVert x - r(x + \epsilon) \rVert^2 = \lVert x - r(x) \rVert^2 + \sigma^2 \Big( \lVert J_r(x) \rVert_F^2 - \big(x - r(x)\big) \cdot \Delta r(x) \Big) + O(\sigma^4) . $$

The check `small-noise-denoising.check.py` computes the expectation by Gauss–Hermite quadrature for a small sigmoid autoencoder of the NumPy reference with $D = 2$, takes $J_r$ from the reference and $\Delta r$ by finite differences, and checks that the gap to statement 1 falls as $\sigma^4$ as $\sigma$ halves. It checks statement 2 exactly by quadrature, and by a Monte Carlo average within its standard error.
