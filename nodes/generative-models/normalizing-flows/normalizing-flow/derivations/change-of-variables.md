---
id: change-of-variables
derives: [c-cov, c-compose]
kind: closed-form
assumes: ["f is a bijection of R^D onto itself, differentiable with a differentiable inverse", "the base density p_z is continuous"]
uses: [probability-density-function@m2, determinant-is-multiplicative@m2, determinant-of-the-inverse@m2]
status: full
check: change-of-variables.check.py
lean: null
sources:
  - {work: papamakarios-2021-normalizing-flows-probabilistic-modeling, locator: "p. 3", note: "eqs. (2) and (3), the volume reading of the determinant, and eqs. (5) and (6) for a composition"}
  - {work: dinh-2014-nice, locator: "p. 1", note: "eq. (1), the log-likelihood of a bijection to a factorized prior"}
---

### Statement

Let $f : \mathbb{R}^D \to \mathbb{R}^D$ be a diffeomorphism, let $\mathbf{z} = f(\mathbf{x})$ and let $\mathbf{z}$ have the density $p_z$.
Then $\mathbf{x} = f^{-1}(\mathbf{z})$ has the density

$$ \log p_x(\mathbf{x}) = \log p_z\big(f(\mathbf{x})\big) + \log \left| \det \frac{\partial f}{\partial \mathbf{x}} \right| . $$

For a composition $f = f_L \circ \cdots \circ f_1$ with $\mathbf{h}_0 = \mathbf{x}$ and $\mathbf{h}_k = f_k(\mathbf{h}_{k-1})$, the log-determinant is the sum of the layers' log-determinants, $\sum_{k=1}^{L} \log |\det \partial f_k / \partial \mathbf{h}_{k-1}|$, and the inverse is $f_1^{-1} \circ \cdots \circ f_L^{-1}$.

### Assumptions

1. $f$ and $f^{-1}$ are continuously differentiable, so the change-of-variables theorem for integrals applies. Used in step 1.
2. Every $f_k$ is a diffeomorphism. Used in step 3.

### Steps

1. For any Borel set $A$, $\Pr(\mathbf{x} \in A) = \Pr(\mathbf{z} \in f(A)) = \int_{f(A)} p_z(\mathbf{z})\, d\mathbf{z}$. The substitution $\mathbf{z} = f(\mathbf{x})$ turns this into $\int_A p_z(f(\mathbf{x}))\, |\det J_f(\mathbf{x})|\, d\mathbf{x}$, so the integrand is the density of $\mathbf{x}$.
2. Taking logarithms gives the statement. Since $J_{f^{-1}}(\mathbf{z}) = J_f(\mathbf{x})^{-1}$ and $\det(A^{-1}) = 1 / \det A$, the same density is $p_z(\mathbf{z})\, |\det J_{f^{-1}}(\mathbf{z})|^{-1}$, the form used when the model is written as the sampler $\mathbf{x} = g(\mathbf{z})$, $g = f^{-1}$.
3. By the chain rule $J_f(\mathbf{x}) = J_{f_L}(\mathbf{h}_{L-1}) \cdots J_{f_1}(\mathbf{h}_0)$, and the determinant of a product is the product of the determinants, so the logarithm of its absolute value is the sum over layers. The inverse of a composition is the composition of the inverses in reverse order.

### Result and check

$$ \log p_x(\mathbf{x}) = \log p_z(\mathbf{h}_L) + \sum_{k=1}^{L} \log \left| \det \frac{\partial \mathbf{h}_k}{\partial \mathbf{h}_{k-1}} \right| . $$

The check `change-of-variables.check.py` integrates the density of a random four-layer coupling flow in the plane on a grid and gets 1, compares the histogram of samples $f^{-1}(\mathbf{z})$ with the density, and compares the sum of the layers' log-determinants with $\log|\det|$ of the finite-difference Jacobian of the whole stack.
