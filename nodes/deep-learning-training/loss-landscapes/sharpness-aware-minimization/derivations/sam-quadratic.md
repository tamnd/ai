---
id: sam-quadratic
derives: [c-quad-sam-grad, c-quad-penalty, c-quad-eigen, c-quad-no-rest, c-quad-cycle, c-valley]
kind: closed-form
assumes: ["L(w) = w^T H w / 2 with H symmetric", "p = 2 and the SAM gradient drops the second-order terms", "Hw != 0"]
uses: [dual-norm-perturbation]
status: full
check: sam-quadratic.check.py
lean: null
sources:
  - {work: foret-2021-sharpness-aware-minimization, locator: "p. 4"}
---

### Statement

Let $L(w) = \tfrac12 w^\top H w$ with $H$ symmetric, $\rho > 0$, and $Hw \ne 0$.
With $p = 2$ the SAM gradient $\nabla L(w + \hat\epsilon(w))$ is

$$ g_{\mathrm{SAM}}(w) = H\Big(w + \rho \frac{Hw}{\lVert Hw \rVert}\Big) = Hw + \rho \frac{H^2 w}{\lVert Hw \rVert} . $$

Then (i) $g_{\mathrm{SAM}}$ is the exact gradient of $L(w) + \rho \lVert Hw \rVert$;
(ii) in an orthonormal eigenbasis $H = \sum_i \lambda_i v_i v_i^\top$, with $c_i = v_i^\top w$, the $i$-th component of $g_{\mathrm{SAM}}$ is $\lambda_i c_i (1 + \rho \lambda_i / \lVert Hw \rVert)$, so SAM multiplies the gradient along each eigendirection by a factor that grows with its curvature;
(iii) if $H$ is positive definite, $g_{\mathrm{SAM}}(w) \ne 0$ at every $w$ with $Hw \ne 0$, so the update never comes to rest away from the minimum;
(iv) in one dimension with $0 < \eta\lambda < 2$ the update $w \leftarrow w - \eta\lambda(w + \rho\,\operatorname{sign} w)$ ends in the 2-cycle $\pm\, \eta\lambda\rho / (2 - \eta\lambda)$;
(v) on the valley $L(a, b) = \tfrac12 (ab - 1)^2$, every minimum $ab = 1$ has Hessian eigenvalues $0$ and $a^2 + b^2 \ge 2$, with equality only at $a = b = \pm 1$.

### Assumptions

1. $L$ is the quadratic above, or for (v) the valley; $H$ is symmetric, and positive definite for (iii) and (iv).
2. The perturbation is the $p = 2$ case of `dual-norm-perturbation`, $\hat\epsilon = \rho\, \nabla L / \lVert \nabla L \rVert$, and the SAM gradient is $\nabla L$ at $w + \hat\epsilon$ with the second-order terms dropped.

### Steps

1. $\nabla L(w) = Hw$, so $\hat\epsilon = \rho Hw / \lVert Hw \rVert$ and $\nabla L(w + \hat\epsilon) = H(w + \hat\epsilon)$, which expands to the display.
2. $\nabla \lVert Hw \rVert = H^\top Hw / \lVert Hw \rVert = H^2 w / \lVert Hw \rVert$ for symmetric $H$, which gives (i). So on a quadratic the SAM update is exactly gradient descent on the linearized objective $L + \rho \lVert \nabla L \rVert_2$ of `dual-norm-perturbation`.
3. $v_i^\top H w = \lambda_i c_i$ and $v_i^\top H^2 w = \lambda_i^2 c_i$, which gives (ii).
4. If $g_{\mathrm{SAM}} = 0$ with $H$ invertible then $w = -\rho Hw / \lVert Hw \rVert$, so $Hw = -(\lVert Hw \rVert/\rho)\, w$ and $w$ is an eigenvector with a negative eigenvalue, which a positive definite $H$ has not.
5. In one dimension write $s = \eta\lambda$. For $w > 0$ small the update gives $(1 - s)w - s\rho < 0$, and two updates map $w$ to $(1 - s)^2 w + s^2 \rho$, a contraction with fixed point $s^2\rho / (1 - (1 - s)^2) = s\rho / (2 - s)$; the next update sends it to its negative.
6. For the valley, $\nabla L = r(b, a)$ with $r = ab - 1$ and $\nabla^2 L = (b, a)(b, a)^\top + r \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$. At $r = 0$ this has rank one with eigenvalue $a^2 + b^2 \ge 2ab = 2$.

### Result and check

The check confirms the closed form and its equality with a finite-difference gradient of $L + \rho\lVert Hw\rVert$ on random positive definite $H$, the eigen-components, that the gradient is never zero and never shorter than $Hw$ along an eigendirection, that with $H = \mathrm{diag}(10, 0.1)$ and $\eta = 0.01$ the ratio of the sharp to the flat component stays below that of gradient descent at every one of 300 steps, the 2-cycle amplitude for $\eta\lambda \in \{0.1, 0.5, 0.9, 1.5\}$, the valley's Hessian eigenvalues, and that on the valley gradient descent stops at a minimum with top eigenvalue above 8.5 while SAM ends near $ab = 1$ with top eigenvalue below 2.5.
