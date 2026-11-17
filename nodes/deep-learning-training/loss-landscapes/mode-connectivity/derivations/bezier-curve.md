---
id: bezier-curve
derives: [c-bezier-points, c-bend-gradient, c-ring-example]
kind: closed-form
assumes: ["w1, w2 and the bend theta are points of R^p", "t lies in [0, 1]", "the loss L is differentiable where the gradient is used"]
uses: []
status: full
check: bezier-curve.check.py
lean: null
sources:
  - {work: garipov-2018-loss-surfaces-mode-connectivity, locator: "p. 4"}
---

### Statement

Let $\phi_\theta(t) = (1-t)^2 w_1 + 2t(1-t)\,\theta + t^2 w_2$ for $t \in [0, 1]$.
Then $\phi_\theta(0) = w_1$, $\phi_\theta(1) = w_2$ and $\phi_\theta(\tfrac12) = \tfrac14(w_1 + 2\theta + w_2)$.
The Jacobian of $\phi_\theta(t)$ in $\theta$ is $2t(1-t)\,I$, so the gradient of the curve loss $\ell(\theta) = \mathbb{E}_{t \sim U(0,1)} L(\phi_\theta(t))$ is $\mathbb{E}_{t}\,[\,2t(1-t)\,\nabla L(\phi_\theta(t))\,]$.
On the ring loss $L(w) = (\lVert w\rVert^2 - 1)^2$ in $\mathbb{R}^2$ with $w_1 = (-1, 0)$ and $w_2 = (1, 0)$, the segment has barrier 1 at $t = \tfrac12$, while the Bezier curve with $\theta = (0, 2)$ has barrier $\tfrac1{16}$, reached where $t(1-t) = \tfrac18$.

### Assumptions

1. The endpoints and the bend are fixed vectors; only $t$ varies along the curve.
2. For the gradient, $L$ is differentiable at the points of the curve, and the expectation and the gradient can be exchanged (true when $\nabla L$ is bounded on the curve).
3. In the example both endpoints are minima with $L = 0$, so both barrier conventions of the page give the same number.

### Steps

1. At $t = 0$ the weights are $1, 0, 0$, and at $t = 1$ they are $0, 0, 1$, which gives the endpoints.
2. At $t = \tfrac12$ the weights are $\tfrac14, 2 \cdot \tfrac14, \tfrac14$, which gives $\tfrac14(w_1 + 2\theta + w_2)$. At $t = \tfrac14$ they are $\tfrac9{16}, \tfrac6{16}, \tfrac1{16}$.
3. $\phi_\theta(t)$ is affine in $\theta$ with coefficient $2t(1-t)$, so by the chain rule $\nabla_\theta L(\phi_\theta(t)) = 2t(1-t)\nabla L(\phi_\theta(t))$. Averaging over $t$ gives the gradient of $\ell$; one sampled $t$ gives an unbiased estimate of it.
4. The segment $(1-t)w_1 + t w_2 = (2t - 1, 0)$ has $L = ((2t-1)^2 - 1)^2$, which is 0 at the ends and 1 at $t = \tfrac12$.
5. On the curve with $\theta = (0, 2)$, the first coordinate is $-(1-t)^2 + t^2 = 2t - 1$ and the second is $4t(1-t)$. With $s = t(1-t) \in [0, \tfrac14]$, $\lVert\phi\rVert^2 = 1 - 4s + 16s^2$, so $L = (16s^2 - 4s)^2 = (4s(1 - 4s))^2$.
6. $4s(1-4s)$ is largest at $s = \tfrac18$, where it equals $\tfrac14$; so the largest loss on the curve is $\tfrac1{16}$, at $t = (1 \pm \sqrt{1/2})/2$, about 0.146 and 0.854. At $t = \tfrac12$ the curve passes through $(0, 1)$, on the circle, where $L = 0$.

### Result and check

The endpoints, the midpoint and the factor $2t(1-t)$ are exact. They match the curve and the training loss of Garipov et al. (2018), whose bend is updated with one sampled $t$ per step.
The example shows a curved path with a barrier sixteen times lower than the segment between the same two minima.
`bezier-curve.check.py` checks the points at $t = 0, \tfrac14, \tfrac12, 1$ on random vectors, compares $2t(1-t)$ with a finite difference in $\theta$, and evaluates the ring loss on a fine grid along both paths.
