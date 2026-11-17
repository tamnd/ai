---
id: one-dimensional-flow
derives: [c-flow-order, c-g1d, c-residual-g1d, c-augment-g1d]
kind: proof
assumes: ["f is Lipschitz continuous in h and continuous in t", "d = D = 1 for the plain model"]
uses: []
status: full
check: one-dimensional-flow.check.py
lean: null
sources:
  - {work: dupont-2019-augmented-neural-odes, locator: "Section 3"}
  - {work: dupont-2019-augmented-neural-odes, locator: "Section 5"}
---

### Statement

Let $\phi_T$ be the map $h(0) \mapsto h(T)$ of eq-ode in one dimension, and $g(-1) = 1$, $g(1) = -1$.

1. $\phi_T$ is strictly increasing: $x_1 < x_2$ gives $\phi_T(x_1) < \phi_T(x_2)$.
2. No such flow represents $g$. With the loss $L = \frac{1}{2} \cdot \frac{1}{2} \big( (\phi_T(-1) - 1)^2 + (\phi_T(1) + 1)^2 \big)$, every flow has $L > \frac12$.
3. One residual step represents $g$: $h + F(h)$ with $F(h) = -2h$ maps $-1$ to $1$ and $1$ to $-1$.
4. With one dimension of augmentation, $D = 2$, the linear field $f(h) = \pi \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix} h$ over $T = 1$ maps $(x, 0)$ to $(-x, 0)$, and so represents $g$ on its first coordinate.

### Assumptions

1. Solutions of eq-ode exist and are unique (the Picard-Lindelöf theorem), and depend continuously on $t$. Used in step 1.

### Steps

1. Let $x_1 < x_2$ and $e(t) = h_2(t) - h_1(t)$, continuous with $e(0) > 0$. If $e(T) \le 0$, the intermediate value theorem gives $t^*$ with $e(t^*) = 0$. Then $h_1$ and $h_2$ are two solutions with the same value at $t^*$, so by uniqueness they are equal for all $t$, and $x_1 = x_2$, a contradiction. So $e(T) > 0$.
2. Write $u = \phi_T(-1)$ and $v = \phi_T(1)$, so $u < v$ by statement 1. On the closed set $u \le v$, the function $(u - 1)^2 + (v + 1)^2$ has its unconstrained minimum $(1, -1)$ outside, so its minimum is on the line $u = v$, where it is $2u^2 + 2 \ge 2$. Hence $(u - 1)^2 + (v + 1)^2 \ge 2$, with equality only at $u = v = 0$, which $u < v$ excludes: $L > \frac{1}{4} \cdot 2 = \frac12$.
3. $-1 + 2 = 1$ and $1 - 2 = -1$. The discrete step is not bound by step 1, because nothing forces its map to be continuous in time: Euler steps with $1 + \Delta t\, \partial f / \partial h < 0$ reverse order.
4. The solution of $dh/dt = \pi R h$ is $e^{\pi R t} h(0)$, and $e^{\theta R}$ is the rotation by the angle $\theta$, so $e^{\pi R} = -I$ and $(x, 0) \mapsto (-x, 0)$. In two dimensions the trajectories of $-1$ and $1$ pass each other on half-circles instead of crossing.

### Result and check

$$ x_1 < x_2 \implies \phi_T(x_1) < \phi_T(x_2), \qquad \inf_{\phi_T} L = \tfrac12 $$

The check `one-dimensional-flow.check.py` solves random one-dimensional fields of the NumPy reference and checks that order is kept, trains the plain model on $g$ and checks that its loss stays above $\frac12$, and solves the rotation with the Runge-Kutta solver.
