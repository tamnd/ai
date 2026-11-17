---
id: euler-resnet
derives: [c-euler-resnet, c-euler-order]
kind: identity
assumes: ["f is continuously differentiable in h and t", "the time steps are equal, dt = T / N"]
uses: []
status: full
check: euler-resnet.check.py
lean: null
sources:
  - {work: chen-2018-neural-ordinary-differential-equations, locator: "Section 1"}
  - {work: haber-2017-stable-architectures-deep-neural-networks, locator: "Section 2"}
---

### Statement

1. One step of Euler's method on eq-ode with the step $\Delta t$ is $h_{k+1} = h_k + \Delta t\, f(h_k, t_k)$: a residual block whose residual branch is $\Delta t\, f$. With $\Delta t = 1$ and a separate field per step it is the residual network of eq-euler.
2. For a linear field $f(h) = A h$, one step of Euler's method multiplies $h$ by $I + \Delta t A$ and one step of the classical Runge-Kutta method by $\sum_{j=0}^{4} (\Delta t A)^j / j!$, the Taylor polynomial of degree 4 of $e^{\Delta t A}$. The error at $T$ is of the order $\Delta t$ for Euler and $\Delta t^4$ for Runge-Kutta: halving the step divides it by about 2 and 16.

### Assumptions

1. The solver takes $N$ equal steps from $0$ to $T$, $\Delta t = T / N$, $t_k = k \Delta t$. Used in all steps.
2. The field is linear, $f(h) = A h$, in statement 2 only.

### Steps

1. Euler's method replaces $dh/dt$ on $[t_k, t_{k+1}]$ by its value at the left end, $h_{k+1} = h_k + \Delta t\, f(h_k, t_k)$. A residual block computes $h + F(h)$; with $F = \Delta t\, f(\cdot, t_k)$ the two agree. With $\Delta t = 1$ and parameters $\theta_k$ for each step, $h_{k+1} = h_k + f(h_k, \theta_k)$, which is eq-euler.
2. For $f(h) = A h$ the Euler step is $h + \Delta t A h = (I + \Delta t A) h$. In the Runge-Kutta step, with $M = \Delta t A$: $\Delta t k_1 = M h$, $\Delta t k_2 = M(h + \tfrac12 M h) = (M + \tfrac12 M^2) h$, $\Delta t k_3 = (M + \tfrac12 M^2 + \tfrac14 M^3) h$, $\Delta t k_4 = (M + M^2 + \tfrac12 M^3 + \tfrac14 M^4) h$. Then $h + \tfrac16 \Delta t (k_1 + 2 k_2 + 2 k_3 + k_4) = (I + M + \tfrac12 M^2 + \tfrac16 M^3 + \tfrac1{24} M^4) h$.
3. The exact solution is $h(T) = e^{TA} h(0) = (e^{\Delta t A})^N h(0)$. One step of a method that matches the Taylor series of $e^{\Delta t A}$ through $\Delta t^p$ makes an error of the order $\Delta t^{p+1}$; over $N = T / \Delta t$ steps the errors add up to the order $\Delta t^p$, with $p = 1$ for Euler and $p = 4$ for Runge-Kutta.

### Result and check

$$ h_{k+1} = h_k + \Delta t\, f(h_k, t_k), \qquad \lVert h_N - h(T) \rVert = O(\Delta t) \text{ (Euler)}, \quad O(\Delta t^4) \text{ (RK4)} $$

The check `euler-resnet.check.py` compares one Euler step of the NumPy reference with a residual block, compares one Runge-Kutta step on a linear field with the Taylor polynomial, and measures the ratio of the errors at $T$ when the step halves.
