---
id: adjoint
derives: [c-adjoint, c-adjoint-theta, c-adjoint-memory]
kind: gradient
assumes: ["f is continuously differentiable in h and theta and continuous in t", "the loss L depends on the trajectory only through h(T)"]
uses: []
status: full
check: adjoint.check.py
lean: null
sources:
  - {work: chen-2018-neural-ordinary-differential-equations, locator: "Section 2"}
---

### Statement

Write $a(t) = \partial L / \partial h(t)$ for the adjoint, a row vector.

1. The adjoint solves $da/dt = -a(t)\, \partial f / \partial h$ backwards from $a(T) = \partial L / \partial h(T)$, and $\partial L / \partial h(0) = a(0)$ (eq-adjoint).
2. The gradient of the parameters is $\partial L / \partial \theta = \int_0^T a(t)\, \partial f(h(t), t, \theta) / \partial \theta \, dt$ (eq-adjoint-theta).
3. Solving the three equations for $h$, $a$ and the running integral together, from $T$ back to $0$, needs $h(T)$ but no state of the forward solve, so the memory is $2 B D + P$ numbers for any number of steps; backpropagation through the steps of the classical Runge-Kutta method keeps the $4 N$ stages of the forward pass, $4 N B D$ numbers.

### Assumptions

1. The flow $h(t) \mapsto h(t + \varepsilon)$ is differentiable, $h(t + \varepsilon) = h(t) + \varepsilon f(h(t), t, \theta) + O(\varepsilon^2)$. Used in steps 1 and 2.
2. Solutions of eq-ode are unique, so that $h$ can be solved backwards from $h(T)$. Used in step 3.

### Steps

1. $L$ depends on $h(t)$ only through $h(t + \varepsilon)$, so by the chain rule $a(t) = a(t + \varepsilon)\, \partial h(t + \varepsilon) / \partial h(t) = a(t + \varepsilon) (I + \varepsilon\, \partial f / \partial h) + O(\varepsilon^2)$. Then $(a(t + \varepsilon) - a(t)) / \varepsilon = -a(t + \varepsilon)\, \partial f / \partial h + O(\varepsilon)$, and the limit $\varepsilon \to 0$ is statement 1.
2. Append $\theta$ to the state with $d\theta/dt = 0$. The augmented field has the Jacobian rows $[\partial f / \partial h, \partial f / \partial \theta]$ for $h$ and zero rows for $\theta$, so statement 1 applied to the augmented state gives $d a_\theta / dt = -a(t)\, \partial f / \partial \theta$ with $a_\theta(T) = 0$, since $L$ does not depend on $\theta$ except through $h$. Integrating from $T$ to $0$, $\partial L / \partial \theta = a_\theta(0) = \int_0^T a\, \partial f / \partial \theta \, dt$, which is statement 2.
3. The right-hand sides at time $t$ need $h(t)$, $a(t)$ and $\theta$. By assumption 2, $h(t)$ is the solution of eq-ode backwards from $h(T)$, so it is solved with $a$ rather than stored. The state of the backward solve is $h$ and $a$, $B D$ numbers each, and the integral, $P$ numbers; the solver keeps a fixed number of copies of it per step, whatever $N$ is. Backpropagation instead differentiates each stage of each step and needs the input of each of the $4N$ stages.

### Result and check

$$ \frac{da}{dt} = -a\, \frac{\partial f}{\partial h}, \qquad \frac{\partial L}{\partial \theta} = \int_0^T a\, \frac{\partial f}{\partial \theta} \, dt $$

The check `adjoint.check.py` compares the adjoint gradient of the NumPy reference with central finite differences of the loss of a finely solved forward pass, checks statement 1 on a linear field, where $a(0) = a(T) e^{TA}$, and records the size of the state of the backward solve.
