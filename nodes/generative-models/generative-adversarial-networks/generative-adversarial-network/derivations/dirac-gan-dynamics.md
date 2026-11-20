---
id: dirac-gan-dynamics
derives: [c-msch-dirac]
kind: closed-form
assumes: ["the Dirac-GAN: the generator is the point theta, the discriminator D(x) = psi x, the data the point 0", "L(theta, psi) = f(psi theta) + f(0) with f differentiable and f'(0) != 0"]
uses: [eigenvalue@m2, saddle-point-of-a-function-on-a-product@m2, stochastic-gradient-descent@m2]
status: full
check: dirac-gan-dynamics.check.py
lean: null
sources:
  - {work: mescheder-2018-which-training-methods-gans-converge, locator: "pp. 3, 4, 6", note: "Definition 2.1, Lemmas 2.2 to 2.4, eq 7 and Lemma 3.3"}
---

### Statement

The generator descends $L$ in $\theta$ and the discriminator ascends it in $\psi$, so training follows the vector field $v(\theta, \psi) = \big(-\partial_\theta L, \partial_\psi L\big) = \big(-f'(\psi\theta)\,\psi,\; f'(\psi\theta)\,\theta\big)$.

1. The only equilibrium is $\theta = \psi = 0$, and the Jacobian of $v$ there has the eigenvalues $\pm f'(0)\, i$.
2. Along the integral curves of $v$, $\theta^2 + \psi^2$ is constant: the continuous dynamics circle the equilibrium and never reach it.
3. Simultaneous gradient steps of size $h$, $F_h = \mathrm{id} + h v$, have at the equilibrium the Jacobian $I + hJ$ with the eigenvalues $1 \pm h f'(0)\, i$, of absolute value $\sqrt{1 + h^2 f'(0)^2} > 1$: near the equilibrium each step multiplies the distance to it by that factor.
4. With the penalty $\tfrac{\gamma}{2}\psi^2$ subtracted from the discriminator's objective, which is R1 on this example, the Jacobian is $\begin{pmatrix} 0 & -f'(0) \\ f'(0) & -\gamma \end{pmatrix}$ with eigenvalues $-\tfrac{\gamma}{2} \pm \sqrt{\tfrac{\gamma^2}{4} - f'(0)^2}$, whose real parts are negative for every $\gamma > 0$; they are real for $\gamma \ge 2|f'(0)|$.

For the original GAN, $f(t) = -\log(1 + e^{-t})$ and $f'(0) = \tfrac12$.

### Assumptions

1. $f$ is differentiable with $f'(0) \ne 0$. Used in steps 1 to 4.
2. The data distribution is the point mass at 0, so the real-data term $f(-D(0)) = f(0)$ does not depend on $\psi$. Used in step 1.

### Steps

1. $\partial_\theta L = f'(\psi\theta)\psi$ and $\partial_\psi L = f'(\psi\theta)\theta$. Both vanish only where $\psi = \theta = 0$, since $f'(\psi\theta) = f'(0) \ne 0$ when either is 0 and the other must then vanish; at that point $J = \begin{pmatrix} 0 & -f'(0) \\ f'(0) & 0 \end{pmatrix}$, whose characteristic polynomial $\lambda^2 + f'(0)^2$ has the roots $\pm f'(0) i$. This is statement 1.
2. $\tfrac{d}{dt}(\theta^2 + \psi^2) = 2\theta(-f'\psi) + 2\psi(f'\theta) = 0$. This is statement 2.
3. $D F_h = I + h\,Dv$; the eigenvalues of $I + hJ$ are $1 + h\lambda$, and $|1 \pm h f'(0) i|^2 = 1 + h^2 f'(0)^2$. Because $J = f'(0) R$ with $R$ the rotation by a right angle, $I + hJ$ is $\sqrt{1 + h^2 f'(0)^2}$ times a rotation, so it scales every vector by that factor. This is statement 3.
4. The penalty adds $-\gamma\psi$ to the $\psi$ component of $v$ and $-\gamma$ to the lower right entry of $J$; the characteristic polynomial becomes $\lambda^2 + \gamma\lambda + f'(0)^2$, with the roots of statement 4. If $\gamma^2 / 4 < f'(0)^2$ the real part is $-\gamma / 2$; otherwise both roots are real and negative because their product $f'(0)^2$ is positive and their sum $-\gamma$ is negative. This is statement 4.

### Result and check

$$ \lambda_{1,2} = \pm f'(0)\, i, \qquad |1 \pm h f'(0)\, i| = \sqrt{1 + h^2 f'(0)^2}, \qquad \lambda_{1,2}^{\gamma} = -\frac{\gamma}{2} \pm \sqrt{\frac{\gamma^2}{4} - f'(0)^2} . $$

The check `dirac-gan-dynamics.check.py` differentiates the vector field and the update operator of the NumPy reference's `dirac_gan` by finite differences at the equilibrium and compares their eigenvalues with statements 1, 3 and 4, integrates the continuous dynamics with small Runge-Kutta steps to check statement 2, and runs the reference near the equilibrium to check the growth factor of statement 3.
