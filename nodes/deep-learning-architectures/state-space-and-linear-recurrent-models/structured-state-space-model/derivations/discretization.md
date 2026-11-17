---
id: discretization
derives: [c-zoh, c-bilinear, c-stability, c-rules-agree, eq-zoh, eq-bilinear]
kind: identity
assumes: ["x'(t) = A x(t) + B u(t) with A an N by N matrix", "the input is sampled at step Δ > 0", "for zero-order hold, u is constant on each interval [kΔ, (k+1)Δ)", "for the bilinear rule, the integral of the right-hand side over one step is replaced by the trapezoid rule"]
uses: [matrix-exponential@m2, eigenvalue@m2, eigendecomposition@m2]
status: full
check: discretization.check.py
lean: null
sources:
  - {work: gu-2020-hippo, locator: "page 21"}
  - {work: gu-2021-structured-state-spaces-s4, locator: "Section 2.3"}
  - {work: gu-2023-mamba, locator: "Section 2"}
---

### Statement

For $x'(t) = A x(t) + B u(t)$ sampled at step $\Delta$, with $x_k = x(k\Delta)$:

1. Zero-order hold, with $u$ constant on each step, is exact and gives $x_k = \bar A x_{k-1} + \bar B u_k$ with
$$ \bar A = e^{\Delta A}, \qquad \bar B = \Big(\int_0^\Delta e^{sA}\, ds\Big) B = A^{-1}(e^{\Delta A} - I) B \ \text{ when } A \text{ is invertible}. $$
For a scalar $a \ne 0$: $\bar a = e^{\Delta a}$ and $\bar b = (e^{\Delta a} - 1)\, b / a$.
2. The bilinear rule (trapezoid rule on the state) gives
$$ \bar A = (I - \tfrac{\Delta}{2} A)^{-1}(I + \tfrac{\Delta}{2} A), \qquad \bar B = (I - \tfrac{\Delta}{2} A)^{-1} \Delta B . $$
3. If every eigenvalue $\lambda$ of $A$ has $\operatorname{Re}\lambda < 0$, then every eigenvalue of $\bar A$ has modulus less than 1 under both rules, for every $\Delta > 0$. The bilinear rule maps the imaginary axis to the unit circle and the right half-plane outside it.
4. For a scalar $a$, the bilinear $\bar a$ minus the ZOH $\bar a$ is $(\Delta a)^3/12 + O(\Delta^4)$.

### Assumptions

1. The system is linear and time-invariant: $A$ and $B$ do not depend on $t$. Used in all steps.
2. Zero-order hold: $u(t) = u_k$ on the interval that ends at $k\Delta$. Used in step 2.
3. The bilinear rule approximates $\int_{t}^{t+\Delta} A x(s)\, ds$ by $\tfrac{\Delta}{2} A (x(t) + x(t+\Delta))$ and the input term by $\Delta B u$. Used in step 4.
4. $I - \tfrac{\Delta}{2} A$ is invertible; this holds when no eigenvalue of $A$ equals $2/\Delta$, in particular when $\operatorname{Re}\lambda < 0$ for all eigenvalues. Used in steps 4 and 6.

### Steps

1. Multiply the equation by $e^{-tA}$: $\frac{d}{dt}\big(e^{-tA} x(t)\big) = e^{-tA}\big(x'(t) - A x(t)\big) = e^{-tA} B u(t)$. Integrate from $t_0$ to $t_0 + \Delta$:
$$ x(t_0 + \Delta) = e^{\Delta A} x(t_0) + \int_0^{\Delta} e^{(\Delta - s) A} B\, u(t_0 + s)\, ds . $$
2. With $u$ constant and equal to $u_k$ on the step (assumption 2), the integral is $\big(\int_0^\Delta e^{rA} dr\big) B u_k$ after the change of variable $r = \Delta - s$. This gives $\bar A$ and the first form of $\bar B$ of statement 1, with no approximation beyond the hold.
3. When $A$ is invertible, $\frac{d}{dr} A^{-1} e^{rA} = e^{rA}$, so $\int_0^\Delta e^{rA} dr = A^{-1}(e^{\Delta A} - I)$. For a scalar, $\bar b = (e^{\Delta a} - 1) b / a$, which tends to $\Delta b$ as $a \to 0$. HiPPO (Appendix B.3) and Mamba (Section 2, equation 4) write the same formula; Mamba writes $\bar B = (\Delta A)^{-1}(e^{\Delta A} - I) \cdot \Delta B$, the same matrix.
4. The bilinear rule (assumption 3): $x_k - x_{k-1} = \tfrac{\Delta}{2} A (x_{k-1} + x_k) + \Delta B u_k$. Collect $x_k$: $(I - \tfrac{\Delta}{2} A) x_k = (I + \tfrac{\Delta}{2} A) x_{k-1} + \Delta B u_k$, and invert the left factor (assumption 4). This is equation (3) of S4 and the bilinear update of HiPPO (Appendix B.3).
5. Stability under zero-order hold: if $A = V \Lambda V^{-1}$, then $e^{\Delta A} = V e^{\Delta\Lambda} V^{-1}$, with eigenvalues $e^{\Delta\lambda}$ and $|e^{\Delta\lambda}| = e^{\Delta \operatorname{Re}\lambda} < 1$. Without diagonalizability, the eigenvalues of $e^{\Delta A}$ are still $e^{\Delta\lambda}$ (spectral mapping), so the bound holds.
6. Stability under the bilinear rule: $\bar A$ is a rational function of $A$, so its eigenvalues are $\mu = (1 + z)/(1 - z)$ with $z = \Delta\lambda/2$. Write $z = p + i q$: $|1 + z|^2 - |1 - z|^2 = 4p$. So $|\mu| < 1$ exactly when $p < 0$, $|\mu| = 1$ when $p = 0$, and $|\mu| > 1$ when $p > 0$. Since $\Delta > 0$, the sign of $p$ is that of $\operatorname{Re}\lambda$, for every step size.
7. Order of agreement: $e^{z} = 1 + z + z^2/2 + z^3/6 + O(z^4)$ and $(1 + z/2)/(1 - z/2) = (1 + z/2)(1 + z/2 + z^2/4 + z^3/8 + \dots) = 1 + z + z^2/2 + z^3/4 + O(z^4)$. The bilinear value minus the exponential is $z^3/4 - z^3/6 = z^3/12$, plus $O(z^4)$, with $z = \Delta a$: both rules agree to second order in $\Delta$.

### Result and check

$$ \bar A_{\text{ZOH}} = e^{\Delta A}, \quad \bar B_{\text{ZOH}} = A^{-1}(e^{\Delta A} - I) B, \qquad \bar A_{\text{bil}} = (I - \tfrac{\Delta}{2}A)^{-1}(I + \tfrac{\Delta}{2}A), \quad \bar B_{\text{bil}} = (I - \tfrac{\Delta}{2}A)^{-1}\Delta B . $$

The check `discretization.check.py` integrates the ODE with a fine Runge-Kutta solver under a piecewise constant input and compares it with the ZOH recurrence, checks the closed form of $\bar B$ against the Van Loan block exponential, compares the scalar formulas with the analytic values, maps random stable, marginal and unstable eigenvalues through the bilinear rule for step sizes from $10^{-3}$ to $10^{4}$, and measures the $z^3/12$ gap.
