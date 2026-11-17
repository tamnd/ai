---
id: quadratic-stability
derives: [c-quad-modes, c-quad-threshold, c-quad-example, c-quad-momentum]
kind: closed-form
assumes: ["the objective is 0.5 x^T H x with H symmetric positive definite", "a fixed step size eta > 0", "full-batch gradient descent, with no noise"]
uses: []
status: full
check: quadratic-stability.check.py
lean: null
sources:
  - {work: cohen-2021-gradient-descent-edge-stability, locator: "p. 2"}
  - {work: cohen-2021-gradient-descent-edge-stability, locator: "p. 3"}
---

### Statement

Let $f(x) = \tfrac12 x^\top H x$ with $H$ symmetric positive definite, eigenvalues $\lambda_1 \ge \dots \ge \lambda_d > 0$ and orthonormal eigenvectors $q_1, \dots, q_d$, and run gradient descent $x_{t+1} = x_t - \eta \nabla f(x_t)$ with a fixed $\eta > 0$.
Then (i) $\langle q_i, x_t\rangle = (1 - \eta\lambda_i)^t \langle q_i, x_0\rangle$ for every $i$ and $t$;
(ii) $x_t \to 0$ from every start if and only if $\eta\lambda_1 < 2$, that is $\eta < 2/\lambda_1$; at $\eta\lambda_1 = 2$ the top coordinate flips sign with constant size, and above it grows geometrically while flipping sign;
(iii) for $\lambda_1 = 20$ and $\lambda_2 = 1$, the top factor is $-0.8$ at $\eta = 0.09$ and $-1.2$ at $\eta = 0.11$, so after 100 steps the top coordinate is multiplied by $0.8^{100} \approx 2.04 \cdot 10^{-10}$ in the first case and by $1.2^{100} \approx 8.28 \cdot 10^{7}$ in the second;
(iv) with heavy-ball (Polyak) momentum $x_{t+1} = x_t - \eta\nabla f(x_t) + \beta(x_t - x_{t-1})$, $0 \le \beta < 1$, the iterates converge if and only if $\eta\lambda_1 < 2 + 2\beta$, and with Nesterov momentum $x_{t+1} = y_t - \eta\nabla f(y_t)$, $y_t = x_t + \beta(x_t - x_{t-1})$, if and only if $\eta\lambda_1 (1 + 2\beta) < 2 + 2\beta$.

### Assumptions

1. $H$ is symmetric with all eigenvalues positive; for a zero eigenvalue the factor is 1 and that coordinate never moves.
2. The step size is fixed, and the gradient is exact.
3. For (iv), $0 \le \beta < 1$.

### Steps

1. The gradient is $Hx$, so $x_{t+1} = (I - \eta H)x_t$. The matrix $I - \eta H$ has the eigenvectors $q_i$ with eigenvalues $1 - \eta\lambda_i$, so projecting on $q_i$ gives $\langle q_i, x_{t+1}\rangle = (1 - \eta\lambda_i)\langle q_i, x_t\rangle$, and induction gives (i).
2. A coordinate goes to zero from every start exactly when $|1 - \eta\lambda_i| < 1$, which for $\lambda_i > 0$ is $0 < \eta\lambda_i < 2$. The condition is hardest for the largest eigenvalue, so all coordinates converge if and only if $\eta\lambda_1 < 2$. At $\eta\lambda_1 = 2$ the factor is $-1$; above, it is below $-1$, so the coordinate alternates in sign and its size grows by $|1 - \eta\lambda_1| > 1$ per step. This is (ii).
3. With $\lambda_1 = 20$: $1 - 0.09 \cdot 20 = -0.8$ and $1 - 0.11 \cdot 20 = -1.2$; the second coordinate has factors $0.91$ and $0.89$ and converges in both runs. Then $0.8^{100} = e^{100\ln 0.8} = 2.037 \cdot 10^{-10}$ and $1.2^{100} = 8.282 \cdot 10^{7}$, which is (iii).
4. With momentum each eigen-coordinate $z_t$ follows a second-order recurrence. Heavy ball: $z_{t+1} = (1 + \beta - \eta\lambda) z_t - \beta z_{t-1}$. Nesterov: $z_{t+1} = (1 - \eta\lambda)(1 + \beta) z_t - (1 - \eta\lambda)\beta z_{t-1}$. A recurrence $z_{t+1} = b z_t - c z_{t-1}$ decays from every start if and only if both roots of $r^2 - b r + c$ lie inside the unit circle, which (Jury's test) is $|c| < 1$ and $|b| < 1 + c$.
5. Heavy ball: $c = \beta < 1$; $|1 + \beta - \eta\lambda| < 1 + \beta$ is $0 < \eta\lambda < 2 + 2\beta$.
6. Nesterov: $c = (1 - \eta\lambda)\beta$. The binding side of $|b| < 1 + c$ for large $\eta\lambda$ is $(\eta\lambda - 1)(1 + \beta) < 1 + (1 - \eta\lambda)\beta$, which rearranges to $\eta\lambda(1 + 2\beta) < 2 + 2\beta$; at that point $|c| = (\eta\lambda - 1)\beta < 1$ holds as well. This is (iv), with the thresholds $\mathrm{MSS}_{\text{Polyak}} = (2 + 2\beta)/\eta$ and $\mathrm{MSS}_{\text{Nesterov}} = (2 + 2\beta)/(\eta(1 + 2\beta))$.

### Result and check

On a quadratic, gradient descent with step size $\eta$ is stable exactly when the sharpness $\lambda_1$ is below $2/\eta$, and the unstable direction is the top eigenvector, along which the iterate flips sign and grows.
Cohen et al. (2021) use this model to define the threshold $2/\eta$ and the maximum stable sharpness with momentum (pp. 2-3); the momentum thresholds here match their Equation 1.
`quadratic-stability.check.py` runs gradient descent on random positive definite matrices at $0.99$ and $1.01$ times $2/\lambda_1$ and compares every iterate with (i), recomputes the numbers of (iii), and runs both momentum methods at $0.98$ and $1.02$ times their thresholds for several $\beta$, together with the roots of the characteristic polynomials.
